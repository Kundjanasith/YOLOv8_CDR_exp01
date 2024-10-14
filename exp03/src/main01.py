import glob, tqdm, cv2, sys
from ultralytics import YOLO
from shapely.geometry import LineString, Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import math

def flip_polygon_coordinates(polygon_coords, image_width):
    flipped_coords = [(image_width - x, y) for (x, y) in polygon_coords]
    return np.array(flipped_coords)

def divide_polygon_into_sections(polygon_coords, num_sections=12):
    centroid_x, centroid_y = find_polygon_centroid(polygon_coords)
    centroid = Point(centroid_x, centroid_y)
    polygon = Polygon(polygon_coords)
    dividing_lines = []
    angle_increment = 360 / num_sections
    radius = max(polygon.bounds) * 2 
    for i in range(num_sections):
        angle_deg = i * angle_increment
        angle_rad = math.radians(angle_deg)
        end_x = centroid_x + radius * math.cos(angle_rad)
        end_y = centroid_y + radius * math.sin(angle_rad)
        ray = LineString([(centroid_x, centroid_y), (end_x, end_y)])
        intersection = ray.intersection(polygon)
        dividing_lines.append(intersection)
    return dividing_lines

def find_polygon_centroid(polygon_coords):
    polygon = Polygon(polygon_coords)
    centroid = polygon.centroid
    return centroid.x, centroid.y

def find_intersection(start, end, polygon_coords):
    line = LineString([start, end])
    polygon = Polygon(polygon_coords)
    intersection = line.intersection(polygon)
    if intersection.is_empty:
        return None  # No intersection
    else:
        return list(intersection.coords)

def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point2 - point1)
    return distance

def draw_dashed_line(image, start_point, end_point, color, thickness, dash_length=10):
    line_length = np.linalg.norm(np.array(end_point) - np.array(start_point))
    num_dashes = int(line_length // dash_length)
    line_vector = (np.array(end_point) - np.array(start_point)) / line_length
    for i in range(num_dashes):
        dash_start = start_point + i * dash_length * line_vector
        dash_end = start_point + (i + 0.5) * dash_length * line_vector
        cv2.line(image, tuple(dash_start.astype(int)), tuple(dash_end.astype(int)), color, thickness)
    return image

def polygon_to_ellipse(polygon_points):
    polygon_points = np.array(polygon_points[0], dtype=np.int32)
    ellipse = cv2.fitEllipse(polygon_points)
    center, axes, angle = ellipse 
    ellipse_points = []
    num_points = len(polygon_points)
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points  # Parametric angle
        x = int(center[0] + (axes[0] / 2) * np.cos(theta))
        y = int(center[1] + (axes[1] / 2) * np.sin(theta))
        ellipse_points.append([x, y])
    return [np.array(ellipse_points, dtype=np.int32)]

def find_perpendicular_line(l1_start, l1_end, length=None):
    l1_start = np.array(l1_start)
    l1_end = np.array(l1_end)
    direction_l1 = l1_end - l1_start
    direction_l2 = np.array([-direction_l1[1], direction_l1[0]])
    midpoint = (l1_start + l1_end) / 2
    if length is None:
        length = np.linalg.norm(direction_l1)
    direction_l2 = direction_l2 / np.linalg.norm(direction_l2) * (length / 2)
    l2_start = midpoint - direction_l2
    l2_end = midpoint + direction_l2
    return l2_start, l2_end

# file_o = open('results_val.csv','w')
# file_o.write('file_name,left_right\n')
model = YOLO('../../model/train/weights/best.pt') 
input_paths = glob.glob('../../../data/SI/test_si/*.jpg')
input_paths.sort()
for input_image_path in tqdm.tqdm(input_paths):
    results = model(input_image_path,conf=0.001,iou=0.8)
    cup_arr = []
    dis_arr = []
    for result in results:
        if result.masks is None:
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            if box.cls[0] == 0:
                if 0 not in points[0]:
                    cup_arr.append(points[0])
            if box.cls[0] == 1:
                if 0 not in points[0]:
                    dis_arr.append(points[0])
    cup_polygons = [Polygon([tuple(point) for point in coords]) for coords in cup_arr]
    dis_polygons = [Polygon([tuple(point) for point in coords]) for coords in dis_arr]
    result = []
    for dis_index, dis_poly in enumerate(dis_polygons):
        contained_cup_polygons = [red_poly for red_poly in cup_polygons if red_poly.within(dis_poly)]
        if contained_cup_polygons:
            highest_cup_poly = max(contained_cup_polygons, key=lambda p: max(point[1] for point in p.exterior.coords)-min(point[1] for point in p.exterior.coords)) 
            result.append({
                "dis_polygon_index": dis_index,
                "dis_polygon_area": dis_poly.area,
                "largest_cup_polygon": highest_cup_poly,
                "largest_cup_polygon_index": cup_polygons.index(highest_cup_poly),
                "largest_cup_polygon_area": highest_cup_poly.area
            })
    if result:
        largest_dis_poly_result = max(result, key=lambda x: x["dis_polygon_area"])
    else:
        print("No red polygons are contained within any green polygons.")
    cup_arr = [cup_arr[largest_dis_poly_result['largest_cup_polygon_index']]]
    dis_arr = [dis_arr[largest_dis_poly_result['dis_polygon_index']]]
    # print(input_image_path)
    # print(np.array(cup_arr).shape)
    # print(np.array(dis_arr).shape)
    fig, axs = plt.subplots(1, 2) 
    image = cv2.imread(input_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb_predict = image_rgb.copy()
    # print(dis_arr) 
    # if len(cup_arr) > 5:
    print('>>',np.array(cup_arr).shape,np.array(dis_arr).shape)
    if np.array(cup_arr).shape[1] >= 5:
        cup_arr = polygon_to_ellipse(cup_arr)
    if np.array(dis_arr).shape[1] >= 5:
        dis_arr = polygon_to_ellipse(dis_arr)
    z_arr = []
    for i in cup_arr[0]:
        z_arr.append((i[0], i[1]))
    center_x, center_y = find_polygon_centroid(z_arr)
    side = None
    if center_x < image_rgb_predict.shape[0]/2:
        side = 'LEFT'
    else:
        side = 'RIGHT'
    if side == 'LEFT':
        cup_arr = np.array(cup_arr)
        dis_arr = np.array(dis_arr)
        cup_arr = cup_arr.reshape(cup_arr.shape[1], cup_arr.shape[2])
        cup_arr = flip_polygon_coordinates(cup_arr, image_rgb_predict.shape[0])
        cup_arr = cup_arr.reshape(1, cup_arr.shape[0], cup_arr.shape[1])
        dis_arr = dis_arr.reshape(dis_arr.shape[1], dis_arr.shape[2])
        dis_arr = flip_polygon_coordinates(dis_arr, image_rgb_predict.shape[0])
        dis_arr = dis_arr.reshape(1, dis_arr.shape[0], dis_arr.shape[1])
        image_rgb_predict = cv2.flip(image_rgb_predict, 1)

    
    input_image_path = input_image_path.replace('../../../data/SI/','figs/')
    # file_o.write('%s,%s\n'%(input_image_path,side))

    cv2.polylines(image_rgb_predict , cup_arr, isClosed=True, color=(255, 0, 0), thickness=20)
    cv2.polylines(image_rgb_predict , dis_arr, isClosed=True, color=(0, 255, 0), thickness=20)
    axs[0].imshow(image_rgb_predict)
    axs[0].axis('off')
    image_rgb_predict = image_rgb.copy()
    if side == 'LEFT':
        image_rgb_predict = cv2.flip(image_rgb_predict, 1)
    crop_arr = []
    x_crop_arr = []
    y_crop_arr = []
    for i in dis_arr[0]:
        crop_arr.append((i[0], i[1]))
        x_crop_arr.append(i[0])
        y_crop_arr.append(i[1])
    if np.array(cup_arr).shape[1] >= 5:
        cup_arr = polygon_to_ellipse(cup_arr)
    if np.array(dis_arr).shape[1] >= 5:
        dis_arr = polygon_to_ellipse(dis_arr)
    
    # sys.exit()
    

    # image_rgb_predict = image_rgb.copy()
    cv2.polylines(image_rgb_predict , cup_arr, isClosed=True, color=(255, 0, 0), thickness=5)
    cv2.polylines(image_rgb_predict , dis_arr, isClosed=True, color=(0, 255, 0), thickness=5)
    # start_point = (x_crop_arr[np.argmin(y_crop_arr)], np.min(y_crop_arr))
    # end_point = (x_crop_arr[np.argmax(y_crop_arr)], np.max(y_crop_arr))
    # print(start_point, end_point)
    # print(np.array(cup_arr).shape)
    
    # # image_rgb_predict = draw_dashed_line(image_rgb_predict, start_point, end_point, (0, 100, 200), 5, dash_length=1)
    # cup_arr = np.array(cup_arr)
    # intersections = find_intersection(start_point, end_point, cup_arr.reshape(cup_arr.shape[1],cup_arr.shape[2]))
    # # print(intersections)
    # if intersections != None:
    #     image_rgb_predict = draw_dashed_line(image_rgb_predict, start_point, intersections[0], (0, 100, 200), 5, dash_length=1)
    #     image_rgb_predict = draw_dashed_line(image_rgb_predict, intersections[1], end_point, (0, 100, 200), 5, dash_length=1)
    XcropMin, XcropMax = np.min(x_crop_arr), np.max(x_crop_arr)
    YcropMin, YcropMax = np.min(y_crop_arr), np.max(y_crop_arr)
    print('>>||',np.array(dis_arr).shape)
    dis_arr = np.array(dis_arr)
    secs = divide_polygon_into_sections(dis_arr.reshape(dis_arr.shape[1],dis_arr.shape[2]))
    # print(secs[0].coords[0][0])
    # cv2.circle(image_rgb_predict, (int(center_x-5), int(center_y-5)), 10, (100, 100, 100), -1)
    # print(secs[0].coords[0])
    # print(int(secs[0].coords[0][0]), int(secs[0].coords[0][1]))
    tem_arr = ['i','h','g','f','e','d','c','b','a','l','k','j']
    for tem in range(12):
        cv2.line(image_rgb_predict, (int(secs[tem].coords[0][0]), int(secs[tem].coords[0][1])), (int(secs[tem].coords[1][0]), int(secs[tem].coords[1][1])), (100, 100, 100), 1)
        mid_point = (int((secs[tem].coords[0][0] + secs[tem].coords[1][0]) // 2), int((secs[tem].coords[0][1] + secs[tem].coords[1][1]) // 2))
        text = str(tem_arr[tem])
        # Define the font, font scale, and color of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_color = (255, 255, 255)  # White color
        text_thickness = 2
        # Put the text on the image at the calculated midpoint
        cv2.putText(image_rgb_predict, text, mid_point, font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    cropped_image = image_rgb_predict[YcropMin-10:YcropMax+10, XcropMin-10:XcropMax+10, :]
    axs[1].imshow(cropped_image)
    axs[1].axis('off')
    plt.savefig('main01.png',bbox_inches='tight')
    # # print(input_image_path)
    # input_image_path = input_image_path.replace('../../../data/SI/','figs/')
    # if intersections != None:
    #     l1 = calculate_distance(start_point, intersections[0])
    #     l2 = calculate_distance(intersections[1], end_point)
    #     # file_o.write('%s,%f,%f\n'%(input_image_path,l1,l2))
    # plt.savefig(input_image_path,bbox_inches='tight')
    break
print("OK")