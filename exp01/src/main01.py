import glob, tqdm, cv2
from ultralytics import YOLO
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt

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

file_o = open('results_test.csv','w')
file_o.write('file_name,l1,l2\n')
model = YOLO('../../model/train/weights/best.pt') 
for input_image_path in tqdm.tqdm(glob.glob('../../../data/SI/test_si/*.jpg')):
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
    cv2.polylines(image_rgb_predict , cup_arr, isClosed=True, color=(255, 0, 0), thickness=20)
    cv2.polylines(image_rgb_predict , dis_arr, isClosed=True, color=(0, 255, 0), thickness=20)
    # axs[0].imshow(image_rgb_predict)
    # axs[0].axis('off')
    image_rgb_predict = image_rgb.copy()
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

    image_rgb_predict = image_rgb.copy()
    cv2.polylines(image_rgb_predict , cup_arr, isClosed=True, color=(255, 0, 0), thickness=5)
    cv2.polylines(image_rgb_predict , dis_arr, isClosed=True, color=(0, 255, 0), thickness=5)
    #1 
    # print(np.where(y_crop_arr==np.min(y_crop_arr)))
    # print(np.where(x_crop_arr==np.min(x_crop_arr)))
    start_point = (x_crop_arr[np.argmin(y_crop_arr)], np.min(y_crop_arr))

    end_point = (x_crop_arr[np.argmax(y_crop_arr)], np.max(y_crop_arr))
    # print(start_point, end_point)
    image_rgb_predict = draw_dashed_line(image_rgb_predict, start_point, end_point, (0, 100, 200), 5, dash_length=1)
    l2start, l2end = find_perpendicular_line(start_point, end_point)
    # print(l2start, l2end)
    l2start[0] = l2start[0]-20
    l2end[0] = l2end[0]+20
    # print(l2start, l2end)
    XcropMin, XcropMax = np.min(x_crop_arr), np.max(x_crop_arr)
    YcropMin, YcropMax = np.min(y_crop_arr), np.max(y_crop_arr)
    image_rgb_predict = draw_dashed_line(image_rgb_predict, l2start, l2end, (0, 100, 200), 5, dash_length=1)
    cropped_image = image_rgb_predict[YcropMin-10:YcropMax+10, XcropMin-10:XcropMax+10, :]
    # axs[1].imshow(cropped_image)
    # axs[1].axis('off')
    # plt.savefig('main01.png',bbox_inches='tight')
    # print(input_image_path)
    input_image_path = input_image_path.replace('../../../data/SI/','figs/')
    l1 = calculate_distance(start_point, end_point)
    l2 = calculate_distance(l2start, l2end)
    file_o.write('%s,%f,%f\n'%(input_image_path,l1,l2))
    # plt.savefig(input_image_path,bbox_inches='tight')
    # break
print("OK")