from ultralytics import YOLO
import glob, cv2
import matplotlib.pyplot as plt 
import numpy as np 
import sys 
import numpy as np 
import tqdm
from shapely.geometry import Polygon

model = YOLO('../model/train/weights/best.pt') 
input_image_path = sys.argv[1]
output_image_path = sys.argv[2]
input_label_path = sys.argv[3]

def compute_CDR(cup,dis):
    if cup is None or dis is None:
        return None
    cup_y = cup[:,:,1]
    dis_y = dis[:,:,1]
    if len(dis_y[0]) == 0 or len(cup_y[0]) == 0:
        return None
    min_cup_y = np.min(cup_y)
    max_cup_y = np.max(cup_y)
    min_dis_y = np.min(dis_y)
    max_dis_y = np.max(dis_y)
    cdr = (max_cup_y-min_cup_y)/(max_dis_y-min_dis_y)
    return cdr

fig, axs = plt.subplots(2, 1) 
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


## ACTUAL
with open(input_label_path, 'r') as file_label:
    lines = file_label.readlines()
l0 = lines[0].split()[1:]
cup_arr = []
for i in range(0,len(l0)-1,2):
    cup_arr.append([float(l0[i])*image_rgb.shape[0],float(l0[i+1])*image_rgb.shape[1]])
cup_arr = np.array([cup_arr], dtype=np.int32)
l1 = lines[1].split()[1:]
dis_arr = []
for i in range(0,len(l1)-1,2):
    dis_arr.append([float(l1[i])*image_rgb.shape[0],float(l1[i+1])*image_rgb.shape[1]])
dis_arr = np.array([dis_arr], dtype=np.int32)

image_rgb_actual = image_rgb.copy()
cv2.polylines(image_rgb_actual, cup_arr, isClosed=True, color=(255, 0, 0), thickness=20)
cv2.polylines(image_rgb_actual, dis_arr, isClosed=True, color=(0, 255, 0), thickness=20)
axs[0].imshow(image_rgb_actual)
axs[0].axis('off')
cdr = compute_CDR(cup_arr,dis_arr)
axs[0].set_title('Actual : CDR=%.2f'%cdr)

## PREDICT
image_rgb_predict = image_rgb.copy()
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
cv2.polylines(image_rgb_predict , cup_arr, isClosed=True, color=(255, 0, 0), thickness=20)
cv2.polylines(image_rgb_predict , dis_arr, isClosed=True, color=(0, 255, 0), thickness=20)
axs[1].imshow(image_rgb_predict)
axs[1].axis('off')
cdr = compute_CDR(np.array(cup_arr),np.array(dis_arr))
axs[1].set_title('Predict : CDR=%.2f'%cdr)
plt.savefig(output_image_path,bbox_inches='tight')    