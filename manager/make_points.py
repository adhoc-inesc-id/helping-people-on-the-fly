import cv2
import yaml
import numpy as np
from utils import planar_homography
from utils.color_segmentation import find_segmented_centers, detect_blobs_centers_of_mass, ColorSegmentation

room_clean = cv2.imread("../room_clean.jpg")
def closest_node(point, centers):
    node = None
    smallest = np.inf
    for n, center in enumerate(centers):
        distance = np.linalg.norm((center-point), 2)
        if distance < smallest:
            smallest = distance
            node = n
    return node

with open('config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
hue = np.array(config["color segmentation"]["hue"])
sat = np.array(config["color segmentation"]["sat"])
val = np.array(config["color segmentation"]["val"])
graph_node_centers_homography_camera_referential = np.array(config["graph nodes camera points"])
graph_node_centers_homography_real_world_referential = []
for point in graph_node_centers_homography_camera_referential:
    graph_node_centers_homography_real_world_referential.append(planar_homography.camera_to_real_world_point(point))
graph_node_centers_homography_real_world_referential = np.array(
    graph_node_centers_homography_real_world_referential)
places = config["node names"]
room_seg = ColorSegmentation(hue, sat, val).segmentation(room_clean)

center_cam = find_segmented_centers(room_seg)
center_real = planar_homography.camera_to_real_world_point(center_cam)
node = closest_node(center_real, graph_node_centers_homography_real_world_referential)
node = f"({places[node]})"

center_real = round(center_real[0], 3), round(center_real[1], 3)

scale = 1
xoff = -400
yoff = 60
color = (255, 255, 255)
circle_color = (0, 0, 255)
def make_points(name, image):
    image = cv2.circle(image.copy(), center_cam, 15, circle_color, thickness=4)
    image = cv2.putText(image, f"px: {center_cam[0]} py: {center_cam[1]}", (center_cam[0] + xoff, center_cam[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA, False)
    image = cv2.putText(image, f"x: {center_real[0]} y: {center_real[1]}", (center_cam[0] + xoff, center_cam[1] + yoff), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA, False)
    image = cv2.putText(image, node, (center_cam[0] + xoff, center_cam[1] + 2 * yoff), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA, False)
    cv2.imwrite(name, image)

make_points("../room_clean_track.jpg", room_clean)
make_points("../room_seg_track.jpg", cv2.cvtColor(room_seg.copy(), cv2.COLOR_GRAY2RGB)
)
