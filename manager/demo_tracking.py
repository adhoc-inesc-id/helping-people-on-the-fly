import argparse
import cv2
import yaml
import numpy as np

from utils import planar_homography
from utils.cameras import RealSenseCamera
from utils.color_segmentation import ColorSegmentation, detect_blobs_centers_of_mass

def closest_node(point, centers):
    node = None
    smallest = np.inf
    for n, center in enumerate(centers):
        distance = np.linalg.norm((center-point), 2)
        if distance < smallest:
            smallest = distance
            node = n
    return node

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-camera_snapshot', default="../resources/images/shoes_far2.jpeg")
    parser.add_argument('-downscale', type=int, default=0.3)
    opt = parser.parse_args()

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)

    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    color_segmentation = ColorSegmentation(hue, sat, val)
    graph_node_centers_homography_camera_referential = np.array(config["graph nodes camera points"])
    nodes = config["node names"]

    cam = RealSenseCamera()
    while True:

        cv2.namedWindow("Feed")
        while True:
            img = cam.take_picture()
            img = color_segmentation.segmentation(img)
            try:
                feet = detect_blobs_centers_of_mass(img)[0]
            except:
                continue
            node = closest_node(feet, graph_node_centers_homography_camera_referential)
            print(nodes[node])
            cv2.imshow("Feed", img)
            k = cv2.waitKey(1) & 0xFF
            if k % 256 == 27:
                break
