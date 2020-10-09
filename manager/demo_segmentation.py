import argparse
import cv2
import yaml
import numpy as np

from utils.cameras import RealSenseCamera
from utils.color_segmentation import ColorSegmentation, detect_blobs_centers_of_mass

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

    cam = RealSenseCamera()
    while True:

        image = cam.take_picture()
        width = int(image.shape[1] * opt.downscale)
        height = int(image.shape[0] * opt.downscale)

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        segmented_image = color_segmentation.segmentation(image)
        #feet = detect_blobs_centers_of_mass(segmented_image)[0]
        #centers_of_mass_image = cv2.circle(image.copy(), feet, 20, (0, 0, 255), thickness=3)

        cv2.imshow("Raw Image", image)
        cv2.imshow("Segmented Image", segmented_image)
        #cv2.imshow(f"Feet detected at {feet}", centers_of_mass_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
