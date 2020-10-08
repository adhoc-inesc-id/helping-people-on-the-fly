import argparse
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from color_segmentation import ColorSegmentation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-camera_snapshot', default="../resources/shoes_far2.jpeg")
    parser.add_argument('-downscale', type=int, default=0.2)
    opt = parser.parse_args()

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    color_segmentation = ColorSegmentation(hue, sat, val)

    image = cv2.imread(opt.camera_snapshot)
    width = int(image.shape[1] * opt.downscale)
    height = int(image.shape[0] * opt.downscale)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    segmented_image = color_segmentation.segmentation(image)

    plt.imshow(image)
    plt.show()

    plt.imshow(segmented_image)
    plt.show()