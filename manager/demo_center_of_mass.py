import cv2
import imutils
import yaml
import numpy as np
import matplotlib.pyplot as plt

from color_segmentation import ColorSegmentation, detect_blobs_centers_of_mass

if __name__ == '__main__':

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    color_segmentation = ColorSegmentation(hue, sat, val)
    image = cv2.imread("shoes_far2.jpeg")
    width = int(image.shape[1] * 0.2)
    height = int(image.shape[0] * 0.2)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    segmented_image = color_segmentation.segmentation(image)

    center = detect_blobs_centers_of_mass(segmented_image)[0]
    print(center)
    plt.imshow(cv2.circle(segmented_image, center, 10, (255, 0, 0)))
    plt.show()