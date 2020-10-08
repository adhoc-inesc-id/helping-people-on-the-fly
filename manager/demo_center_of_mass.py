import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from color_segmentation import ColorSegmentation, detect_blobs_centers_of_mass, find_segmented_centers

if __name__ == '__main__':

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    color_segmentation = ColorSegmentation(hue, sat, val)
    image = cv2.imread("../resources/images/shoes_near.jpeg")
    width = int(image.shape[1] * 0.2)
    height = int(image.shape[0] * 0.2)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    segmented_image = color_segmentation.segmentation(image)

    center = detect_blobs_centers_of_mass(segmented_image)[0]
    print(center)
    plt.imshow(cv2.circle(image, center, 20, (255, 0, 0), thickness=2))
    plt.show()