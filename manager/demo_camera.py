import argparse

import yaml
from utils.cameras import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-camera', default="segmentation")
    parser.add_argument('-picture', default="../resources/images/sheet_for_homography.jpg")
    parser.add_argument('-downscale', type=int, default=20)

    opt = parser.parse_args()

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
    if opt.camera == "picture":
        cam = ImageWrapperCamera("../resources/images/shoes_near.jpeg", 0.2)
    elif opt.camera == "segmentation":
        hue = np.array(config["color segmentation"]["hue"])
        sat = np.array(config["color segmentation"]["sat"])
        val = np.array(config["color segmentation"]["val"])
        cam = YellowFeetSegmentationCamera(-1, hue, sat, val)
    elif opt.camera == "realsense":
        cam = RealSenseCamera()
    else:
        cam = CV2VideoCapture(0)
    cv2.namedWindow("Feed")
    while True:
        img = cam.take_picture()
        cv2.imshow("Feed", img)
        k = cv2.waitKey(1) & 0xFF
        if k % 256 == 27:
            break
    cv2.destroyAllWindows()
