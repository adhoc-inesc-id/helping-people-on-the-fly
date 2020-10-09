import argparse

import cv2

from utils.cameras import RealSenseCamera


def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"- [{x}, {y}]")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-downscale', type=int, default=1)
    opt = parser.parse_args()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', callback)

    image = RealSenseCamera().take_picture()
    width = int(image.shape[1] * opt.downscale)
    height = int(image.shape[0] * opt.downscale)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    print("camera frame points:")
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
