import argparse

import cv2


def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"- [{x}, {y}]")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-camera_snapshot', default="../resources/images/sheet_for_homography.jpg")
    parser.add_argument('-downscale', type=int, default=20)
    opt = parser.parse_args()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', callback)

    image = cv2.imread(opt.camera_snapshot)
    width = int(image.shape[1] * opt.downscale / 100)
    height = int(image.shape[0] * opt.downscale / 100)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

    print("camera frame points:")
    while(1):
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break