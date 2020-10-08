import numpy as np
import cv2


class ColorSegmentation(object):

    def __init__(self, hue=np.array([0, 255]), saturation=np.array([0, 255]), value=np.array([0, 255])):

        self._hue = hue
        self._saturation = saturation
        self._value = value

    @property
    def hue(self):
        return self._hue

    @hue.setter
    def hue(self, hue):
        if type(hue) is np.ndarray:
            self._hue = hue
        else:
            print('New value for hue not a numpy array, ignoring new value')

    @property
    def saturation(self):
        return self._saturation

    @saturation.setter
    def saturation(self, saturation):
        if type(saturation) is np.ndarray:
            self._saturation = saturation
        else:
            print('New value for saturation not a numpy array, ignoring new value')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if type(value) is np.ndarray:
            self._value = value
        else:
            print('New value for value not a numpy array, ignoring new value')

    def segmentation(self, img, use_mask=False):

        blur = cv2.blur(img, (5, 5))

        blur0 = cv2.medianBlur(blur, 5)

        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)

        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

        hsv_img = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        low_thresh = np.array([self._hue[0], self._saturation[0], self._value[0]])
        high_thresh = np.array([self._hue[1], self._saturation[1], self._value[1]])

        mask_img = cv2.inRange(hsv_img, low_thresh, high_thresh)

        if use_mask:
            mask_img = cv2.bitwise_and(img, img, mask= mask_img)

        return mask_img

    def offline_segmentation_config(self, imagepath, scale, img_counter, use_mask=False):

        def nothing(x):
            pass

        img = cv2.imread(imagepath)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        cv2.imshow("video", img)

        print('Initial Segmentation')
        cv2.namedWindow('segmented_image')
        segmented_img = self.segmentation(img, use_mask)
        cv2.imshow('segmented_image', segmented_img)

        cv2.createTrackbar('Hue_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Hue_Max', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Saturation_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Saturation_Max', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Value_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Value_Max', 'segmented_image', 0, 255, nothing)

        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'segmented_image', 0, 1, nothing)

        while True:

            k = cv2.waitKey(1) & 0xFF
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, img)
                print("{} written!".format(img_name))
                img_counter += 1

            cv2.imshow("video", img)

            # get current positions of four trackbars
            hue_min = cv2.getTrackbarPos('Hue_Min', 'segmented_image')
            hue_max = cv2.getTrackbarPos('Hue_Max', 'segmented_image')
            saturation_min = cv2.getTrackbarPos('Saturation_Min', 'segmented_image')
            saturation_max = cv2.getTrackbarPos('Saturation_Max', 'segmented_image')
            value_min = cv2.getTrackbarPos('Value_Min', 'segmented_image')
            value_max = cv2.getTrackbarPos('Value_Max', 'segmented_image')
            s = cv2.getTrackbarPos(switch, 'segmented_image')

            if s == 1:
                if hue_max > hue_min:
                    self._hue = np.array([hue_min, hue_max])
                if saturation_max > saturation_min:
                    self._saturation = np.array([saturation_min, saturation_max])
                if value_max > value_min:
                    self._value = np.array([value_min, value_max])

            segmented_img = self.segmentation(img, use_mask)

            cv2.imshow('segmented_image', segmented_img)

        cv2.destroyAllWindows()

    def online_segmentation_config(self, cam, img_counter, use_mask=False):

        def nothing(x):
            pass

        print('Get first frame')
        ret, img = cam.read()
        if not ret:
            print("failed to grab frame")
            return
        cv2.imshow("video", img)

        print('Initial Segmentation')
        cv2.namedWindow('segmented_image')
        segmented_img = self.segmentation(img, use_mask)
        cv2.imshow('segmented_image', segmented_img)

        cv2.createTrackbar('Hue_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Hue_Max', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Saturation_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Saturation_Max', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Value_Min', 'segmented_image', 0, 255, nothing)
        cv2.createTrackbar('Value_Max', 'segmented_image', 0, 255, nothing)

        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'segmented_image', 0, 1, nothing)

        while True:

            k = cv2.waitKey(1) & 0xFF
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, img)
                print("{} written!".format(img_name))
                img_counter += 1

            print('Get new frame')
            ret, img = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("video", img)

            print('Update segmentation values')
            # get current positions of four trackbars
            hue_min = cv2.getTrackbarPos('Hue_Min', 'segmented_image')
            hue_max = cv2.getTrackbarPos('Hue_Max', 'segmented_image')
            saturation_min = cv2.getTrackbarPos('Saturation_Min', 'segmented_image')
            saturation_max = cv2.getTrackbarPos('Saturation_Max', 'segmented_image')
            value_min = cv2.getTrackbarPos('Value_Min', 'segmented_image')
            value_max = cv2.getTrackbarPos('Value_Max', 'segmented_image')
            s = cv2.getTrackbarPos(switch, 'segmented_image')

            if s == 1:
                print('Updating segmentation')
                if hue_max > hue_min:
                    self._hue = np.array([hue_min, hue_max])
                if saturation_max > saturation_min:
                    self._saturation = np.array([saturation_min, saturation_max])
                if value_max > value_min:
                    self._value = np.array([value_min, value_max])

            segmented_img = self.segmentation(img, use_mask)

            cv2.imshow('segmented_image', segmented_img)

        cv2.destroyAllWindows()

def test_offline(imgpath, scale=1.0):

    img_counter = 0
    color_seg = ColorSegmentation()
    color_seg.offline_segmentation_config(imgpath, scale, img_counter)
    img = cv2.imread(imgpath)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    cv2.namedWindow("video")
    cv2.namedWindow('segmented_image')

    while True:
        cv2.imshow("video", img)
        segmented_img = color_seg.segmentation(img)
        cv2.imshow('segmented_image', segmented_img)


def test_online():

    print('Acquiring Camera')
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("video")
    if (cam.isOpened() == False):
        print("Error opening video stream or file")
        return
    img_counter = 0
    color_seg = ColorSegmentation()
    color_seg.online_segmentation_config(cam, img_counter)

    cv2.namedWindow("video")
    cv2.namedWindow('segmented_image')

    while True:

        print('Get new frame')
        ret, img = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("video", img)

        segmented_img = color_seg.segmentation(img)

        cv2.imshow('segmented_image', segmented_img)


if __name__ == '__main__':
    test_offline("shoes_far1.jpeg", 0.2)
