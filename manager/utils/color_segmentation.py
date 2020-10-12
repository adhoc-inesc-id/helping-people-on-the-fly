import imutils
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

        # Image filtering to remove impurities
        blur = cv2.blur(img, (5, 5))
        blur0 = cv2.medianBlur(blur, 5)
        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

        # Convert to HSV space to lose dependency on lighting conditions
        hsv_img = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        # Segment image according to hue, saturation and value bounds defined
        low_thresh = np.array([self._hue[0], self._saturation[0], self._value[0]])
        high_thresh = np.array([self._hue[1], self._saturation[1], self._value[1]])
        segmented_image = cv2.inRange(hsv_img, low_thresh, high_thresh)

        # If use_mask was given, show only segmented portion of initial image
        if use_mask:
            segmented_image = cv2.bitwise_and(img, img, mask=segmented_image)

        return segmented_image

    def still_segmentation_config(self, img, use_mask=False):

        def nothing(x):
            pass

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

        while True:

            k = cv2.waitKey(1) & 0xFF
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            print('Update segmentation values')
            # get current positions of four trackbars
            hue_min = cv2.getTrackbarPos('Hue_Min', 'segmented_image')
            hue_max = cv2.getTrackbarPos('Hue_Max', 'segmented_image')
            saturation_min = cv2.getTrackbarPos('Saturation_Min', 'segmented_image')
            saturation_max = cv2.getTrackbarPos('Saturation_Max', 'segmented_image')
            value_min = cv2.getTrackbarPos('Value_Min', 'segmented_image')
            value_max = cv2.getTrackbarPos('Value_Max', 'segmented_image')

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

    def stream_segmentation_config(self, cam, img_counter, use_mask=False):

        def nothing(x):
            pass

        print('Get first frame')
        ret, img = 1, cam.take_picture()
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
            ret, img = 1, cam.take_picture()
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

    color_seg = ColorSegmentation()
    img = cv2.imread(imgpath)
    cv2.namedWindow("orig_image")
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("orig_image", img)
    color_seg.still_segmentation_config(img)

    if cv2.getWindowProperty('orig_image', cv2.WND_PROP_VISIBLE) != 1.0:
        cv2.namedWindow("orig_image")
    if cv2.getWindowProperty('segmented_image', cv2.WND_PROP_VISIBLE) != 1.0:
        cv2.namedWindow('segmented_image')

    while True:

        k = cv2.waitKey(1) & 0xFF
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        cv2.imshow("orig_image", img)
        segmented_img = color_seg.segmentation(img)
        cv2.imshow('segmented_image', segmented_img)

    cv2.destroyAllWindows()


def test_online():

    print('Acquiring Camera')

    from utils.cameras import RealSenseCamera

    cam = RealSenseCamera()

    cv2.namedWindow("video")

    img_counter = 0
    color_seg = ColorSegmentation()
    color_seg.stream_segmentation_config(cam, img_counter)

    if cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) != 1.0:
        cv2.namedWindow("video")
    if cv2.getWindowProperty('segmented_image', cv2.WND_PROP_VISIBLE) != 1.0:
        cv2.namedWindow('segmented_image')

    while True:
        print('Get new frame')
        ret, img = 1, cam.take_picture()
        if not ret:
            print("failed to grab frame")
            break

        segmented_img = color_seg.segmentation(img)
        center_of_mass = find_segmented_centers(segmented_img, 'averaging')
        cv2.circle(img, (int(center_of_mass[0]), int(center_of_mass[1])), 7, (255, 255, 255), -1)

        cv2.imshow("video", img)
        cv2.imshow('segmented_image', segmented_img)

        k = cv2.waitKey(1) & 0xFF
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cv2.destroyAllWindows()

def test_detect():
    print('Acquiring Camera')

    from utils.cameras import RealSenseCamera

    cam = RealSenseCamera()

    cv2.namedWindow("video")

    color_seg = ColorSegmentation(np.array([38, 67]), np.array([32, 255]), np.array([156, 255]))

    cv2.namedWindow('segmented_image')

    while True:
        print('Get new frame')
        ret, img = 1, cam.take_picture()
        if not ret:
            print("failed to grab frame")
            break

        segmented_img = color_seg.segmentation(img)
        center_of_mass = find_segmented_centers(segmented_img, 'averaging')
        cv2.circle(img, (int(center_of_mass[0]), int(center_of_mass[1])), 7, (255, 255, 255), -1)

        cv2.imshow("video", img)
        cv2.imshow('segmented_image', segmented_img)

        k = cv2.waitKey(1) & 0xFF
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cv2.destroyAllWindows()

def detect_blobs_centers_of_mass(blobbed_image):
    contours = cv2.findContours(blobbed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    centers = []
    for contour in contours:
        try:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = cX, cY
            centers.append(center)
        except ZeroDivisionError:
            pass
    return centers


def find_segmented_centers(segmented_img, mode='averaging', max_contours=2):

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(segmented_img, 127, 255, 0)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    contours_areas = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate area for each contour
        area = cv2.contourArea(c)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # store centers and areas for post-processing
        centers += [(cX, cY)]
        contours_areas += [area]

    if len(centers) > 0:
        # to find the center of mass of the feet
        if mode.find('averaging') != -1:

            largest_areas = []
            # find the contours for the feet
            for i in range(len(contours_areas)):
                n_areas = len(largest_areas)
                current_area = contours_areas[i]
                if n_areas == 0:
                    largest_areas += [(i, current_area)]

                elif n_areas < max_contours:
                    j = 0
                    while j < n_areas:
                        if current_area > largest_areas[j][1]:
                            largest_areas = largest_areas[:j] + [(i, current_area)] + largest_areas[j:]
                            break
                        j += 1
                    if j == n_areas:
                        largest_areas = largest_areas[:j] + [(i, current_area)]

                else:
                    j = 0
                    while j < n_areas:
                        if current_area > largest_areas[j][1]:
                            largest_areas = largest_areas[:j] + [(i, current_area)] + largest_areas[j:-1]
                            break
                        j += 1

            avg_cX, avg_cY = 0, 0
            for area_t in largest_areas:
                center = centers[area_t[0]]
                avg_cX += center[0]
                avg_cY += center[1]
            avg_cX, avg_cY = avg_cX/len(largest_areas), avg_cY/len(largest_areas)

            return int(avg_cX), int(avg_cY)

        # retrieves only the center of mass of biggest body
        elif mode.find('single') != -1:
            largest_contour = 0
            largest_area = 0
            for i in range(len(contours_areas)):
                current_area = contours_areas[i]
                if current_area > largest_area:
                    largest_contour = i

            return centers[largest_contour][0], centers[largest_contour][1]

        else:
            print('Invalid center of mass retrieving mode')
            return 0, 0
    else:
        return 0, 0

if __name__ == '__main__':
    # test_offline("../resources/images/shoes_far1.jpeg", 0.5)
    test_online()
    #test_detect()
