import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=1280, height=720):
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self._cam = rs.pipeline()
        self._cam.start(config)
    def take_picture(self):
        frame = self._cam.wait_for_frames().get_color_frame()
        frame = np.asanyarray(frame.as_frame().get_data())
        return frame
    def __del__(self):
        self._cam.stop()

class CV2VideoCaptureCamera:
    def __init__(self, id):
        self._cam = cv2.VideoCapture(id)
    def take_picture(self):
        frame = self._cam.read()[1]
        return frame

class ImageWrapperCamera:
    def __init__(self, path, scale=1.0):
        self._image = cv2.imread(path)
        width = int(self._image.shape[1] * scale)
        height = int(self._image.shape[0] * scale)
        self._image = cv2.resize(self._image, (width, height), interpolation=cv2.INTER_AREA)
    def take_picture(self):
        return self._image

class VideoWrapperCamera:
    def __init__(self, path):
        self._video = cv2.VideoCapture(path)
