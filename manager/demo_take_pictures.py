import yaml

from utils import planar_homography
from utils.cameras import *
import cv2

if __name__ == '__main__':


    with open('config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    #hue = np.array([30, 59])
    #sat = np.array([36, 255])
    #val = np.array([183, 251])

    bgr = RealSenseCamera()
    room_clean = bgr.take_picture()
    #bgr_picture = cv2.imread("../resources/images/shoes_near.jpeg")

    room_seg = ColorSegmentation(hue, sat, val).segmentation(room_clean)
    center_cam = find_segmented_centers(room_seg)

    center_real = planar_homography.camera_to_real_world_point(center_cam)
    #center_real = center_cam

    xoff = -700
    yoff = 100
    seg_track_picture = cv2.circle(cv2.cvtColor(room_seg.copy(), cv2.COLOR_GRAY2RGB), center_cam, 15, (0, 0, 255), thickness=10)
    seg_track_picture = cv2.putText(seg_track_picture, f"px: {center_cam[0]} py: {center_cam[1]}", (center_cam[0] + xoff, center_cam[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA, False)
    seg_track_picture = cv2.putText(seg_track_picture, f"x: {center_cam[0]} y: {center_cam[1]}", (center_cam[0] + xoff, center_cam[1] + yoff), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA, False)

    room_clean_track_picture = cv2.circle(room_clean.copy(), center_cam, 15, (0, 0, 255), thickness=10)
    room_clean_track_picture = cv2.putText(room_clean_track_picture, f"px: {center_cam[0]} py: {center_cam[1]}", (center_cam[0] - 700, center_cam[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA, False)
    room_clean_track_picture = cv2.putText(room_clean_track_picture, f"x: {center_cam[0]} y: {center_cam[1]}", (center_cam[0] - 700, center_cam[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA, False)

    cv2.imwrite("../room_clean.jpg", room_clean)
    cv2.imwrite("../room_seg.jpg", room_seg)
    cv2.imwrite("../room_seg_track.jpg", seg_track_picture)
    cv2.imwrite("../room_clean_track.jpg", room_clean_track_picture)
