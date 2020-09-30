import os

from cv2.cv2 import findHomography
import numpy as np

def setup_homography_matrix(load_cached=True):
    """
    Loads the homography matrix (either from cached file or from scratch)
    """
    if load_cached:
        try:
            homography_matrix = np.load("homography_matrix.npy")
        except FileNotFoundError:
            homography_matrix = setup_homography_matrix(load_cached=False)
    else:
        import yaml
        with open('homography_points.yml', 'r') as file:
            points = yaml.load(file, Loader=yaml.FullLoader)
            real_world_points = np.array([points["real world points"]])
            camera_frame_points = np.array([points["camera frame points"]])
            homography_matrix, _ = findHomography(camera_frame_points, real_world_points)

    np.save("homography_matrix", homography_matrix)
    return homography_matrix

H = setup_homography_matrix()

def reinitialize_homography_matrix():
    """
    Creates new homography matrix using points in homography_points.yml
    """
    global H
    os.remove("homography_matrix.npy")
    H = setup_homography_matrix(load_cached=False)

def planar_homography(point, inverted=False):
    augmented_point = np.concatenate((point, np.array([1])))
    homography_matrix = np.linalg.inv(H) if inverted else H
    homography_point = np.dot(homography_matrix, augmented_point)
    homography_point = homography_point / homography_point[2]
    return homography_point[:-1]

def camera_to_real_world_point(camera_point):
    """
    Maps a camera point into its real world point
    """
    return planar_homography(camera_point)

def real_world_to_camera_point(real_world_point):
    """
    Maps a real world point into its camera point
    """
    return planar_homography(real_world_point, inverted=True)