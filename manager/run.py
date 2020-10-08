"""
Manager node main
Python 3
"""
import time
import yaml
import rospy

from std_msgs.msg import String
from argparse import ArgumentParser

import utils.planar_homography as planar_homography
from utils.cameras import *
from utils.color_segmentation import ColorSegmentation, detect_blobs_centers_of_mass

# ######### #
# Auxiliary #
# ######### #

def row_index(row, matrix):
    if not isinstance(row, np.ndarray):
        row = np.array(row)
    possible = np.where(np.all(matrix == row, axis=1))
    if len(possible) != 1:
        raise ValueError("Not a valid row in the matrix")
    else:
        return possible[0]

def find_next_node(next_index):
    adjacencies = np.where(adjacency_matrix[last_known_robot_location] == 1)[0]
    downgrade_to_lower_index = int(next_index) >= len(adjacencies)
    next_index = 0 if downgrade_to_lower_index else next_index
    next_node = adjacencies[next_index]
    return next_node

def send_astro_order(order: str):
    rospy.loginfo(f"Sent Astro node order {order}")
    astro_publisher.publish(order)

def map_astro_order(action: int):
    if "move" in action_meanings[action]:
        next_node = find_next_node(action)
        x, y = graph_node_centers_astro_referential[next_node]
        order = f"go to {x}, {y}"
    elif action == 3: order = "stay"
    else: raise ValueError("Should be unreachable - invalid action index")
    rospy.loginfo(f"Mapped action {action} to order {order}")
    return order

def closest_node(point, centers):
    node = None
    smallest = np.inf
    for n, center in enumerate(centers):
        distance = np.linalg.norm((center-point), 2)
        if distance < smallest:
            smallest = distance
            node = n
    return node

def read_astro_node(dead_reckoning):

    x, y = dead_reckoning.split(",")
    x, y = float(x), float(y)

    global last_known_robot_location
    try:
        n_astro = closest_node(np.array(x, y), graph_node_centers_astro_referential)
        rospy.loginfo(f"Astro is closest to {places[n_astro]}")
        last_known_robot_location = n_astro
    except ValueError:
        n_astro = last_known_robot_location
        rospy.logwarn(f"Astro's coordinates {x, y} do not map to any valid known node. Using last known location ({places[n_astro]})")

    return n_astro

def read_human_feet_camera():

    # 1 - Take picture using camera (camera object created in main)
    image = camera.take_picture()
    if image is not None:
        rospy.loginfo("Successfully took picture")

    # 2 - Run color segmentation (also created in main)
    segmented_image = color_segmentation.segmentation(image)
    rospy.loginfo("Ran color segmentation")

    # 3 - Take center of mass from detected feet
    feet = detect_blobs_centers_of_mass(segmented_image)[0]
    rospy.loginfo(f"Detected feet at {feet}")

    return feet

def read_human_node():

    # 1 - Access camera and take human's feet location on camera via background
    feet_on_camera = read_human_feet_camera()

    # 2 - Get mapped 2d coordinate via planar homography
    feet_on_real_world = planar_homography.camera_to_real_world_point(feet_on_camera)

    # 4 - Map real world ground position to correct node
    global last_known_human_location
    try:
        n_human = closest_node(feet_on_real_world, graph_node_centers_homography_real_world_referential)
        rospy.loginfo(f"Human is closest to {places[n_human]}")
        last_known_human_location = n_human
    except ValueError:
        n_human = last_known_human_location
        rospy.logwarn(f"Human's feet {feet_on_real_world} do not map to any valid known node. Using last known location ({places[n_human]})")

    return n_human

def make_current_state(dead_reckoning):

    global explored_bits

    n_astro = read_astro_node(dead_reckoning)
    n_human = read_human_node()

    if n_astro in nodes_to_explore:
        i = nodes_to_explore.index(n_astro)
        explored_bits[i] = 1

    if n_human in nodes_to_explore:
        i = nodes_to_explore.index(n_human)
        explored_bits[i] = 1

    state = np.array([n_astro, n_human] + [explored_bits])
    rospy.loginfo(f"Successfully built state st: {state}")

    return state

# ######## #
# Sequence #
# ######## #

# Step 1
def request_action_from_decision_node(state: np.ndarray):
    message = " ".join([str(entry) for entry in state])
    rospy.loginfo(f"Sent state to Decision node: '{message}'")
    decision_publisher.publish(message)

# Step 2
def receive_decision_node_message(message: String):

    rospy.loginfo("############")
    rospy.loginfo("New Timestep")

    action = int(message.data)
    rospy.loginfo(f"Received action #{action} ({action_meanings[action]}) from Decision node")

    # 1 - Processar objectivo da ação (O que o astro tem de fazer)
    order = map_astro_order(action)

    # 2 - Enviar ordens ao Astro
    send_astro_order(order)

    # Aguardar mensagens do astro
    rospy.loginfo(f"Awaiting Astro node's message")

# Step 3
def receive_astro_node_message(message: String):
    data = message.data
    dead_reckoning = data.split(";")
    rospy.loginfo(f"Received Astro node message: '{dead_reckoning};'")
    state = make_current_state(dead_reckoning)
    rospy.loginfo(f"Built state array {state}'")
    request_action_from_decision_node(state)


if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_mmdp_manager")

    opt.add_argument("-decision_subscriber_topic", default="/adhoc_mmdp/decision_manager")
    opt.add_argument("-decision_publisher_topic", default="/adhoc_mmdp/manager_decision")

    opt.add_argument("-astro_subscriber_topic", default="/adhoc_mmdp/astro_manager")
    opt.add_argument("-astro_publisher_topic", default="/adhoc_mmdp/manager_astro")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=10)

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)

    # ######### #
    # Auxiliary #
    # ######### #

    with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)

    rospy.loginfo(f"Initializing auxiliary structures")

    # Environment Reckon Task #
    action_meanings = (
        "move to lower-index node",
        "move to second-lower-index node",
        "move to third-lower-index node",
        "stay",
    )
    places = config["node names"]
    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    nodes_to_explore = config["nodes to explore"]
    graph_node_centers_astro_referential = np.array(config["graph nodes astro points"])

    explored_bits = [1, 0, 0]
    last_known_robot_location = 0
    last_known_human_location = 0

    initial_state = np.array([last_known_robot_location, last_known_human_location] + explored_bits)

    # Homography
    graph_node_centers_homography_real_world_referential = np.array(["graph nodes real world points"])

    # Camera
    camera = ImageWrapperCamera("../resources/shoes_near.jpeg")
    hue = np.array(config["color segmentation"]["hue"])
    sat = np.array(config["color segmentation"]["sat"])
    val = np.array(config["color segmentation"]["val"])
    color_segmentation = ColorSegmentation(hue, sat, val)

    # ### #
    # ROS #
    # ### #

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Decision node subscriber (local topic at {opt.decision_subscriber_topic})")
    callback = lambda message: receive_decision_node_message(message)
    decision_subscriber = rospy.Subscriber(opt.decision_subscriber_topic, String, receive_decision_node_message)

    rospy.loginfo(f"Setting up Astro node subscriber (local topic at {opt.astro_subscriber_topic})")
    astro_subscriber = rospy.Subscriber(opt.astro_subscriber_topic, String, receive_astro_node_message)

    # ########## #
    # Publishers #
    # ########## #

    rospy.loginfo(f"Setting up Decision node publisher (topic at {opt.decision_publisher_topic})")
    decision_publisher = rospy.Publisher(opt.decision_publisher_topic, String, queue_size=opt.publisher_queue_size)

    rospy.loginfo(f"Setting up Astro node publisher (topic at {opt.astro_publisher_topic})")
    astro_publisher = rospy.Publisher(opt.astro_publisher_topic, String, queue_size=opt.publisher_queue_size)

    # ### #
    # Run #
    # ### #

    for t in reversed(range(5)):
        rospy.loginfo(f"Starting in {t+1}")
        time.sleep(1)

    rospy.loginfo("Starting")

    rate = rospy.Rate(opt.communication_refresh_rate)

    request_action_from_decision_node(initial_state)

    rospy.spin()
