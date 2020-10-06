"""
Manager node main
Python 3
"""
import time

import rospy
from std_msgs.msg import String
from argparse import ArgumentParser
import numpy as np

import planar_homography

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
        x, y = dead_reckoning_coordinate_map[next_node]
        order = f"go to {x}, {y}"
    elif action == 3: order = "stay"
    else: raise ValueError("Should be unreachable - invalid action index")
    rospy.loginfo(f"Mapped action {action} to order {order}")
    return order

def read_astro_node(dead_reckoning):

    x, y, _ = dead_reckoning.split(",")
    x, y = float(x), float(y)

    global last_known_robot_location
    try:
        n_astro = row_index((x, y), dead_reckoning_coordinate_map)
        last_known_robot_location = n_astro
    except ValueError:
        n_astro = last_known_robot_location
        rospy.loginfo(f"Astro's coordinates {x, y} do not map to any valid known node")

    return n_astro


def read_human_feet_camera():

    # TODO - Miguel aqui background subtraction teu

    # 1 - Take picture using camera

    # 2 - Run background subtraction and take x, y of feet in camera coordinates
    x, y = 0, 0

    return x, y


def read_human_node():

    # 1 - Access camera and take human's feet location on camera via background
    feet_center_of_mass = read_human_feet_camera()
    camera_frame_point = np.array(feet_center_of_mass)

    # 2 - Get mapped 2d coordinate via planar homography
    real_world_point = planar_homography.camera_to_real_world_point(camera_frame_point)

    # 4 - Map real world ground position to correct node
    global last_known_human_location
    try:
        n_human = row_index(real_world_point, homography_coordinate_map)
        last_known_human_location = n_human
    except ValueError:
        n_human = last_known_human_location

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
    rospy.loginfo(f"State: {state}")

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

    rospy.loginfo("")
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

    rospy.loginfo(f"Initializing auxiliary structures")

    action_meanings = (
        "move to lower-index node",
        "move to second-lower-index node",
        "move to third-lower-index node",
        "stay",
        "locate human",
        "locate robot"
    )
    places = ("porta", "baxter", "bancada", "bancada dupla", "mesa")

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])

    import yaml
    with open('config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    nodes_to_explore = config["nodes to explore"]
    explored_bits = [1, 0, 0]
    dead_reckoning_coordinate_map = np.array(config["graph nodes astro points"])
    homography_coordinate_map = np.array(config["homography"]["graph nodes real world points"])

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

    time.sleep(2)

    rospy.loginfo("Ready")

    initial_state = np.array([0, 0, 1, 0, 0])

    last_known_robot_location = 0
    last_known_human_location = 0

    rate = rospy.Rate(opt.communication_refresh_rate)

    request_action_from_decision_node(initial_state)

    rospy.spin()
