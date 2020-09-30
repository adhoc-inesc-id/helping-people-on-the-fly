"""
Manager node main
Python 3
"""
import rospy
from std_msgs.msg import String
from argparse import ArgumentParser
import numpy as np

import planar_homography


def find_next_node(next_index):
    adjacencies = np.where(adjacency_matrix[current_node] == 1)[0]
    downgrade_to_lower_index = int(next_index) >= len(adjacencies)
    next_index = 0 if downgrade_to_lower_index else next_index
    next_node = adjacencies[next_index]
    return next_node


def request_action_from_decision_node(state: np.ndarray):
    message = " ".join([str(entry) for entry in state])
    rospy.loginfo(f"Sent state to Decision node: '{message}'")
    decision_publisher.publish(message)


def send_astro_order(order: str):
    rospy.loginfo(f"Sent Astro node order {order}")
    astro_publisher.publish(order)


def map_astro_order(action: int):

    if "move" in action_meanings[action]:
        next_node = find_next_node(action)
        x, y, z = dead_reckoning_coordinate_map[next_node]
        order = f"go to {x}, {y}, {z}"
    elif action == 3: order = "stay"
    else: raise ValueError("Should be unreachable - invalid action index")

    rospy.loginfo(f"Mapped action {action} to order {order}")

    return order


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



def read_astro_node(dead_reckoning):
    x, y, _ = dead_reckoning.split(",")
    x, y = float(x), float(y)
    possible_node = np.where(np.all(dead_reckoning_coordinate_map == np.array([x, y]), axis=1))
    n_astro = int(possible_node[0]) if len(possible_node) == 1 else current_node
    return n_astro

def read_human_node():

    # 1 - Access camera and take human's feet location on camera via background
    # TODO
    camera_frame_point = np.array([0, 0])

    # 2 - Get mapped 2d coordinate via planar homography
    real_world_point = planar_homography.camera_to_real_world_point(camera_frame_point)

    # 4 - Map real world ground position to correct node
    # TODO
    n_human = 0

    return n_human


def make_current_state(dead_reckoning):

    global current_node
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

    nodes_to_explore = [0, 1, 4]
    explored_bits = [1, 0, 0]
    current_node = 0

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

    # FIXME - Handcode here the X, Y of astro's referencial
    dead_reckoning_coordinate_map = np.array([
        [0.0, 0.0],    # 0
        [1.0, 0.0],    # 1
        [2.0, 0.0],    # 2
        [3.0, 0.0],    # 3
        [4.0, 0.0],    # 4
    ])

    # FIXME - Handcode here the X, Y of homography's real world referencial
    homography_coordinate_map = np.array([
        [0.0, 0.0],    # 0
        [1.0, 0.0],    # 1
        [2.0, 0.0],    # 2
        [3.0, 0.0],    # 3
        [4.0, 0.0],    # 4
    ])

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

    rospy.loginfo("Ready")

    rate = rospy.Rate(opt.communication_refresh_rate)

    rospy.spin()
