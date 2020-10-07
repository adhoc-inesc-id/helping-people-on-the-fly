"""
Decision node main
Python 3
"""

import rospy
from std_msgs.msg import String
from argparse import ArgumentParser
import numpy as np
from yaaf.agents import RandomAgent


def send_manager_message(message: String):
    manager_publisher.publish(message)


def receive_manager_message(message: String):
    rospy.loginfo("")
    rospy.loginfo("New Timestep")
    message = message.data
    state = np.array([int(o) for o in message.split(" ")])
    rospy.loginfo(f"Received state from Manager node: {state}")
    action = agent.action(state)
    rospy.loginfo(f"Sending action #{action} to Manager node ({action_meanings[action]})")
    send_manager_message(f"{action}")


def setup_possible_tasks():

    n_astro, n_human = 0, 0
    movement_failure_prob = 0.0
    goals = (
        [0, 1, 4],
        [2, 3, 4],
        [0, 2, 4]
    )

    return [make_task(nodes, n_astro, n_human, movement_failure_prob, t) for t, nodes in enumerate(goals)]


def make_task(nodes_to_explore, n_astro, n_human, movement_failture_probability, task_no):

    from environment.EnvironmentReckonMMDP import EnvironmentReckonMMDP

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])

    initial_state = [n_astro, n_human] + [0 for _ in range(len(nodes_to_explore))]

    if n_astro in nodes_to_explore:
        initial_state[2 + nodes_to_explore.index(n_astro)] = 1

    if n_human in nodes_to_explore:
        initial_state[2 + nodes_to_explore.index(n_human)] = 1

    task = EnvironmentReckonMMDP(
        adjacency_matrix,
        nodes_to_explore,
        movement_failture_probability,
        initial_state=np.array(initial_state),
        id=f"env-reckon-mmdp-v{task_no}")

    return task


def setup_agent(possible_tasks):
    return RandomAgent(num_actions=possible_tasks[0].num_actions)
    #return BOPA(possible_tasks)


if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_mmdp_decision")

    opt.add_argument("-manager_subscriber_topic", default="/adhoc_mmdp/manager_decision")
    opt.add_argument("-manager_publisher_topic", default="/adhoc_mmdp/decision_manager")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=10)

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)

    rospy.loginfo("Setting up algorithm and auxiliary structures")

    possible_tasks = setup_possible_tasks()
    agent = setup_agent(possible_tasks)
    action_meanings = possible_tasks[0].action_meanings

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Manager node subscriber (local topic at {opt.manager_subscriber_topic})")
    manager_subscriber = rospy.Subscriber(opt.manager_subscriber_topic, String, receive_manager_message)

    # ########## #
    # Publishers #
    # ########## #

    rospy.loginfo(f"Setting up Manager node publisher (topic at {opt.manager_publisher_topic})")
    manager_publisher = rospy.Publisher(opt.manager_publisher_topic, String, queue_size=opt.publisher_queue_size)

    # ### #
    # Run #
    # ### #

    rospy.loginfo("Ready")
    rate = rospy.Rate(opt.communication_refresh_rate)
    rospy.spin()
