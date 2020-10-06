"""
Decision node main
Python 3
"""
import time

import rospy
from std_msgs.msg import String
from argparse import ArgumentParser
import numpy as np

from agents.BOPA import BOPA
from run_full_empirical_evaluation import task_factory


def send_manager_message(message: String):
    manager_publisher.publish(message)


def receive_manager_message(message: String):
    rospy.loginfo("")
    rospy.loginfo("New Timestep")
    message = message.data
    state = np.array([int(o) for o in message.split(" ")])
    rospy.loginfo(f"Received state from Manager node: {state}")
    action = 0
    #action = agent.action(state)
    rospy.loginfo(f"Sending action #{action} to Manager node ({action_meanings[action]})")
    send_manager_message(f"{action}")


def setup_possible_tasks():
    return task_factory(size="small", teammate="greedy")

def setup_agent(possible_tasks):
    return BOPA(possible_tasks)



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
