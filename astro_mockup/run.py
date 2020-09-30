"""
Astro Mockup
Final version will run on Astro (Python2)
"""
from typing import List

import rospy
from std_msgs.msg import String
from argparse import ArgumentParser

import pyttsx3

def say_tts(text, engine):
    rospy.loginfo(f"TTS: {text}")
    engine.say(text)
    engine.runAndWait()

def move_astro(new_x, new_y, new_z):
    rospy.loginfo(f"Moving astro to {new_x}, {new_y}, {new_z}")
    # TODO


def receive_order_from_manager(message: String, current_coordinates: List[float]):

    rospy.loginfo("")
    rospy.loginfo("New Timestep")

    order = message.data
    rospy.loginfo(f"Received Manager node order: '{order}'")

    if "goto" in order:
        _, next_coordinates = order.split(" ")
        new_x, new_y, new_z = next_coordinates.split(",")
        new_x, new_y, new_z = float(new_x), float(new_y), float(new_z)
        move_astro(new_x, new_y, new_z)
        new_coordinates = f"{new_x}, {new_y}, {new_z}"
        current_coordinates[0] = new_x
        current_coordinates[1] = new_y
        current_coordinates[2] = new_z
    else:
        new_coordinates = ",".join([str(i) for i in current_coordinates])

    send_manager_message(new_coordinates)


def send_manager_message(message: str):
    rospy.loginfo(f"Sent Manager node message {message}")
    manager_publisher.publish(message)


if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_mmdp_astro")

    opt.add_argument("-manager_subscriber_topic", default="/adhoc_mmdp/manager_astro")
    opt.add_argument("-manager_publisher_topic", default="/adhoc_mmdp/astro_manager")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=1)
    opt.add_argument("-tts", action="store_true")

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)

    rospy.loginfo(f"Initializing auxiliary structures")

    if opt.tts:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 150)
        speak = lambda text: say_tts(text, tts_engine)
    else:
        speak = lambda text: rospy.loginfo(f"TTS: {text}")

    # Initial position
    coordinates = [0.0, 0.0, 0.0]

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Manager node subscriber (local topic at {opt.manager_subscriber_topic})")
    callback = lambda message: receive_order_from_manager(message, coordinates)
    manager_subscriber = rospy.Subscriber(opt.manager_subscriber_topic, String, callback)

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
