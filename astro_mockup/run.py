"""
Astro Mockup
Final version will run on Astro (Python2)
"""
from typing import List

import rospy
import yaml
from std_msgs.msg import String
from argparse import ArgumentParser

import pyttsx3

def say_tts(text, engine):
    rospy.loginfo(f"TTS: {text}")
    engine.say(text)
    engine.runAndWait()

# ######## #
# Sequence #
# ######## #

# Step 1
def receive_order_from_manager(message: String):

    rospy.loginfo("")
    rospy.loginfo("New Timestep")

    order = message.data
    rospy.loginfo(f"Received Manager node order: '{order}'")

    global current_coordinates

    if "goto" in order:
        _, next_coordinates = order.split(" ")
        new_x, new_y = next_coordinates.split(",")
        new_x, new_y = float(new_x), float(new_y)
        move_astro(new_x, new_y)
        current_coordinates[0] = new_x
        current_coordinates[1] = new_y
        message = f"{new_x}, {new_y}"
    else:
        message = ",".join([str(i) for i in current_coordinates])

    rospy.loginfo(f"After order new coordinates are {message}")
    send_manager_message(message)

# Step 2
def move_astro(new_x, new_y):
    rospy.loginfo(f"Moving astro to {new_x}, {new_y}")
    # TODO Miguel Código para enviar movimento ao astro aqui


# Step 3
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

    with open('config.yml') as f:
        config = yaml.load(f.read(), yaml.FullLoader)

    if opt.tts:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 150)
        speak = lambda text: say_tts(text, tts_engine)
    else:
        speak = lambda text: rospy.loginfo(f"TTS: {text}")

    # Initial position
    current_coordinates = config["initial position"]

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Manager node subscriber (local topic at {opt.manager_subscriber_topic})")
    callback = lambda message: receive_order_from_manager(message)
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
