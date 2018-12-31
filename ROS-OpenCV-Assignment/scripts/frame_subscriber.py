#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import lane_detection
import equation_publisher


def process_frame(frame):

    bridge = CvBridge()

    print("Frame received..")

    image = bridge.imgmsg_to_cv2(frame)
    (_, (left_slope, left_inter, right_slope, right_inter)) = lane_detection.mark_image(image)

    equation_publisher.publisher(("Left: y = %sm + %s | Right: y = %sm + %s" %(left_slope, left_inter, right_slope, right_inter)))


def subscriber():

    rospy.init_node("frame_subscriber", anonymous=True)
    rospy.Subscriber("lane_image", Image, process_frame)

    rospy.spin()


if __name__ == '__main__':
    print('Running frame_subscriber.')

    try:
        subscriber()
    except Exception, e:
        print("Error occurred. %s", e)

    print("DONE..")
