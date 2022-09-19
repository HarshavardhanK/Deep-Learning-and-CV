#!/usr/bin/env python

import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def image_publisher(path):

    publisher = rospy.Publisher('lane_image', Image, queue_size=101)
    rospy.init_node('frame_publisher', anonymous=True)

    if publisher != None:
        print("Publisher node created.")
    else:
        print("Publisher node cannot be created.")

    bridge = CvBridge()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error reading video file at {}".format(path))
        exit(0)

    print("Video file is opened.. Showing feed.")

    rval, frame = cap.read()

    while rval:
        cv2.imshow("Stream: " + path, frame)
        rval, frame = cap.read()

        if frame is not None:
            #frame = np.unint8(frame)
            frame = np.array(frame,'uint8')

        image_message = bridge.cv2_to_imgmsg(frame, encoding='passthrough')
        #print(image_message)

        publisher.publish(image_message)

        key = cv2.waitKey(40)

        if key == 27 or key == 1048603:
            break

        cv2.destroyWindow('preview')

if __name__ == '__main__':
    path = "/home/tsarshah/Desktop/Code Files/OpenCV/With-Python/challenge_video.mp4"

    print("Lane detection direc")

    try:
        image_publisher(path)
    except rospy.ROSInterruptException, e:
        print(e)
        pass

    print("DONE.")
