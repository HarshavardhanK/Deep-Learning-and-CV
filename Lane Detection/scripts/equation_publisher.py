#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def publisher(data):

    _publisher = rospy.Publisher('equation', String, queue_size=101)
    #rospy.init_node('equation_publisher', anonymous=True)

    _publisher.publish(data)
