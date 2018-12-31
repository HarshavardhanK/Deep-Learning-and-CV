#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def print_equation(data):
    print(data.data)

def subscriber():

    rospy.init_node('equation_subscriber', anonymous=True)
    rospy.Subscriber('equation', String, print_equation)

    rospy.spin()

if __name__ == '__main__':
    print('Equation pub/sub..')

    try:
        subscriber()
    except Exception, e:
        print("Exception: %s", e)

    print("DONE.")
