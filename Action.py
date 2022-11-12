#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import rospy
import random
import time


from geometry_msgs.msg import Twist


rospy.init_node('vel_node')
vel_pub = rospy.Publisher('/mobile_base/commands/velocuity', Twist, queue_size=10)


def execute(action):
    move_cmd = Twist()
    for i in action['angular_vel']:
        if i == 0:
            move_cmd.linear.x = 0.5
            move_cmd.angular.z = 0.5
        elif i == 1:
            move_cmd.linear.x = 0.5
            move_cmd.angular.z = 1.0
        elif i == 2:
            move_cmd.linear.x = 0.5
            move_cmd.angular.z = -0.5
        elif i == 3:
            move_cmd.linear.x = 0.5
            move_cmd.angular.z = -1.0
        else:
            raise Exception('Error discrete action: {}'.format(i))
        time.sleep(0.5)
if __name__ == "__main__":
    action = dict(angular_vel=[])
    for i in range(20):
        action['angular_vel'].append(random.randrange(0, 4, 1))
    execute(action)

