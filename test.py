#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import time

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import env


# def takeoff_callback(msg):
#     twist_msg.linear.z = 0.4
#
#     while True:
#         cmd_vel_pub.publish(twist_msg)
#         # i = i + 1
#         rate.sleep()
#
#     twist_msg.linear.z = 0
#     cmd_vel_pub.publish(twist_msg)


# cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
# rospy.init_node("test111")
# rospy.wait_for_service('/gazebo/reset_world')
# reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
# reset_world()

test = env.GazeboUAV()
test.reset()
# move_cmd = Twist()

# move_cmd.linear.z = 1
# cmd_vel.publish(move_cmd)
# time.sleep(0.5)
# move_cmd.linear.z = 0
# cmd_vel.publish(move_cmd)
# time.sleep(0.5)
# test.reset()
# test.GetDepthImageObservation()
# test.execute(1)

# rate = rospy.Rate(5)
# takeoff_sub = rospy.Subscriber('/takeoff', Empty, takeoff_callback)
# cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
# twist_msg = Twist()
# empty_msg = Empty()
# rospy.spin()
# test.takeoff_no_SITL()