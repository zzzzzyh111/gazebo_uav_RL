#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import rospy
import tf
import math as math
import random
import time

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


def set_start(x, y, theta):
    state = ModelState()
    state.model_name = 'iris'
    state.reference_frame = 'world'  # ''ground_plane'
    # pose
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = 0
    quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
    state.pose.orientation.x = quaternion[0]
    state.pose.orientation.y = quaternion[1]
    state.pose.orientation.z = quaternion[2]
    state.pose.orientation.w = quaternion[3]
    # twist
    state.twist.linear.x = 0
    state.twist.linear.y = 0
    state.twist.linear.z = 0
    state.twist.angular.x = 0
    state.twist.angular.y = 0
    state.twist.angular.z = 0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        result = set_state(state)
        assert result.success is True
        print("set the model state successfully")
    except rospy.ServiceException:
        print("/gazebo/get_model_state service call failed")


def execute(action_num):
    vel_cmd = [0.0]
    move_cmd = TwistStamped()
    if action_num == 0:
        move_cmd.linear.y = 0.0
        move_cmd.angular.z = 1.0
    elif action_num == 1:
        # move_cmd.twist.linear.x = 0.5
        move_cmd.twist.linear.x = 1
        move_cmd.twist.angular.z = 1.0
    elif action_num == 2:
        move_cmd.linear.y = -0.5
        move_cmd.angular.z = -0.5
    elif action_num == 3:
        move_cmd.linear.y = -0.5
        move_cmd.angular.z = -1.0
    else:
        raise Exception('Error discrete action: {}'.format(i))
    # vel_cmd = [move_cmd.linear.x, move_cmd.angular.z]
    time_st = time.time()
    while time.time() - time_st <= 1:
        vel_pub.publish(move_cmd)


def takeoff():
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    pose = PoseStamped()
    quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 1

    # Send a few setpoints before starting
    for i in range(100):
        if (rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()
    start_time = rospy.Time.now()

    # while (not rospy.is_shutdown()):
    while ((rospy.Time.now() - start_time) < rospy.Duration(15.0)):
        if (current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(2.0)):
            if (set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if (not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(2.0)):
                if (arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()

        local_pos_pub.publish(pose)

        rate.sleep()


if __name__ == "__main__":
    start = [0, 9]
    theta = -1.0 / 2 * math.pi
    rospy.init_node("offb_node_py")
    set_start(start[0], start[1], theta)
    print('111')
    takeoff()
    print('222')
    vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    # action = dict(angular_vel=[])
    for i in range(20):
        execute(1)
    # for i in range(20):
    #     execute(0)

