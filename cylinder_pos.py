#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import time
import tf
import math
import random
import config
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates




# def ModelStateCallBack(data):
#     global cylinder_state
#     idx = data.name.index("unit_cylinder" + str(1))
#     cylinder_state = [data.pose[idx].position.x, data.pose[idx].position.y]

rospy.init_node('GazeboUAV', anonymous=False)
set_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
# object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, ModelStateCallBack)
rospy.sleep(1.0)




state = ModelState()
# state.model_name = 'quadrotor'
for i in range(10):
    state.model_name = 'unit_cylinder' + str(i)
    state.reference_frame = 'world'  # ''ground_plane'
    # pose
    state.pose.position.x = config.obstacle_position[i][0] + random.uniform(-1.0, 1.0)
    state.pose.position.y = config.obstacle_position[i][1] + random.uniform(-1.0, 1.0)
    state.pose.position.z = 1
    set_state.publish(state)

rospy.sleep(0.5)
start_index = np.random.choice(len(config.start_space))
goal_index = np.random.choice(len(config.goal_space))
start = config.start_space[start_index]
state.model_name = 'quadrotor'
state.reference_frame = 'world'
state.pose.position.x = start[0]
state.pose.position.y = start[1]
state.pose.position.z = 1.5
set_state.publish(state)