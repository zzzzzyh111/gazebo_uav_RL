#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import rospy
import numpy as np
import cv2
import tf
import math as math
import random
import copy
import config

from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
from std_srvs.srv import Empty


class GazeboUAV():
    def __init__(self):
        rospy.init_node('GazeboUAV', anonymous=False)

        # -----------Params--------------------------------------------------
        self.depth_image_size = [160, 120]
        self.bridge = CvBridge()
        self.vel_cmd = [0.0]

        self.default_states = None
        self.depth_image = None

        self.goal_space = config.goal_space
        self.start_space = config.start_space
        self.obstacle_pos = config.obstacle_position
        self.des = Point32()

        self.p = [21.0, 0.0]
        self.success = False
        self.dist_init = 0
        self.dist = 0
        self.reward = 0
        self.obstacle_state = []
        self._actions = []
        self.cylinder_pos = [[] for i in range(10)]
        self.uav_trajectory = [[], []]
        self.stacked_imgs = None

        # ------------------------Publisher and Subscriber---------------------------
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.resized_depth_img = rospy.Publisher('/camera/depth/image_resized', Image, queue_size=10)
        self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image, self.DepthImageCallBack)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.sleep(1.)

    def ModelStateCallBack(self, data):
        idx = data.name.index("quadrotor")
        quaternion = (data.pose[idx].orientation.x,
                      data.pose[idx].orientation.y,
                      data.pose[idx].orientation.z,
                      data.pose[idx].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.self_state = [data.pose[idx].position.x,
                           data.pose[idx].position.y,
                           yaw,
                           data.twist[idx].linear.x,
                           data.twist[idx].linear.y,
                           data.twist[idx].angular.z]
        if self.default_states is None:
            self.default_states = copy.deepcopy(data)
        for i in range(10):
            idx = data.name.index("unit_cylinder" + str(i))
            self.cylinder_pos[i] = [data.pose[idx].position.x, data.pose[idx].position.y]

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def GetDepthImageObservation(self): \
            # Ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "bgr8")
            cv_img = np.array(cv_img, dtype=np.int8)
            cv_img[np.isnan(cv_img)] = 0
            return cv_img
        except Exception as err:
            print("Ros_to_Cv2 Failure: %s" % err)

    def getdist(self):
        theta = self.self_state[2]
        a_x = self.des.x - self.self_state[0]
        a_y = self.des.y - self.self_state[1]
        c = math.sqrt(a_x ** 2 + a_y ** 2)
        alpha = math.atan2(a_y, a_x) - theta
        return c, alpha

    def goal2robot(self, d_x, d_y, theta):
        d = math.sqrt(d_x ** 2 + d_y ** 2)
        alpha = math.atan2(d_y, d_x) - theta
        return d, alpha

    def obstacle2robot(self, e_x, e_y):
        s_x = e_x - self.self_state[0]
        s_y = e_y - self.self_state[1]
        e = math.sqrt(s_x ** 2 + s_y ** 2)
        beta = math.atan2(s_x, s_y) - self.self_state[2]
        return e, beta

    def DetectCollision(self):
        collision = False
        for i in range(len(self.cylinder_pos)):
            e, _ = self.obstacle2robot(self.cylinder_pos[i][0], self.cylinder_pos[i][1])
            if e < 1.2:
                collision = True

        return collision

    def get_states(self):
        self.stacked_imgs = np.dstack([self.stacked_imgs[:, :, -9:], self.GetDepthImageObservation()])
        return self.stacked_imgs, self.p

    def get_actions(self):
        return self._actions

    def GetSelfSpeed(self):
        return self.vel_cmd

    def SetUAVPose(self, x, y, theta):
        state = ModelState()
        state.model_name = 'quadrotor'
        state.reference_frame = 'world'  # ''ground_plane'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 1.5
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
        self.set_state.publish(state)

    def SetObjectPose(self):
        state = ModelState()
        for i in range(10):
            state.model_name = 'unit_cylinder' + str(i)
            state.reference_frame = 'world'  # ''ground_plane'
            state.pose.position.x = self.cylinder_pos[i][0]
            state.pose.position.y = self.cylinder_pos[i][1]
            state.pose.position.z = 1
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            self.set_state.publish(state)

    def SetObjectPose_random(self):
        state = ModelState()
        for i in range(10):
            state.model_name = 'unit_cylinder' + str(i)
            state.reference_frame = 'world'  # ''ground_plane'
            # pose
            state.pose.position.x = config.obstacle_position[i][0] + random.uniform(-1.0, 1.0)
            state.pose.position.y = config.obstacle_position[i][1] + random.uniform(-1.0, 1.0)
            state.pose.position.z = 1
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            self.set_state.publish(state)

    def set_goal(self, x, y):
        self.des.x = x
        self.des.y = y
        self.des.z = 1


    def reset(self):
        start_index = np.random.choice(len(self.start_space))
        goal_index = np.random.choice(len(self.goal_space))
        start = self.start_space[start_index]
        goal = self.goal_space[goal_index]
        # -------------------------------------
        theta = - math.pi / 2
        self.SetUAVPose(start[0], start[1], theta)
        self.set_goal(goal[0], goal[1])
        d0, alpha0 = self.goal2robot(goal[0] - start[0], goal[1] - start[1], theta)
        self.p = [d0, alpha0]
        self.reward = 0
        self.dist_init = d0
        self.vel_cmd = [0.0]
        self.success = False
        self.stacked_imgs = np.dstack([self.GetDepthImageObservation()] * 4)
        img, pos = self.get_states()
        return img, pos

    def execute(self, action_num):

        move_cmd = Twist()
        if action_num == 0:
            angular_z = 0.5
        elif action_num == 1:
            angular_z = 1.0
        elif action_num == 2:
            angular_z = -0.5
        elif action_num == 3:
            angular_z = -1.0
        elif action_num == 4:
            angular_z = 0
        else:
            raise Exception('Error discrete action')
        move_cmd.linear.x = 1.0
        move_cmd.angular.z = angular_z
        self.vel_pub.publish(move_cmd)

    def step(self):
        d1, alpha1 = self.getdist()
        self.p = [d1, alpha1]
        self.dist = d1
        terminal, reward = self.GetRewardAndTerminate()
        self.reward = reward
        self.dist_init = self.dist
        img, pos = self.get_states()

        return img, pos, terminal, reward

    def GetRewardAndTerminate(self):
        terminal = False

        reward = (10 * (self.dist_init - self.dist) - 0.2)

        if (self.dist < 1):
            reward = 1000.0
            print("Arrival!")
            terminal = True
            self.success = True

        if (self.self_state[0] >= 6.50) or (self.self_state[0] <= -6.50) or (self.self_state[1] >= 11.5) or (
                self.self_state[1] <= -11.5):
            reward = -100.0
            print("Out!")
            terminal = True

        if self.DetectCollision():
            reward = -100.0
            print("Collision!")
            terminal = True

        return terminal, reward
