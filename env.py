#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import rospy
import numpy as np
import cv2
import time
import tf
import math as math
import copy
import config

from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point32
# from mavros_msgs.msg import State
# from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
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

        self.goal_space = config.goal_space[0]
        self.start_space = config.start_space[0]
        self.obstacle_pos = config.obstacle_position
        self.des = Point32()

        self.p = [21.0, 0.0]
        self.success = False
        self.dist_init = 0
        self.dist = 0
        self.reward = 0
        self.obstacle_state = []
        self._actions = []
        self.stacked_imgs = None

        # ------------------------Publisher and Subscriber---------------------------
        # self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        # self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

        self.resized_depth_img = rospy.Publisher('/camera/depth/image_resized', Image, queue_size=10)
        self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
        # self.state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)
        # self.image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.DepthImageCallBack)
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image, self.DepthImageCallBack)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.sleep(1.)

    def ModelStateCallBack(self, data):
        idx = data.name.index("quadrotor")
        quaternion = (data.pose[idx].orientation.x,
                      data.pose[idx].orientation.y,
                      data.pose[idx].orientation.z,
                      data.pose[idx].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        self.self_state = [data.pose[idx].position.x,
                           data.pose[idx].position.y,
                           yaw,
                           data.twist[idx].linear.x,
                           data.twist[idx].linear.y,
                           data.twist[idx].angular.z]

        if self.default_states is None:
            self.default_states = copy.deepcopy(data)

        # print("UAV has been reset!")

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def GetDepthImageObservation(self): \
            # Ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "bgr8")
            # cv2.imwrite('input1.png', cv_img)
            # print("cv_img's shape1 = ", cv_img.shape, type(cv_img), cv_img.dtype)
            cv_img = np.array(cv_img, dtype=np.int8)
            # Resize
            # dim = (self.depth_image_size[0], self.depth_image_size[1])  # 160, 120
            # cv_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
            # cv_img = cv2.resize(cv_img, dim)
            cv_img[np.isnan(cv_img)] = 0
            # cv_img[cv_img < 0.5] = 0
            # cv2.imwrite('input.png', cv_img)
            # np.save('state_cv.npy', cv_img)
            # print("cv_img's shape2 = ", cv_img.shape)
            # cv_img = cv_img.reshape(48, 64, 1)
            # print("Received an image!")
            return cv_img
        except Exception as err:
            print("Ros_to_Cv2 Failure: %s" % err)

        # Cv2 image to ros image and publish
        # try:
        # resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        # self.resized_depth_img.publish(resized_img)

        # except Exception as err:
        #     print("Cv2_to_Ros Failure: %s" % err)

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
        for i in range(len(self.obstacle_pos)):
            e, _ = self.obstacle2robot(self.obstacle_pos[i][0], self.obstacle_pos[i][1])
            if e < 1.2:
            # if self.obstacle2robot(self.obstacle_pos[i][0], self.obstacle_pos[i][1]) < 1.2:
                collision = True

        return collision

    def get_states(self):
        self.stacked_imgs = np.dstack([self.stacked_imgs[:, :, -9:], self.GetDepthImageObservation()])
        return self.stacked_imgs, self.p

    def get_actions(self):
        return self._actions

    def GetSelfSpeed(self):
        return self.vel_cmd

    def SetObjectPose(self, x, y, theta):
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

        # rospy.wait_for_service('/gazebo/set_model_state')
        # try:
        #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #     result = set_state(state)
        #     assert result.success is True
        #     print("set the model state successfully")
        # except rospy.ServiceException:
        #     print("/gazebo/get_model_state service call failed")

    def set_goal(self, x, y):
        self.des.x = x
        self.des.y = y
        self.des.z = 1

    def state_cb(self, msg):
        global current_state
        current_state = msg

    def takeoff_SITL(self):
        state_sub = self.state_sub
        local_pos_pub = self.local_pos_pub
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
        theta = -1.0 / 2 * math.pi
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
        offb_set_mode.custom_mode = 'offboard'

        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        last_req = rospy.Time.now()
        start_time = rospy.Time.now()

        # while (not rospy.is_shutdown()):
        while ((rospy.Time.now() - start_time) < rospy.Duration(25.0)):
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


    def takeoff_no_SITL(self):
        state = ModelState()
        state.model_name = 'quadrotor'
        state.reference_frame = 'world'
        state.pose.position.x = 0
        state.pose.position.y = 11
        state.pose.position.z = 1
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.set_state.publish(state)
            rate.sleep()

    def takeoff_gazebo(self):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # self.unpause
            rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        move_cmd = Twist()
        move_cmd.linear.z = 1
        self.vel_pub.publish(move_cmd)
        time.sleep(0.1)
        move_cmd.linear.z = 0
        self.vel_pub.publish(move_cmd)
        time.sleep(0.5)
        # print("Start Position: ", self.start_space[0], self.start_space[1], "Goal Position: ", self.goal_space[0],
        #         self.goal_space[1])

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")


    def reset(self):
        start = self.start_space
        theta = - math.pi / 2
        self.SetObjectPose(start[0], start[1], theta)
        goal = self.goal_space
        self.set_goal(goal[0], goal[1])
        # self.takeoff()
        # print("Start Position: ", start[0], start[1], "Goal Position: ", goal[0], goal[1])
        d0, alpha0 = self.goal2robot(goal[0] - start[0], goal[1] - start[1], theta)
        self.p = [d0, alpha0]
        self.reward = 0
        self.dist_init = d0
        self.vel_cmd = [0.0]
        self.success = False
        self.stacked_imgs = np.dstack([self.GetDepthImageObservation()] * 4)
        # np.save('state_stacked_before', self.GetDepthImageObservation())
        # self.obstacle_state = []
        # for i in range(len(self.obstacle_pos)):
        #     e, beta = self.obstacle2robot(self.obstacle_pos[i][0], self.obstacle_pos[i][1])
        #     self.obstacle_state.append(e)
        #     self.obstacle_state.append(beta)
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

        ts = time.time()
        while time.time() - ts <= 0.5:
            if self.DetectCollision():
                move_cmd.linear.x = 0
                move_cmd.linear.y = 0
                move_cmd.angular.z = 0
                self.vel_pub.publish(move_cmd)
                break
        # time.sleep(0.2)  # execute time

        self.vel_cmd = [angular_z]
        d1, alpha1 = self.getdist()
        self.p = [d1, alpha1]
        # self.obstacle_state = []
        # for i in range(len(self.obstacle_pos)):
        #     e, beta = self.obstacle2robot(self.obstacle_pos[i][0], self.obstacle_pos[i][1])
        #     self.obstacle_state.append(e)
        #     self.obstacle_state.append(beta)
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
