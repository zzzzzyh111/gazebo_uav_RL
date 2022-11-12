#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        # cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2_img = bridge.imgmsg_to_cv2(msg, "32FC1")
    except CvBridgeError as err:
        print("Receive Failure: %s" %err)
    else:
        #Save your OpenCV2 image as a png
        cv2.imwrite('/home/yuhang/catkin_ws/src/uav/Images/camera_image_depth.png', cv2_img)
        # cv2.imwrite('/home/zyh/catkin_ws/src/vision/Images/camera_image.png', cv2_img)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    # image_topic = "/iris/usb_cam/image_raw"
    image_topic = "/camera/depth/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()











