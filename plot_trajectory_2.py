#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import env
import time
import ddqn

GazeboUAV = env.GazeboUAV()
GazeboUAV.unpause()
agent = ddqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
                gamma=0.99, learning_rate=1e-4, eps=0, eps_min=0, eps_period=5000, network='Duel')
success = False
# param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/ddqn_lab1.pth'
# param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/ddqn_lab1.pth'
param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Duel_DQN_Reward_home2_sup.pth'
pos_path='/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Obstacle_Pos_scene3.txt'
tra_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Path_scene3_DQN.txt'

agent.load_model(param_path, map_location=torch.device('cpu'))
for i_episode in range(100):
    if not success:

        GazeboUAV.SetObjectPose()
        state1, state2 = GazeboUAV.reset()
        time.sleep(0.5)
        GazeboUAV.uav_trajectory = [[], []]
        for i in range(100):
            action = agent.get_action(state1, state2)
            GazeboUAV.execute(action)
            time.sleep(0.3)
            next_state1, next_state2, terminal, reward = GazeboUAV.step()
            if GazeboUAV.success == True:
                uav_trajectory = GazeboUAV.uav_trajectory
                cylinder_pos = GazeboUAV.cylinder_pos
                np.savetxt(tra_path, uav_trajectory)
                np.savetxt(pos_path, cylinder_pos)
                print('step = ', i)
                # GazeboUAV.pause()
                success = True
                break
            if terminal:
                break
            state1 = next_state1
            state2 = next_state2
    # else:
    #     uav_trajectory = GazeboUAV.uav_trajectory
    #     cylinder_pos = GazeboUAV.cylinder_pos
    #     np.savetxt(tra_path, uav_trajectory)
    #     np.savetxt(pos_path, cylinder_pos)
#         # GazeboUAV.reset()
        # time.sleep(0.5)
        # break