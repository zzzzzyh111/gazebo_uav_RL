#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import env
import time
import ddqn

# GazeboUAV = env.GazeboUAV()
# agent = ddqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
#                 gamma=0.99, learning_rate=1e-4, eps=0, eps_min=0, eps_period=5000, network='Duel')
# success = False
# param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Duel_DQN_Reward_home2_sup.pth'
pos_path='/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Obstacle_Pos.txt'
tra_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Path.txt'
# agent.load_model(param_path, map_location=torch.device('cpu'))
# for i_episode in range(100):
#     if not success:
#         if (i_episode % 10 == 0 and i_episode != 0):
#             GazeboUAV.SetObjectPose_random()
#         else:
#             GazeboUAV.SetObjectPose()
#         state1, state2 = GazeboUAV.reset()
#         time.sleep(0.5)
#         GazeboUAV.uav_trajectory = [[], []]
#         for i in range(100):
#             action = agent.get_action(state1, state2)
#             GazeboUAV.execute(action)
#             time.sleep(0.3)
#             next_state1, next_state2, terminal, reward = GazeboUAV.step()
#             if GazeboUAV.success == True:
#                 success = True
#                 break
#             if terminal:
#                 break
#             state1 = next_state1
#             state2 = next_state2
#     else:
#         uav_trajectory = GazeboUAV.uav_trajectory
#         cylinder_pos = GazeboUAV.cylinder_pos
#         np.savetxt(tra_path, uav_trajectory)
#         np.savetxt(pos_path, cylinder_pos)
# #         # GazeboUAV.reset()
#         # time.sleep(0.5)
#         break

figure = plt.figure(1, figsize=(10,10))
ax = figure.add_subplot(111)
a = np.loadtxt(pos_path)
b = np.loadtxt(tra_path)
print(b[0][0], b[1][0])
print(b[0][-1], b[1][-1])
# a = [[a[i][0] for i in range(len(a))], [a[i][1] for i in range(len(a))]]
# for i in range(len(a[0])):
#     circle = Circle(xy=(a[0][i], a[1][i]), radius=0.5, color='k', alpha=1)
#     ax.add_patch(circle)
# plt.plot(b[0], b[1], color='r', linewidth=2)
# plt.grid(linestyle='-.')
# plt.axis('equal')
# plt.show()




# plt.plot(b[0], b[1], color='b', linewidth=3)

# cir1 = Circle(xy=[0.5,0.5], radius=0.5, color='k', alpha=0.5)
# cir2 = Circle(xy=[1,1], radius=0.5, color='k', alpha=0.5)
# ax = fig.add_subplot(111)
# ax.add_patch(cir1)
# ax.add_patch(cir2)
# plt.plot(a[0], a[1])
