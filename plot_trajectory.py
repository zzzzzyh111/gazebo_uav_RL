#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib import rcParams
import numpy as np
import env
import time
import ddqn


config = {
    "font.family":'Times New Roman',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}

rcParams.update(config)

# GazeboUAV = env.GazeboUAV()
# agent = ddqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
#                 gamma=0.99, learning_rate=1e-4, eps=0, eps_min=0, eps_period=5000, network='Duel')
# success = False
# param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Duel_DQN_Reward_home2_sup.pth'
pos_path='/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Obstacle_Pos_scene1.txt'
tra_path_1 = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Path_scene1_DQN_1.txt'
tra_path_2 = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Path_scene1_Double_DQN_1.txt'
tra_path_3 = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Path_scene1_Duel_DQN_1.txt'

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
b = np.loadtxt(tra_path_1)
c = np.loadtxt(tra_path_2)
d = np.loadtxt(tra_path_3)
a = [[a[i][0] for i in range(len(a))], [a[i][1] for i in range(len(a))]]
# plt.fill([7.425, 7.425, -7.425, -7.425], [13, -13, -13, 13], 'k', alpha=0)
plt.plot((7.45, 7.45), (11, -11), 'saddlebrown', linewidth=7, linestyle='-')
plt.plot((-7.45, -7.45), (11, -11), 'saddlebrown', linewidth=7, linestyle='-')
for i in range(len(a[0])):
    circle = Circle(xy=(a[0][i], a[1][i]), radius=0.5, color='darkorange', alpha=1)
    ax.add_patch(circle)
plt.plot(b[0], b[1], c='dodgerblue', linewidth=3, linestyle=':')
plt.plot(c[0], c[1], c='darkviolet', linewidth=3, linestyle=':')
plt.plot(d[0], d[1], c='crimson', linewidth=3, linestyle=':')
plt.plot(b[0][0], b[1][0], c='green', marker='v', markersize=20,)
# plt.text(b[0][0], b[1][0], 'Starting Point', color='r', fontsize=18, position=(b[0][0]+0.5, b[1][0]),
#          verticalalignment='bottom')
plt.plot(b[0][-1], b[1][-1], c='red', marker='*', markersize=20,)
# plt.text(b[0][-1], b[1][-1], 'Target Point', color='r', fontsize=18, position=(b[0][-1]+0.5, b[1][-1]),
#          verticalalignment='bottom')
# plt.text(a[1][0], a[1][1], 'Obstacle', fontsize=18, position=(5.5, a[1][1]),
#          verticalalignment='bottom')
plt.grid(linestyle='-')
plt.axis('equal')
plt.xlim((-17, 17))
plt.ylim((-17, 17))
plt.yticks(size=14)
plt.xticks(size=14)
plt.xlabel('$x \ [\mathrm{m}]$', fontsize=16)
plt.ylabel('$y \ [\mathrm{m}]$', fontsize=16)
legend_elements = [Line2D([0], [0],linestyle=':', linewidth=3,color='dodgerblue', label='DQN'),
                   Line2D([0], [0],linestyle=':', linewidth=3,color='darkviolet', label='Double DQN'),
                   Line2D([0], [0],linestyle=':', linewidth=3,color='crimson', label='Dueling DQN'),
                   Line2D([0], [0],marker='v', color='w', label='Starting Point', markerfacecolor='g',
                          markersize=15),
                   Line2D([0], [0], marker='*', color='w', label='Target Point', markerfacecolor='r',
                          markersize=20),
                   Line2D([0], [0], marker='o', color='w', label='Obstacle', markerfacecolor='darkorange',
                          markersize=13),
                   Line2D([0], [0],linestyle='-', linewidth=7,color='saddlebrown', label='Bound')
                   ]
plt.legend(handles=legend_elements, loc='best', fontsize=14)
plt.title('Flight Trajectory', fontsize='24')
plt.savefig('/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Tra_Scene1.svg', dpi=600)
plt.show()




# plt.plot(b[0], b[1], color='b', linewidth=3)

# cir1 = Circle(xy=[0.5,0.5], radius=0.5, color='k', alpha=0.5)
# cir2 = Circle(xy=[1,1], radius=0.5, color='k', alpha=0.5)
# ax = fig.add_subplot(111)
# ax.add_patch(cir1)
# ax.add_patch(cir2)
# plt.plot(a[0], a[1])
