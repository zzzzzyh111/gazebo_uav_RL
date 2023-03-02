#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}

rcParams.update(config)
m = 100


# path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/'
path = '/home/zyh/catkin_ws/src/UAV/scripts/Record/'
ep_reward_arr_1 = np.loadtxt(path + 'DDQN_Reward_lab1.txt')
# ep_reward_arr_2 = np.loadtxt(path + 'DDQN_Reward_home1.txt')
# ep_reward_arr_3 = np.loadtxt(path + 'Duel_DQN_Reward_home2_sup.txt')
n = 10000 // m
avg_reward_arr_1 = np.mean(np.reshape(ep_reward_arr_1[:m*n], [n, m]), 1)
print(len(avg_reward_arr_1))
# n = 15000// m
# avg_reward_arr_2 = np.mean(np.reshape(ep_reward_arr_2[:m*n], [n, m]), 1)
# n = len(ep_reward_arr_3) // m
# avg_reward_arr_3 = np.mean(np.reshape(ep_reward_arr_3[:m*n], [n, m]), 1)
plt.xlim((0, 150))
x_ticks = np.linspace(0, 150, 16)
plt.xticks(x_ticks)
plt.plot(avg_reward_arr_1, color='dodgerblue', linewidth=2.0,label='DQN')
# plt.plot(avg_reward_arr_2, label='Double-DQN')
# plt.plot(avg_reward_arr_3, label='Duel-DQN')
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Average Reward', fontsize=15)
plt.legend(loc='best')
# plt.savefig('./reward.jpg')
plt.show()

# print(x)
# print(y)
# plt.plot(y, x)

# plt.title('Reward-Episode Curve')
# plt.show()
