#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

m = 100
path = '/home/zyh/catkin_ws/src/UAV/scripts/Record/'
ep_reward_arr_1 = np.loadtxt(path + 'DQN_Reward_1st.txt')
ep_reward_arr_2 = np.loadtxt(path + 'DDQN_Reward_lab1.txt')
n = len(ep_reward_arr_1) // m
avg_reward_arr_1 = np.mean(np.reshape(ep_reward_arr_1[:m*n], [n, m]), 1)
n = len(ep_reward_arr_2) // m
avg_reward_arr_2 = np.mean(np.reshape(ep_reward_arr_2[:m*n], [n, m]), 1)

plt.plot(avg_reward_arr_1, label='DQN')
plt.plot(avg_reward_arr_2, label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='best')
# plt.savefig('./reward.jpg')
plt.show()

# print(x)
# print(y)
# plt.plot(y, x)

# plt.title('Reward-Episode Curve')
# plt.show()
