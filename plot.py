#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

m = 100
path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/'
ep_reward_arr = np.loadtxt(path + 'DQN_Reward_1st.txt')
n = len(ep_reward_arr) // m
avg_reward_arr = np.mean(np.reshape(ep_reward_arr[:m*n], [n, m]), 1)
plt.plot(avg_reward_arr)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('./reward.jpg')
plt.show()

# print(x)
# print(y)
# plt.plot(y, x)

# plt.title('Reward-Episode Curve')
# plt.show()
