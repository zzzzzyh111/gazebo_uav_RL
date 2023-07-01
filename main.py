#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import ddqn
import numpy as np
import time
import torch
import rospy
from std_srvs.srv import Empty

GazeboUAV = env.GazeboUAV()
agent = ddqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
                gamma=0.99, learning_rate=1e-4, eps=0.95, eps_min=0.1, eps_period=5000, network='DQN')

model_path = '/home/zyh/catkin_ws/src/Uav/scripts/Record/'
# Load pre-trained model
param_path = '/home/zyh/catkin_ws/src/Uav/scripts/Record/'
# agent.load_model(param_path, map_location=torch.device('cuda：0'))
if not os.path.exists(model_path):
    os.makedirs(model_path)
total_episode = 15000
max_step_per_episode = 70

ep_reward_list = []
for i_episode in range(total_episode + 1):
    if (i_episode % 10 == 0 and i_episode != 0):
        GazeboUAV.SetObjectPose_random()
    else:
        GazeboUAV.SetObjectPose()
    state1, state2 = GazeboUAV.reset()
    time.sleep(0.5)

    ep_reward = 0
    # print('开始交互')
    for t in range(max_step_per_episode):
        action = agent.get_action(state1, state2)
        # print('action = ', action)
        # next_state1, next_state2, terminal, reward = GazeboUAV.execute(action)
        GazeboUAV.execute(action)
        ts = time.time()
        if len(agent.replay_buffer.memory) > 64:
            agent.learn()
        while time.time() - ts <= 0.5:
            continue
        # print("学习时长：", time.time() - ts)
        # ts = time.time()
        next_state1, next_state2, terminal, reward = GazeboUAV.step()
        # print("state获取时长：", time.time() - ts)
        # ts = time.time()
        ep_reward += reward
        # print('ep_reward = ', reward)
        agent.replay_buffer.add(state1, state2, action, reward, next_state1, next_state2, terminal)
        # print("buffer_add时长：", time.time() - ts)
        # print(time.time() - ts)
        if terminal:
            break

        state1 = next_state1
        state2 = next_state2

    ep_reward_list.append(ep_reward)
    print("Episode:{} step:{} ep_reward:{} epsilon:{}".format(
        i_episode, t, ep_reward, round(agent.eps, 4)))

    if (i_episode + 1) % 100 == 0:
        np.savetxt(model_path + 'DQN_home_sup.txt', ep_reward_list)
        agent.save_model(model_path + 'DQN_home_sup.pth')
