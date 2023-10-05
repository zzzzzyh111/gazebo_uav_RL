#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import env
import ddqn
import time
import torch



GazeboUAV = env.GazeboUAV()
agent = ddqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
                gamma=0.99, learning_rate=1e-4, eps=0, eps_min=0, eps_period=5000)
param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Duel_DQN_Reward_home2_sup.pth'
# param_path = '/home/zyh/catkin_ws/src/UAV/scripts/Record/dqn_1st.pth'
agent.load_model(param_path, map_location=torch.device('cpu'))
while True:
    state1, state2 = GazeboUAV.reset()
    time.sleep(0.5)
    for i in range(100):
        action = agent.get_action(state1, state2)
        GazeboUAV.execute(action)
        time.sleep(0.3)
        next_state1, next_state2, terminal, reward = GazeboUAV.step()
        if terminal:
            break
        state1 = next_state1
        state2 = next_state2
