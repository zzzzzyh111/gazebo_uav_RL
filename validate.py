#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import env
import dqn
import time



GazeboUAV = env.GazeboUAV()
agent = dqn.DQN(GazeboUAV, batch_size=64, memory_size=10000, target_update=4,
                gamma=0.99, learning_rate=1e-4, eps_min=0, eps_period=5000)
param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/'
agent.load_model(param_path + '/dqn_1st.pth')
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