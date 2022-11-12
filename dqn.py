#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch
import rospy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import env
import config
import random
import numpy as np
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    # Add the replay memory
    def add(self, state1, state2, action, reward, next_state1, next_state2, done):
        self.memory.append((state1, state2, action, reward, next_state1, next_state2, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states1, states2, actions, rewards, next_states1, next_states2, dones = map(np.stack, zip(*batch))
        return states1, states2, actions, rewards, next_states1, next_states2, dones


class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 6), stride=8)
        self.cnn_2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.cnn_3 = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.fc_target = nn.Linear(2, 64)
        self.fc_1 = nn.Linear(64 + 64, 256)
        # self.fc_1 = nn.Linear(192 + 64, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 5)

        self.fc_test1 = nn.Linear(22, 128)
        self.fc_test2 = nn.Linear(128, 128)
        self.fc_test3 = nn.Linear(128, 5)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1/255
        x1 = F.relu(self.cnn_1(img.transpose(1, 3)))
        x2 = F.relu(self.cnn_2(x1))
        x3 = self.pool_1(x2)
        x4 = F.relu(self.cnn_3(x3))

        x_target = F.relu(self.fc_target(state2))
        x_merge = torch.cat((x4.view(batch_size, -1), x_target), axis=1)
        fc_1 = F.relu(self.fc_1(x_merge))
        fc_2 = F.relu(self.fc_2(fc_1))
        fc_3 = F.relu(self.fc_2(fc_2))
        x_output = self.output(fc_3)
        # x1 = F.relu(self.fc_test1(state))
        # x2 = F.relu(self.fc_test2(x1))
        # x_output = self.fc_test3(x2)
        return x_output


class DQN():
    def __init__(self, env, memory_size=50000, learning_rate=4e-5, batch_size=32, target_update=1000,
                 gamma=0.95, eps=1, eps_min=0.1, eps_period=2000):
        super(DQN, self).__init__()
        self.env = env

        # Torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Deep Q network
        self.predict_net = DQNNet().to(self.device)
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Target network
        self.target_net = DQNNet().to(self.device)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.target_update = target_update
        self.update_count = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Learning setting
        self.gamma = gamma

        # Exploration setting
        self.eps = eps
        self.eps_min = eps_min
        self.eps_period = eps_period

    # Get the action
    def get_action(self, state1, state2):
        # Random action
        if np.random.rand() < self.eps:
            self.eps = self.eps - (1 - self.eps_min) / self.eps_period if self.eps > self.eps_min else self.eps_min
            return np.random.randint(0, 5)

        # Get the action
        state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
        state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
        q_values = self.predict_net(state1, state2).cpu().detach().numpy()
        action = np.argmax(q_values)

        return action

    # Learn the policy
    def learn(self):
        # Replay buffer
        states1, states2, actions, rewards, next_states1, next_states2, dones = self.replay_buffer.sample(self.batch_size)
        states1 = torch.FloatTensor(states1).to(self.device)
        states2 = torch.FloatTensor(states2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states1 = torch.FloatTensor(next_states1).to(self.device)
        next_states2 = torch.FloatTensor(next_states2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # Calculate values and target values
        target_values = (rewards + self.gamma * torch.max(self.target_net(next_states1, next_states2),
                                                          axis=1)[0] * (1 - dones)).view(-1, 1)
        predict_values = self.predict_net(states1, states2).gather(1, actions.view(-1, 1))

        # Calculate the loss and optimize the network
        loss = self.loss_fn(predict_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update the target network
        self.update_count += 1
        if self.update_count == self.target_update:
            self.target_net.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0

    def save_model(self, filename):
        torch.save(self.predict_net.state_dict(), filename)
