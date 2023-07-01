#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import spline

ddqn = np.loadtxt('./dqn_model_DDQN2/rMean.txt')
d3qn = np.loadtxt('./dqn_model_D3QN3/rMean.txt')

x = np.array([i for i in range(1000)])
# xnew = np.linspace(x.min(),x.max(),100)

# ddqn_smooth = spline(x,ddqn,xnew)
# d3qn_smooth = spline(x,d3qn,xnew)


rMat1 = np.resize(np.array(d3qn),[len(d3qn)//10,10])
rMean_d3qn = np.average(rMat1,1)

rMat2 = np.resize(np.array(ddqn),[len(d3qn)//10,10])
rMean_ddqn = np.average(rMat2,1)

plt.plot(x/10, ddqn+15, color = '#CDC9C9',linewidth=5)
plt.plot(x/10, d3qn+5, color = '#F5DEB3',linewidth=5)
plt.plot(rMean_ddqn+15,'--',label = 'DDQN',linewidth=2.5)
plt.plot(rMean_d3qn+5,'-',label = 'D3QN',linewidth=2.5)

plt.xlabel('Episodes')
plt.ylabel('Cumulative rewards')
plt.legend()
plt.show()
