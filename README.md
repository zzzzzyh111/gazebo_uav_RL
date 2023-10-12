# UAV Obstacle Avoidance-DQN-Gazebo
Based on the Gazebo simulation, this repository focuses on the UAV obstacle avoidance problem, which leverages visual inputs and the DQN algorithm.
![](Record/Image/Trajectory.gif)  
If you have your own gazebo world file, please run the code **after roslaunch**! From my side, the roslaunch command is
```
roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch
```
Please make sure that the relevant dependencies have been installed before the training process
```
sudo apt-get update
sudo apt-get upgrade
pip install rospkg
pip install catkin-tools
pip install opencv-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # I'm using torch2.0.1+cuda11.7
```
As for the other dependencies, please install them according to your error messages.  

After that, you can start training straight away by running
```
python3 main.py
```
Moreover, you can define your own goal/start/obstacle positions in [config.py](./config.py). Definitions regarding the simulation environment, such as resetting the environment, can be found in [env.py](./env,py). [ddqn.py](./ddqn.py) describes the DQN algorithm and its derivatives. [validate.py](./validate.py) is used to test the parameters.

Since the training process requires real physical time to give the drone time to perform the maneuvers, the training time is long (about 10 hours or so to see the REWARD noticeably rise phenomenon), so please be patient. If you need to do gazebo simulation acceleration, please refer to [this website](https://answers.gazebosim.org//question/12477/speeding-up-gazebo-physics-simulation-considering-ros-plugin-usage/). The training curves are shown below.
<div align="center">
  <img src="Record/Image/Reward_Comparison-1.png" alt="Reward Comparison" width="400">
</div>
The collision-free trajectories are shown below.
<div align="center">
    <img src="Record/Image/Tra_Env1-1.png" alt="Tra1" width="150">
    <img src="Record/Image/Tra_Env2-1.png" alt="Tra2" width="150">
    <img src="Record/Image/Tra_Env3-1.png" alt="Tra3" width="150">
</div>
This work has been published in [AIAA Aviation 2023 Forum](https://arc.aiaa.org/doi/abs/10.2514/6.2023-3813). Please kindly refer to the following format for citations if needed

```bibtex
@inproceedings{zhang2023partially,
  title={Partially-Observable Monocular Autonomous Navigation for UAV through Deep Reinforcement Learning},
  author={Zhang, Yuhang and Low, Kin Huat and Lyu, Chen},
  booktitle={AIAA AVIATION 2023 Forum},
  pages={3813},
  year={2023}
}
```

