
# Reinforcement-learning-based Soft-Actor-Critic Trajectory Planning Agent for Robot-assisted Transcranial Magnetic Stimulation

An experimental Reinforcement learning project with KUKA lbr iiwa 14 using ROS2. The RL algorithm used in this project is the Soft-Actor-Critic (SAC). As of now, everything works with ROS2 Jazzy, Gazebo Harmonic (binary installation), running on Ubuntu 24.04.
ROS 2 packages for the KUKA LBR, including Gazebo simulation support is from [lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack).

The SAC-algorithm used in this project is modified from [this repository](https://github.com/pranz24/pytorch-soft-actor-critic).

All RL in this project is done with PyTorch. The code features a custom environment utilizing OpenAI gymnasium. Our goal is to train the KUKA cobot to move the end-effector near any arbitrary point sampled 
from the surface of a head model, with the end-effector oriented along the normal of the head surface. The cobot is equipped with a 3D replica of a transcranial magnetic stimulation coil.

The project features a few gazebo assets [here](https://github.com/Miroryt/rl-tms-navigation/tree/master/src/gazebo), including a world "my_world.sdf", which has a head model spawned on the air along with the KUKA cobot. Additionally, a red transparent TMS coil ("coil_xy_short.stl") is spawned to the goal position in the desired
orientation to help visualize the goal of the agent and better evaluate the performance of the neural network.

<img src="https://github.com/Miroryt/rl-tms-navigation/blob/main/sac_demo.gif" alt="demo gif" width="500"/>


## Quick Start (Linux)

> [!CAUTION]
> It is recommended to run this program using a system with a dedicated GPU.

To run this program, follow the steps in [lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack): 
- Install ROS2 dev tools:
`sudo apt install ros-dev-tools`

- Download Gazebo (Harmonic, again other versions might work as well)

- Create a workspace, clone, and install dependencies

```shell  
source /opt/ros/jazzy/setup.bash
export FRI_CLIENT_VERSION=1.15
mkdir -p lbr-stack/src && cd lbr-stack
vcs import src --input https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/jazzy/lbr_fri_ros2_stack/repos-fri-${FRI_CLIENT_VERSION}.yaml
rosdep install --from-paths src -i -r -y`
```

- Build

`colcon build --symlink-install`

And now you're done with lbr-stack. Next

- Install python 3 (I had python 3.12):
- Create a virtual environment:

```shell
sudo apt install python3-venv -y
python3 -m venv my_env
source my_env/bin/activate
cd ..
```

- In this virtual env:
- Clone this repository
- Install requirements

```shell
cd rl-tms-navigation/src
pip install -r requirements.txt
```

- Also make sure you have these ROS2 packages installed:

```shell
sudo apt update
sudo apt install ros-jazzy-ament-index-python \
                 ros-jazzy-rclpy \
                 ros-jazzy-launch-ros \
                 ros-jazzy-moveit \
                 ros-jazzy-geometry-msgs \
                 ros-jazzy-sensor-msgs \
                 ros-jazzy-std-msgs \
                 ros-jazzy-trajectory-msgs
```

> [!CAUTION]
As of now, there is a few pathing issues that need to be resolved. In order to launch the "my_world.sdf" -file, you need to change a few files and paths.
First, inside lbr-stack [gazebo.launch.py](https://github.com/lbr-stack/lbr_fri_ros2_stack/blob/jazzy/lbr_bringup/launch/gazebo.launch.py) and [gazebo.py](https://github.com/lbr-stack/lbr_fri_ros2_stack/blob/jazzy/lbr_bringup/lbr_bringup/gazebo.py),
the default world file in the launch arguments is "empty.sdf". If you want the head model to spawn, and have the KUKA oriented similarly to our experimental setup, you need to launch the "my_world.sdf" file, found in \src\gazebo inside this repository.
Furthermore, if you want to mount the TMS coil model to the KUKA end-effector, you need to replace the "lbr_iiwa14_r820_macro.xacro" file with the one found in this repository inside /gazebo, 
and change the path to "coil_xy.stl" to match your device. The file "lbr_iiwa14_r820_macro.xacro" contains the robot description, and it has been modified such that
the end-effector link has a mesh called "coil_xy.stl" (TMS-coil).


Now the lbr-stack directory handles the simulation. In a new terminal:

```shell
cd lbr-stack
source install/setup.bash
```

- And now you can launch Gazebo with
  
```shell
ros2 launch lbr_bringup gazebo.launch.py \
ctrl:=joint_trajectory_controller \
model:=iiwa14
```

- The repository contains a pre-trained model in \checkpoints\sac_checkpoint_agent. In another terminal:
  
```shell
cd lbr-stack
source install/setup.bash
source ~my_env/.venv/bin/activate
cd --
cd /rl-tms-navigation/src
python3 Evaluation.py
```

--> And the Robot should start moving in Gazebo.



## Training neural networks

For neural network training, the environment features a 'training-mode', where actual commands to the cobot via a ROS2 topic is skipped to reduce training times. Instead of subscribing to a joint state topic to gather joint angle information,
joint states are calculated from the initial joint states by adding the actions done by the SAC algorithm. You can train new agents by running main.py.

In addition, the program includes the option to set an SAC agent's actions to be the cobot's end-effector cartesian coordinates + orientation as Euler angles (3+3 vector). If an agent is trained with EE position + orientation as action, the cobot's corresponding joint angles are computed with inverse kinematics via "ikpy" library (https://github.com/Phylliade/ikpy).
**According to our experiments, having the EE position + orientation as action yields better performance in comparison to having the cobot's joint angles as action.**

## Data

The performance evaluation data of the pre-trained model is logged in [/src/evaluation_data](https://github.com/Miroryt/rl-tms-navigation/tree/master/src/evaluation_data), with 150 episodes of navigation data with both a moving target and a static target.