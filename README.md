# ADARL ENVIRONMENTS

This repository is a collection of environments and tools for Reinforcement Learing implemented with ADARL (https://github.com/c-rizz/adarl).
Environments implemented with adarl can be executed on simulation or real hardware interchangeably.
No need for deploy implementations or configs.

## Setup

To use the implemented environments and build upon them you will need to set up a python workspace with the aappropriate dependencies.
I suggest you use uv instead of pip.

You will need to install:
 * [ADARL](https://github.com/c-rizz/adarl): provides the interfaces to simulators (MJX, Genesis, PyBullet) and real robots (via Xbot zmq)
 * [RREAL](https://github.com/c-rizz/rreal): provides RL algorithms implementations
 * [PyDescriptions](https://github.com/c-rizz/pydescriptions): wraps URDF description packages in a python-installable package

For now, I suggest you install those as editable-installs.

```
cd ~
mkdir -p adarl_ws/src
cd adarl_ws/src
git clone https://github.com/c-rizz/adarl --branch crzz-dev
git clone https://github.com/c-rizz/adarl_envs --branch crzz-dev
git clone https://github.com/c-rizz/rreal --branch crzz-dev
git clone https://github.com/c-rizz/pydescriptions --branch crzz-dev
cd ..
mkdir virtualenv
uv venv --python=3.13 virtualenv/adarl313
. virtualenv/adarl313/bin/activate

uv pip install -r src/adarl/requirements_py313.txt
uv pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pydescriptions
uv pip install git+https://github.com/c-rizz/joyzmq # For joystick deployment
```

# Running an Example

If you followed the instructions before you should be able to launch an example training with:

```
cd ~/adarl_ws
. virtualenv/adarl313/bin/activate
./src/adarl_envs/src/adarl_envs/experiments/loco_kyon_ppo.py --comment "example training"
```

On your first run you may get in the terminal a prompt from wandb asking you to log in.

You can monitor the training in the terminal and pause it by typing 'pause' and pressing enter (It may take a bit to pause).
Once it is paused you can resume or stop the training.
You can also terminate it by repatedly pressing CTRL+C.

Outputs for the runs will be saved in adarl_ws/lrg_exps/loco_kyon_ppo.py. There you can find checkpints, videos, and all information regarding the run.

# Running a checkpoint

Once the policy is trained you can deploy it through the smae python environment on which it was trained.
You just need a simple python control loop.

You can deploy the the policy trained with loco_kyon_ppo using loco_play_vec1.py.
With this script you can deploy to XBot (either simulation or reality ) or Mujoco C++ by using the --mode option.
For controlling the robot you can use the [joyzmq](https://github.com/c-rizz/joyzmq) package.

```
cd ~/adarl_ws
. virtualenv/adarl313/bin/activate
./src/adarl_envs/src/adarl_envs/experiments/loco_play_vec1.py --comment "example play" --pretrained lrg_exps/loco_kyon_ppo.py/<run_id>/seed_0/checkpoints/
```

In a separate terminal (use joyzmq-joystick in place of joyzmq-keyboard if you have a proper joystick):
```
uvx --from git+https://github.com/c-rizz/joyzmq joyzmq-keyboard --bind tcp://127.0.0.1:5666
```



# Framework Overview

The overall framework is composed of 3 main packages: 

* **ADARL** - **Adapters for Robot Learning**: Provides interfaces to simulators and real-world hardware, together with general tools and utilities. The interfaces follow a common structure, allowing to develop environments that can be deployed on different simlators/hardwares by simply swapping the underlying adapter. Interfaces are provided for MJX and PyBullet, interfaces for ROS/XBOT are also available, but are for now provided in the ADARL_ROS package (for dependency reasons).
* **RREAL** - **Robot Reinforcement Algorithms**: Proides implementations of reinforcement learnign algorithms for robotics, with interfaces for ADARL environments.
* **ADARL_ENVS**: Environment implementations and examples based on ADARL and RREAL. Currently focuses on quadruped locomotion, with the LocomotionVecEnv environment.

In addition to this othere relevant packages are **pykyon** and **pycentauro**, which wrap the the HHCM kyon and centauro repositories in standard python packages.
In this way they can be indexed sing standard python tooling, and, using this [fork of xacro](https://github.com/c-rizz/xacro_standalone/tree/extra_find_pkg_path), XACRO files can be directly compiled without using ROS (hopefully the xaro fork will eventually be merged, see [xacro issue #356](https://github.com/ros/xacro/pull/356)).


# Package Overview

In this package you can find:

* **env/RobotVecEnv.py**, a base environment for implementing robotics environment
* **env/LocomotionVecEnv.py**, a locomotion environment, based on RobotVecEnv
* **env/GraspVecEnv**, a simple grasping envoronment, again based on RobotVecEnv (still very much a work in progress)