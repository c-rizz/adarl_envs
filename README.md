# ADARL ENVIRONMENTS

This repository is a collection of environments and tools for robotic environments implemented with ADARL.
For now it is focused on an environment for quadruped locomotion.

### Setup - Auto

You can try to install everything using the provided setup script.
It hasn't been thoroughly tested, so it may not work, but it's worth a try.

```
bash <(curl -s https://gitlab.com/-/snippets/4863010/raw/main/setup_full.sh)
```

You can find the script also in the repo main folder.
Let me know if it doesn't work.

If you want to use the environments with ROS1+XBOT you will have to create a separate workspace,
you should be able to do it with:

```
bash <(curl -s https://gitlab.com/-/snippets/4863010/raw/main/setup_full.sh) ros1xbot
```


### Setup - Manual

If the setup script didn't work, or if you want do do things differently, here are some instructions for setting things manually.
You will need to create a python virtualenv with adarl, rreal, and optionally pykyon or pycentauro, and their dependencies.
Notice Some dependencies are "non-standard", maining you cannot use the official version, at least until their upstream repositories don't get updated/fixed (for example mujoco and xacro).

You can use the adarl_envs and its requirements directly on your machine, but I sugegst you use a docker container to manage the environment.
Some docker images are provided in the adarl_docker_utils at https://github.com/c-rizz/adarl_docker_utils.
With it you can launch a container with the following:

```
cd ~
git clone https://github.com/c-rizz/adarl_docker_utils
./adarl_docker_utils/basic/launch_persisting.sh
```

Now that you have a suitable docker container, you can create the workspace as follows:

Outside the Docker:

```
cd ~
mkdir adarl_ws
cd adarl_ws
mkdir src
cd src
git clone git@github.com:ADVRHumanoids/adarl_envs.git
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
git clone --recurse-submodules https://github.com/c-rizz/pycentauro
```

Inside the Docker:

```
cd /home/host/adarl_ws
mkdir virtualenv
python3 -m venv virtualenv/adarl
. virtualev/adarl/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
pip install -r src/adarl/mjx_requirements_2204_v1.txt
```


# Running an Example

If you followed the instructions before you should be able to start the container with:

```
~/adarl_ws/src/adarl_docker_utils/basic/launch_persisting.sh
```

At this point, inside the container, you should be able to train kyon with SAC on MJX with the following command (trains in about 50 minutes on an RTX 4080):

```
byobu
cd /home/host/adarl_ws
. virtualenv/adarl/bin/activate
./src/adarl_envs/src/adarl_envs/experiments/loco_kyon.py --algorithm sac --mode mjx --comment kyon_training
```

On your first run you may get in the terminal a prompt from wandb asking you to log in.

You can monitor the training in the terminal and pause it by typing 'pause' and pressing enter (It may take a bit to pause).
Once it is paused you can resume or stop the training.
You can also terminate it by repatedly pressing CTRL+C.

Outputs for the runs will be saved in adarl_ws/lrg_exps/vec_loco_env_test. There you can find checkpints, videos, and all information regarding the run.

You can find some info for deploying on Xbot in readme_ros_xbot.md


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