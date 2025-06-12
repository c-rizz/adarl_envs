# ADARL ENVIRONMENTS

This repository is a collection of environments and tools for robotic environments implemented with ADARL.

For now it is focused on an environment for quadruped locomotion.

### Setup

You will need to create a python virtualenv with adarl, rreal, and optionally pykyon or pycentauro, and their dependencies.
Notice Some dependencies are also "non-standard", at least until their upstream repositories don't get updated/fixed (for example mujoco 
and xacro).

You can use the adarl_envs and its requirements directly on your machine, but it is preferable to use a docker
container to manage the environment.
Some docker images a provided in the adarl_docker_utils at https://github.com/c-rizz/adarl_docker_utils.
You can launch a container with the following:

```
cd ~
git clone https://github.com/c-rizz/adarl_docker_utils
./adarl_docker_utils/basic/launch_persisting.sh
```

To create the workspace you can do as follows:

Outside the docker:

```
cd ~
mkdir adarl_ws
cd adarl_ws
mkdir src
cd src
git clone https://github.com/c-rizz/adarl_envs
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone https://github.com/ADVRHumanoids/pykyon
git clone https://github.com/c-rizz/pycentauro
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


At this point you should be able to train kyon with SAC on MJX with the following (trains in about 50 minutes on an RTX 4080):

```
./src/adarl_envs/src/adarl_envs/experiments/vec_loco_env_test.py --algorithm sac --mode mjx --comment kyon_training --robot kyon
```