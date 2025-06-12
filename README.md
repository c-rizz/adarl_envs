# ADARL ENVIRONMENTS

This repository is a collection of environments and tools for robotic environments implemented with ADARL.


For now it is focused on an environment for quadruped locomotion.

### Setup - Auto

You can try to install everything using the provided setup script.
It hasn't been thoroughly tested, so it may not work, but it's worth a try.

```
curl https://gitlab.com/-/snippets/4863010/raw/main/setup_full.sh | bash
```

You can find the script also in the repo main folder.
Let me know if it doesn't work.

### Setup - Manual

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
./src/adarl_envs/src/adarl_envs/experiments/vec_loco_env_test.py --algorithm sac --mode mjx --comment kyon_training --robot kyon
```

On your first run you may get in the terminal a prompt from wandb asking you to log in.

You can monitor the training in the terminal and pause it by typing 'pause' and pressing enter (It may take a bit to pause).
Once it is paused you can resume or stop the training.
You can also terminate it by repatedly pressing CTRL+C.

Outputs for the runs will be saved in adarl_ws/lrg_exps/vec_loco_env_test. There you can find checkpints, videos, and all information regarding the run.
