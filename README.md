# ADARL ENVIRONMENTS

This repository is meant to collect environments and tools for robotic environments.

For now it is focused on the environment for quadruped locomotion.

### Setting up:

You will need to create a python virtualenv with adarl, rreal, and optionally pykyon or pycentauro, and their dependencies.
Notice Some dependencies are also "non-standard", at least until their upstream repositories don't get updated/fixed (for example mujoco 
and xacro).

I suggest you do as follows:

```
cd ~
mkdir adarl_ws
cd adarl_ws
mkdir src
cd src
git clone https://github.com/c-rizz/adarl_envs
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone https://github.com/???/pykyon
git clone https://github.com/???/pycentauro

cd ..
mkdir virtualenv
cd virtualenv
python3 -m venv adarl
cd ..
. virtualev/adarl/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e src/adarl
pip install -e src/adarl_envs
pip install -e src/rreal
pip install -e src/pykyon
pip install -e src/pycentauro
pip install -r src/adarl/mjx_requirements_2204_v1.txt
```



At this point you should be able to train kyon with SAC on MJX with (trin s in about 50 minutes on an RTX 4080):

```
./src/adarl_envs/src/adarl_envs/experiments/vec_loco_env_test.py --algorithm sac --mode mjx --comment easier --robot kyon
```