#!/bin/bash

# In a clean container with ros noetic and xbot

mkdir forest_ws



echo "Opening your ssh key for cloning github repos, will store the credentials for 10 minutes"
ssh-add -l &>/dev/null # check if ssh agent running
if [ "$?" == 2 ]; then # was it not running?
    eval `ssh-agent -s` #then run it
    stop_sshagent_afterwards=true
else
    stop_sshagent_afterwards=false
fi
grep -slR "PRIVATE" /home/host/.ssh/ | xargs -o ssh-add
ssh-add -l

set -Eeo pipefail

sudo apt update

cd forest_ws
python3 -m venv virtualenv
. virtualenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install hhcm-forest
forest init
. setup.bash
forest add-recipes git@github.com:ADVRHumanoids/multidof_recipes.git --tag ros2

# mkdir ros_src
# cd ros_src
# git clone git@github.com:c-rizz/adarl_ros # Would be nice to avoid this, just needed to be able to launch scripts from roslaunch
# cd ..
pip install empy==3.3.4 colcon-common-extensions==0.3.0 lark==1.3.0
apt install -y libboost-all-dev libhdf5-dev libqhull-dev libassimp-dev liboctomap-dev
forest grow -j10 xbot2_mujoco
forest grow -j10 iit-kyon-ros-pkg
if [ stop_sshagent_afterwards ]; then
    ssh-agent -k
fi

# cd ..
# pip install catkin_pkg
# pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro -e src/adarl_ros/adarl_ros -e src/adarl_ros/adarl_ros_utils
# pip install -r src/adarl/requirements_2004.txt
# apt remove -y liboctomap-dev
# apt install -y python3-lxml
