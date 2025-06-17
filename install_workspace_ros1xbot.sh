#!/bin/bash

# In a clean container with ros noetic and xbot

cd /home/host/adarl_ws_ros1
. /opt/ros/noetic/setup.bash
. /opt/xbot/setup.sh
mkdir forest_ws
mkdir virtualenv
python3.8 -m venv virtualenv/adarl
. virtualenv/adarl/bin/activate
pip install --upgrade pip setuptools wheel
pip install hhcm-forest



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
forest init
. setup.bash
forest add-recipes git@github.com:ADVRHumanoids/multidof_recipes.git --tag mjx_adarl

mkdir ros_src
cd ros_src
git clone git@github.com:c-rizz/adarl_ros # Would be nice to avoid this, just needed to be able to launch scripts from roslaunch
cd ..
forest grow -j10 xbot2_mujoco
forest grow -j10 iit-kyon-ros-pkg
if [ stop_sshagent_afterwards ]; then
    ssh-agent -k
fi

cd ..
pip install catkin_pkg
pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro -e src/adarl_ros/adarl_ros -e src/adarl_ros/adarl_ros_utils
pip install -r src/adarl/requirements_2004.txt
apt remove -y liboctomap-dev
apt install -y python3-lxml
