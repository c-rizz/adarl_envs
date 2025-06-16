#!/bin/bash

if [[ "$1" != "ros1xbot" ]]; then
    echo "ADARL AUTOINSTALL SCRIPT"
    prepare_for_ros1xbot=true
    ws_name="adarl_ws"
else
    echo "ADARL AUTOINSTALL SCRIPT - With ROS1 and XBOT"
    prepare_for_ros1xbot=false
    ws_name="adarl_ws_ros1"
fi

echo "This script will:"
echo " - Create a workspace in your home at ~/$ws_name"
echo " - Clone the required git repositories in it"
echo " - Create a suitable docker container, which has access to your home"
echo " - Set up a virtual environment inside the docker and inside the workspace"
echo "Press ENTER to continue."
read response


cd $HOME
mkdir -p "$ws_name/src"
cd "$ws_name/src"

echo "Opening your ssh key for cloning github repos, will store the credentials for 10 minutes"
ssh-add -l &>/dev/null # check if ssh agent running
if [ "$?" == 2 ]; then # was it not running?
    eval `ssh-agent -s` #then run it
    stop_sshagent_afterwards=true
else
    stop_sshagent_afterwards=false
fi
ssh-add -t 600

echo "Cloning repositories..."
sleep 3
git clone https://github.com/c-rizz/adarl_docker_utils
git clone git@github.com:ADVRHumanoids/adarl_envs.git
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
git clone --recurse-submodules https://github.com/c-rizz/pycentauro
if [[ prepare_for_ros1xbot ]]; then
    # git clone https://gitlab.com/crzz/adarl_ros
    git clone git@github.com:c-rizz/adarl_ros
fi
cd ..

if [ stop_sshagent_afterwards ]; then
    ssh-agent -k
fi


echo "Creating docker container..."
sleep 3

if [[ prepare_for_ros1xbot ]]; then
    ./src/adarl_docker_utils/ros1-xbot/launch_persisting.sh --no-start
    docker start adarl-xbot-2004
    docker exec adarl-xbot-2004 bash /home/host/$ws_name/src/adarl_envs/setup_workspace_ros1xbot.sh
    docker stop adarl-xbot-2004
else
    ./src/adarl_docker_utils/basic/launch_persisting.sh --no-start
    docker start adarl-2204-opengl-basic
    docker exec adarl-2204-opengl-basic bash "/home/host/$ws_name/src/adarl_envs/setup_workspace.sh"
    docker stop adarl-2204-opengl-basic
fi

echo ""
echo ""
echo ""
echo "  Everything should now be installed."
echo "  You should be able to start up the docker with:"
echo "      ~/$ws_name/src/adarl_docker_utils/basic/launch_persisting.sh"
echo "  Then you can move to /home/host/$ws_name to find the workspace in your own home folder"
echo "  You can then start the virtualenv with:"
echo "      source virtualenv/adarl/bin/activate"
echo "  And you can test that everything works with, for example:"
echo "      ./src/adarl_envs/src/adarl_envs/experiments/vec_loco_env_test.py --algorithm sac_small --mode mjx --comment kyon_training --robot kyon"
