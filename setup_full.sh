#!/bin/bash


cd $HOME
mkdir -p adarl_ws/src
cd adarl_ws/src
git clone https://github.com/c-rizz/adarl_docker_utils
git clone git@github.com:ADVRHumanoids/adarl_envs.git
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
git clone --recurse-submodules https://github.com/c-rizz/pycentauro
cd ..

pwd
./src/adarl_docker_utils/basic/launch_persisting.sh --no-start
docker start adarl-2204-opengl-basic
docker exec adarl-2204-opengl-basic bash /home/host/adarl_ws/src/adarl_envs/setup_env.sh
docker stop adarl-2204-opengl-basic
