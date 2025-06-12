#!/bin/bash


cd $HOME
mkdir -p adarl_ws/src
cd adarl_ws/src
git clone https://github.com/c-rizz/adarl_docker_utils
git clone https://gitlab.com/crzz/adarl_envs
git clone https://github.com/c-rizz/adarl
git clone https://github.com/c-rizz/rreal
git clone --recurse-submodules https://github.com/ADVRHumanoids/pykyon
git clone --recurse-submodules https://github.com/c-rizz/pycentauro
cd ..

pwd
./src/adarl_docker_utils/basic/launch_persisting.sh --no-start
docker run crizzard/adarl:2204-opengl-basic /home/host/adarl_ws/src/adarl_envs/setup_env.sh