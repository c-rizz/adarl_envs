#!/bin/bash

if [[ "$*" == *"--nodevgpu"* ]] ; then
    dev_mode=true
else
    dev_mode=false
fi

if [[ "$dev_mode" == true ]]; then
    echo "Cloning dev repositories..."
    git clone https://github.com/c-rizz/adarl_docker_utils.git
    git clone git@github.com:c-rizz/adarl.git --branch crzz-dev
    git clone git@github.com:c-rizz/adarl_envs.git --branch crzz-dev
    git clone git@github.com:c-rizz/rreal.git --branch crzz-dev
    git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
    git clone --recurse-submodules https://github.com/c-rizz/pycentauro
else
    echo "Cloning public repositories..."
    git clone https://github.com/c-rizz/adarl_docker_utils.git
    git clone git@github.com:c-rizz/adarl.git
    git clone git@github.com:c-rizz/adarl_envs.git
    git clone git@github.com:c-rizz/rreal.git
    git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
    git clone --recurse-submodules https://github.com/c-rizz/pycentauro
fi