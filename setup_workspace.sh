#!/bin/bash

cd /home/host/adarl_ws
mkdir virtualenv
python3 -m venv virtualenv/adarl
. virtualenv/adarl/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
pip install -r src/adarl/mjx_requirements_2204_v1.txt