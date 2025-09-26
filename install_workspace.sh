#!/bin/bash

cd /home/host/adarl_ws

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.13 virtualenv/adarl313
. virtualenv/adarl313/bin/activate
uv pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
uv pip install -r src/adarl/mjx_requirements_py313.txt