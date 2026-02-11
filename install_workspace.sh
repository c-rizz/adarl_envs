#!/bin/bash

cd /home/host/adarl_ws

# Avoid clobbering an existing environment.
if [ -d virtualenv/adarl313 ]; then
	echo "virtualenv/adarl313 already exists; aborting install."
	exit 0
fi

# Install uv only if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# Ensure the default installation location is on PATH for the remainder of the script.
	export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.13 virtualenv/adarl313
. virtualenv/adarl313/bin/activate
uv pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
uv pip install -r src/adarl/mjx_requirements_py313.txt