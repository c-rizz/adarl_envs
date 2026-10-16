#!/bin/bash

# Usage: install_workspace.sh [--nodev] [venv_name]   (default venv: adarl313)
VENV_NAME="adarl313"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dev_mode=true
for arg in "$@"; do
    case "$arg" in
        --nodev) dev_mode=false ;;
        -*) echo "Unknown option: $arg"; exit 1 ;;
        *) VENV_NAME="$arg" ;;
    esac
done

cd $HOME
mkdir -p adarl_ws/src
cd adarl_ws

cd src
if [[ "$dev_mode" == true ]]; then
    echo "Cloning dev repositories..."
    git clone git@github.com:c-rizz/adarl.git --branch crzz-dev
    git clone git@github.com:c-rizz/adarl_envs.git --branch crzz-dev
    git clone git@github.com:c-rizz/rreal.git --branch crzz-dev
    git clone git@github.com:c-rizz/pydescriptions.git
else
    echo "Cloning public repositories..."
    git clone https://github.com/c-rizz/adarl_docker_utils.git
    git clone git@github.com:c-rizz/adarl.git
    git clone git@github.com:c-rizz/adarl_envs.git
    git clone git@github.com:c-rizz/rreal.git
    git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
    git clone --recurse-submodules https://github.com/c-rizz/pycentauro
fi
cd ..

# Avoid clobbering an existing environment.
if [ -d "virtualenv/$VENV_NAME" ]; then
	echo "virtualenv/$VENV_NAME already exists; aborting install."
	exit 0
fi

# Install uv only if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# Ensure the default installation location is on PATH for the remainder of the script.
	export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.13 "virtualenv/$VENV_NAME"
. "virtualenv/$VENV_NAME/bin/activate"
uv pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pydescriptions
uv pip install -r src/adarl/requirements_py313.txt
