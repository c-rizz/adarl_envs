#!/bin/bash

# Default values
prepare_for_ros1xbot=false
dev_mode=false
ws_name="adarl_ws"
verbose=false
help_requested=false

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "ADARL AUTOINSTALL SCRIPT"
    echo ""
    echo "COMMANDS:"
    echo "  ros1xbot          Install with ROS1 and XBOT support"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --dev         Enable development mode"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -h, --help        Show this help message"
    echo "  --ws-name NAME    Set custom workspace name (default: adarl_ws)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                    # Basic installation"
    echo "  $0 ros1xbot           # Install with ROS1 and XBOT"
    echo "  $0 --dev ros1xbot     # Development mode with ROS1 and XBOT"
    echo "  $0 --ws-name my_ws    # Custom workspace name"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            help_requested=true
            shift
            ;;
        -d|--dev)
            dev_mode=true
            shift
            ;;
        -v|--verbose)
            verbose=true
            shift
            ;;
        --ws-name)
            if [[ -n "$2" && "$2" != -* ]]; then
                ws_name="$2"
                shift 2
            else
                echo "Error: --ws-name requires a value"
                exit 1
            fi
            ;;
        ros1xbot)
            prepare_for_ros1xbot=true
            ws_name="adarl_ws_ros1"
            shift
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            echo "Error: Unknown command $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [[ "$help_requested" == true ]]; then
    show_help
    exit 0
fi

# Verbose output function
log_verbose() {
    if [[ "$verbose" == true ]]; then
        echo "[VERBOSE] $*"
    fi
}

# =============================================================================
# MAIN SCRIPT LOGIC
# =============================================================================

if [[ "$prepare_for_ros1xbot" == true ]]; then
    echo "ADARL AUTOINSTALL SCRIPT - With ROS1 and XBOT"
else
    echo "ADARL AUTOINSTALL SCRIPT"
fi

log_verbose "Configuration:"
log_verbose "  prepare_for_ros1xbot: $prepare_for_ros1xbot"
log_verbose "  dev_mode: $dev_mode"
log_verbose "  ws_name: $ws_name"
log_verbose "  verbose: $verbose"

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

sleep 1
if [[ "$dev_mode" == true ]]; then
    echo "Cloning dev repositories..."
    git clone https://github.com/c-rizz/adarl_docker_utils.git
    git clone https://gitlab.com/crzz/adarl_envs.git --branch crzz-dev
    git clone https://gitlab.com/crzz/adarl.git --branch crzz-dev
    git clone https://gitlab.com/crzz/rreal.git --branch crzz-dev
    git clone --recurse-submodules git@github.com:ADVRHumanoids/pykyon.git
    git clone --recurse-submodules https://github.com/c-rizz/pycentauro
    if [[ prepare_for_ros1xbot ]]; then
        # git clone https://gitlab.com/crzz/adarl_ros
        git clone https://gitlab.com/crzz/adarl_ros.git
    fi
else
    echo "Cloning public repositories..."
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
fi

cd ..

if [ stop_sshagent_afterwards ]; then
    ssh-agent -k
fi


echo "Creating docker container..."
sleep 1

if [[ "$prepare_for_ros1xbot" == true ]]; then
    docker_launcher=./src/adarl_docker_utils/ros1-xbot/launch_persisting.sh
    docker_name=adarl-xbot-2004
    ws_installer=/home/host/$ws_name/src/adarl_envs/install_workspace_ros1xbot.sh
else
    docker_launcher=./src/adarl_docker_utils/basic/launch_persisting.sh
    docker_name=adarl-2204-opengl-basic
    ws_installer=/home/host/$ws_name/src/adarl_envs/install_workspace.sh
fi
$docker_launcher --no-start
docker start $docker_name
docker exec -it $docker_name bash $ws_installer
docker stop $docker_name


echo ""
echo ""
echo ""
echo "  Everything should be correctly installed."
echo "  You should be able to start up the docker with:"
echo "      $docker_launcher"
echo "  Then you can move to /home/host/$ws_name to find the workspace in your own home folder"
echo "  You can then start the virtualenv with:"
echo "      source virtualenv/adarl/bin/activate"

if [[ "$1" == "ros1xbot" ]]; then
echo "  Then can check that everything works following the instructions in "
echo "  $ws_name/src/adarl_envs/ros_xbot.md"
else
echo "  Then can check that everything works with, for example:"
echo "      ./src/adarl_envs/src/adarl_envs/experiments/vec_loco_env_test.py --algorithm sac_small --mode mjx --comment kyon_training --robot kyon"
fi
