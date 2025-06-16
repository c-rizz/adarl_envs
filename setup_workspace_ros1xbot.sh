# In a clean container with ros noetic and xbot

cd /home/host/adarl_ws_ros1
. /opt/ros/noetic/setup.bash
. /opt/xbot/setup.sh
mkdir forest_ws
mkdir virtualenv
python3.8 -m venv virtualenv/venv38
. virtualenv/venv38/bin/activate
pip install --upgrade pip setuptools wheel
pip install hhcm-forest



echo "TTY available" || echo "NO TTY"
export SSH_ASKPASS=/bin/false 
export GIT_ASKPASS=/bin/false
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

cd forest_ws
forest init
. setup.bash
forest add-recipes git@github.com:ADVRHumanoids/multidof_recipes.git --tag master
forest grow -j10 xbot2_mujoco
forest grow -j10 iit-kyon-ros-pkg

if [ stop_sshagent_afterwards ]; then
    ssh-agent -k
fi

cd ..
pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
pip install -r src/adarl/requirements_2004.txt
apt remove -y liboctomap-dev
