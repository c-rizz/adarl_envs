# In a clean container with ros noetic and xbot

cd /home/host/adarl_ws_ros1
. /opt/ros/noetic/setup.bash
. /opt/xbot/setup.sh
mkdir forest_ws
mkdir virtualenv
python3.8 -m venv virtualenv/venv38
. virtualenv/venv38/bin/activate
pip install --upgrade pip setuptools wheel
pip installl hhcm-forest

cd forest_ws
forest init
. setup.bash
forest add-recipes git@github.com:ADVRHumanoids/multidof_recipes.git --tag master
forest grow -j10 xbot2_mujoco
forest grow -j10 iit-kyon-ros-pkg

cd ..
pip install -e src/adarl -e src/adarl_envs -e src/rreal -e src/pykyon -e src/pycentauro
pip install -r src/adarl/requrements_2004.txt
apt remove liboctomap-dev