# Some Instructions on deploying ADARL envs on ros+xbot

For now I only tried the deployment with ROS1 and XBOT2, using either xbot2_mujoco or Gazebo.

# KYON_MUJOCO

Using a workspace setup as done in install_workspace_ros1xbot.sh, it should be possible to
run an adarl environment controlling Kyon in the interactive mujoco simulation.
Due to conflicts between dependencies it is quite finicky brittle setup.
You will need to run the Simulator and the environment with slightly different terminal configurations.
The simulation terminal must source the setup scripts of ros, xbot and forest, BUT NOT the python virtual environment.
The adarl-based script will need ros, xbot, forest and the virtual environment.

All of this will be inside the ros1-xbot docker container, I suggest you used byobu to split the terminal in two (or more
it you want xbot2-gui).

So, launch the docker with:

```
~/adarl_ws_ros1/src/adarl_docker_utils/ros1-xbot/launch_persisting.sh
```

Launch byobu:

```
byobu
```

Open 3 splits with SHIFT+F2 and then CTRL-F2, enable the mouse pressing twice ALT-F12.


In one terminal, start the simulation:

```
cd /home/host/adarl_ws_ros1
. src/adarl_envs/setup_ros1xbot_sim.sh
roslaunch src/adarl_envs/src/adarl_envs/ros/all_kyon_mujoco.launch
```

In a second terminal, start xbot gui:

```
cd /home/host/adarl_ws_ros1
. src/adarl_envs/setup_ros1xbot_sim.sh
xbot2-gui
```


In the ADARL run:
```
cd /home/host/adarl_ws_ros1
. src/adarl_envs/setup_ros1xbot.sh
./src/adarl_envs/src/adarl_envs/experiments/loco_play_vec1.py --comment t --robot kyon --mode xbot --record --norender --control sine --deterministic```

This will run a simple hardcoded policy that oscillates joints according to a sine function.

