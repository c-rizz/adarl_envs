#!/bin/bash
script_dir=$(dirname $(realpath $0))
ros2 launch ./src/adarl_envs/src/adarl_envs/ros/all_kyon_ros2.launch.xml \
 simulator:=mujoco scene_camera:=true camera_front_down:=true arms:=false \
 world:=$script_dir/world_follow_camera.xml
