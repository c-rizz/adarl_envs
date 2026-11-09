#!/bin/bash
camzmq-ros2 --name scene --port 5710  --size 853x480 --info-topic /scene_camera/color/camera_info --rgb-topic /scene_camera/color/image_raw --no-depth --use-sim-time