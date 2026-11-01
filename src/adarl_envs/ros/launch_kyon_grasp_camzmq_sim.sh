#!/bin/bash
camzmq-ros2 --name zed --port 5700 --size 102x64  --info-topic /zed_front_down/camera_info --rgb-topic /zed_front_down/image --depth-topic /zed_front_down/depth_image --use-sim-time