#!/bin/bash

scriptdir="$(dirname "$0")"
cd "$scriptdir"
xbot2-core -V -H mj -C "$scriptdir/xbotcore_kyon_ros2.yaml"