#!/bin/bash

scriptdir="$(cd "$(dirname "$0")" && pwd)"
cd "$scriptdir"
xbot2-core -V -H mj -C "$scriptdir/xbotcore_kyon_ros2.yaml"