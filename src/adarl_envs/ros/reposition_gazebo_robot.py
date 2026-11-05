#!/usr/bin/env python3
"""Reposition the robot in a running Gazebo (gz sim), standing upright.

By default the robot is put back upright where it is, keeping its heading, and dropped from --z: the
quick way to recover it after a fall without restarting the simulation. --x/--y/--yaw move it
elsewhere instead.

Only the base pose is set: Gazebo keeps the joint positions and the velocities the robot had, so the
controller (e.g. xbot) has to bring the legs back under it.

As spawn_gazebo_cube.py, the gz commands run through `docker exec` by default; pass --container ""
to run them directly.
"""
from __future__ import annotations

import argparse
import math

from adarl_envs.ros.spawn_gazebo_cube import run_gz, robot_pose_xy_yaw, protobuf_string


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, default=None, help="World x of the base, in metres (default: where it is now)")
    ap.add_argument("--y", type=float, default=None, help="World y of the base, in metres (default: where it is now)")
    ap.add_argument("--z", type=float, default=0.5, help="World z the base is dropped from, in metres")
    ap.add_argument("--yaw", type=float, default=None, help="Heading, in degrees (default: the current one)")
    ap.add_argument("--robot-model", default="kyon", help="Gazebo model to reposition")
    ap.add_argument("--world", default="default", help="Gazebo world name")
    ap.add_argument("--container", default="adarl-jazzy-2404-xbot-mujoco",
                    help="Docker container Gazebo runs in; empty to run gz directly")
    ap.add_argument("--dry-run", action="store_true", help="Print the new pose, without touching Gazebo")
    args = ap.parse_args()

    robot_x, robot_y, robot_yaw = robot_pose_xy_yaw(args.robot_model, args.container)
    x = robot_x if args.x is None else args.x
    y = robot_y if args.y is None else args.y
    yaw = robot_yaw if args.yaw is None else math.radians(args.yaw)

    print(f"robot '{args.robot_model}' at x={robot_x:.3f} y={robot_y:.3f} yaw={math.degrees(robot_yaw):.1f} deg")
    print(f"repositioning to x={x:.3f} y={y:.3f} z={args.z:.3f} yaw={math.degrees(yaw):.1f} deg, upright")
    pose_req = (f"name: {protobuf_string(args.robot_model)} "
                f"position: {{ x: {x} y: {y} z: {args.z} }} "
                f"orientation: {{ x: 0 y: 0 z: {math.sin(yaw/2)} w: {math.cos(yaw/2)} }}")
    if args.dry_run:
        print(f"dry run, not repositioning. The request is:\n{pose_req}")
        return

    out = run_gz(["service", "-s", f"/world/{args.world}/set_pose", "--reqtype", "gz.msgs.Pose",
                  "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", pose_req], args.container)
    if "data: true" not in out:
        raise RuntimeError(f"Gazebo did not set the pose of '{args.robot_model}'. Reply was:\n{out}")
    print("repositioned")


if __name__ == "__main__":
    main()
