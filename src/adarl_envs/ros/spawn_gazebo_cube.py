#!/usr/bin/env python3
"""Spawn the grasping cube into a running Gazebo (gz sim), placed in front of the robot.

The cube is compiled from the same models/cube.urdf.xacro the grasping environment uses, at the same
size and mass, so the policy meets the object it was trained on. It is placed relative to the robot's
current pose rather than at fixed world coordinates, since the robot moves around: by default in the
middle of the area GraspVecEnv samples it from, on the ground (kyon is trained with the table sunk to
table_height=0.001, i.e. grasping from the floor).

Gazebo usually runs inside the xbot docker container while the play scripts run on the host, so the
gz commands are run through `docker exec` by default; pass --container "" to run them directly.

Running it again replaces the cube, so it doubles as a reset between episodes.
"""
from __future__ import annotations

import argparse
import math
import random
import re
import shlex
import subprocess
from pathlib import Path

import adarl.utils.utils
from adarl.utils.base_utils import compile_xacro_string

# As in GraspVecEnv: the cube's edge (self._cube_size) and the area its initial position is sampled
# from (manipulation_area_minmax_xy), which is in world coordinates there, with kyon's pelvis at
# x=-0.2 (homing_body_pose_xyz_xyzw in grasp_builder.get_kyon_args). Expressed relative to the pelvis:
CUBE_SIZE = 0.04
FORWARD_RANGE = (0.65, 0.85)
LEFT_RANGE = (-0.15, 0.15)


def run_gz(gz_args : list[str], container : str, timeout : float = 20.0) -> str:
    """Run a gz command, inside the docker container unless container is empty."""
    gz_cmd = " ".join(shlex.quote(a) for a in ["gz", *gz_args])
    if container:
        # gz comes from ROS' gz_tools_vendor, which is only on the PATH once ROS is sourced.
        cmd = ["docker", "exec", container, "bash", "-lc", f"source /opt/ros/jazzy/setup.bash && {gz_cmd}"]
    else:
        cmd = ["gz", *gz_args]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"'{gz_cmd}' failed with exit {done.returncode}:\n{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def robot_pose_xy_yaw(robot_model : str, container : str) -> tuple[float, float, float]:
    """The robot model's position on the ground plane and heading, from `gz model -p`."""
    out = run_gz(["model", "-m", robot_model, "-p"], container)
    triplets = re.findall(r"\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]", out.split("Pose", 1)[-1])
    if len(triplets) < 2:
        raise RuntimeError(f"Could not read the pose of model '{robot_model}' from gz. Is it spawned? Output was:\n{out}")
    (x, y, _z), (_roll, _pitch, yaw) = [tuple(float(v) for v in t) for t in triplets[:2]]
    return x, y, yaw


def cube_urdf(size : float, mass : float) -> str:
    """The cube GraspVecEnv spawns, as a free body: no joint to the world link, so gz lets it fall and slide."""
    xacro_string = Path(adarl.utils.utils.pkgutil_get_path("adarl_envs", "models/cube.urdf.xacro")).read_text()
    return compile_xacro_string(xacro_string, model_kwargs={"size" : str(size),
                                                            "mass" : str(mass),
                                                            "add_floating_joint" : "false"})


def protobuf_string(s : str) -> str:
    """s as a protobuf text-format string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", type=float, default=sum(FORWARD_RANGE)/2,
                    help=f"Distance in front of the robot's pelvis, in metres (training range {FORWARD_RANGE})")
    ap.add_argument("--left", type=float, default=0.0,
                    help=f"Lateral offset from the robot's pelvis, positive to its left, in metres (training range {LEFT_RANGE})")
    ap.add_argument("--yaw", type=float, default=0.0, help="Cube yaw relative to the robot, in degrees")
    ap.add_argument("--random", action="store_true",
                    help="Sample position and yaw like GraspVecEnv does, ignoring --forward/--left/--yaw")
    ap.add_argument("--size", type=float, default=CUBE_SIZE, help="Cube edge, in metres")
    ap.add_argument("--mass", type=float, default=0.1, help="Cube mass, in kg")
    ap.add_argument("--name", default="cube", help="Gazebo model name; an existing model with this name is replaced")
    ap.add_argument("--robot-model", default="kyon", help="Gazebo model the cube is placed relative to")
    ap.add_argument("--world", default="default", help="Gazebo world name")
    ap.add_argument("--container", default="adarl-jazzy-2404-xbot-mujoco",
                    help="Docker container Gazebo runs in; empty to run gz directly")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be spawned, without touching Gazebo")
    args = ap.parse_args()

    if args.random:
        forward, left = random.uniform(*FORWARD_RANGE), random.uniform(*LEFT_RANGE)
        rel_yaw = random.uniform(0, 2*math.pi)
    else:
        forward, left, rel_yaw = args.forward, args.left, math.radians(args.yaw)

    robot_x, robot_y, robot_yaw = robot_pose_xy_yaw(args.robot_model, args.container)
    x = robot_x + forward*math.cos(robot_yaw) - left*math.sin(robot_yaw)
    y = robot_y + forward*math.sin(robot_yaw) + left*math.cos(robot_yaw)
    z = args.size/2 + 0.001 # resting on the ground, a millimetre up so it does not start interpenetrating
    yaw = robot_yaw + rel_yaw

    print(f"robot '{args.robot_model}' at x={robot_x:.3f} y={robot_y:.3f} yaw={math.degrees(robot_yaw):.1f} deg")
    print(f"cube '{args.name}' ({args.size} m, {args.mass} kg): {forward:.3f} m ahead, {left:.3f} m left, "
          f"relative yaw {math.degrees(rel_yaw):.1f} deg -> world x={x:.3f} y={y:.3f} z={z:.3f}")
    urdf = cube_urdf(args.size, args.mass)
    factory_req = (f"sdf: {protobuf_string(urdf)} name: {protobuf_string(args.name)} allow_renaming: false "
                   f"pose: {{ position: {{ x: {x} y: {y} z: {z} }} "
                   f"orientation: {{ x: 0 y: 0 z: {math.sin(yaw/2)} w: {math.cos(yaw/2)} }} }}")
    if args.dry_run:
        print("dry run, not spawning. The description is:\n" + urdf)
        return

    # Replacing rather than refusing makes re-running the script the way to reset the cube. Removing a
    # model that does not exist just reports false, so this is safe on the first run too.
    run_gz(["service", "-s", f"/world/{args.world}/remove", "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean", "--timeout", "3000",
            "--req", f"name: {protobuf_string(args.name)} type: MODEL"], args.container)
    out = run_gz(["service", "-s", f"/world/{args.world}/create", "--reqtype", "gz.msgs.EntityFactory",
                  "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", factory_req], args.container)
    if "data: true" not in out:
        raise RuntimeError(f"Gazebo did not create the cube. Reply was:\n{out}")
    print("spawned")


if __name__ == "__main__":
    main()
