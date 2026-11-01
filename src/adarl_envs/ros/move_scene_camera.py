#!/usr/bin/env python3
"""Move the scene camera in a running Gazebo (gz sim), from the command line or from the keyboard.

The scene camera is the static `scene_camera` model all_kyon_ros2.launch.xml spawns with
scene_camera:=true. It is moved with the world's set_pose service, so it does not need respawning.

One-shot: pass any of --x --y --z --roll --pitch --yaw; what is left out keeps its current value.
--look-at-robot sets pitch and yaw so the camera points at the robot. With no pose arguments the
current pose is just printed.

Interactive (--interactive), each key applied at once:
    w/s  forward/back      a/d  left/right      (horizontal, along the camera heading)
    r/f  up/down
    j/l  yaw left/right    i/k  pitch up/down
    o    aim at the robot (reads its current position) and keep aiming at it while moving
    +/-  double/halve the steps
    p    print the pose    q    quit
Rotating by hand stops the aiming.

At the end the pose is printed as all_kyon_ros2.launch.xml arguments, to make it the default there.
Angles are in degrees on the command line and in the printouts, in radians in the launch arguments.
Positive pitch looks down, as in the launch file.

As spawn_gazebo_cube.py, the gz commands run through `docker exec` by default; pass --container ""
to run them directly.
"""
from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass, replace

from adarl_envs.ros.spawn_gazebo_cube import protobuf_string


class GzShell:
    """A shell kept open to run gz commands in, inside the docker container unless container is empty.

    Starting `docker exec` and sourcing ROS for each command would add about half a second to every
    key press in interactive mode."""
    _DONE = "__gz_shell_done__"

    def __init__(self, container : str):
        cmd = ["docker", "exec", "-i", container, "bash"] if container else ["bash"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
        if container:
            # gz comes from ROS' gz_tools_vendor, which is only on the PATH once ROS is sourced.
            self._run("source /opt/ros/jazzy/setup.bash")

    def _run(self, cmd : str) -> str:
        self._proc.stdin.write(f"{cmd} 2>&1; printf '\\n%s %d\\n' {self._DONE} $?\n")
        self._proc.stdin.flush()
        lines = []
        for line in self._proc.stdout:
            if line.startswith(self._DONE):
                exit_code = int(line.split()[1])
                out = "".join(lines)
                if exit_code != 0:
                    raise RuntimeError(f"'{cmd}' failed with exit {exit_code}:\n{out.strip()}")
                return out
            lines.append(line)
        raise RuntimeError(f"The shell running gz exited during '{cmd}'")

    def gz(self, gz_args : list[str]) -> str:
        return self._run(" ".join(shlex.quote(a) for a in ["gz", *gz_args]))

    def close(self):
        self._proc.stdin.close()
        self._proc.wait(timeout=5)


@dataclass
class Pose:
    """World pose, in metres and radians, with gz's fixed-axis roll-pitch-yaw."""
    x : float
    y : float
    z : float
    roll : float
    pitch : float
    yaw : float

    def __str__(self):
        return (f"x={self.x:.3f} y={self.y:.3f} z={self.z:.3f} roll={math.degrees(self.roll):.1f} "
                f"pitch={math.degrees(self.pitch):.1f} yaw={math.degrees(self.yaw):.1f} deg")

    def launch_args(self) -> str:
        return (f"scene_camera_x:={self.x:.3f} scene_camera_y:={self.y:.3f} scene_camera_z:={self.z:.3f} "
                f"scene_camera_roll:={self.roll:.4f} scene_camera_pitch:={self.pitch:.4f} "
                f"scene_camera_yaw:={self.yaw:.4f}")

    def quaternion_xyzw(self) -> tuple[float, float, float, float]:
        cr, sr = math.cos(self.roll/2), math.sin(self.roll/2)
        cp, sp = math.cos(self.pitch/2), math.sin(self.pitch/2)
        cy, sy = math.cos(self.yaw/2), math.sin(self.yaw/2)
        return (sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,
                cr*cp*cy + sr*sp*sy)


def model_pose(shell : GzShell, model : str) -> Pose:
    """The model's world pose, from `gz model -p`."""
    out = shell.gz(["model", "-m", model, "-p"])
    triplets = re.findall(r"\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]", out.split("Pose", 1)[-1])
    if len(triplets) < 2:
        raise RuntimeError(f"Could not read the pose of model '{model}' from gz. Is it spawned? Output was:\n{out}")
    (x, y, z), (roll, pitch, yaw) = [tuple(float(v) for v in t) for t in triplets[:2]]
    return Pose(x, y, z, roll, pitch, yaw)


def set_model_pose(shell : GzShell, model : str, world : str, pose : Pose):
    qx, qy, qz, qw = pose.quaternion_xyzw()
    pose_req = (f"name: {protobuf_string(model)} "
                f"position: {{ x: {pose.x} y: {pose.y} z: {pose.z} }} "
                f"orientation: {{ x: {qx} y: {qy} z: {qz} w: {qw} }}")
    out = shell.gz(["service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", pose_req])
    if "data: true" not in out:
        raise RuntimeError(f"Gazebo did not set the pose of '{model}'. Reply was:\n{out}")


def aimed_at(pose : Pose, target_xyz : tuple[float, float, float]) -> Pose:
    """pose, turned to look at target_xyz: gz cameras look along their +x, and positive pitch looks down."""
    dx, dy, dz = target_xyz[0] - pose.x, target_xyz[1] - pose.y, target_xyz[2] - pose.z
    return replace(pose, roll=0.0, pitch=math.atan2(-dz, math.hypot(dx, dy)), yaw=math.atan2(dy, dx))


def robot_target(shell : GzShell, robot_model : str, target_dz : float) -> tuple[float, float, float]:
    robot = model_pose(shell, robot_model)
    print(f"robot '{robot_model}' at x={robot.x:.3f} y={robot.y:.3f} z={robot.z:.3f}")
    return robot.x, robot.y, robot.z + target_dz


def interactive(shell : GzShell, args : argparse.Namespace, pose : Pose,
                target : tuple[float, float, float] | None) -> Pose:
    """Move the camera from the keyboard until q or ctrl-c, and return where it was left."""
    step, angle_step = args.step, math.radians(args.angle_step)
    max_pitch = math.radians(89)
    print(__doc__[__doc__.index("    w/s"):__doc__.index("Rotating")].rstrip())
    print(f"steps: {step:.3f} m, {math.degrees(angle_step):.2f} deg")
    fd = sys.stdin.fileno()
    terminal_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = sys.stdin.read(1)
            forward = (math.cos(pose.yaw), math.sin(pose.yaw))
            left = (-math.sin(pose.yaw), math.cos(pose.yaw))
            moves = {"w" : (*forward, 0.0), "s" : (-forward[0], -forward[1], 0.0),
                     "a" : (*left, 0.0),    "d" : (-left[0], -left[1], 0.0),
                     "r" : (0.0, 0.0, 1.0), "f" : (0.0, 0.0, -1.0)}
            turns = {"j" : (0.0, 1.0), "l" : (0.0, -1.0), "i" : (-1.0, 0.0), "k" : (1.0, 0.0)}
            if key == "q":
                break
            elif key == "p":
                print(pose)
                continue
            elif key in "+=":
                step, angle_step = step*2, angle_step*2
                print(f"steps: {step:.3f} m, {math.degrees(angle_step):.2f} deg")
                continue
            elif key in "-_":
                step, angle_step = step/2, angle_step/2
                print(f"steps: {step:.3f} m, {math.degrees(angle_step):.2f} deg")
                continue
            elif key == "o":
                target = robot_target(shell, args.robot_model, args.target_dz)
                pose = aimed_at(pose, target)
            elif key in moves:
                mx, my, mz = moves[key]
                pose = replace(pose, x=pose.x + step*mx, y=pose.y + step*my, z=pose.z + step*mz)
                if target is not None:
                    pose = aimed_at(pose, target)
            elif key in turns:
                if target is not None:
                    print("stopped aiming at the robot")
                    target = None
                dpitch, dyaw = turns[key]
                pose = replace(pose, pitch=max(-max_pitch, min(max_pitch, pose.pitch + angle_step*dpitch)),
                               yaw=pose.yaw + angle_step*dyaw)
            else:
                continue
            set_model_pose(shell, args.model, args.world, pose)
            print(pose)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, terminal_settings)
    return pose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, default=None, help="World x, in metres (default: the current one)")
    ap.add_argument("--y", type=float, default=None, help="World y, in metres (default: the current one)")
    ap.add_argument("--z", type=float, default=None, help="World z, in metres (default: the current one)")
    ap.add_argument("--roll", type=float, default=None, help="Roll, in degrees (default: the current one)")
    ap.add_argument("--pitch", type=float, default=None,
                    help="Pitch, in degrees, positive looks down (default: the current one)")
    ap.add_argument("--yaw", type=float, default=None, help="Yaw, in degrees (default: the current one)")
    ap.add_argument("--look-at-robot", action="store_true",
                    help="Point the camera at the robot, setting pitch and yaw and zeroing roll. "
                         "In interactive mode, keep aiming at it while moving")
    ap.add_argument("--target-dz", type=float, default=0.0,
                    help="Height added to the robot's position when aiming at it, in metres")
    ap.add_argument("--interactive", action="store_true", help="Move the camera from the keyboard")
    ap.add_argument("--step", type=float, default=0.1, help="Interactive translation step, in metres")
    ap.add_argument("--angle-step", type=float, default=5.0, help="Interactive rotation step, in degrees")
    ap.add_argument("--model", default="scene_camera", help="Gazebo model of the camera")
    ap.add_argument("--robot-model", default="kyon", help="Gazebo model --look-at-robot aims at")
    ap.add_argument("--world", default="default", help="Gazebo world name")
    ap.add_argument("--container", default="adarl-jazzy-2404-xbot-mujoco",
                    help="Docker container Gazebo runs in; empty to run gz directly")
    ap.add_argument("--dry-run", action="store_true", help="Print the new pose, without touching Gazebo")
    args = ap.parse_args()
    if args.look_at_robot and (args.pitch is not None or args.yaw is not None):
        ap.error("--look-at-robot sets pitch and yaw itself, it cannot be combined with --pitch or --yaw")
    if args.interactive and args.dry_run:
        ap.error("--dry-run is for one-shot moves, it cannot be combined with --interactive")
    if args.interactive and not sys.stdin.isatty():
        ap.error("--interactive needs a terminal")

    shell = GzShell(args.container)
    try:
        pose = model_pose(shell, args.model)
        print(f"camera '{args.model}' at {pose}")
        given = {k : getattr(args, k) for k in ("x", "y", "z", "roll", "pitch", "yaw") if getattr(args, k) is not None}
        pose = replace(pose, **{k : (math.radians(v) if k in ("roll", "pitch", "yaw") else v) for k, v in given.items()})
        target = None
        if args.look_at_robot:
            target = robot_target(shell, args.robot_model, args.target_dz)
            pose = aimed_at(pose, target)

        if args.interactive:
            if given or target is not None:
                set_model_pose(shell, args.model, args.world, pose)
                print(f"moved to {pose}")
            pose = interactive(shell, args, pose, target)
        elif given or target is not None:
            if args.dry_run:
                print(f"dry run, not moving. The new pose would be {pose}")
            else:
                set_model_pose(shell, args.model, args.world, pose)
                print(f"moved to {pose}")
        print(f"launch arguments for this pose:\n{pose.launch_args()}")
    finally:
        shell.close()


if __name__ == "__main__":
    main()
