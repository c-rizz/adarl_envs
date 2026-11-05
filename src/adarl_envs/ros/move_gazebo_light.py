#!/usr/bin/env python3
"""Change where a directional light shines from in a running Gazebo (gz sim).

The world has a single light, the directional `sun`: its position has no effect on the lighting, so
moving it means changing its direction, the vector the light travels along (z negative points down).
With no --direction the light is just printed.

The light is changed with the world's light_config service, which replaces all of the light's
properties: the request is the light as the scene reports it, with only the direction changed.
set_pose is accepted for lights too, but does not change the rendering.

The scene keeps reporting the direction from the world file, so the printed direction is not the
one set by a previous run of this script.

As spawn_gazebo_cube.py, the gz commands run through `docker exec` by default; pass --container ""
to run them directly.
"""
from __future__ import annotations

import argparse
import re

from adarl_envs.ros.spawn_gazebo_cube import run_gz, protobuf_string


def light_fields(scene : str, light : str) -> str:
    """The fields of the named light, in protobuf text format, from the output of the scene/info service."""
    match = re.search(r"light \{\s*name: " + re.escape(protobuf_string(light)), scene)
    if match is None:
        names = re.findall(r"light \{\s*name: (\".*\")", scene)
        raise RuntimeError(f"No light named '{light}' in the scene, it has: {', '.join(names) or 'none'}")
    start = scene.index("{", match.start())+1
    depth = 1
    for i in range(start, len(scene)):
        depth += {"{" : 1, "}" : -1}.get(scene[i], 0)
        if depth == 0:
            return scene[start:i].strip()
    raise RuntimeError(f"Could not parse light '{light}' from the scene:\n{scene[match.start():]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", type=float, nargs=3, default=None, metavar=("X","Y","Z"),
                    help="Vector the light travels along, in the world frame")
    ap.add_argument("--light", default="sun", help="Gazebo light to change")
    ap.add_argument("--world", default="default", help="Gazebo world name")
    ap.add_argument("--container", default="adarl-jazzy-2404-xbot-mujoco",
                    help="Docker container Gazebo runs in; empty to run gz directly")
    ap.add_argument("--dry-run", action="store_true", help="Print the request, without touching Gazebo")
    args = ap.parse_args()

    scene = run_gz(["service", "-s", f"/world/{args.world}/scene/info", "--reqtype", "gz.msgs.Empty",
                    "--reptype", "gz.msgs.Scene", "--timeout", "3000", "--req", ""], args.container)
    fields = light_fields(scene, args.light)
    if "type: DIRECTIONAL" not in fields:
        raise RuntimeError(f"Light '{args.light}' is not directional, it has no direction to change:\n{fields}")
    direction_field = r"direction \{[^}]*\}"
    current = re.search(direction_field, fields)
    print(f"light '{args.light}', direction in the world file: {' '.join(current[0].split()) if current else 'none'}")
    if args.direction is None:
        return

    x, y, z = args.direction
    new_direction = f"direction {{ x: {x} y: {y} z: {z} }}"
    light_req = re.sub(direction_field, new_direction, fields) if current else f"{fields}\n{new_direction}"
    if args.dry_run:
        print(f"dry run, not changing the light. The request is:\n{light_req}")
        return

    out = run_gz(["service", "-s", f"/world/{args.world}/light_config", "--reqtype", "gz.msgs.Light",
                  "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", light_req], args.container)
    if "data: true" not in out:
        raise RuntimeError(f"Gazebo did not change light '{args.light}'. Reply was:\n{out}")
    print(f"direction set to x={x} y={y} z={z}")


if __name__ == "__main__":
    main()
