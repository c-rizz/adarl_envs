#!/usr/bin/env python3
"""Record a sequence of depth images from a ZED camera.

Each saved frame produces NNNN_depth.npy (float32 depth in metres, NaN/inf where invalid),
NNNN_depth.png (the SDK's greyscale depth visualisation) and NNNN_left.png (the matching left image).
timestamps.csv records the camera timestamp of each saved frame, so the actual spacing can be checked.

Frames are grabbed continuously and saved only when due, so the spacing is never shorter than
--spacing, rounded up to the next camera frame. If saving takes longer than --spacing, frames are
saved as fast as possible.

Needs the ZED SDK and its python API (pyzed): run it on the host where the SDK is installed, after
`python3 /usr/local/zed/get_python_api.py`.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pyzed.sl as sl

N_FRAMES = 20       # number of frames to save
SPACING_S = 0.5     # seconds between saved frames
SETTLE_FRAMES = 30  # frames grabbed and discarded at start, to let auto-exposure and depth settle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=N_FRAMES, help="Number of frames to save")
    ap.add_argument("--spacing", type=float, default=SPACING_S, help="Seconds between saved frames")
    ap.add_argument("--out-dir", default="~/zed_out", help="Folder the frames are saved into")
    ap.add_argument("--serial", type=int, default=None,
                    help="Serial number of the camera to open; by default the first one found")
    ap.add_argument("--depth-mode", default="NEURAL",
                    help="sl.DEPTH_MODE name, e.g. NEURAL or PERFORMANCE. NEURAL spends a few minutes "
                         "optimising its model on the first run")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    if args.serial is not None:
        init.set_from_serial_number(args.serial)
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")

    try:
        runtime = sl.RuntimeParameters()
        depth, depth_view, left = sl.Mat(), sl.Mat(), sl.Mat()
        for _ in range(SETTLE_FRAMES):
            zed.grab(runtime)

        with open(os.path.join(out_dir, "timestamps.csv"), "w") as ts_file:
            ts_file.write("frame,camera_timestamp_ns\n")
            next_save = time.monotonic()
            saved = 0
            while saved < args.frames:
                # Grab continuously so exposure tracking stays live; save only when due.
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                if time.monotonic() < next_save:
                    continue
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                zed.retrieve_image(depth_view, sl.VIEW.DEPTH)
                zed.retrieve_image(left, sl.VIEW.LEFT)
                stem = os.path.join(out_dir, f"{saved:04d}")
                np.save(f"{stem}_depth.npy", depth.get_data())
                depth_view.write(f"{stem}_depth.png")
                left.write(f"{stem}_left.png")
                ts_file.write(f"{saved},{zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()}\n")
                print(f"saved frame {saved + 1}/{args.frames}")
                saved += 1
                next_save += args.spacing
    finally:
        zed.close()
    print(f"saved {args.frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
