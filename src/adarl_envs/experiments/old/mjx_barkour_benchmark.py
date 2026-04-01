#!/usr/bin/env python3

"""Simple MJX XML benchmark with random controls.

Inspired by the MJX tutorial notebook:
https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

This intentionally avoids Brax/Playground env wrappers. It just:
1. loads a MuJoCo XML,
2. resets to the home keyframe,
3. applies random piecewise-constant actuator targets or joint torques,
4. runs a jitted MJX rollout,
5. prints throughput numbers.

If `--xml-path` is not provided, it falls back to the Barkour scene from
`mujoco_playground`'s menagerie path.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_triton_gemm_any=True" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    ).strip()

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np


def tree_nbytes(tree) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def print_largest_dataclass_fields(obj, name: str, top_k: int = 8) -> None:
    try:
        obj_fields = dataclasses.fields(obj)
    except TypeError:
        print(f"Cannot summarize fields for {name}: object is not a dataclass")
        return

    field_infos: list[tuple[str, int, str, str]] = []
    total_nbytes = 0
    for field in obj_fields:
        value = getattr(obj, field.name)
        nbytes = tree_nbytes(value)
        total_nbytes += nbytes
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        shape_str = str(tuple(shape)) if shape is not None else "-"
        dtype_str = str(dtype) if dtype is not None else "-"
        field_infos.append((field.name, nbytes, shape_str, dtype_str))

    field_infos = [fi for fi in field_infos if fi[1] > 0]
    field_infos.sort(key=lambda x: x[1], reverse=True)
    if not field_infos:
        print(f"No array-backed fields found for {name}")
        return

    print(f"Largest {min(top_k, len(field_infos))} fields in {name}:")
    denom = max(total_nbytes, 1)
    for field_name, nbytes, shape_str, dtype_str in field_infos[:top_k]:
        print(
            f"  {field_name:<24} {nbytes/1024**2:9.3f} MB"
            f"  ({100*nbytes/denom:5.1f}%)"
            f"  shape={shape_str}"
            f"  dtype={dtype_str}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml-path", type=str, default=None, help="Path to any MuJoCo XML file.")
    ap.add_argument(
        "--download-menagerie",
        action="store_true",
        help="If no --xml-path is given, ask mujoco_playground to download the default Barkour menagerie asset.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="JAX device: auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    ap.add_argument(
        "--impl",
        type=str,
        default="jax",
        choices=("jax", "warp"),
        help="MJX backend implementation.",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Number of simulations to run in parallel.")
    ap.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of outer-loop control steps per rollout. This is also the number of lax.scan calls.",
    )
    ap.add_argument("--sim-dt", type=float, default=0.004, help="MuJoCo timestep.")
    ap.add_argument(
        "--substeps",
        "--control-repeat-steps",
        dest="substeps",
        type=int,
        default=10,
        help="Number of MJX step calls inside each lax.scan call.",
    )
    ap.add_argument(
        "--control-scale",
        type=float,
        default=0.30,
        help="Uniform random target scale around the home control target.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for controls.")
    ap.add_argument(
        "--no-playground-tuning",
        action="store_true",
        help="Disable the Barkour-specific damping / actuator tuning used when the default Barkour model is loaded.",
    )
    return ap.parse_args()


def available_devices(kind: str) -> list[jax.Device]:
    try:
        return list(jax.devices(kind))
    except RuntimeError:
        return []


def resolve_device(device_str: str) -> jax.Device:
    if device_str == "auto":
        gpus = available_devices("gpu")
        return gpus[0] if gpus else jax.devices("cpu")[0]
    if device_str == "cpu":
        return jax.devices("cpu")[0]
    if device_str in {"cuda", "gpu"}:
        gpus = available_devices("gpu")
        if not gpus:
            raise RuntimeError("GPU device requested, but JAX does not see any GPU devices.")
        return gpus[0]
    if device_str.startswith("cuda:") or device_str.startswith("gpu:"):
        index = int(device_str.split(":")[1])
        gpus = available_devices("gpu")
        if index >= len(gpus):
            raise RuntimeError(
                f"GPU device {device_str} requested, but only {len(gpus)} GPU device(s) are visible to JAX."
            )
        return gpus[index]
    raise ValueError(f"Unsupported device string: {device_str}")


def resolve_xml_path(xml_path_arg: str | None, download_menagerie: bool) -> Path:
    if xml_path_arg is not None:
        xml_path = Path(xml_path_arg).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")
        return xml_path

    try:
        from mujoco_playground._src import mjx_env
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mujoco_playground is not importable. Run this script with the same Python "
            "environment you use for playground, or pass --xml-path explicitly."
        ) from exc

    if download_menagerie:
        mjx_env.ensure_menagerie_exists()

    xml_path = mjx_env.MENAGERIE_PATH / "google_barkour_vb" / "scene_mjx.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Default Barkour XML not found at {xml_path}.\n"
            "Pass --xml-path to a local MuJoCo XML, or rerun with "
            "--download-menagerie if your environment can fetch mujoco_menagerie."
        )
    return Path(xml_path)


def load_model(
    xml_path: Path,
    sim_dt: float,
    apply_barkour_tuning: bool,
) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_model.opt.timestep = sim_dt

    if apply_barkour_tuning:
        if mj_model.nv > 6:
            mj_model.dof_damping[6:] = 0.5239
        if mj_model.nu > 0 and mj_model.actuator_gainprm.shape[1] >= 1:
            mj_model.actuator_gainprm[:, 0] = 35.0
        if mj_model.nu > 0 and mj_model.actuator_biasprm.shape[1] >= 2:
            mj_model.actuator_biasprm[:, 1] = -35.0

    mj_data = mujoco.MjData(mj_model)
    home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, home_key_id)
        home_ctrl = np.array(mj_model.key_ctrl[home_key_id], dtype=np.float32)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
        home_ctrl = np.zeros((mj_model.nu,), dtype=np.float32)
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data, home_ctrl


def get_torque_dof_ids(mj_model: mujoco.MjModel) -> np.ndarray:
    if mj_model.nv <= 0:
        return np.zeros((0,), dtype=np.int32)
    dof_joint_types = mj_model.jnt_type[mj_model.dof_jntid]
    # Exclude free-joint wrench entries: for torque fallback we only want actual
    # joint-space DOFs, not floating-base generalized forces.
    return np.nonzero(dof_joint_types != mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)


def batch_data(single_data: mjx.Data, batch_size: int) -> mjx.Data:
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape),
        single_data,
    )


def build_input_sequence(
    seed: int,
    device: jax.Device,
    num_steps: int,
    batch_size: int,
    input_dim: int,
    control_scale: float,
    use_actuators: bool,
    home_ctrl: np.ndarray | None,
    ctrl_range: np.ndarray | None,
) -> list[jax.Array]:
    np_rng = np.random.default_rng(seed)
    if use_actuators:
        if home_ctrl is None or ctrl_range is None:
            raise ValueError("Actuator mode requires home_ctrl and ctrl_range.")
        ctrl_low = np.asarray(ctrl_range[:, 0], dtype=np.float32)
        ctrl_high = np.asarray(ctrl_range[:, 1], dtype=np.float32)
        home_ctrl_np = np.asarray(home_ctrl, dtype=np.float32)
        step_inputs = np_rng.uniform(
            low=-control_scale,
            high=control_scale,
            size=(num_steps, batch_size, input_dim),
        ).astype(np.float32)
        step_inputs = np.clip(home_ctrl_np[None, None, :] + step_inputs, ctrl_low, ctrl_high)
    else:
        step_inputs = np_rng.uniform(
            low=-control_scale,
            high=control_scale,
            size=(num_steps, batch_size, input_dim),
        ).astype(np.float32)
    return [jax.device_put(step_inputs[step_idx], device=device) for step_idx in range(num_steps)]


def make_rollout_fn(
    use_actuators: bool,
    torque_dof_ids: np.ndarray | None = None,
    num_substeps: int = 1,
):
    batched_step = jax.vmap(mjx.step, in_axes=(None, 0))
    torque_dof_ids_jax = (
        None if torque_dof_ids is None else jnp.asarray(torque_dof_ids, dtype=jnp.int32)
    )

    def rollout(
        model_and_data: tuple[mjx.Model, mjx.Data], inputs: jax.Array
    ) -> tuple[mjx.Model, mjx.Data]:
        def scan_step(carry: tuple[mjx.Model, mjx.Data], _):
            model, data = carry
            if use_actuators:
                data = data.replace(ctrl=inputs)
            else:
                qfrc_applied = jnp.zeros_like(data.qfrc_applied)
                qfrc_applied = qfrc_applied.at[:, torque_dof_ids_jax].set(inputs)
                data = data.replace(qfrc_applied=qfrc_applied)
            data = batched_step(model, data)
            return (model, data), None

        final_model_and_data, _ = jax.lax.scan(scan_step, model_and_data, xs=None, length=num_substeps)
        return final_model_and_data

    return jax.jit(rollout)


def main() -> None:
    args = parse_args()

    xml_path = resolve_xml_path(args.xml_path, args.download_menagerie)
    device = resolve_device(args.device)
    using_default_barkour = args.xml_path is None
    apply_barkour_tuning = using_default_barkour and not args.no_playground_tuning

    mj_model, mj_data, home_ctrl = load_model(
        xml_path=xml_path,
        sim_dt=args.sim_dt,
        apply_barkour_tuning=apply_barkour_tuning,
    )
    use_actuators = False # mj_model.nu > 0
    torque_dof_ids = None if use_actuators else get_torque_dof_ids(mj_model)
    if not use_actuators and torque_dof_ids.size == 0:
        raise RuntimeError(
            "The loaded model has no actuators and no non-free joint DOFs to torque-control."
        )
    ctrl_range = np.array(mj_model.actuator_ctrlrange, dtype=np.float32) if use_actuators else None
    input_dim = mj_model.nu if use_actuators else int(torque_dof_ids.size)
    num_steps = args.steps
    if num_steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive.")
    num_scan_iterations = num_steps * args.substeps
    duration_sec = num_scan_iterations * mj_model.opt.timestep

    mjx_model = mjx.put_model(mj_model, device=device, impl=args.impl)
    mjx_data = mjx.put_data(mj_model, mj_data, device=device, impl=args.impl)
    mjx_data_batched = batch_data(mjx_data, args.batch_size)
    mjx_model_mb = tree_nbytes(mjx_model) / 1024**2
    mjx_data_mb = tree_nbytes(mjx_data) / 1024**2
    mjx_data_batched_mb = tree_nbytes(mjx_data_batched) / 1024**2

    print(f"XML path:        {xml_path}")
    print(f"Default Barkour: {using_default_barkour}")
    print(f"Barkour tuning:  {apply_barkour_tuning}")
    print(f"Device:          {device}")
    print(f"MJX impl:        {args.impl}")
    print(f"Batch size:      {args.batch_size}")
    print(f"nq / nv / nu:    {mj_model.nq} / {mj_model.nv} / {mj_model.nu}")
    print(f"Control mode:    {'actuator_ctrl' if use_actuators else 'joint_torque'}")
    print(f"Control dim:     {input_dim}")
    print(f"Sim dt:          {mj_model.opt.timestep:.6f} s")
    print(f"Control steps:   {num_steps}")
    print(f"Substeps:        {args.substeps}")
    print(f"lax.scan calls:  {num_steps}")
    print(f"MJX step calls:  {num_scan_iterations}")
    print(f"Rollout time:    {duration_sec:.3f} s per env")
    if use_actuators:
        print(f"Control scale:   +/-{args.control_scale:.3f} around home target")
    else:
        print(f"Torque scale:    +/-{args.control_scale:.3f} generalized force units")
    print(f"MJX model size:  {mjx_model_mb:.3f} MB")
    print(f"MJX data size:   {mjx_data_mb:.3f} MB")
    print(f"Batched data:    {mjx_data_batched_mb:.3f} MB")
    print_largest_dataclass_fields(mjx_model, "mjx_model")
    print_largest_dataclass_fields(mjx_data, "mjx_data")
    print_largest_dataclass_fields(mjx_data_batched, f"batched mjx_data (batch_size={args.batch_size})")

    input_seq = build_input_sequence(
        seed=args.seed,
        device=device,
        num_steps=num_steps,
        batch_size=args.batch_size,
        input_dim=input_dim,
        control_scale=args.control_scale,
        use_actuators=use_actuators,
        home_ctrl=home_ctrl if use_actuators else None,
        ctrl_range=ctrl_range,
    )
    print(
        f"Per-step inputs: {len(input_seq)} arrays, each {input_seq[0].shape}, "
        f"dtype: {input_seq[0].dtype}"
    )

    rollout_fn = make_rollout_fn(
        use_actuators=use_actuators,
        torque_dof_ids=torque_dof_ids,
        num_substeps=args.substeps,
    )

    t0 = time.monotonic()
    print("Compiling MJX rollout function...")
    warmup_out = (mjx_model, mjx_data_batched)
    for step_idx in range(min(num_steps, 10)):
        warmup_out = rollout_fn(warmup_out, input_seq[step_idx])
        print(".", end="", flush=True)
    print()
    # jax.block_until_ready(warmup_out[1].qpos)
    warmup_time = time.monotonic() - t0
    print(f"Warmup compile+run: {warmup_time:.3f} s")

    print("Running timed rollout...")
    t0 = time.monotonic()
    final_out = (mjx_model, mjx_data_batched)
    for step_idx in range(num_steps):
        final_out = rollout_fn(final_out, input_seq[step_idx])
    # jax.block_until_ready(final_out[1].qpos)
    t1 = time.monotonic()

    tot_time = t1 - t0
    avg_substep_time = tot_time / (num_steps * args.substeps)
    avg_step_time = tot_time / (num_steps)

    print(f"Total rollout time: {tot_time:.3f} s")
    print(f"total substeps:     {num_steps * args.substeps}")
    print(f"Avg substep time:     {avg_substep_time * 1e6:.3f} µs")
    print(f"Avg step time:     {avg_step_time * 1e6:.3f} µs")


if __name__ == "__main__":
    main()
