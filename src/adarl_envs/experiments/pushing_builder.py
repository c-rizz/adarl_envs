#!/usr/bin/env python3
"""Builders for the simplified pushing environment (adarl port of panda_pushing)."""
from __future__ import annotations

import copy
import os
import torch as th

import adarl.utils.dbg.ggLog as ggLog
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
from adarl_envs.env.PushingVecEnv import PushingVecEnv

EE_LINK = ("pushing", "pushing_ee")
X_JOINT = ("pushing", "ee_x_slider")
Y_JOINT = ("pushing", "ee_y_slider")

DEFAULT_ENV_BUILDER_ARGS = {
    "mode" : "mjx",
    "th_device" : th.device("cuda", 0),
    # Control period: how often the blocking step advances the trajectory reference and checks
    # whether the end effector arrived. It is NOT the duration of an environment step -- that is
    # variable, since a step lasts as long as the commanded movement takes (see max_step_duration_sec
    # for its cap). Keep it a multiple of 1/1024 so it stays representable in binary.
    "control_period_sec" : 20/1024,
    "max_steps_per_episode" : 50,
    "observe_camera" : True,
    "enable_rendering" : True,
    "enable_ui_camera" : True,
    "show_gui" : False,
    "record_video" : False,
    "video_save_freq" : -1,
    # Frames per second of the recorded video. One frame is one environment step, and a step
    # simulates a variable amount of time (~0.7s under a pushing policy, since it lasts as long as
    # the commanded movement takes), so no fixed value is real time -- real time would be ~1.4fps,
    # which is unwatchable. 10fps shows each step for 100ms, i.e. ~5s for a 50-step episode.
    "video_fps" : 10.0,
    "quiet" : False,
    # End effector position controller. Since the step is blocking, the settling time of these gains
    # sets how much simulation an agent step costs: keep the controller near critical damping, i.e.
    # actuator_kv ~= 2*sqrt(actuator_kp*ee_mass) (ee_mass is 0.1kg by default). Overdamping it is
    # expensive -- kv=30 at kp=200 (zeta=3.35) needs ~20 control chunks per step instead of ~6.
    "actuator_kp" : 200.0,
    "actuator_kv" : 9.0,
    "max_actuator_force" : 50.0,
    "position_tolerance" : 0.002,
    "max_step_duration_sec" : 2.0,
    "blocking_movement" : True,
    # A command is followed as a quintic, velocity/acceleration-limited trajectory, like the original
    # PyBullet pushing setup did. These defaults are that setup's effective limits (PyBullet's
    # defaults of 1 m/s and 10 m/s^2, scaled by its velocity_scaling=0.9 and acceleration_scaling=0.5).
    # They set how long a movement takes, and so how much simulation an agent step costs.
    "ee_max_velocity" : 0.9,
    "ee_max_acceleration" : 5.0,
    "use_trajectory" : True,
    "sim_step_dt" : 1/1024,
}


def build_adapter(num_envs : int, run_folder : str, env_builder_args : dict):
    """Build the adapter driving the floating end effector."""
    mode = env_builder_args["mode"].strip().lower()
    th_device = env_builder_args["th_device"]
    if mode != "mjx":
        raise NotImplementedError(f"Only the 'mjx' mode is implemented for the pushing environment, got '{mode}'")
    from adarl.adapters.Mjx2DofCartesianAdapter import Mjx2DofCartesianAdapter
    import jax
    sim_step_dt = env_builder_args["sim_step_dt"]
    # The blocking step runs the simulation in chunks of this length, advancing the trajectory
    # reference and checking the end-effector tracking error once per chunk. It must stay constant
    # to avoid jit recompilations.
    control_chunk_sec = env_builder_args["control_period_sec"]
    return Mjx2DofCartesianAdapter(end_effector_link=EE_LINK,
                                   xjoint=X_JOINT,
                                   yjoint=Y_JOINT,
                                   position_tolerance=env_builder_args["position_tolerance"],
                                   blocking_movement=env_builder_args["blocking_movement"],
                                   max_step_duration_sec=env_builder_args["max_step_duration_sec"],
                                   sim_dt=control_chunk_sec,
                                   max_velocity=env_builder_args["ee_max_velocity"],
                                   max_acceleration=env_builder_args["ee_max_acceleration"],
                                   use_trajectory=env_builder_args["use_trajectory"],
                                   vec_size=num_envs,
                                   enable_rendering=env_builder_args["enable_rendering"],
                                   jax_device=(jax.devices("gpu")[th_device.index]
                                               if th_device.type == "cuda" else jax.devices("cpu")[0]),
                                   output_th_device=th_device,
                                   sim_step_dt=sim_step_dt,
                                   step_length_sec=control_chunk_sec,
                                   realtime_factor=-1.0,
                                   show_gui=env_builder_args["show_gui"],
                                   gui_env_index=0,
                                   add_ground=True,
                                   log_folder=run_folder,
                                   opt_preset=env_builder_args.get("mjx_opt_preset", "fast"),
                                   opt_override=env_builder_args.get("mjx_opt_override", {}),
                                   default_actuator_kp=env_builder_args["actuator_kp"],
                                   default_actuator_kv=env_builder_args["actuator_kv"],
                                   default_max_actuator_force=env_builder_args["max_actuator_force"])


def build_pushing_env(seed : int, run_folder : str, num_envs : int, env_builder_args : dict) -> PushingVecEnv:
    """Build a PushingVecEnv and its adapter. No training-framework dependency."""
    env_builder_args = copy.deepcopy(DEFAULT_ENV_BUILDER_ARGS) | copy.deepcopy(env_builder_args)
    os.makedirs(run_folder, exist_ok=True)
    adapter = build_adapter(num_envs=num_envs, run_folder=run_folder, env_builder_args=env_builder_args)
    env_kwargs = {k:v for k,v in env_builder_args.items()
                  if k in ("observe_camera", "history_length", "frame_stack_length",
                           "obs_camera_resolution_hw", "obs_camera_render_resolution_hw",
                           "img_crop_ltrb", "ui_camera_resolution_hw", "operating_area_xy", "goal_tolerance",
                           "max_position_change", "ee_height", "ee_diameter", "cube_size", "cube_mass",
                           "prevent_ee_out", "terminate_on_success", "sparse_reward", "reward_cube_pos_weight",
                           "reward_tip_pos_weight", "reward_cube_move_weight", "reward_scale",
                           "goal_spawn_border_dist", "cube_spawn_border_dist", "ee_spawn_border_dist",
                           "allow_successful_initial_cube_position", "spawn_rejection_candidates",
                           "cube_color", "camera_position_xyz", "camera_orientation_rpy",
                           "camera_offset_xyz", "camera_offset_rpy", "enable_ui_camera")}
    return PushingVecEnv(adapter=adapter,
                         th_device=env_builder_args["th_device"],
                         max_episode_steps=env_builder_args["max_steps_per_episode"],
                         # ControlledVecEnv's name for its nominal step length; inert for this env,
                         # whose step duration is decided by the adapter (see PushingVecEnv).
                         step_duration_sec=env_builder_args["control_period_sec"],
                         seed=seed,
                         **env_kwargs)


def overlay_text_func(vo, a, r, te, tr, info, extra_info):
    def fmt(t, precision=3):
        if t is None:
            return "None"
        t = t.squeeze().cpu().tolist()
        if not isinstance(t, list):
            t = [t]
        return "[" + ", ".join(f"{e: .{precision}f}" if isinstance(e, float) else str(e) for e in t) + "]"
    return (f"\n"
            f"Step             {fmt(info.get('ep_step_count', None), 0)}\n"
            f"cube2goal_dist   {fmt(info.get('cube2goal_dist', None))}\n"
            f"cube2tip_dist    {fmt(info.get('cube2tip_dist', None))}\n"
            f"ee_tracking_err  {fmt(info.get('ee_tracking_error', None))}\n"
            f"success          {fmt(info.get('success', None), 0)}\n")


def runner_builder(seed,
                   run_folder,
                   num_envs : int,
                   env_builder_args : dict,
                   env_name : str = "",
                   autoreset : bool = True,
                   quiet : bool = False) -> EnvRunner:
    ggLog.info(f"Building pushing env: env_builder_args = {env_builder_args}")
    args = copy.deepcopy(DEFAULT_ENV_BUILDER_ARGS) | copy.deepcopy(env_builder_args)
    lrenv = build_pushing_env(seed=seed, run_folder=run_folder, num_envs=num_envs, env_builder_args=args)
    max_steps = args["max_steps_per_episode"]
    vrunner : EnvRunner = EnvRunner(env=lrenv,
                                    verbose=not quiet,
                                    quiet=quiet,
                                    episodeInfoLogFile=run_folder+"/vec_runner.log",
                                    ui_render_envs=[0],
                                    autoreset=autoreset,
                                    log_freq=max_steps)
    if args.get("video_save_freq", -1) > 0:
        vrunner = EnvRunnerRecorderWrapper(vrunner,
                                           fps=args["video_fps"],
                                           outFolder=run_folder+"/RunnerRecorder",
                                           env_index=0,
                                           saveFrequency_ep=args["video_save_freq"],
                                           publish=False,
                                           stream=True,
                                           vec_obs_keys=["base.vec","privileged.vec"],
                                           overlay_text_xy=(0.025, 0.025),
                                           overlay_text_height=0.035,
                                           overlay_text_color_rgb=(255, 150, 0),
                                           overlay_text_func=overlay_text_func)
    return vrunner


def pushing_vecenv_builder(seed, run_folder, num_envs : int, env_builder_args : dict, env_name : str = ""):
    """Gym-style vectorized env builder, for the rreal training helpers."""
    from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
    quiet = env_builder_args.get("quiet", False)
    vrunner = runner_builder(seed=seed,
                             run_folder=run_folder,
                             num_envs=num_envs,
                             env_builder_args=env_builder_args,
                             quiet=quiet)
    env = Runner2VecGymWrapper(runner=vrunner, quiet=quiet)
    env.reset(seed=seed)
    return env
