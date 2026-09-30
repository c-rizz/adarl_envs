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
    # Visual variation preset of the original setup, chosen from the seed (see sample_variation).
    # None keeps the scene as configured.
    "variations_sampler" : None,
}

# Cube colors of the original setup's variation presets
_VARIATION_COLORS = [(0.9, 0.1, 0.1), # 0 red
                     (0.1, 0.9, 0.1), # 1 green
                     (0.1, 0.1, 0.9), # 2 blue
                     (0.1, 0.9, 0.9), # 3 cyan
                     (0.9, 0.9, 0.1), # 4 yellow
                     (0.9, 0.1, 0.9), # 5 magenta
                     (0.7, 0.7, 0.7), # 6 grey70
                     (0.5, 0.5, 0.5), # 7 grey50
                     (0.3, 0.3, 0.3), # 8 grey30
                     (0.9, 0.9, 0.9)] # 9 grey90


def sample_variation(variations_sampler : str | None, seed : int) -> dict:
    """Scene settings of one of the original setup's visual variation presets
    (PandaPushingEnv2DOF's variations_sampler), chosen from the seed.

    Returns the env builder args to override (a subset of cube_color, camera_offset_xyz,
    camera_offset_rpy and light_direction). Every environment of a run gets the same variation: the
    seed offsets rreal gives the collector (+100000*i) and evaluation (+100000000) environments leave
    all the indices below unchanged.

    Presets:
        sim-minimal : cube color, one of 10.
        sim-small   : red cube, one of 4 camera position offsets and one of 4 light directions.
        sim-medium  : one of 4 camera position+orientation offsets, one of 4 light directions and
                      one of 10 cube colors.
        sim-large   : red cube, one of 2 camera placements (turned by +-90 degrees).
        real-minimal, real-small, real-large : no change, they only applied to the real setup.
    """
    if variations_sampler is None:
        return {}
    if variations_sampler == "sim-minimal":
        return {"cube_color" : _VARIATION_COLORS[seed%len(_VARIATION_COLORS)]}
    elif variations_sampler == "sim-small":
        cam_offsets = [(0.05, 0.00,  0.0),   (-0.05,  0.0, 0.0),  (0.0,  0.0, 0.05),  (0.0,   0.0, -0.05)]
        light_dirs =  [(0.50, 0.01, -1.0),   (-0.5,  0.01, -1.0), (0.01, 0.5, -1.0),  (0.01, -0.5, -1.0)]
        return {"cube_color" : _VARIATION_COLORS[0],
                "camera_offset_xyz" : cam_offsets[(seed*5)%4],
                "light_direction" : light_dirs[int((seed*5)/4)%4]}
    elif variations_sampler == "sim-medium":
        cam_offsets =     [(0.05,  0.0,  0.15), ( 0.10,  0.0, -0.15), (0.05,   0.15,  0.0), (0.05,  -0.15,  0.0)]
        cam_rpy_offsets = [(0.00,  0.1,  0.00), ( 0.00, -0.2,  0.00), (0.00,   0.00,  0.4), (0.00,   0.00, -0.3)]
        light_dirs =      [(0.5,   0.01, -1.0), (-0.5,   0.01, -1.0), (0.01,   0.5,  -1.0), (0.01,  -0.5,  -1.0)]
        return {"camera_offset_xyz" : cam_offsets[(seed*5)%4],
                "camera_offset_rpy" : cam_rpy_offsets[(seed*5)%4],
                "light_direction" : light_dirs[int((seed*5)/4)%4],
                "cube_color" : _VARIATION_COLORS[int((seed/16)%10)]}
    elif variations_sampler == "sim-large":
        cam_rpy_offsets = [(0.0, 0.0, 1.5707), (0.0, 0.0, -1.5707)]
        cam_offsets =     [(-0.45, 0.5, 0.0), (-0.5, -0.5, 0.0)]
        return {"cube_color" : _VARIATION_COLORS[0],
                "camera_offset_xyz" : cam_offsets[seed%2],
                "camera_offset_rpy" : cam_rpy_offsets[seed%2],
                "light_direction" : (0.01, 0.5, -1.0)}
    elif variations_sampler in ("real-minimal", "real-small", "real-large"):
        return {}
    else:
        raise AttributeError(f"Unknown variations_sampler {variations_sampler}")


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
                                   # the scene brings its own light, whose direction the variation presets change
                                   add_ground_light=False,
                                   log_folder=run_folder,
                                   opt_preset=env_builder_args.get("mjx_opt_preset", "fast"),
                                   opt_override=env_builder_args.get("mjx_opt_override", {}),
                                   default_actuator_kp=env_builder_args["actuator_kp"],
                                   default_actuator_kv=env_builder_args["actuator_kv"],
                                   default_max_actuator_force=env_builder_args["max_actuator_force"],
                                   mjx_impl="warp")


def build_pushing_env(seed : int, run_folder : str, num_envs : int, env_builder_args : dict) -> PushingVecEnv:
    """Build a PushingVecEnv and its adapter. No training-framework dependency."""
    env_builder_args = copy.deepcopy(DEFAULT_ENV_BUILDER_ARGS) | copy.deepcopy(env_builder_args)
    variation = sample_variation(env_builder_args["variations_sampler"], seed)
    if variation:
        ggLog.info(f"Pushing env variation '{env_builder_args['variations_sampler']}' for seed {seed}: {variation}")
    env_builder_args |= variation
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
                           "camera_offset_xyz", "camera_offset_rpy", "light_direction", "enable_ui_camera")}
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
