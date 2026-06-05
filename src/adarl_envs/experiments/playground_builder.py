#!/usr/bin/env python3  
from __future__ import annotations

import json
import os
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from adarl.envs.vec.PlaygroundMjxEnvRunner import PlaygroundMjxEnvRunner, wrap_for_adarl_training
from ml_collections import config_dict
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
import torch as th
import gymnasium as gym
from mujoco_playground._src import wrapper


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in registry.manipulation._envs:
        return manipulation_params.rsl_rl_config(env_name)
    elif env_name in registry.locomotion._envs:
        return locomotion_params.rsl_rl_config(env_name)
    else:
        raise ValueError(f"No RL config for {env_name}")

def loco_mujoco_playground_runner_builder(seed,
                        run_folder,
                        num_envs : int,
                        env_builder_args : dict,
                        env_name : str = "",
                        autoreset : bool = True,
                        quiet : bool = False) -> PlaygroundMjxEnvRunner:

    if not autoreset:
        raise NotImplementedError("autoreset=False is not currently supported for the playground runner, because the underlying MjxEnv does not support it. Please set autoreset=True or use a different runner if you need this feature.")
    # Possibly parse the device for multi-GPU
    video_save_freq =   env_builder_args.pop("video_save_freq")
    record_video =      env_builder_args["record_video"]
    env_name =          env_builder_args["env_name"]
    device =            env_builder_args.get("device", "cuda:0")
    impl =              env_builder_args.get("impl", "jax")
    camera=             env_builder_args.get("camera", None)
    episode_length =    env_builder_args.get("episode_length", 1000)
    playground_config_overrides = env_builder_args.get("playground_config_overrides", None)
    

    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

    if env_name is None:
        raise ValueError("env_builder_args must contain 'env_name':"
                        f"One of: {', '.join(mujoco_playground.registry.ALL_ENVS)}")

    if env_name == "SpotFlatTerrainJoystick":
        _joints = ["fl_hx","fl_hy","fl_kn","fr_hx","fr_hy","fr_kn",
                   "hl_hx","hl_hy","hl_kn","hr_hx","hr_hy","hr_kn"]
        _feet = ["FL","FR","HL","HR"]
        _xyz = ["x","y","z"]
        _state_labels = (
            [f"gyro_{a}" for a in _xyz]                                       # 3
            + [f"gravity_{a}" for a in _xyz]                                  # 3
            + [f"jpos_minus_default_{j}" for j in _joints]                    # 12
            + [f"qpos_err_t-{t}_{j}" for t in range(3) for j in _joints]      # 36 (history_len=3)
            + [f"feet_pos_{f}_{a}" for f in _feet for a in _xyz]              # 12
            + [f"last_act_{j}" for j in _joints]                              # 12
            + ["cmd_lin_vel_x","cmd_lin_vel_y","cmd_ang_vel_yaw"]             # 3
        )                                                                     # = 81
        _privileged_extra = (
            [f"gyro_clean_{a}" for a in _xyz]                                 # 3
            + [f"accel_{a}" for a in _xyz]                                    # 3
            + [f"gravity_clean_{a}" for a in _xyz]                            # 3
            + [f"local_linvel_{a}" for a in _xyz]                             # 3
            + [f"global_angvel_{a}" for a in _xyz]                            # 3
            + [f"jpos_clean_minus_default_{j}" for j in _joints]              # 12
            + [f"feet_pos_clean_{f}_{a}" for f in _feet for a in _xyz]        # 12
            + [f"qvel_{j}" for j in _joints]                                  # 12
            + [f"actuator_force_{j}" for j in _joints]                        # 12
            + [f"last_contact_{f}" for f in _feet]                            # 4
            + [f"foot_linvel_{f}_{a}" for f in _feet for a in _xyz]           # 12
            + [f"feet_air_time_{f}" for f in _feet]                           # 4
            + [f"xfrc_torso_{a}" for a in _xyz]                               # 3
        )                                                                     # = 86
        obs_labels = {"state":            _state_labels,
                      "privileged_state": _state_labels + _privileged_extra}  # 81 + 86 = 167
    else:
        obs_labels = None

    # Load default config from registry
    env_cfg = registry.get_default_config(env_name)
    env_cfg.impl = impl
    print(f"Environment config:\n{env_cfg}")
    if isinstance(playground_config_overrides, dict):
        env_cfg_overrides = playground_config_overrides
    else:
        env_cfg_overrides = json.loads(playground_config_overrides) if playground_config_overrides is not None else {}
    print(f"Environment config overrides:\n{env_cfg_overrides}\n")

    # Domain randomization
    # randomizer = registry.get_domain_randomizer(env_name)
    # We'll store environment states during rendering
    #   render_trajectory = []

    # # Callback to gather states for rendering
    # def render_callback(env : mujoco_playground.MjxEnv, state):
    #     # render_trajectory.append(state)
    #     return env.render(  [state],
    #                         camera=camera,
    #                         height=480,
    #                         width=640,
    #                         # scene_option=scene_option,
    #                         )[0]

    # Create the environment
    raw_env = registry.load(
        env_name, config=env_cfg, config_overrides=env_cfg_overrides
    )

    from adarl.adapters.mujoco_utils import print_mj_model
    os.makedirs(run_folder, exist_ok=True)
    print_mj_model(raw_env.mj_model, full_dump=True, file=run_folder+"/mj_model_full.txt")

    env = wrap_for_adarl_training(
        raw_env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=None
    )

    vrunner = PlaygroundMjxEnvRunner(env,
                                     num_envs=num_envs,
                                     seed=seed,
                                     device=th.device(device),
                                     ui_render_envs=[0],
                                     render_camera_name=camera,
                                     randomize_step_timeout_counters=env_builder_args["randomize_step_timeout_counters"]
                                     )

    stepLength_sec : float = env_cfg.ctrl_dt
    if video_save_freq>0:
        vrunner = EnvRunnerRecorderWrapper(vrunner,
                                        fps = 1/stepLength_sec,
                                        outFolder=run_folder+"/RunnerRecorder",
                                        env_index=0,
                                        saveFrequency_ep=video_save_freq,
                                        publish=False,
                                        stream=True,
                                        vec_obs_keys = ["state","privileged_state"], #"base.vec", #TODO: somehow pass multiple keys and include privileged, or auto-detect which keys to save
                                        record_video=record_video,
                                        overlay_text_xy=(0.025,0.025),
                                        overlay_text_height=0.035,
                                        overlay_text_color_rgb=(255,150,0),
                                        overlay_text_func=None,
                                        override_obs_labels=obs_labels)
    return vrunner
    
def playground_venv_builder(seed : int,
                                run_folder : str,
                                num_envs : int, 
                                env_builder_args : dict,
                                env_name : str = "") -> gym.vector.VectorEnv:
    vrunner = loco_mujoco_playground_runner_builder(seed, run_folder, num_envs, env_builder_args, env_name)
    env = Runner2VecGymWrapper(runner=vrunner, quiet=env_builder_args["quiet"])        
    env.reset(seed=seed)
    return env

from loco_builder import single_env_builder
def playground_single_env_builder(seed : int,
                    log_folder : str,
                    is_eval : bool, 
                    env_builder_args : dict) -> tuple[gym.Env,float]:    
    env_builder_args["stepLength_sec"] = registry.get_default_config(env_builder_args["env_name"]).ctrl_dt
    print(f"playground_single_env_builder with env_builder_args = {env_builder_args}")
    return single_env_builder(seed = seed,
                            log_folder = log_folder,
                            env_builder_args = env_builder_args,
                            is_eval=is_eval,
                            runner_builder=loco_mujoco_playground_runner_builder)

