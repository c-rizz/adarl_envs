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
                        env_name : str = ""):

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

    # Load default config from registry
    env_cfg = registry.get_default_config(env_name)
    env_cfg.impl = impl
    print(f"Environment config:\n{env_cfg}")
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
    vrunner = EnvRunnerRecorderWrapper(vrunner,
                                    fps = 1/stepLength_sec,
                                    outFolder=run_folder+"/RunnerRecorder",
                                    env_index=0,
                                    saveFrequency_ep=video_save_freq,
                                    publish=False,
                                    stream=True,
                                    vec_obs_key= None, #"base.vec", #TODO: somehow pass multiple keys and include privileged, or auto-detect which keys to save
                                    record_video=record_video,
                                    overlay_text_xy=(0.025,0.025),
                                    overlay_text_height=0.035,
                                    overlay_text_color_rgb=(255,150,0),
                                    overlay_text_func=None)
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
