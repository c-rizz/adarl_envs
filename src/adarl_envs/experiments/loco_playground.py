#!/usr/bin/env python3  
from __future__ import annotations

import json
import os

from ml_collections import config_dict
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
from adarl.envs.vec.GymEnvRunner import GymEnvRunner
from adarl.envs.vec.PlaygroundMjxEnvRunner import PlaygroundMjxEnvRunner
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
import torch as th
import gymnasium as gym

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

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
    randomizer = registry.get_domain_randomizer(env_name)
    # We'll store environment states during rendering
    #   render_trajectory = []

    # Callback to gather states for rendering
    def render_callback(env : mujoco_playground.MjxEnv, state):
        # render_trajectory.append(state)
        return env.render(  [state],
                            camera=camera,
                            height=480,
                            width=640,
                            # scene_option=scene_option,
                            )[0]

    # Create the environment
    raw_env = registry.load(
        env_name, config=env_cfg, config_overrides=env_cfg_overrides
    )

    vrunner = PlaygroundMjxEnvRunner(raw_env,
                                     num_envs=num_envs,
                                     seed=seed,
                                     device=th.device(device),
                                     ui_render_envs=[0],
                                     render_camera_name=camera,
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
    
def loco_playground_venv_builder(seed : int,
                                run_folder : str,
                                num_envs : int, 
                                env_builder_args : dict,
                                env_name : str = "") -> gym.vector.VectorEnv:
    vrunner = loco_mujoco_playground_runner_builder(seed, run_folder, num_envs, env_builder_args, env_name)
    env = Runner2VecGymWrapper(runner=vrunner, quiet=env_builder_args["quiet"])        
    env.reset(seed=seed)
    return env


































def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams, TargetEntropyAnnealer
    from adarl_envs.experiments.loco_builder import named_loco_venv_builder
    from adarl_envs.env.LocomotionVecEnv import TransitionAugmentor
    import math
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=1000 #int(ep_duration_sec/step_length_sec)

    algo = args["algorithm"]                                
    if algo == "sac" or algo == "ppo":
        train_envs = 4096
    elif algo == "sac_small":
        train_envs = 8
    else:
        raise RuntimeError(f"Unexpected algo '{algo}'")
    
    if mode == "pybullet":
        env_device = th.device("cpu",0)
    elif mode == "mjx":
        env_device = th.device("cuda",0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")
    
    eval_freq = 10
    env_builder_args = {
        "video_save_freq" : -1,
        "record_video" : False,
        "env_name" : "Go1JoystickFlatTerrain"   
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args.update({
        "video_save_freq" : 1,
        "record_video" : True
    })
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 100,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 100,
        "skip_first_eval": False
    }

    eval_configurations = [eval_conf_video_stoch]
       

    if algo.lower() == "ppo":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=loco_playground_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_hyperparams(  minibatch_size=512,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(512,256),
                                                    critic_network_arch=(512,256),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=3,
                                                    total_steps=train_envs*max_steps_per_episode*1000,
                                                    num_envs=train_envs,
                                                    num_steps=40,
                                                    gamma=0.98,
                                                    log_freq_vstep=int(max_steps_per_episode/10),
                                                    # actor_observation_filter=["base.vec","base.last_action_raw", "base.reward_weights"],
                                                    # critic_observation_filter=["base.vec","base.last_action_raw","privileged.vec", "base.reward_weights"],
                                                    ),
                max_episode_duration=max_steps_per_episode,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0,
                checkpoint_freq=-1,
                collector_device=env_device,
                eval_configurations=eval_configurations,
                debug_level=1)
    else:
        raise RuntimeError(f"Unknown algorithm '{algo}'")



if __name__ == "__main__":

    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--algorithm", default="ppo", type=str, help="Algorithm to use ('sac'/'ppo')")
    ap.add_argument("--mode", default="mjx", type=str, help="Simulator to use ('mjx'/'pybullet')")
    ap.add_argument("--no-wandb", default=False, action='store_true', help="Disable Weight&Biases")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                start_adarl=False,
                pkgs_to_save=["adarl","adarl_envs","rreal"],
                use_wandb=not args["no_wandb"])
