#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from adarl_envs.experiments.playground_builder import playground_venv_builder
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=500 #int(ep_duration_sec/step_length_sec)

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
    
    eval_freq = 5
    env_builder_args = {
        "video_save_freq" : -1,
        "record_video" : False,
        "env_name" : "SpotFlatTerrainJoystick",
        "quiet" : False,
        "th_device" : env_device,
        "log_info_stats": True,
        "randomize_step_timeout_counters": True,
        "camera" : "track",
        "episode_length" : max_steps_per_episode,
        "playground_config_overrides": {"reward_config.scales.feet_clearance":0.0,
                                        "reward_config.scales.feet_height":0.0
                                        }
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args.update({
        "video_save_freq" : 1,
        "record_video" : True,
        "randomize_step_timeout_counters": False
    })
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "skip_first_eval": True
    }

    eval_configurations = [eval_conf_video_stoch]
       

    if algo.lower() == "ppo":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=playground_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_hyperparams(  minibatch_size=None,
                                                    minibatch_num=4,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(512,256),
                                                    critic_network_arch=(512,256),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=5,
                                                    total_steps=train_envs*max_steps_per_episode*1000,
                                                    num_envs=train_envs,
                                                    num_steps=24,
                                                    gamma=0.99,
                                                    loss_value_weight=1.0,
                                                    loss_entropy_coeff=0.001,
                                                    log_freq_vstep=int(max_steps_per_episode/10),
                                                    epsilon_policy_ratio_clip=0.2,
                                                    epsilon_value_clip_epsilon=0.2,
                                                    gae_lambda=0.95,
                                                    max_grad_norm=1.0,
                                                    init_actor_logstd=-1.0,
                                                    actor_observation_filter=["state"],
                                                    critic_observation_filter=["privileged_state"],
                                                    ),
                max_episode_duration=max_steps_per_episode,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0,
                checkpoint_freq=-1,
                collector_device=env_device,
                eval_configurations=eval_configurations,
                debug_level=1,
                env_checker_max_obs_value=500.0,
                env_checker_max_rew_value=100.0,)
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
