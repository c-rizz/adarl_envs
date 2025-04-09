#!/usr/bin/env python3  
from __future__ import annotations
from adarl_envs.experiments.loco_builder import named_loco_venv_builder

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams
    import os
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)

    algo = args["algorithm"]                                
    if algo == "sac" or algo == "ppo":
        train_envs = 1024
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
        "action_delay_mustd" : (0.0,0.01),
        "action_noise_mustd" : (0.0,0.001),
        "action_smoothing_halflife_sec" : 0.2,
        "control_mode" : "position",
        "robot_model" : args["robot"],
        "enable_rendering" : False,
        "goal_err_smoothing_halflife_sec" : 0.0,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : mode,
        "quiet" : False,
        "reward_acceleration_weight" :      2.0,
        "reward_actdiff_weight" :           1.0,
        "reward_actacc_weight" :            0.1,
        "reward_contacts_weight" :          0.0,
        "reward_energy_weight" :            0.0,
        "reward_health_weight" :            0.25,
        "reward_position_limit_weight" :    0.1,
        "reward_torque_limit_weight" :      0.0,
        "reward_torque_weight" :            5.0,
        "reward_torquediff_weight" :        0.0,
        "reward_tracking_weight" :          0.0,
        "reward_velocity_limit_weight" :    0.1,
        "reward_velocity_weight" :          1.0,
        "reward_height_weight" :            0.0,
        "reward_pitchnroll_weight" :        0.1,
        "reward_position_weight" :          1.0,
        "reward_feet_air_time_weight" :     20.0,
        "reward_heading_weight" :           0.0,
        "safe_stiffness" : 400,
        "safe_damping" : 5,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "fail_on_safety" : True,
        "th_device" : env_device,
        "video_save_freq" : 0,
        "goal_speed_minmax" : (0,0),
        "use_contacts" : False,
        "frame_stack_length" : 3,
        "verbose_infos" : False,
        "terminate_on_body_contact" : False,
        "use_wandb" : False,
        "init_on_reset_ratio" : 0.8,
        "obs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "obs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_posz_ep_mustd_step_std" :        (0.0, 0.02, 0.02),
        "obs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.05, 0.05),
        "ui_camera_resolution_hw" : (144,256),
        "log_info_stats" : True,
        "initial_pose_randomization" : 0.5,
        "mass_randomization_ratio" : 0.0,
        "friction_slide_spin_roll_randomization_ratios" : (0.0,0.0,0.0),
        "impulse_probability_per_sec" : 0.5,
        "impulse_duration_minmax" : (0.01, 2.5),
        "impulse_mean_std" : (50.0,50.0)
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
    video_eval_env_builder_args["verbose_infos"] = True
    video_eval_env_builder_args["video_save_freq"] = 1
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "init_on_reset_ratio" : 1.0
    }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "init_on_reset_ratio" : 1.0
    }
    # video_norand_eval_env_builder_args = copy.deepcopy(env_builder_args)
    # video_norand_eval_env_builder_args["enable_rendering"] = True
    # video_norand_eval_env_builder_args["verbose_infos"] = True
    # video_norand_eval_env_builder_args["video_save_freq"] = 1
    # video_norand_eval_env_builder_args["initial_pose_randomization"] = 0.0
    # eval_conf_video_norand_det = {
    #     "name" : "video_norand_stoch",
    #     "deterministic" : False,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_norand_eval_env_builder_args,
    #     "num_envs" : 1
    # }
    # run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    # run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    # run_1ms_env_builder_args["enable_rendering"] = True
    # run_1ms_env_builder_args["verbose_infos"] = True
    # run_1ms_env_builder_args["video_save_freq"] = 1
    # run_1ms_env_builder_args["initial_pose_randomization"] = 0.0
    # eval_conf_run_1ms = {
    #     "name" : "run_1ms",
    #     "deterministic" : False,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : run_1ms_env_builder_args,
    #     "num_envs" : 1
    # }
    # video_feasible_env_builder_args = copy.deepcopy(feasible_env_builder_args)
    # video_feasible_env_builder_args["enable_rendering"] = True
    # video_feasible_env_builder_args["video_save_freq"] = 1
    # video_feasible_env_builder_args["randomize_initial_pose"] = False
    # eval_conf_video_feasible = {
    #     "name" : "video_feasible",
    #     "deterministic" : True,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_feasible_env_builder_args,
    #     "num_envs" : 1
    # }
    # video_feasible_jump_env_builder_args = copy.deepcopy(video_feasible_env_builder_args)
    # video_feasible_jump_env_builder_args["leg_min_jump"] = 0.2
    # eval_conf_video_jump_feasible = {
    #     "name" : "video_jump_feasible",
    #     "deterministic" : True,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_feasible_jump_env_builder_args,
    #     "num_envs" : 1
    # }
    eval_configurations = [  
                            # eval_conf_video_det,
                            eval_conf_video_stoch,
                            # eval_conf_run_1ms,
                            # eval_conf_video_norand_det,
                            # #  eval_conf_feasible,
                            # #  eval_conf_video_feasible,
                            # #  eval_conf_video_jump_feasible
                        ]
    if algo.lower() == "sac":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = named_loco_venv_builder,
                    env_builder = None,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_hyperparams(  device = "cuda",
                                                    q_network_arch=[512,128],
                                                    q_lr=0.001,
                                                    policy_lr=0.0003,
                                                    policy_network_arch=[256,256],
                                                    gamma=0.99,
                                                    target_tau = 0.005,
                                                    batch_size=4096,
                                                    buffer_size=1_000_000,
                                                    total_steps=300_000_000,
                                                    train_freq_vstep=10,
                                                    grad_steps=40,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -0.5
                                                    ),
                    checkpoint_freq=5,
                    collector_device=env_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=2)                           
    elif algo.lower() == "sac_small":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = named_loco_venv_builder,
                    env_builder = None,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_hyperparams(  device = "cuda",
                                                    q_network_arch=[256,128],
                                                    q_lr=0.001,
                                                    policy_lr=0.0003,
                                                    policy_network_arch=[128,128],
                                                    gamma=0.99,
                                                    target_tau = 0.005,
                                                    batch_size=512,
                                                    buffer_size=100_000,
                                                    total_steps=300_000_000,
                                                    train_freq_vstep=10,
                                                    grad_steps=40,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -0.5
                                                    ),
                    checkpoint_freq=5,
                    collector_device=env_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=2)                           
    elif algo.lower() == "ppo":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=named_loco_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_hyperparams(  minibatch_size=8192,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(64,64),
                                                    critic_network_arch=(64,64),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=5,
                                                    total_steps=train_envs*max_steps_per_episode*1000,
                                                    num_envs=train_envs,
                                                    num_steps=20,
                                                    gamma=0.99,
                                                    log_freq_vstep=int(max_steps_per_episode/10)),
                max_episode_duration=max_steps_per_episode,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0,
                checkpoint_freq=-1,
                collector_device=th.device("cpu"),
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
    ap.add_argument("--algorithm", default="sac", type=str, help="Algorithm to use ('sac'/'ppo')")
    ap.add_argument("--mode", default="mjx", type=str, help="Simulator to use ('mjx'/'pybullet')")
    ap.add_argument("--robot", default="quad", type=str, help="Robot to be used ('quad'/'kyon'/'centauro')")
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
                pkgs_to_save=["adarl","adarl_envs","rreal"])
