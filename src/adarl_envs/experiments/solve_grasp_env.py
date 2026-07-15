#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams
    from adarl_envs.experiments.grasp_builder import centgrasp_vecenv_builder
    import os
    import math

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
    elif mode == "mujoco":
        env_device = th.device("cpu",0)
    elif mode == "mjx":
        env_device = th.device("cuda",0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")

    eval_freq = 5
    r = 0.0
    n = 0.0
    env_builder_args = {
        "reward_health_weight" :                0.0,
        "reward_joint_power_weight" :           0.0,
        "reward_joint_actacc_weight" :          0.0,
        "reward_joint_actdiff_weight" :         0.1,
        "reward_joint_torque_weight" :          0.002, 
        "reward_joint_position_limit_weight" :  0.0,
        "reward_joint_position_weight" :        2.0,
        "reward_object_pose_weight" :           1.0,
        "reward_gripper_pose_weight" :          1.0,
        "reward_height_position_weight" :       1.0,
        "reward_pitchnroll_weight" :            1.0,
        "reward_velocity_tracking_weight" :     0.0,
        "reward_yaw_vel_track_weight" :         0.0,
        "neutral_body_height" : 0.45,
        "reward_safety_weight" : 0.0,
        "target_object_link" : ("cube","cube"),
        "observe_object_pose" : True,
        "noise_action_delay_mustd_std" : (0.008, 0.005*n, 0.0025*n),
        "noise_action_mustd" : (0.0, 0.0),
        "observe_actor_safety_state" : False,
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position_delta",
        "desired_foot_clearance" : 0.05,
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "enable_reference_filter" : True,
        "fail_on_safety" : False,
        "frame_stack_length" : 3,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.79,0.79],
        "goal_resampling_probability_per_sec" : 0.1,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "goal_yaw_vel_zero_ratio" : 0.25,
        "goal_yaw_vel_minmax" : (-1.0, 1.0),
        "held_joints_damping" : 20.0,
        "held_joints_stiffness" : 2000.0,
        "history_length_action_smoothed" : 0,
        "history_length_action_raw" : 1,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 0.7,
        "randomization_initial_height_range_meters" : 0.0,
        "randomization_initial_joint_pose_range" : {"default" : 0.05, 
                                                    **{("kyon",jn) : 0.9 for jn in ["shoulder_yaw_", "shoulder_pitch_", "elbow_pitch_", "wrist_pitch_", "wrist_yaw_"]}},
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 0.0, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "step_max_good_air_duration" : 0.5,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "step_min_good_air_duration" : 0.1,
        "mode" : mode,
        "noise_abs_obs_angvel_ep_mustd_step_std" :      [0.0, 0.02*n,  0.1*n],
        "noise_abs_obs_gravity_ep_mustd_step_std" :     [0.0, 0.0*n,   0.1*n],
        "noise_abs_obs_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.1*n],
        "noise_abs_obs_linacc_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_linvel_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_posz_ep_mustd_step_std" :        [0.0, 0.0,     0.0],
        "observe_full_robot_state" : False,
        "offset_envs_ep_starts" : False,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomized_dof_armature_ratios" :      0.05*r,
        "randomized_dof_frictionloss_ratios":   0.1*r,
        "randomized_dof_damping_ratios":        0.2*r,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : ("uniform", ([0.8*r,0.8*r,0.8*r],[1.5*r,1.5*r,1.5*r])),
        "randomized_gains_damping_ratio_epstd"       : 0.2*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.2*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_reference_filter_distribution" : ("uniform", (20.0, 50.0)),
        "record_video" : True,
        "randomization_recycle_init_pose" : True,
        "robot_model" : args["robot"],
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : True if algo=="sac" else False,
        "stepLength_sec" : step_length_sec,
        "terminate_on_safety" : False,
        "terminate_on_crash" : True,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0,
        "minimal_infos" : True,
        "playground_style_reward" : False,
        "extrinsics_only_privileged" : False,
        "posref_err_history_length" : 0,
        # "mjx_geom_overrides" : {
        #     "cube" : {"friction" : [2.0,0.001,0.0005]}
        # },
        "mjx_opt_override" : {"impratio" : 1.0,
                            #   "ccd_iterations" : 50
                              },
        "robot_options" : {
            "spawn_legs" : True,
            "ctrl_legs" : True
        },
        "mjx_warp_nccdmax" : 20,
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
    video_eval_env_builder_args["verbose_infos"] = True
    video_eval_env_builder_args["minimal_infos"] = False
    video_eval_env_builder_args["video_save_freq"] = 1
    video_eval_env_builder_args["ui_camera_resolution_hw"] = (270,480)
    video_eval_env_builder_args["randomization_recycle_init_pose"] = False
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
    # video_norand_eval_env_builder_args["initial_pose_randomization_range"] = 0.0
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
    # run_1ms_env_builder_args["initial_pose_randomization_range"] = 0.0
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
                    vec_env_builder = centgrasp_vecenv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams(  model_th_device = "cuda",
                                                    q_network_arch=[512,128],
                                                    q_lr=0.001,
                                                    policy_lr=0.0003,
                                                    policy_arch=[256,256],
                                                    gamma=0.99,
                                                    target_tau = 0.005,
                                                    batch_size=16384,
                                                    buffer_size=5*1024*1000,
                                                    total_steps=10_00_000_000,
                                                    train_freq_vstep=5,
                                                    grad_steps=80,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -1.0,
                                                    actor_log_std_init = -2.0,
                                                    actor_mean_bounds_ratio = 0.9
                                                    ),
                    checkpoint_freq=5,
                    collector_device=env_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=0)
    elif algo.lower() == "sac_small":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = centgrasp_vecenv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams(  model_th_device = "cuda",
                                                    q_network_arch=[512,128],
                                                    q_lr=0.001,
                                                    policy_lr=0.0003,
                                                    policy_arch=[256,256],
                                                    gamma=0.99,
                                                    target_tau = 0.005,
                                                    batch_size=512,
                                                    buffer_size=300*1024,
                                                    total_steps=300_000_000,
                                                    train_freq_vstep=10,
                                                    grad_steps=40,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -0.5,
                                                    actor_log_std_init = -2.0
                                                    ),
                    checkpoint_freq=5,
                    collector_device=env_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=0)                       
    elif algo.lower() == "ppo":
        from rreal.algorithms.ppo2 import ppo_train, PPO_init_hyperparams
        ppo_train(  seed=seed,
                    folderName=folderName,
                    run_id=run_id,
                    args=args,
                    env_builder=None,
                    vec_env_builder=centgrasp_vecenv_builder,
                    env_builder_args=env_builder_args,
                    agent_hyperparams=PPO_init_hyperparams(  minibatch_size=train_envs*24//4,
                                                        minibatch_num=4,
                                                        th_device=th.device("cuda"),
                                                        actor_network_arch=(512,256,64,64),
                                                        critic_network_arch=(512,256,64,64),
                                                        q_lr=None,
                                                        policy_lr=3e-4,
                                                        update_epochs=5,
                                                        total_steps=500_000_000,
                                                        num_envs=train_envs,
                                                        num_steps=24,
                                                        gamma=0.98,
                                                        loss_value_weight=1.0,
                                                        loss_entropy_coeff=1e-3,
                                                        log_freq_vstep=int(max_steps_per_episode/10),
                                                        epsilon_policy_ratio_clip=0.2,
                                                        epsilon_value_clip_epsilon=0.2,
                                                        gae_lambda=0.95,
                                                        max_grad_norm=1.0,
                                                        init_actor_logstd=-1.0,
                                                        actor_observation_filter=["base.vec"], #, "base.reward_weights"],
                                                        critic_observation_filter=["privileged.vec"], #, "base.reward_weights"],
                                                        actor_mean_bounds_ratio=0.8),
                    max_episode_duration=max_steps_per_episode,
                    validation_batch_size=0,
                    validation_buffer_size=0,
                    validation_holdout_ratio=0,
                    checkpoint_freq_vec_ep=10,
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
    ap.add_argument("--algorithm", default="sac", type=str, help="Algorithm to use ('sac'/'ppo')")
    ap.add_argument("--mode", default="mjx", type=str, help="Simulator to use ('mjx'/'pybullet')")
    ap.add_argument("--robot", default="centauro", type=str, help="Robot to be used ('centauro'/'franka')")
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
