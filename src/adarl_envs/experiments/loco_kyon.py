#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams
    from adarl_envs.experiments.loco_builder import named_loco_venv_builder
    import math
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=500 #int(ep_duration_sec/step_length_sec)

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
    
    eval_freq = 10
    r = 1.0 # randomization strength
    n = 1.0 # noise strength
    p = 1.0 # penalties strength
    eps = 0 #1e-6 # For disabled things (but no zero, so I can still see how they would behave)
    env_builder_args = {
        "action_delay_mustd_std" : (0.003, 0.001*n, 0.0025*n),
        "action_noise_mustd" : (0.0,   0.0),
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position",
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "fail_on_safety" : False,
        "frame_stack_length" : 5,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.45,0.45],
        "goal_resampling_probability_per_sec" : 0.1,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "held_joints_damping" : 10.0,
        "held_joints_stiffness" : 500.0,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 0.2,
        "initial_height_randomization_range_meters" : 0.0,
        "initial_joint_pose_randomization_range" : 0.5,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 0.05, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "max_good_step_duration" : 0.3,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "min_good_step_duration" : 0.1,
        "mode" : mode,
        "obs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.02*n, 0.05*n],
        "obs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.01*n, 0.02*n],
        "obs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.02*n],
        "obs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0,    0.0],
        "observe_full_robot_state" : False,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomized_armature_ratios" : 0.1*r,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_frictionloss_ratios"             : 0.2*r,
        "randomized_gains_damping_ratio_epstd"       : 0.2*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.2*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_reference_filter_distribution" : ("uniform", (20.0-15*r, 20.0+15*r)),
        "record_video" : True,
        "recycle_pose_randomization" : True,
        "reward_acceleration_weight" :        eps,
        "reward_actacc_weight" :              eps,
        "reward_actdiff_weight" :             1.0,
        "reward_contacts_weight" :            eps,
        "reward_energy_weight" :              eps,
        "reward_failure_weight" :             eps,
        "reward_feet_air_time_weight" :       eps,
        "reward_feet_ground_time_weight" :    eps,
        "reward_feet_on_ground_weight" :      eps,
        "reward_heading_velocity_weight" :    1.0,
        "reward_heading_weight" :             0.1,
        "reward_health_weight" :              eps,
        "reward_height_position_weight" :     0.5,
        "reward_height_velocity_weight" :     eps,
        "reward_pitchnroll_velocity_weight" : 20.0,
        "reward_pitchnroll_weight" :          0.5,
        "reward_position_limit_weight" :      eps,
        "reward_position_weight" :            eps,
        "reward_posref_vel_weight" :          1.0*p,       
        "reward_sensed_effort_weight" :       eps,
        "reward_slip_weight" :                eps,
        "reward_stand_position_weight" :      1.0,
        "reward_torque_limit_weight" :        eps,
        "reward_torque_weight" :              1.0,
        "reward_torquediff_weight" :          eps,
        "reward_torqueref_weight" :           eps,
        "reward_tracking_weight" :            4.0,
        "reward_velocity_limit_weight" :      eps,
        "reward_velocity_weight" :            eps,
        "reward_velref_weight" :              eps,
        "robot_model" : "kyon",
        "safe_damping" : 5,
        "safe_stiffness" : 600,
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : True,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args.update({        
        "enable_rendering" : True,
        "verbose_infos" : True,
        "video_save_freq" : 1,
        "init_on_reset_ratio" : 1.0,
        # "initial_joint_pose_randomization_range" : 0.02,
        # "mass_randomization_ratio" : 0.0,
        # "friction_slide_spin_roll_randomization_ratios" : (0.0,0.0,0.0),
        "recycle_pose_randomization" : False,
        # "action_delay_mustd" : (0.0,0.0),
        # "action_noise_mustd" : (0.0,0.0),
        # "obs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.0, 0.0],
        # "obs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.0, 0.0],
        # "obs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0, 0.0],
        # "th_device" : th.device("cpu",0) # this segfaults
    })
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 10,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 10
    }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 10,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 10,
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

    gammas = {
        "acceleration" :        0.98,
        "actacc" :              0.0,
        "actdiff" :             0.0,
        "contacts" :            0.98,
        "failure" :             0.98,
        "feet_air_time" :       0.98,
        "feet_ground_time" :    0.98,
        "feet_on_ground" :      0.98,
        "heading" :             0.98,
        "heading_velocity" :    0.98,
        "health" :              0.98,
        "height_position" :     0.98,
        "height_velocity" :     0.98,
        "pitchnroll" :          0.98,
        "pitchnroll_velocity" : 0.98,
        "position" :            0.98,
        "position_limit" :      0.98,
        "posref_vel" :          0.0,
        "sensed_effort" :       0.98,
        "slip" :                0.98,
        "stand_position" :      0.98,
        "torque" :              0.98,
        "torque_limit" :        0.98,
        "torque_refs" :         0.0,
        "torquediff" :          0.9,
        "tracking" :            0.98,
        "velocity" :            0.9,
        "velocity_refs" :       0.0,
        "velocity_limit" :      0.98
    }

    # disabled_rewards = []
    # for k, v in env_builder_args.items():
    #     if k.startswith("reward_") and k.endswith("_weight") and v == 0.0:
    #         reward_name = k[len("reward_"):-len("_weight")]
    #         disabled_rewards.append(reward_name)
    # reward_enable_mask = {k:0 for k in disabled_rewards}


    eval_configurations = [  
                            eval_conf_video_det,
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
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams( model_th_device = "cuda",
                                                    q_network_arch=[1024,512],
                                                    q_lr=0.0005,
                                                    policy_lr=0.0005,
                                                    policy_arch=[512,256,256],
                                                    gamma=gammas,
                                                    target_tau = 0.005,
                                                    batch_size=4096,
                                                    buffer_size=(6*1024)*1_000, # 10_240_000 Should fit in 16Gb of VRAM
                                                    total_steps=400_000_000,
                                                    train_freq_vstep=5,
                                                    grad_steps=50,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -1.0,
                                                    actor_log_std_init = -2.0,
                                                    actor_observation_filter=["base.vec","base.last_action_raw"],
                                                    critic_observation_filter=["base.vec","base.last_action_raw","privileged.vec"],
                                                    # target_entropy_factor_annealing=("ramp",[100*1e6, 150*1e6, -1, -5]),
                                                    action_reference_obs_key="base.last_action_raw",
                                                    actor_weight_decay=0.0,
                                                    critic_weight_decay=0.0,
                                                    policy_update_freq=2,
                                                    deterministic_collection_ratio=0.01,
                                                    actor_mean_bounds_ratio = 0.8
                                                    ),
                    checkpoint_freq=20,
                    collector_device=env_device,
                    buffer_device=None,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=2,
                    log_weights_and_grads=False)                             
    elif algo.lower() == "sac_small":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = named_loco_venv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams(  model_th_device = "cuda",
                                                    q_network_arch=[256,128],
                                                    q_lr=0.001,
                                                    policy_lr=0.0003,
                                                    policy_arch=[128,128],
                                                    gamma=gammas,
                                                    target_tau = 0.005,
                                                    batch_size=512,
                                                    buffer_size=100_000,
                                                    total_steps=300_000_000,
                                                    train_freq_vstep=10,
                                                    grad_steps=40,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 50),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -0.5,
                                                    actor_log_std_init = -3.0,
                                                    actor_observation_filter=["base.vec"],
                                                    critic_observation_filter=["base.vec","privileged.vec"]
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
                agent_hyperparams=PPO_hyperparams(  minibatch_size=2048,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(512,256),
                                                    critic_network_arch=(512,256),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=3,
                                                    total_steps=train_envs*max_steps_per_episode*1000,
                                                    num_envs=train_envs,
                                                    num_steps=10,
                                                    gamma=0.99,
                                                    log_freq_vstep=int(max_steps_per_episode/10)),
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
    ap.add_argument("--algorithm", default="sac", type=str, help="Algorithm to use ('sac'/'ppo')")
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