#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams, TargetEntropyAnnealer
    from adarl_envs.experiments.loco_builder import named_loco_venv_builder
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
    elif mode == "mjx":
        env_device = th.device("cuda",0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")
    
    split_rewards = True
    degree2rad = math.pi/180
    eval_freq = 10
    r = 1.0 # randomization strength
    n = 1.0 # noise strength
    p = 0.25 # penalties strength
    eps = 0 #1e-6 # For disabled things (but no zero, so I can still see how they would behave)
    env_builder_args = {
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position",
        "desired_foot_clearance" : 0.1,
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_reference_filter" : True,
        "enable_rendering" : False,
        "extrinsics_only_privileged" : False,
        "fail_on_safety" : False,
        "frame_stack_length" : 3,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_resampling_probability_per_sec" : 0.1,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "goal_yaw_vel_minmax" : (-1.0, 1.0),
        "goal_yaw_vel_zero_ratio" : 0.25,
        "goal_height_minmax" : {"kyon" : [0.47,0.47],
                                "go1" : [0.30,0.30],
                                "centauro" : [0.79,0.79],
                                }.get(args["robot"].lower(), [0.5,0.5]),
        "held_joints_damping" :   {("centauro","twisting_pelvis_joint"):5.0,
                                   ("centauro","j_wheel_1"):50.0,
                                   ("centauro","j_wheel_2"):50.0,
                                   ("centauro","j_wheel_3"):50.0,
                                   ("centauro","j_wheel_4"):50.0,
                                    "default": 500.0},
        "held_joints_stiffness" : {("centauro","twisting_pelvis_joint"):2000.0,
                                   ("centauro","j_wheel_1"):0.0,
                                   ("centauro","j_wheel_2"):0.0,
                                   ("centauro","j_wheel_3"):0.0,
                                   ("centauro","j_wheel_4"):0.0,
                                    "default": 500.0},
        "history_length_action_raw" : 3,
        "history_length_action_smoothed" : 1,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 0.7,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 1.0, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "minimal_infos" : True,
        "mode" : mode,
        "no_infos" : False,
        "noise_abs_obs_angvel_ep_mustd_step_std" :      [0.0, 0.02*n,  0.1*n],
        "noise_abs_obs_gravity_ep_mustd_step_std" :     [0.0, 0.0*n,   0.1*n],
        "noise_abs_obs_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.1*n],
        "noise_abs_obs_linacc_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_linvel_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_posz_ep_mustd_step_std" :        [0.0, 0.0,     0.0],
        "noise_action_delay_mustd_std" : (0.008, 0.01*n, 0.0025*n),
        "noise_action_mustd" : (0.0,   0.0),
        "observe_actor_safety_state" : True,
        "observe_full_robot_state" : False,
        "offset_envs_ep_starts" : algo=="ppo",
        "pitchnroll_reward_settle_point" : 15*degree2rad, # ~zero reward after this angle
        "playground_style_reward" : False,
        "posref_err_history_length" : 5,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomization_initial_height_range_meters" : 0.1,
        "randomization_initial_joint_pose_range" : 0.1,
        "randomization_recycle_init_pose" : True,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_dof_armature_ratios" :      ("uniform", [0.1*r,10.0*r]),
        "randomized_dof_damping_ratios":        ("uniform", [0.1*r,10.0*r]),
        "randomized_dof_frictionloss_ratios":   ("uniform", [0.1*r,10.0*r]),
        "randomized_friction_slide_spin_roll_ratios" : ("uniform", ([0.8*r,0.8*r,0.8*r],[1.5*r,1.5*r,1.5*r])),
        "randomized_gains_damping_ratio_epstd"       : 0.2*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.2*r,
        "randomized_mass_ratios" : ("normal", (1.0, 0.1*r)),
        "randomization_recycle_model_alterations" : mode=="genesis", # Genesis has an issue with changing the armature, it's super slow
        "randomized_reference_filter_distribution" : ("uniform", (20.0, 50.0)),
        "record_video" : True,
        "reward_contacts_weight" :                  eps,
        "reward_failure_weight" :                   eps,
        "reward_feet_air_time_weight" :             10.0,
        "reward_feet_ground_time_weight" :          eps,
        "reward_feet_on_ground_weight" :            1.0,
        "reward_feet_step_height_weight" :          1.0,
        "reward_heading_velocity_weight" :          eps,
        "reward_heading_weight" :                   eps,
        "reward_health_weight" :                    eps,
        "reward_height_position_weight" :           0.5,
        "reward_height_velocity_weight" :           0.1,
        "reward_joint_acc_on_vel_weight" :          eps,
        "reward_joint_acceleration_weight" :        eps,
        "reward_joint_actacc_weight" :              0.0*p,
        "reward_joint_actdiff_weight" :             0.0*p,
        "reward_joint_energy_weight" :              eps,
        "reward_joint_position_limit_weight" :      eps,
        "reward_joint_position_weight" :            0.0*p,
        "reward_joint_posref_acc_weight":           eps,
        "reward_joint_posref_vel_weight" :          eps,
        "reward_joint_power_weight" :               0.0*p,
        "reward_joint_sensed_effort_weight" :       eps,
        "reward_joint_stand_position_weight" :      5.0,
        "reward_joint_stand_velocity_weight" :      1.0,
        "reward_joint_torque_limit_weight" :        eps,
        "reward_joint_torque_weight" :              0.0*p,
        "reward_joint_torquediff_weight" :          eps,
        "reward_joint_torqueref_weight" :           eps,
        "reward_joint_velocity_limit_weight" :      eps,
        "reward_joint_velocity_weight" :            eps,
        "reward_joint_velref_weight" :              eps,
        "reward_pitchnroll_velocity_weight" :       0.5,
        "reward_pitchnroll_weight" :                0.5,
        "reward_safety_triggered_weight" :          0.1,
        "reward_scale_nolength":                    0.01,
        "reward_slip_weight" :                      1.0,
        "reward_superweight_joint_penalties" :      1.0,
        "reward_tracking_weight" :                  1.0,
        "reward_yaw_vel_tracking_weight" :          1.0,
        "robot_model" : args["robot"],
        "robot_options" : {
            "add_twisting_pelvis" : False
        },
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : split_rewards,
        "step_max_good_air_duration" : 0.5,
        "step_max_good_ground_duration" : 0.5,
        "step_min_good_air_duration" : 0.3,
        "step_min_good_ground_duration" : 0.1,
        "stepLength_sec" : step_length_sec,
        "terminate_on_body_contact" : False,
        "terminate_on_crash" : True,
        "terminate_on_safety" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0,
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args.update({
        "enable_rendering" : True,
        "verbose_infos" : True,
        "minimal_infos" : False,
        "video_save_freq" : 1,
        "init_on_reset_ratio" : 1.0,
        "offset_envs_ep_starts" : False,
        # "randomization_initial_joint_pose_range" : 0.02,
        # "mass_randomization_ratio" : 0.0,
        # "friction_slide_spin_roll_randomization_ratios" : (0.0,0.0,0.0),
        "randomization_recycle_init_pose" : False,
        # "action_delay_mustd" : (0.0,0.0),
        # "noise_action_mustd" : (0.0,0.0),
        # "obs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.0, 0.0],
        # "obs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.0, 0.0],
        # "obs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0, 0.0],
        # "obs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0, 0.0],
        # "th_device" : th.device("cpu",0) # this segfaults
        # "reward_superweight_joint_penalties" : 1.0
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
    det_video_eval_env_builder_args = copy.deepcopy(video_eval_env_builder_args)
    det_video_eval_env_builder_args.update({
        "reward_superweight_joint_penalties" : 1.0
    })
    eval_conf_video_det = {
        "name" : "video_stoch_det",
        "deterministic" : True,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 100,
        "env_builder_args" : det_video_eval_env_builder_args,
        "num_envs" : 100,
        "skip_first_eval": True
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

    # gamma_long =    0.99
    # gamma_short =   0.90
    # gamma_instant = 0.0
    # gammas = {
    #     "contacts" :                  gamma_long,
    #     "failure" :                   gamma_long,
    #     "feet_air_time" :             gamma_long,
    #     "feet_ground_time" :          gamma_long,
    #     "feet_on_ground" :            gamma_long,
    #     "feet_step_height" :          gamma_long,
    #     "heading_velocity" :          gamma_long,
    #     "heading" :                   gamma_long,
    #     "health" :                    gamma_long,
    #     "height_position" :           gamma_long,
    #     "height_velocity" :           gamma_long,
    #     "joint_acc_on_vel" :          gamma_long,
    #     "joint_acceleration" :        gamma_long,
    #     "joint_actacc" :              gamma_instant,
    #     "joint_actdiff" :             gamma_instant,
    #     "joint_energy" :              gamma_long,
    #     "joint_position_limit" :      gamma_long,
    #     "joint_position" :            gamma_long,
    #     "joint_posref_acc":           gamma_instant,
    #     "joint_posref_vel" :          gamma_instant,
    #     "joint_power" :               gamma_short,
    #     "joint_sensed_effort" :       gamma_long,
    #     "joint_stand_position" :      gamma_long,
    #     "joint_stand_velocity" :      gamma_long,
    #     "joint_torque_limit" :        gamma_long,
    #     "joint_torque" :              gamma_long,
    #     "joint_torquediff" :          gamma_short,
    #     "joint_torqueref" :           gamma_instant,
    #     "joint_velocity_limit" :      gamma_long,
    #     "joint_velocity" :            gamma_short,
    #     "joint_velref" :              gamma_instant,
    #     "pitchnroll_velocity" :       gamma_long,
    #     "pitchnroll" :                gamma_long,
    #     "safety_triggered" :          gamma_long,
    #     "scale_n":                    gamma_long,
    #     "slip" :                      gamma_long,
    #     "tracking" :                  gamma_long,
    #     "yaw_vel_tracking" :          gamma_long,
    # }
    gammas = 0.99

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
    
    annealer = TargetEntropyAnnealer(reference_key="linvel_avg",
                                     start_target=-1.0,
                                     end_target=-3.0,
                                     start_reference_threshold=0.4,
                                     end_reference_threshold=0.0)

    if algo.lower() == "sac":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = named_loco_venv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams( model_th_device = "cuda",
                                                    q_network_arch=[512,256],
                                                    q_lr=0.0003,
                                                    policy_lr=0.0003,
                                                    policy_arch=[512,256],
                                                    gamma=gammas,
                                                    target_tau = 0.005,
                                                    batch_size=24576,
                                                    buffer_size=(8*1024)*1_000, # 10_240_000 Should fit in 16Gb of VRAM
                                                    total_steps=400_000_000,
                                                    train_freq_vstep=5,
                                                    grad_steps=50,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -1.0,
                                                    actor_log_std_init = -1.0,
                                                    actor_observation_filter=["base.vec"],
                                                    critic_observation_filter=["privileged.vec"],
                                                    target_entropy_factor_annealing=annealer.anneal,
                                                    action_reference_obs_key=None, #"base.last_action_raw",
                                                    actor_weight_decay=0.0,
                                                    critic_weight_decay=0.0,
                                                    policy_update_freq=2,
                                                    deterministic_collection_ratio=0.00,
                                                    actor_mean_bounds_ratio = 0.95,
                                                    alpha_lr_factor = 1.0,
                                                    alpha_initial_value = 0.0001,
                                                    independent_entropy_q=split_rewards
                                                    ),
                    checkpoint_freq=20,
                    collector_device=env_device,
                    buffer_device=None,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size=0,
                    validation_batch_size=0,
                    validation_holdout_ratio=0,
                    no_wandb=args["no_wandb"],
                    debug_level=0,
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
        from rreal.algorithms.ppo2 import ppo_train, PPO_init_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=named_loco_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_init_hyperparams(  minibatch_size=2048,
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
                checkpoint_freq_vec_ep=-1,
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
    ap.add_argument("--robot", default="centauro", type=str, help="Which robot to use")
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