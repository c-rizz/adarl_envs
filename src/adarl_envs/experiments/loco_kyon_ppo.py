#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams, TargetEntropyAnnealer
    from adarl_envs.experiments.loco_builder import named_loco_venv_builder
    from adarl_envs.env.LocomotionVecEnv import TransitionAugmentor
    import math
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=500 #int(ep_duration_sec/step_length_sec)

    algo = args["algorithm"]                                
    if algo == "sac" or algo == "ppo":
        train_envs = 4096
    elif algo == "sac_small" or algo == "ppo_small":
        train_envs = 8
    else:
        raise RuntimeError(f"Unexpected algo '{algo}'")
    
    if mode == "pybullet":
        env_device = th.device("cpu",0)
    elif mode == "mjx":
        env_device = th.device("cuda",0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")
    
    eval_freq = 20
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
        "frame_stack_length" : 3,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.47,0.47],
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
        "merge_privileged" : True,
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
        "randomized_reference_filter_distribution" : ("uniform", (20.0, 20.0)),
        "record_video" : True,
        "recycle_pose_randomization" : True,
        "reward_superweight_joint_penalties" : 1.0,
        "reward_acceleration_weight" :        eps,
        "reward_actacc_weight" :              1.0,
        "reward_actdiff_weight" :             1.0,
        "reward_contacts_weight" :            eps,
        "reward_energy_weight" :              eps,
        "reward_failure_weight" :             eps,
        "reward_feet_air_time_weight" :       10.0,
        "reward_feet_ground_time_weight" :    eps,
        "reward_feet_on_ground_weight" :      eps,
        "reward_heading_velocity_weight" :    eps,
        "reward_heading_weight" :             eps,
        "reward_health_weight" :              eps,
        "reward_height_position_weight" :     1.0,
        "reward_height_velocity_weight" :     eps,
        "reward_pitchnroll_velocity_weight" : 1.0,
        "reward_pitchnroll_weight" :          1.0,
        "reward_position_limit_weight" :      1.0,
        "reward_position_weight" :            eps,
        "reward_posref_vel_weight" :          0.0,
        "reward_posref_acc_weight":           0.0,
        "reward_scale_nolength":              0.1,
        "reward_sensed_effort_weight" :       eps,
        "reward_slip_weight" :                eps,
        "reward_stand_position_weight" :      1.0,
        "reward_torque_limit_weight" :        eps,
        "reward_torque_weight" :              0.5,
        "reward_torquediff_weight" :          eps,
        "reward_torqueref_weight" :           eps,
        "reward_tracking_weight" :            3.0,
        "reward_velocity_limit_weight" :      eps,
        "reward_velocity_weight" :            eps,
        "reward_velref_weight" :              eps,
        "robot_model" : "kyon",
        "safe_damping" : 5,
        "safe_stiffness" : 600,
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : False,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0,
        "minimal_infos" : False
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
        # "reward_superweight_joint_penalties" : 1.0
    })
    highpen_video_eval_env_builder_args = copy.deepcopy(video_eval_env_builder_args)
    highpen_video_eval_env_builder_args.update({
        "reward_superweight_joint_penalties" : 0.5
    })
    # video_stoch_highpen = {
    #     "name" : "video_stoch_highpen",
    #     "deterministic" : False,
    #     "eval_freq_ep" : eval_freq*train_envs,
    #     "eval_eps" : 10,
    #     "env_builder_args" : highpen_video_eval_env_builder_args,
    #     "num_envs" : 10,
    #     "skip_first_eval": True
    # }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 10,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 10,
        "skip_first_eval": False
    }
    video_det_eval_env_builder_args = copy.deepcopy(video_eval_env_builder_args)
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 10,
        "env_builder_args" : video_det_eval_env_builder_args,
        "num_envs" : 10,
        "skip_first_eval": False
    }
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

    if env_builder_args["split_rewards"]:
        gamma_long = 0.99
        gamma_short = 0.0
        gammas = {
            "acceleration" :        gamma_long,
            "actacc" :              gamma_short,
            "actdiff" :             gamma_short,
            "contacts" :            gamma_long,
            "failure" :             gamma_long,
            "feet_air_time" :       gamma_long,
            "feet_ground_time" :    gamma_long,
            "feet_on_ground" :      gamma_long,
            "heading" :             gamma_long,
            "heading_velocity" :    gamma_long,
            "health" :              gamma_long,
            "height_position" :     gamma_long,
            "height_velocity" :     gamma_long,
            "pitchnroll" :          gamma_long,
            "pitchnroll_velocity" : gamma_long,
            "position" :            gamma_long,
            "position_limit" :      gamma_long,
            "posref_vel" :          gamma_short,
            "posref_acc" :          gamma_short,
            "sensed_effort" :       gamma_long,
            "slip" :                gamma_long,
            "stand_position" :      gamma_long,
            "torque" :              gamma_long,
            "torque_limit" :        gamma_long,
            "torque_refs" :         gamma_short,
            "torquediff" :          gamma_short,
            "tracking" :            gamma_long,
            "velocity" :            gamma_long,
            "velocity_refs" :       gamma_short,
            "velocity_limit" :      gamma_long
        }
        def transition_augmentor_builder(observation_space, action_space, reward_space):
            return TransitionAugmentor(reward_space=reward_space,
                                    reward_weights_distr=("uniform", (0.01, 1.0)),
                                    augmented_samples=4096).augment_transition
    else:
        gammas = 0.98
        transition_augmentor_builder = None

    eval_configurations = [  
                            # video_stoch_highpen,
                            # eval_conf_video_stoch,
                            # eval_conf_run_1ms,
                            eval_conf_video_det,
                            # #  eval_conf_feasible,
                            # #  eval_conf_video_feasible,
                            # #  eval_conf_video_jump_feasible
                        ]
    
    annealer = TargetEntropyAnnealer(reference_key="linvel_avg",
                                     start_target=-2.0,
                                     end_target=-5.0,
                                     start_reference_threshold=0.5)
    

    if algo.lower() == "sac":
        raise NotImplementedError()
        # sac_train(  seed,
        #             folderName,
        #             run_id,
        #             args,
        #             vec_env_builder = named_loco_venv_builder,
        #             env_builder_args = env_builder_args,
        #             eval_configurations = eval_configurations,
        #             hyperparams = SAC_init_hparams( model_th_device = "cuda",
        #                                             q_network_arch=[1024,512],
        #                                             q_lr=0.0005,
        #                                             policy_lr=0.0005,
        #                                             policy_arch=[512,256],
        #                                             gamma=gammas,
        #                                             target_tau = 0.001,
        #                                             batch_size=4096,
        #                                             buffer_size=(5*1024)*1_000, # 10_240_000 Should fit in 16Gb of VRAM
        #                                             total_steps=5_000_000_000,
        #                                             train_freq_vstep=5,
        #                                             grad_steps=50,
        #                                             learning_starts=max_steps_per_episode*max(train_envs*1, 100),
        #                                             parallel_envs=train_envs,
        #                                             log_freq_vstep=max_steps_per_episode,
        #                                             reference_init_args =   {   "env_builder_args" : env_builder_args,
        #                                                                         "eval_configuration" : eval_configurations},
        #                                             target_entropy_factor = None,
        #                                             actor_log_std_init = -2.0,
        #                                             actor_observation_filter=["base.vec","base.last_action_raw", "base.reward_weights"],
        #                                             critic_observation_filter=["base.vec","base.last_action_raw","privileged.vec", "base.reward_weights"],
        #                                             target_entropy_factor_annealing=annealer.anneal,
        #                                             action_reference_obs_key="base.last_action_raw",
        #                                             actor_weight_decay=1e-5,
        #                                             critic_weight_decay=0.0,
        #                                             policy_update_freq=2,
        #                                             deterministic_collection_ratio=0.00,
        #                                             actor_mean_bounds_ratio = 0.8,
        #                                             alpha_lr_factor = 1.0,
        #                                             independent_entropy_q=True
        #                                             ),
        #             checkpoint_freq=20,
        #             collector_device=env_device,
        #             buffer_device=None,
        #             max_episode_duration=max_steps_per_episode,
        #             validation_buffer_size=0,
        #             validation_batch_size=0,
        #             validation_holdout_ratio=0,
        #             no_wandb=args["no_wandb"],
        #             debug_level=2,
        #             log_weights_and_grads=False,
        #             transition_augmentor_builder=transition_augmentor_builder)
    elif algo.lower() == "sac_small":
        raise NotImplementedError()
        # sac_train(  seed,
        #             folderName,
        #             run_id,
        #             args,
        #             vec_env_builder = named_loco_venv_builder,
        #             env_builder_args = env_builder_args,
        #             eval_configurations = eval_configurations,
        #             hyperparams = SAC_init_hparams( model_th_device = "cuda",
        #                                             q_network_arch=[256,128],
        #                                             q_lr=0.001,
        #                                             policy_lr=0.0003,
        #                                             policy_arch=[128,128],
        #                                             gamma=gammas,
        #                                             target_tau = 0.005,
        #                                             batch_size=512,
        #                                             buffer_size=100_000,
        #                                             total_steps=300_000_000,
        #                                             train_freq_vstep=10,
        #                                             grad_steps=40,
        #                                             learning_starts=max_steps_per_episode*max(train_envs*1, 50),
        #                                             parallel_envs=train_envs,
        #                                             log_freq_vstep=max_steps_per_episode,
        #                                             reference_init_args =   {   "env_builder_args" : env_builder_args,
        #                                                                         "eval_configuration" : eval_configurations},
        #                                             target_entropy_factor = -0.5,
        #                                             actor_log_std_init = -3.0,
        #                                             actor_observation_filter=["base.vec"],
        #                                             critic_observation_filter=["base.vec","privileged.vec"]
        #                                             ),
        #             checkpoint_freq=5,
        #             collector_device=env_device,
        #             max_episode_duration=max_steps_per_episode,
        #             validation_buffer_size=0,
        #             validation_batch_size=0,
        #             validation_holdout_ratio=0,
        #             no_wandb=args["no_wandb"],
        #             debug_level=2)                           
    elif algo.lower() == "ppo" or algo.lower() == "ppo_small":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=named_loco_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_hyperparams(  minibatch_size=None,
                                                    minibatch_num=4,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(512,256),
                                                    critic_network_arch=(512,256),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=4,
                                                    total_steps=200_000_000,
                                                    num_envs=train_envs,
                                                    num_steps=24,
                                                    gamma=0.99,
                                                    loss_value_weight=1.0,
                                                    loss_entropy_coeff=1e-4,
                                                    log_freq_vstep=int(max_steps_per_episode/10),
                                                    epsilon_policy_ratio_clip=0.2,
                                                    epsilon_value_clip_epsilon=0.2,
                                                    gae_lambda=0.95,
                                                    max_grad_norm=1.0,
                                                    init_actor_logstd=-2.0,
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
