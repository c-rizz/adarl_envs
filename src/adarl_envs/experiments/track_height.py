#!/usr/bin/env python3  
from __future__ import annotations

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams
    from adarl_envs.experiments.loco_builder import named_loco_venv_builder
    
    mode = args["mode"].lower()
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=500 #int(ep_duration_sec/step_length_sec)

    algo = args["algorithm"]                                
    if algo == "sac" or algo == "ppo":
        train_envs = 512
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
    
    eval_freq = 20
    env_builder_args = {
        "noise_action_delay_mustd_std" : (0.005, 0.002, 0.0025),
        "noise_action_mustd" : (0.0,   0.0005),
        "action_smoothing_halflife_sec" : 0.05,
        "control_mode" : "position",
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "fail_on_safety" : False,
        "frame_stack_length" : 5,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.3,0.57],
        "goal_resampling_probability_per_sec" : 0.3,
        "goal_speed_minmax" : [0,0.0],
        "goal_yaw_minmax" : [0.0,0.0],
        "held_joints_damping" : 10.0,
        "held_joints_stiffness" : 500.0,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [10.0,50.0],
        "impulse_probability_per_sec" : 0.1,
        "init_on_reset_ratio" : 0.2,
        "randomization_initial_joint_pose_range" : 0.8,
        "initial_pose_randomization_range" : 0.0,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 0.1, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "step_max_good_air_duration" : 1.5,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "step_min_good_air_duration" : 0.2,
        "mode" : mode,
        "obs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.02, 0.02],
        "obs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.02, 0.02],
        "obs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.0, 0.02],
        "obs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0, 0.02],
        "obs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0, 0.02],
        "obs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0, 0.02],
        "observe_full_robot_state" : False,
        "posref_safety_period" : 0.005,
        "quiet" : False,
        "randomized_dof_armature_ratios" : 0.2,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.15,0.05,0.03])),
        "randomized_friction_slide_spin_roll_ratios" : [0.3,0.3,0.3],
        "randomized_dof_frictionloss_ratios" : 0.2,
        "randomized_gains_damping_ratio_epstd" : 0.3,
        "randomized_gains_stiffness_ratio_epstd" : 0.3,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1)),
        "randomized_reference_filter_distribution" : ("uniform", (5.0, 40.0)),
        "record_video" : True,
        "randomization_recycle_init_pose" : True,
        "reward_joint_acceleration_weight" :      0.0,
        "reward_joint_actacc_weight" :            1000.0,
        "reward_joint_actdiff_weight" :           20.0,
        "reward_contacts_weight" :          0.0,
        "reward_joint_energy_weight" :            0.0,
        "reward_failure_weight" :           1.0,
        "reward_feet_air_time_weight" :     0.0,
        "reward_feet_on_ground_weight" :    1.0,
        "reward_heading_weight" :           0.5,
        "reward_health_weight" :            0.0,
        "reward_height_weight" :            4.0,
        "reward_pitchnroll_weight" :        1.75,
        "reward_pos2posref_weight" :        0.0,       
        "reward_joint_position_limit_weight" :    0.1,
        "reward_joint_position_weight" :          0.25,
        "reward_slip_weight" :              2.0,
        "reward_joint_torque_limit_weight" :      0.0,
        "reward_joint_torque_weight" :            2.0,
        "reward_joint_torquediff_weight" :        0.0,
        "reward_joint_torqueref_weight" :         0.0,
        "reward_tracking_weight" :          0.0,
        "reward_joint_velocity_limit_weight" :    0.0,
        "reward_joint_velocity_weight" :          0.0,
        "reward_joint_velref_weight" :            0.0,
        "robot_model" : args["robot"],
        "safe_damping" : 5,
        "safe_stiffness" : 400,
        "saturate_jimp_ref_limits" : False,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args.update({        
        "enable_rendering" : True,
        "verbose_infos" : True,
        "video_save_freq" : 1,
        "init_on_reset_ratio" : 1.0,
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
                                                    q_network_arch=[2048,512,256],
                                                    q_lr=0.0005,
                                                    policy_lr=0.0001,
                                                    policy_arch=[512,256],
                                                    gamma=0.99,
                                                    target_tau = 0.001,
                                                    batch_size=4096,
                                                    buffer_size=5_000_000,
                                                    total_steps=1_000_000_000,
                                                    train_freq_vstep=5,
                                                    grad_steps=20,
                                                    learning_starts=max_steps_per_episode*max(train_envs*1, 100),
                                                    parallel_envs=train_envs,
                                                    log_freq_vstep=max_steps_per_episode,
                                                    reference_init_args =   {   "env_builder_args" : env_builder_args,
                                                                                "eval_configuration" : eval_configurations},
                                                    target_entropy_factor = -2.0,
                                                    actor_log_std_init = -3.0,
                                                    actor_observation_filter=["base.vec","base.last_action_raw"],
                                                    critic_observation_filter=["base.vec","base.last_action_raw","privileged.vec"],
                                                    # target_entropy_factor_annealing=("ramp",[150*1e3, 300*1e3, -1, -5]),
                                                    action_reference_obs_key="base.last_action_raw",
                                                    actor_weight_decay=0.0001,
                                                    critic_weight_decay=0.0,
                                                    policy_update_freq=2,
                                                    deterministic_collection_ratio=0.01,
                                                    actor_mean_bounds_ratio = 0.9
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
