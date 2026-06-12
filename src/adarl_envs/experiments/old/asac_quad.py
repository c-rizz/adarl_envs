#!/usr/bin/env python3  
from __future__ import annotations
from adarl_envs.experiments.loco_builder import named_loco_venv_builder

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams
    from autoencoding_rl.experiments.asac3.asac3_train import asac3_train
    import os
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 80
    env_device = th.device("cpu")
    env_builder_args = {
        "action_delay_mustd" : (0.0,0.01),
        "noise_action_mustd" : (0.0,0.001),
        "action_smoothing_halflife_sec" : 0.2,
        "control_mode" : "position",
        "robot_model" : args["robot"],
        "enable_rendering" : False,
        "goal_err_smoothing_halflife_sec" : 0.2,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : "mjx",
        "quiet" : False,
        "reward_joint_acceleration_weight" :      2.0,
        "reward_joint_actdiff_weight" :           0.6,
        "reward_joint_actacc_weight" :            0.1,
        "reward_contacts_weight" :          0.0,
        "reward_joint_energy_weight" :            0.0,
        "reward_health_weight" :            0.0,
        "reward_joint_position_limit_weight" :    0.1,
        "reward_joint_torque_limit_weight" :      0.0,
        "reward_joint_torque_weight" :            5.0,
        "reward_joint_torquediff_weight" :        0.0,
        "reward_tracking_weight" :          1.0,
        "reward_joint_velocity_limit_weight" :    0.0,
        "reward_joint_velocity_weight" :          1.0,
        "reward_height_weight" :            0.15,
        "reward_pitchnroll_weight" :        0.15,
        "reward_joint_position_weight" :          5.0,
        "reward_feet_air_time_weight" :     20.0,
        "reward_heading_weight" :           0.05,
        "safe_stiffness" : 200,
        "safe_damping" : 5,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "th_device" : env_device,
        "video_save_freq" : 0,
        "goal_speed_minmax" : (0,1),
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
        "initial_pose_randomization_range" : 0.8,
        "mass_randomization_ratio" : 0.3,
        "friction_slide_spin_roll_randomization_ratios" : (0.3,0.3,0.3),
        "impulse_probability_per_sec" : 0.5,
        "impulse_duration_minmax" : (0.01, 2.5),
        "impulse_mean_std" : (100.0,50.0)
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
    video_eval_env_builder_args["verbose_infos"] = True
    video_eval_env_builder_args["video_save_freq"] = 1
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    video_norand_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_norand_eval_env_builder_args["enable_rendering"] = True
    video_norand_eval_env_builder_args["verbose_infos"] = True
    video_norand_eval_env_builder_args["video_save_freq"] = 1
    video_norand_eval_env_builder_args["initial_pose_randomization_range"] = 0.0
    eval_conf_video_norand_det = {
        "name" : "video_norand_stoch",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_norand_eval_env_builder_args,
        "num_envs" : 1
    }
    run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    run_1ms_env_builder_args["enable_rendering"] = True
    run_1ms_env_builder_args["verbose_infos"] = True
    run_1ms_env_builder_args["video_save_freq"] = 1
    run_1ms_env_builder_args["initial_pose_randomization_range"] = 0.0
    eval_conf_run_1ms = {
        "name" : "run_1ms",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : run_1ms_env_builder_args,
        "num_envs" : 1
    }
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
    
    asac3_train(seed=seed,
                train_steps=50_000_000,
                run_folder=folderName,
                gpuid = None, # seed%th.cuda.device_count(),
                pretrained_model_file=resumeModelFile,
                vec_env_builder=named_loco_venv_builder,
                env_builder_args = env_builder_args,
                eval_configurations=[eval_conf_video_stoch,eval_conf_video_det,eval_conf_run_1ms,eval_conf_video_norand_det],
                eval_le_freq_ep=10,
                buffer_storage_torch_device="cpu",
                run_id = run_id,
                debug_level=-10,
                check_infs_nans=False,
                
                automatic_grad_steps=False,
                le_always_deterministic=False,
                le_arch_img_encoder_backbone = None,
                le_arch_img_decoder_backbone = None,
                le_arch_dyn_ensemble_size = 1,
                le_arch_dynamics = [512,512],
                le_arch_enc_ensemble_size = 1,
                le_arch_frame_stack_size=1,
                le_arch_img_dec_ensemble_size = 1,
                le_arch_img_dec_learn_background = False,
                le_arch_img_encoding_size = 0,            
                le_arch_latent_space_size = 192,
                le_arch_optimizer="adam",
                le_arch_reward = [128],
                le_arch_reward_ensemble_size=1,
                le_arch_state_combiner = "identity",
                le_arch_state_decombiner = "identity",
                le_arch_type = "dvae4_3",
                le_arch_use_coord_conv=True,
                le_arch_vec_decoder=[256,256],
                le_arch_vec_decoder_activation=("scaledtanh", 10),
                le_arch_vec_decoder_ensemble_size=1,
                le_arch_vec_encoder=[256,256],
                le_arch_vec_encoder_ensemble_size=1,
                le_arch_vec_encoding_size=192,
                le_batch_size = 128,
                le_bestModelThreshold = -1,
                le_consistency_target = None,
                le_decoder_weight_decay_lambda=None,
                le_validation_set_disable = False,
                le_direct_vec_into_latent = False,
                le_dont_train_prediction_decoder=False,
                le_dont_train_prediction_encoder=False,
                le_dont_train_reconstruction_encoder=False,
                le_fake_encoding_noise_std=0.0,
                le_freeze_decoders=False,
                le_freeze_dynamics=False,
                le_freeze_encoders=False,
                le_grad_steps = max_steps_per_episode*2,
                le_internal_enc_dropout = 0.0,
                le_latent_space_activation="identity",
                le_learning_starts = max_steps_per_episode*train_envs*3,
                le_loss_consistency_weight = 0.0,
                le_loss_img_weight = 1.0,
                le_loss_kld_weight = 0.0001,
                le_loss_latent_prediction_weight = 0.0,
                le_loss_obs_prediction_weight = 1.0,
                le_loss_obs_prediction_discount = 1.0,
                le_loss_reward_prediction_discount = 1.0,
                le_loss_latent_prediction_discount = 1.0,
                le_loss_reconstruction_weight=0.2,
                le_loss_reward_weight = 0.1,
                le_loss_use_log=False,
                le_loss_vec_weight = 0.2,
                le_lr = 0.001,
                le_no_resampling_on_dynamics=False,
                le_policy_input_resampling=True,
                le_pretrain_grad_steps = 10000,
                le_reset_decoder=False,
                le_reset_dynamics=False,
                le_reset_encoder=False,
                le_reset_obs_stats=False,
                le_residual_dynamics=True,
                le_retrain_period = -1,
                le_reward_scaling = 10.0,
                le_reset_on_retrain = False,
                le_tau=1.0,
                le_train_period_vstep=max_steps_per_episode,
                le_train_trajectories_length = 5,
                le_use_log_reward = False,
                le_validation_batch_size = 128,
                le_validation_ratio=0.05,
                le_validation_buffer_size=max_steps_per_episode*train_envs*100,
                le_weight_decay = 1e-5,
                parallelize_experience_collection=False,
                random_vsteps = max_steps_per_episode*2,
                buffer_size = max_steps_per_episode*train_envs*500,

                sac_arch_actor_ensemble_size=3,
                sac_arch_critic = [256,64],
                sac_arch_policy = [1024,128],
                sac_batch_size = 16384,
                sac_ent_coef = "auto",
                sac_gamma = 0.99,
                sac_grad_steps = 20,
                sac_learning_starts = max_steps_per_episode*train_envs*4,
                sac_lr_actor = 0.0005,
                sac_lr_critic = 0.005,
                sac_target_entropy="auto",
                sac_tau=0.005,
                sac_train_period_vstep=5,
                collection_device=env_device,
                comment = args["comment"],
                checkpointing_freq_ep=None,
                debug_level=-10,
                buffer_reward_exploration_augmentation = False,
                buffer_reward_expl_augmentation_weight = 0.0,
                asac_use_rnd_novelty_exploration_bonus = False,
                asac_exploration_bonus_weight = 0.0,
                asac_rnd_nn_arch=[128,128],
                asac_rnd_novelty_estimator_learning_rate = 0.00001,
                parallel_envs=train_envs,
                asac_running_avgs_alpha = 0.999,
                asac_running_avg_rew_alpha = 0.9999,
                max_episode_duration=max_steps_per_episode,
                min_episode_duration=max_steps_per_episode,
                allow_tf32=True,
                purely_numpy_env = False
                )



if __name__ == "__main__":

    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","adarl_envs","rreal"])