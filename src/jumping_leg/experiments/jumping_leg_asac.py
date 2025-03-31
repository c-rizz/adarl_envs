#!/usr/bin/env python3  
from __future__ import annotations


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams
    from autoencoding_rl.experiments.asac3.asac3_train import asac3_train
    import os
    from jumping_leg.experiments.build_jumping_leg_env import env_builder
    
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    ep_duration_sec = 5
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 100
    env_device = th.device("cpu")
    env_builder_args = {
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 0.0,
        "reward_velocity_limit_weight" : 0.0,
        "reward_torque_limit_weight" : 0.0,
        "reward_torque_weight" : 0.0,
        "reward_torquediff_weight" : 0.0,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.0,
        "reward_acceleration_weight" : 0.0,
        "th_device" : env_device,
        "control_mode" : "impedance",
        "video_save_freq" : 0,
        "stepLength_sec" : step_length_sec,
        "platform_randomization" : "single",
        "quiet" : False,
        "mode" : "pybullet",
        "use_contacts" : False,
        "ep_obs_noise_mustd" : (0.0, 0.001),
        "step_obs_noise_std" : 0.001,
        "stop_on_safety" : False,
        "action_delay_mustd" : (0.01,0.01),
        "max_steps_per_episode" : max_steps_per_episode,
        "obs_only_vec":True,
        "action_smoothing_halflife_sec" : 0.01,
        "leg_min_height" : 0.4,
        "leg_max_height" : 0.65,
        "leg_max_jump" : 0.6,
        "leg_min_jump" : -0.1,
        "goal_dist_smoothing_halflife_sec" : 0.0,
        "enable_rendering" : False,
        "randomize_initial_pose" : False}
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
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
    feasible_env_builder_args = copy.deepcopy(env_builder_args)
    feasible_env_builder_args["leg_max_jump"] = 0.2
    feasible_env_builder_args["ep_obs_noise_mustd"] = (0.0, 0.0)
    feasible_env_builder_args["step_obs_noise_std"] = 0.0
    feasible_env_builder_args["randomize_initial_pose"] = False
    feasible_env_builder_args["platform_randomization"] = "single"
    # feasible_env_builder_args["enable_rendering"] = True
    # feasible_env_builder_args["video_save_freq"] = 1
    eval_conf_feasible = {
        "name" : "feasible",
        "deterministic" : True,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 32,
        "env_builder_args" : feasible_env_builder_args,
        "num_envs" : 16
    }
    video_feasible_env_builder_args = copy.deepcopy(feasible_env_builder_args)
    video_feasible_env_builder_args["enable_rendering"] = True
    video_feasible_env_builder_args["video_save_freq"] = 1
    video_feasible_env_builder_args["randomize_initial_pose"] = False
    eval_conf_video_feasible = {
        "name" : "video_feasible",
        "deterministic" : True,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_feasible_env_builder_args,
        "num_envs" : 1
    }
    video_feasible_jump_env_builder_args = copy.deepcopy(video_feasible_env_builder_args)
    video_feasible_jump_env_builder_args["leg_min_jump"] = 0.2
    eval_conf_video_jump_feasible = {
        "name" : "video_jump_feasible",
        "deterministic" : True,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_feasible_jump_env_builder_args,
        "num_envs" : 1
    }

    eval_confs = [eval_conf_video_stoch,eval_conf_video_det,eval_conf_feasible, eval_conf_video_feasible, eval_conf_video_jump_feasible]

    asac3_train(seed=seed,
                train_steps=50_000_000,
                run_folder=folderName,
                gpuid = None, # seed%th.cuda.device_count(),
                pretrained_model_file=resumeModelFile,
                env_builder=env_builder,
                env_builder_args = env_builder_args,
                eval_configurations=eval_confs,
                eval_le_freq_ep=10,
                storage_torch_device=th.device("cuda:0"),
                run_id = run_id,
                debug=-10,
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
                le_disable_validation_set = False,
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
                le_loss_consistency_weight = 0.1,
                le_loss_img_weight = 1.0,
                le_loss_kld_weight = 0.0001,
                le_loss_latent_weight = 0.0,
                le_loss_obs_pred_weight = 1.0,
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
                replay_buffer_size = max_steps_per_episode*train_envs*500,

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
                use_rnd_novelty_exploration_bonus = False,
                exploration_bonus_weight = 0.0,
                rnd_nn_arch=[128,128],
                rnd_novelty_estimator_learning_rate = 0.00001,
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
                pkgs_to_save=["adarl","jumping_leg","rreal"])