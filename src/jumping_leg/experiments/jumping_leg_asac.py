#!/usr/bin/env python3




if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun
    from autoencoding_rl.experiments.asac3.asac3_train import asac3_train
    from jumping_leg.experiments.build_jumping_leg_env import env_builder
    import torch as th

    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--resumeFolder", default=None, type=str, help="Resume an entire run composed of multiple seeds")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--xvfb", default=False, action='store_true', help="Run with xvfb")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    ap.add_argument("--offline", default=False, action='store_true', help="Train offline")
    # group = ap.add_mutually_exclusive_group()
    # group.add_argument("--gazebo",     default=False, action='store_true',     help="Use gazebo classic env")
    # group.add_argument("--gz",         default=False, action='store_true',         help="Use ignition gazebo env")
    # group.add_argument("--simplified", default=False, action='store_true', help="Use simplified pybullet env")
    # group.add_argument("--real", default=False, action='store_true', help="Run on real robot")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    # if args["real"] and args["maxProcs"]>0:
    #     raise AttributeError("Cannot run multiple processes in the real")


    # if args["simplified"]:
    #     mode = "simplified"
    # elif args["gz"]:
    #     mode = "gz"
    # elif args["gazebo"]:
    #     mode = "gazebo_classic"
    # else:
    #     raise RuntimeError("No mode was specified, use either --gazebo --gz or --simplified")

    action_repeat = 4
    ep_duration = int(1000/action_repeat)
    parallel_envs = 16

    def runFunction(seed, folderName, resumeModelFile, run_id, args):
        if resumeModelFile is not None:
            raise AttributeError("resume is not supported")
        return asac3_train( seed=seed,
                            train_steps=ep_duration*8000*parallel_envs,
                            run_folder=folderName,
                            gpuid = None, # seed%th.cuda.device_count(),
                            modelFile=resumeModelFile,
                            env_builder=env_builder,
                            env_builder_args = {"reward_contacts_weight" : 0.0,
                                                "reward_energy_weight" : 0.0,
                                                "reward_position_limit_weight" : 1.0,
                                                "reward_torque_limit_weight" : 1.0,
                                                "reward_torque_weight" : 0.5,
                                                "reward_tracking_weight" : 1.0,
                                                "reward_velocity_weight" : 0.1,
                                                "th_device" : th.device("cuda"),
                                                "control_mode" : "position_and_gains",
                                                "video_save_freq" : 0,
                                                "stepLength_sec" : 20/1024, # about 50Hz
                                                "platform_randomization" : "single",
                                                "quiet" : False,
                                                "mode" : "pybullet",
                                                "use_contacts" : True,
                                                "ep_obs_noise_mustd" : (0.01, 0.01),
                                                "step_obs_noise_std" : 0.01,
                                                "stop_on_safety" : True,
                                                "action_delay_mustd" : (0.01,0.01),
                                                "obs_only_vec" : False,
                                                "max_steps_per_episode" : ep_duration},
                            eval_le_freq_ep=100,
                            eval_period_ep=100,
                            eval_episodes_num=1,
                            storage_torch_device="cpu",                            
                            experiment_name=os.path.basename(__file__),
                            run_id = run_id,
                            main_file_path=__file__,
                            debug=-10,
                            check_infs_nans=False,
                            
                            automatic_grad_steps=False,
                            le_always_deterministic=False,
                            le_arch_img_encoder_backbone = "conv_small2",
                            le_arch_img_decoder_backbone = "conv_small",
                            le_arch_dyn_ensemble_size = 5,
                            le_arch_dynamics = [256,256],
                            le_arch_enc_ensemble_size = 1,
                            le_arch_frame_stack_size=1,
                            le_arch_img_dec_ensemble_size = 1,
                            le_arch_img_dec_learn_background = False,
                            le_arch_img_encoding_size = 16,            
                            le_arch_latent_space_size = 73,
                            le_arch_optimizer="adam",
                            le_arch_reward = [256],
                            le_arch_reward_ensemble_size=5,
                            le_arch_state_combiner = "identity",
                            le_arch_state_decombiner = "identity",
                            le_arch_type = "dvae4_3",
                            le_arch_use_coord_conv=True,
                            le_arch_vec_decoder=[256],
                            le_arch_vec_decoder_activation=("scaledtanh", 10),
                            le_arch_vec_decoder_ensemble_size=5,
                            le_arch_vec_encoder="identity",
                            le_arch_vec_encoder_ensemble_size=5,
                            le_arch_vec_encoding_size=57,
                            le_batch_size = 128,
                            le_bestModelThreshold = -1,
                            le_consistency_target = None,
                            le_decoder_weight_decay_lambda=None,
                            le_disable_validation_set = True,
                            le_direct_vec_into_latent = False,
                            le_dont_train_prediction_decoder=False,
                            le_dont_train_prediction_encoder=False,
                            le_dont_train_reconstruction_encoder=False,
                            le_fake_encoding_noise_std=0.0,
                            le_freeze_decoders=False,
                            le_freeze_dynamics=False,
                            le_freeze_encoders=False,
                            le_grad_steps = 50,
                            le_internal_enc_dropout = 0.0,
                            le_latent_space_activation="identity",
                            le_learning_starts = 1000,
                            le_loss_consistency_weight = 0.0,
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
                            le_pretrain_grad_steps = 16,
                            le_reset_decoder=False,
                            le_reset_dynamics=False,
                            le_reset_encoder=False,
                            le_reset_obs_stats=False,
                            le_residual_dynamics=True,
                            le_retrain_epochs = 30,
                            le_retrain_period = -1,
                            le_reward_scaling = 10.0,
                            le_reset_on_retrain = False,
                            le_tau=1.0,
                            le_train_freq=(ep_duration,"step"),
                            le_train_trajectories_length = 5,
                            le_use_log_reward = False,
                            le_validation_batch_size = 128,
                            le_validation_ratio=0.05,
                            le_validation_buffer_size=ep_duration*parallel_envs*10,
                            le_weight_decay = 1e-5,
                            parallelize_experience_collection=False,
                            random_steps = 1000,
                            replay_buffer_size = ep_duration*parallel_envs*30,

                            sac_arch_actor_ensemble_size=3,
                            sac_arch_critic = [64,64],
                            sac_arch_policy = [64,64],
                            sac_batch_size = 512,
                            sac_ent_coef = "auto",
                            sac_gamma = 0.99,
                            sac_grad_steps = 50,
                            sac_learning_starts = ep_duration*parallel_envs*5,
                            sac_lr_actor = 0.0005,
                            sac_lr_critic = 0.005,
                            sac_target_entropy="auto",
                            sac_tau=0.005,
                            sac_train_freq=(ep_duration,"step"),
                            comment = args["comment"],
                            checkpointing_freq_ep=100,
                            buffer_reward_exploration_augmentation = False,
                            buffer_reward_expl_augmentation_weight = 0.0,
                            use_rnd_novelty_exploration_bonus = False,
                            exploration_bonus_weight = 0.0,
                            rnd_nn_arch=[128,128],
                            rnd_novelty_estimator_learning_rate = 0.00001,
                            parallel_envs=parallel_envs,
                            asac_running_avgs_alpha = 0.999,
                            asac_running_avg_rew_alpha = 0.9999,
                            max_episode_duration=ep_duration,
                            min_episode_duration=ep_duration,
                            allow_tf32=True,
                            purely_numpy_env = False
                            )

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = args["resumeFolder"],
                args = args,
                start_adarl=False)
