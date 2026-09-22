#!/usr/bin/env python3
"""Train a policy on the simplified pushing environment.

    python -m adarl_envs.experiments.solve_pushing_env --comment "first pushing run"
    python -m adarl_envs.experiments.solve_pushing_env --comment "camera obs" --obs-cam
    python -m adarl_envs.experiments.solve_pushing_env --comment "debug" --algorithm sac_small --no-wandb

The environment is described in adarl_envs.env.PushingVecEnv: the agent commands a planar
displacement of a floating end effector and has to push a cube onto a goal. The actor reads the
"base" observation (end effector position, goal and time, plus the camera image when --obs-cam is
given), while the critic reads the "privileged" one, which always contains the full cube pose.
"""
from __future__ import annotations


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams
    from rreal.feature_extractors.mixed_feature_extractor import MixedFeatureExtractorInitArgs
    from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractorInitArgs
    from adarl_envs.experiments.pushing_builder import pushing_vecenv_builder

    debug_level = 1
    mode = args["mode"].lower()
    obs_cam = args["obs_cam"]
    algo = args["algorithm"].lower()
    # Control period of the end-effector controller, not the duration of an environment step (that
    # is variable). Use multiples of 1/1024 to keep it representable in binary.
    control_period_sec = 20/1024
    max_steps_per_episode = 50 # as in the original PandaPushingEnv2DOF

    if algo == "sac":
        # The environment's step is blocking (it lasts as long as the commanded movement takes), and
        # rendering the observation camera costs more on top, so fewer envs than the grasp setup.
        train_envs = 256 if obs_cam else 1024
    elif algo == "sac_small":
        train_envs = 8
    else:
        raise RuntimeError(f"Unexpected algo '{algo}'")
    if args["envs"] is not None:
        train_envs = args["envs"]

    if mode == "mjx":
        env_device = th.device("cuda",0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}' (the pushing environment only supports 'mjx')")

    # Images are big, so the buffer holds far fewer transitions when the camera is observed.
    buffer_size = 300*1024 if obs_cam else 1024*1024

    eval_freq = 5
    env_builder_args = {
        "mode" : mode,
        "th_device" : env_device,
        "control_period_sec" : control_period_sec,
        "max_steps_per_episode" : max_steps_per_episode,
        "observe_camera" : obs_cam,
        "enable_rendering" : obs_cam,
        "enable_ui_camera" : False,
        "frame_stack_length" : 1,
        "history_length" : 1,
        "sparse_reward" : False,
        "log_info_stats" : True, # required by sac_train's VectorEnvLogger wrapper
        "quiet" : True,
        "record_video" : False,
        "video_save_freq" : -1,
    }

    # Recording the eval video goes through hdf5plot; --no-eval-video keeps the periodic evaluation
    # but skips the recording, which is useful when that path is unavailable.
    record_eval_video = not args["no_eval_video"]
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = record_eval_video
    video_eval_env_builder_args["enable_ui_camera"] = record_eval_video
    video_eval_env_builder_args["record_video"] = record_eval_video
    video_eval_env_builder_args["video_save_freq"] = 1 if record_eval_video else -1
    video_eval_env_builder_args["ui_camera_resolution_hw"] = (270,480)
    video_eval_env_builder_args["quiet"] = False
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "skip_first_eval" : True
    }
    eval_configurations = [eval_conf_video_stoch]

    # The actor only sees what a real setup could measure; the critic always gets the cube pose.
    actor_observation_filter = ["base.vec", "base.camera"] if obs_cam else ["base.vec"]
    critic_observation_filter = ["privileged.vec"]

    if algo == "sac":
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = pushing_vecenv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams(
                        alpha_initial_value = 0.001,
                        actor_log_std_init = -2.0,
                        actor_mean_bounds_ratio = 0.9,
                        actor_observation_filter = actor_observation_filter,
                        batch_size = 4096,
                        buffer_size = buffer_size,
                        critic_observation_filter = critic_observation_filter,
                        gamma = 0.99, # episodes are only 50 steps long
                        grad_steps = 40,
                        learning_starts = max_steps_per_episode*max(train_envs, 100),
                        log_freq_vstep = max_steps_per_episode,
                        model_th_device = "cuda",
                        parallel_envs = train_envs,
                        policy_arch = [256,256],
                        policy_lr = 0.0003,
                        q_lr = 0.001,
                        q_network_arch = [512,128],
                        reference_init_args = {"env_builder_args" : env_builder_args,
                                               "eval_configuration" : eval_configurations},
                        target_entropy_factor = -1.0,
                        target_tau = 0.005,
                        total_steps = 100_000_000,
                        train_freq_vstep = 5,
                        ),
                    checkpoint_freq = 5,
                    collector_device = env_device,
                    max_episode_duration = max_steps_per_episode,
                    validation_buffer_size = 0,
                    validation_batch_size = 0,
                    validation_holdout_ratio = 0,
                    no_wandb = args["no_wandb"],
                    debug_level = debug_level,
                    # The image observation needs an extractor that can mix it with the vector part;
                    # without the camera the observations are plain vectors and need none.
                    actor_feature_extractor_name = "MixedFeatureExtractor" if obs_cam else None,
                    actor_fe_hparams = MixedFeatureExtractorInitArgs(device=th.device("cuda"),
                                                                     vec_encoder_arch="identity",
                                                                     vec_encoding_size=None,
                                                                     combiner_arch="identity",
                                                                     encoding_size=None) if obs_cam else None,
                    critic_feature_extractor_name = None,
                    critic_fe_hparams = StackVectorsFeatureExtractorInitArgs(device=th.device("cuda")),
                    use_rnd_exploration = False
                    )
    elif algo == "sac_small":
        # Small-scale configuration, for checking that the whole pipeline runs.
        sac_train(  seed,
                    folderName,
                    run_id,
                    args,
                    vec_env_builder = pushing_vecenv_builder,
                    env_builder_args = env_builder_args,
                    eval_configurations = eval_configurations,
                    hyperparams = SAC_init_hparams(
                        actor_log_std_init = -2.0,
                        actor_observation_filter = actor_observation_filter,
                        batch_size = 512,
                        buffer_size = 100*1024,
                        critic_observation_filter = critic_observation_filter,
                        gamma = 0.98,
                        grad_steps = 20,
                        learning_starts = max_steps_per_episode*max(train_envs, 100),
                        log_freq_vstep = max_steps_per_episode,
                        model_th_device = "cuda",
                        parallel_envs = train_envs,
                        policy_arch = [256,256],
                        policy_lr = 0.0003,
                        q_lr = 0.001,
                        q_network_arch = [512,128],
                        reference_init_args = {"env_builder_args" : env_builder_args,
                                               "eval_configuration" : eval_configurations},
                        target_entropy_factor = -1.0,
                        target_tau = 0.005,
                        total_steps = 100_000_000,
                        train_freq_vstep = 10
                        ),
                    checkpoint_freq = 5,
                    collector_device = env_device,
                    max_episode_duration = max_steps_per_episode,
                    validation_buffer_size = 0,
                    validation_batch_size = 0,
                    validation_holdout_ratio = 0,
                    no_wandb = args["no_wandb"],
                    debug_level = debug_level,
                    actor_feature_extractor_name = "MixedFeatureExtractor" if obs_cam else None,
                    actor_fe_hparams = MixedFeatureExtractorInitArgs(device=th.device("cuda"),
                                                                     vec_encoder_arch="identity",
                                                                     vec_encoding_size=None,
                                                                     combiner_arch="identity",
                                                                     encoding_size=None) if obs_cam else None)
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
    ap.add_argument("--algorithm", default="sac", type=str, help="Algorithm to use ('sac'/'sac_small')")
    ap.add_argument("--mode", default="mjx", type=str, help="Simulator to use ('mjx')")
    ap.add_argument("--envs", default=None, type=int, help="Override the number of parallel environments")
    ap.add_argument("--no-wandb", default=False, action='store_true', help="Disable Weight&Biases")
    ap.add_argument("--obs-cam", default=False, action='store_true', help="Observe the camera image instead of the cube pose")
    ap.add_argument("--no-eval-video", default=False, action='store_true', help="Evaluate periodically but do not record the evaluation video")

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
