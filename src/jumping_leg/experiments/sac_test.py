#!/usr/bin/env python3  


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
    from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    ep_duration_sec = 5
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 32
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
        "th_device" : th.device("cpu"),
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
    
    eval_confs = [  eval_conf_video_det,
                    eval_conf_video_stoch,
                    eval_conf_feasible,
                    eval_conf_video_feasible,
                    eval_conf_video_jump_feasible]

    sac_train(  seed,
                folderName,
                run_id,
                args,
                env_builder = build_jumping_leg_env.env_builder,
                env_builder_args = env_builder_args,
                eval_configurations = eval_confs,
                hyperparams = SAC_hyperparams(  device = "cuda",
                                                q_network_arch=[256,128],
                                                q_lr=0.001,
                                                policy_lr=0.0005,
                                                policy_network_arch=[256,256],
                                                gamma=0.99,
                                                target_tau = 0.005,
                                                batch_size=16384,
                                                buffer_size=1_000_000,
                                                total_steps=100_000_000,
                                                train_freq_vstep=25,
                                                grad_steps=50,
                                                learning_starts=max_steps_per_episode*train_envs*5,
                                                parallel_envs=train_envs,
                                                log_freq_vstep=max_steps_per_episode,
                                                reference_init_args={"env_builder_args":env_builder_args,
                                                                     "eval_configuration" : eval_confs}),
                checkpoint_freq=100,
                collector_device=th.device("cpu"),
                debug_level=-1,
                max_episode_duration=max_steps_per_episode,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0)



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