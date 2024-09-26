#!/usr/bin/env python3  


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    import jumping_leg.experiments.build_locomotion_env as build_locomotion_env
    from rreal.examples.solve_sac import sac_train, SAC_hyperparams
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    ep_duration_sec = 5
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 64
    env_builder_args = {
        "action_delay_mustd" : (0.01,0.01),
        "action_noise_mustd" : (0.0,0.001),
        "action_smoothing_halflife_sec" : 0.2,
        "control_mode" : "position",
        "enable_rendering" : False,
        "goal_err_smoothing_halflife_sec" : 0.2,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : "pybullet",
        "quiet" : False,
        "reward_acceleration_weight" : 0.5,
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_health_weight" : 0.0,
        "reward_position_limit_weight" : 0.1,
        "reward_torque_limit_weight" : 0.1,
        "reward_torque_weight" : 0.0,
        "reward_torquediff_weight" : 0.0,
        "reward_tracking_weight" : 0.0,
        "reward_velocity_limit_weight" : 0.1,
        "reward_velocity_weight" : 0.1,
        "reward_height_weight" : 0.1,
        "reward_pitchnroll_weight" : 0.1,
        "safe_stiffness" : 200,
        "safe_damping" : 10,
        "stepLength_sec" : step_length_sec,
        "obs_noise_step_std" : 0.0,
        "obs_noise_ep_mustd" : (0.0, 0.0),
        "stop_on_safety" : False,
        "th_device" : th.device("cpu"),
        "video_save_freq" : 0,
        "goal_speed_minmax" : (0,2),
        "use_contacts" : True,
        "frame_stack_length" : 1}
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
    run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    run_1ms_env_builder_args["enable_rendering"] = True
    run_1ms_env_builder_args["video_save_freq"] = 1
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
    

    sac_train(  seed,
                folderName,
                run_id,
                args,
                env_builder = build_locomotion_env.env_builder,
                env_builder_args = env_builder_args,
                eval_env_builder_args = [
                                        eval_conf_video_det,
                                        eval_conf_video_stoch,
                                        eval_conf_run_1ms,
                                        #  eval_conf_feasible,
                                        #  eval_conf_video_feasible,
                                        #  eval_conf_video_jump_feasible
                                         ],
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
                                                train_freq=25,
                                                grad_steps=100,
                                                learning_starts=max_steps_per_episode*train_envs*5,
                                                parallel_envs=train_envs,
                                                log_freq_vstep=max_steps_per_episode),
                video_recorder_kwargs=build_locomotion_env.video_recorder_kwargs,
                checkpoint_freq=100,
                collector_device=th.device("cpu"))



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