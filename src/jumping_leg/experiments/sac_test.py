#!/usr/bin/env python3  


def runFunction(seed, folderName, resumeModelFile, run_id, args):
    step_length_sec = 20/1024  # about 50Hz
    ep_duration_sec = 5
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    num_envs = 32
    env_builder_args = {
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 1.0,
        "reward_torque_limit_weight" : 1.0,
        "reward_torque_weight" : 0.1,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.01,
        "th_device" : th.device("cpu"),
        "control_mode" : "position_and_gains",
        "video_save_freq" : 0,
        "stepLength_sec" : step_length_sec,
        "platform_randomization" : "single",
        "quiet" : False,
        "mode" : "pybullet",
        "use_contacts" : False,
        "ep_obs_noise_mustd" : (0.01, 0.01),
        "step_obs_noise_std" : 0.01,
        "stop_on_safety" : True,
        "action_delay_mustd" : (0.01,0.01),
        "max_steps_per_episode" : max_steps_per_episode,
        "obs_only_vec":True,
        "action_smoothing_halflife_sec" : 0.01}

    sac_train(  seed,
                folderName,
                run_id,
                args,
                env_builder = build_jumping_leg_env.env_builder,
                env_builder_args = env_builder_args,
                hyperparams = SAC_hyperparams( device = "cuda",
                                                q_network_arch=[256,128],
                                                q_lr=0.005,
                                                policy_lr=0.0005,
                                                policy_network_arch=[256,128],
                                                gamma=0.99,
                                                target_tau = 0.005,
                                                batch_size=16384,
                                                buffer_size=1_000_000,
                                                total_steps=10_000_000,
                                                train_freq=25,
                                                grad_steps=50,
                                                learning_starts=max_steps_per_episode*num_envs*5,
                                                parallel_envs=num_envs,
                                                log_freq_vstep=max_steps_per_episode,
                                                eval_freq_ep=10*num_envs),
                video_recorder_kwargs=build_jumping_leg_env.video_recorder_kwargs)



if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun
    import torch as th
    import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
    from rreal.examples.solve_sac import sac_train, SAC_hyperparams

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