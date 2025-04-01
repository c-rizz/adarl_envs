#!/usr/bin/env python3

import adarl.utils.dbg.dbg_img
from rreal.algorithms.sac import SAC
from rreal.algorithms.random_policy import RandomPolicy
import adarl.utils.utils

import torch as th
import adarl.utils.session
from adarl_envs.experiments.build_adarl_envs_env import env_builder, video_recorder_kwargs
import adarl.utils.sigint_handler


import adarl.utils.dbg
from rreal.examples.evaluate import evaluate
import adarl.utils.dbg.ggLog as ggLog
import numpy as np

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    adarl.utils.dbg.dbg_img.helper.enable_web_dbg(True)
    step_length_sec = 20/1024  # about 50Hz
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
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
        "action_smoothing_halflife_sec" : 0.01,
        "leg_min_height" : 0.4,
        "leg_max_height" : 0.65,
        "leg_max_jump" : 0.3,
        "enable_rendering" : True,
        "goal_dist_smoothing_halflife_sec" : 0.01}

    results = evaluate( seed=seed,
                        folderName=folderName,
                        run_id=run_id,
                        args=args,
                        env_builder=env_builder,
                        env_builder_args=env_builder_args,
                        video_recorder_kwargs=video_recorder_kwargs,
                        model_builder= lambda obs_space, act_space, hyperparams:SAC.load(path=args["pretrained"]),
                        # model_builder = lambda obs_space, act_space, hyperparams: RandomPolicy(int(np.prod(act_space.shape)),
                        #                                                         act_space.low.tolist(),
                        #                                                         act_space.high.tolist()),
                        model_kwargs = {},
                        num_envs=args["num_envs"],
                        episodes=args["vepisodes"]*args["num_envs"],
                        extra_info_stats=["avg10_dist","safety_triggered"],
                        deterministic=True)
    n = "\n"
    results_strings = [' - '+k+': '+str(v) for k,v in results.items()]
    results_strings = sorted(results_strings)
    ggLog.info(f"Results =\n{n.join(results_strings)}")
    return results

if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--pretrained", required = True, type=str, help="Model to load")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use")
    ap.add_argument("--vepisodes", default=10, type=int, help="Evaluate the policy with this number of parallelized episodes (acutal numberof episodes in num_envs*vepisodes)")
    ap.add_argument("--num_envs", default=16, type=int, help="Use this number of parallel envs")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=1,
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","adarl_envs"])
