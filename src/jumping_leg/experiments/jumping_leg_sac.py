#!/usr/bin/env python3

import time
import inspect
import numpy as np

from stable_baselines3.sac.policies import MultiInputPolicy
# from stable_baselines3 import SAC
# from jumping_leg.utils.original_sac_with_timing import SAC
from jumping_leg.utils.modded_sac import SAC
from stable_baselines3.common.noise import NormalActionNoise
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils

from lr_gym.envs.GymToLr import GymToLr
from lr_gym.envs.lr_wrappers.ObsToDict import ObsToDict
import os
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
from lr_gym.utils.sb3_buffers import ThDictReplayBuffer
from lr_gym.envs.NestedDictFlattenerGymWrapper import NestedDictFlattenerGymWrapper
from lr_gym.envs.lr_wrappers.ObsToImgVecDict import ObsToImgVecDict
import torch as th
import lr_gym.utils.session
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from wandb.integration.sb3 import WandbCallback
# from stable_baselines3.common.vec_env import SubprocVecEnv
from lr_gym.utils.subproc_vec_env import SubprocVecEnv
from lr_gym.envs.VecEnvLogger import VecEnvLogger
from lr_gym.utils.sb3_callbacks import EvalCallback_ep, SigintHaltCallback



def runFunction(seed, folderName, resumeModelFile, run_id, args):
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
    ggLog.info(f"Starting run")
    log_folder = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        folderName=folderName,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"])

    th.cuda.set_sync_debug_mode("warn")
    device = lr_gym.utils.utils.torch_selectBestGpu()
    ggLog.info("Building env...")

    #setup seeds for reproducibility
    env_builder_args = {"th_device" : th.device("cpu"),
                        "video_save_freq" : 0,
                        "reward_torque_weight" : 0.0,
                        "reward_position_weight" : 1.0,
                        "reward_velocity_weight" : 0.0,
                        "reward_energy_weight" : 0.01,
                        "reward_tracking_weight" : 1.0}
    
    parallel_envs = 16
    if False: #parallel_envs == 1:
        env = env_builder(log_folder=log_folder, seed = seed, env_builder_args = {"th_device" : th.device("cpu"),
                                                                                "video_save_freq" : 0})
        
    else:
        builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(parallel_envs)]
        env = SubprocVecEnv(builders, start_method = "forkserver")
        env = VecEnvLogger(env)
    eval_env_builder_args = {}
    eval_env_builder_args.update(env_builder_args)
    env_builder_args["video_save_freq"] = 1
    eval_env = env_builder(log_folder=log_folder+"/eval", seed = seed+100000000, 
                         env_builder_args = env_builder_args)

    ggLog.info("Built")
#    env.action_space.seed(seed)

    max_steps = 500

    model = SAC("MultiInputPolicy", env, verbose=1,
                    batch_size=4096,
                    buffer_size=10_000_000,
                    gamma=0.99,
                    learning_rate=0.005,
                    ent_coef="auto",
                    learning_starts=10_000,
                    tau=0.005,
                    gradient_steps=int(max_steps),
                    train_freq=max_steps,
                    target_entropy="auto",
                    seed = seed,
                    device=device,
                    policy_kwargs=dict(net_arch=[256,256]),
                    replay_buffer_class = ThDictReplayBuffer,
                    replay_buffer_kwargs = {"storage_torch_device" : device},
                    tensorboard_log=folderName+f"/tensorboard")

    callbacks = []
    callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=log_folder+"/eval/EvalCallback",
                                            log_path=log_folder+"/eval/EvalCallback", eval_freq_ep=10,
                                            deterministic=False, render=False, verbose=True,
                                            n_eval_episodes = 1))
    callbacks.append(WandbCallback( model_save_path=f"{folderName}/wandb/save",
                                    verbose=2))
    callbacks.append(SigintHaltCallback())
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=10_000_000,
                callback=callbacks)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    # res = lr_gym.utils.utils.evaluatePolicy(env = eval_env, model = None, episodes = 10, predict_func=model.predict)
    # print(f"Summary:\n{res}")

if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from lr_gym.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--resumeFolder", default=None, type=str, help="Resume an entire run composed of multiple seeds")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    # ap.add_argument("--xvfb", default=False, action='store_true', help="Run with xvfb")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    # ap.add_argument("--offline", default=False, action='store_true', help="Train offline")
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
    parallel_envs = 1


    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = args["resumeFolder"],
                args = args,
                debug_level = -10)
