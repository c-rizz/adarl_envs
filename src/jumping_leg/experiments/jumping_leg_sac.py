#!/usr/bin/env python3

import time
import inspect
from jumping_leg.utils.modded_sac import SAC
from stable_baselines3.ppo import PPO
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils

import os
from lr_gym.utils.sb3_buffers import ThDictReplayBuffer
import torch as th
import lr_gym.utils.session
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from wandb.integration.sb3 import WandbCallback
# from stable_baselines3.common.vec_env import SubprocVecEnv
from lr_gym.utils.subproc_vec_env import SubprocVecEnv
from lr_gym.envs.VecEnvLogger import VecEnvLogger
from lr_gym.utils.sb3_callbacks import EvalCallback_ep, SigintHaltCallback, PrintLrRunInfo
import stable_baselines3.common.base_class
import stable_baselines3.common.on_policy_algorithm

# def collect_rollout(env, steps_to_collect, callback):
#     callback.on_rollout_start()
#     rollout_data = ???
#     num_collected_steps = 0
#     while num_collected_steps < steps_to_collect:
#         ... = env.step()
#         num_collected_steps += 1
#         # Give access to local variables
#         callback.update_locals(locals())
#         # Only stop training if return value is False, not when it is None.
#         if not callback.on_step():
#             return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

#     callback.on_rollot_end()
#     return rollout_data

# def learn(model, callback, timesteps):
#     if isinstance(model,stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm):
#         # In sb3's on policy algorithms collection and training are not separated in a pretty way
#         # could go around it but it's a bit ugly
#         # Just use their learn()
#         model.learn(total_timesteps=timesteps,
#                     callback=callback)
#         return

#     if isinstance(model,stable_baselines3.common.base_class.BaseAlgorithm):
#         # If we are dealing with an sb3 model
#         model._setup_learn(total_timesteps=timesteps, callback=callback)
#         #define these locals
#         log_interval: int = 4
#         tb_log_name: str = "run"
#         reset_num_timesteps: bool = True
#         progress_bar: bool = False
    
#     callback.on_training_start(locals(), globals())

#     num_collected_steps = 0
#     while num_collected_steps <= timesteps:
#         collected_steps = collect_rollout(env, collected_steps_per_iteration, callback)
#         model.replay_buffer.update(collected_steps)
#         callback.on_rollout
#         train(gradient_steps, batch_size)




def runFunction(seed, folderName, resumeModelFile, run_id, args):
    env_builder_args = {
        "reward_contacts_weight" : 0.0, # ("uniform", 0, 1),
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 10.0,
        "reward_torque_limit_weight" : 1.0,
        "reward_torque_weight" : 0.0,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.0,
        "th_device" : th.device("cpu"),
        "control_mode" : "torque",
        "video_save_freq" : 0,
        "stepLength_sec" : 0.01,
        "platform_randomization" : "single"
                        }
    args.update({
        "batch_size" : 16384,
        "lr" : 0.005,
        "train_steps" : 50,
        "train_freq" : 50,
        "algorithm" : "sac"
                 })
    return solve_sac(seed, folderName, run_id, args, env_builder_args)

def solve_sac(seed, folderName, run_id, args, env_builder_args):
    
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
    eval_env = env_builder(log_folder=log_folder+"/eval_env", seed = seed+100000000, 
                         env_builder_args = env_builder_args)

    ggLog.info("Built")
#    env.action_space.seed(seed)

    ep_duration = 5/env_builder_args["stepLength_sec"]

    if args["algorithm"] == "sac":
        model = SAC("MultiInputPolicy", env, verbose=0,
                        batch_size=args["batch_size"],
                        buffer_size=10_000_000,
                        gamma=0.99,
                        learning_rate=args["lr"],
                        ent_coef="auto",
                        learning_starts=10_000,
                        tau=0.005,
                        gradient_steps=args["train_steps"],
                        train_freq=args["train_freq"],
                        target_entropy="auto",
                        seed = seed,
                        device=device,
                        policy_kwargs=dict(net_arch=[512,256]),
                        replay_buffer_class = ThDictReplayBuffer,
                        replay_buffer_kwargs = {"storage_torch_device" : device},
                        tensorboard_log=folderName+f"/tensorboard")
    elif args["algorithm"] == "ppo":    
        model = PPO("MultiInputPolicy", env, verbose=1,
                    n_steps = args["train_freq"],
                    batch_size=args["batch_size"],
                    learning_rate=args["lr"],
                    seed = seed,
                    device=device,
                    policy_kwargs=dict(net_arch=[512,256]),
                    # replay_buffer_class = ThDictReplayBuffer,
                    # replay_buffer_kwargs = {"storage_torch_device" : device},
                    tensorboard_log=folderName+f"/tensorboard")
    else:
        raise RuntimeError(f"Invalid algorithm '{args['algorithm']}")

    callbacks = []
    callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=log_folder+"/eval/EvalCallback",
                                            log_path=log_folder+"/eval/EvalCallback", eval_freq_ep=10,
                                            deterministic=True, render=False, verbose=True,
                                            n_eval_episodes = 1))
    callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=log_folder+"/eval100/EvalCallback",
                                            log_path=log_folder+"/eval100/EvalCallback", eval_freq_ep=100,
                                            deterministic=True, render=False, verbose=True,
                                            n_eval_episodes = 10))
    callbacks.append(WandbCallback( model_save_path=f"{folderName}/wandb/save",
                                    verbose=2))
    callbacks.append(SigintHaltCallback())
    callbacks.append(PrintLrRunInfo(print_freq_ep=parallel_envs))
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=10_000_000,
                callback=callbacks)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")



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

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = args["resumeFolder"],
                args = args,
                debug_level = -10,
                start_lr_gym=False,
                pkgs_to_save=["lr_gym","jumping_leg"])
