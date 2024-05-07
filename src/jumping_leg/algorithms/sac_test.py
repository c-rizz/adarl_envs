#!/usr/bin/env python3  

import os
import random
import time

import numpy as np
import torch
import torch as th
import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
from lr_gym.utils.async_vector_env import AsyncVectorEnvShmem
import inspect
import lr_gym.utils.session
from lr_gym.envs.vector_env_logger import VectorEnvLogger
from lr_gym.utils.buffers import ThDReplayBuffer
import lr_gym.utils.sigint_handler
from jumping_leg.algorithms.sac import SAC, train_off_policy
from jumping_leg.algorithms.collectors import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector
import wandb 
from lr_gym.utils.callbacks import EvalCallback, CheckpointCallbackRB


def build_vec_env(env_builder_args, log_folder, seed, num_envs):
    builders = [(lambda i: (lambda: build_jumping_leg_env.env_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(num_envs)]
    envs = AsyncVectorEnvShmem(builders, context="forkserver", purely_numpy=False, shared_mem_device = th.device("cpu"), copy_data=False)
    envs = VectorEnvLogger(env = envs)
    return envs

def build_sac(obs_space, act_space, hyperparams):
    return SAC(observation_space=obs_space,
                action_size=int(np.prod(act_space.shape)),
                q_network_arch=[512,256],
                q_lr=hyperparams["q_lr"],
                policy_lr=hyperparams["policy_lr"],
                policy_arch=[512,256],
                action_min = act_space.low.tolist(),
                action_max = act_space.high.tolist(),
                torch_device=hyperparams["device"],
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=0.99,
                target_tau = 0.005,
                policy_update_freq=2,
                target_update_freq=1)

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    env_builder_args = {
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 10.0,
        "reward_torque_limit_weight" : 1.0,
        "reward_torque_weight" : 0.1,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.0,
        "th_device" : th.device("cpu"),
        "control_mode" : "position_and_gains",
        "video_save_freq" : 0,
        "stepLength_sec" : 0.02,
        "platform_randomization" : "single",
        "quiet" : False}

    hyperparams = {"train_freq" : 50,
                   "grad_steps" : 25,
                   "q_lr" : 0.005,
                   "policy_lr" : 0.0005}
    main(seed, folderName, run_id, args, env_builder_args, hyperparams)

# import traceback
# import threading
# t = threading.Thread.__init__
# def threadwrapper(self : threading.Thread, *args, **kwargs):
#     t(self, *args, **kwargs)
#     print(f" created thread {self.name}")
#     traceback.print_stack()

def main(seed, folderName, run_id, args, env_builder_args, hyperparams):

    # print(f"active threads = {threading.enumerate()}")
    # threading.Thread.__init__ = threadwrapper
    # torchexplorer.setup()
    log_folder, session = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams["device"] = device
    # env setup
    num_envs = 16
    use_processes = True
    if use_processes:
        vec_env_builder = lambda: build_vec_env(env_builder_args=env_builder_args, log_folder=log_folder, seed=seed, num_envs=num_envs)
        collector = AsyncProcessExperienceCollector(vec_env_builder=vec_env_builder, 
                                            base_model_builder=lambda o,a: build_sac(o,a,hyperparams),
                                            storage_torch_device=device,
                                            buffer_size=hyperparams["train_freq"]*num_envs,
                                            session=session)
        observation_space = collector.observation_space()
        action_space = collector.action_space()
        model = build_sac(observation_space, action_space, hyperparams)
    else:
        vec_env = build_vec_env(env_builder_args=env_builder_args,
                            log_folder=log_folder,
                            seed=seed,
                            num_envs=num_envs)
        observation_space = vec_env.single_observation_space
        action_space = vec_env.single_action_space
        model = build_sac(observation_space, action_space, hyperparams)
        collector = AsyncThreadExperienceCollector(vec_env=vec_env,
                                    base_model=model,
                                    buffer_size=hyperparams["train_freq"]*num_envs,
                                    storage_torch_device=device)

    # torchexplorer.watch(model, backend="wandb")
    wandb.watch((model, model._actor, model._q_net), log="all", log_freq=1000, log_graph=True)

    # compiled_model = th.compile(model)
    # envs.single_observation_space.dtype = np.float32
    # rb = ThDictReplayBuffer(
    #     buffer_size=1000_000,
    #     observation_space=envs.single_observation_space,
    #     action_space=envs.single_action_space,
    #     device=device,
    #     storage_torch_device=device,
    #     handle_timeout_termination=False,
    #     n_envs=num_envs,
    #     disable_validation_set=True)
    rb = ThDReplayBuffer(
        buffer_size=10_000_000,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        storage_torch_device=device,
        handle_timeout_termination=True,
        n_envs=num_envs)
    start_time = time.time()

    eval_env = build_jumping_leg_env.env_builder(log_folder=log_folder+"/eval",
                            seed=seed+100000000,
                            env_builder_args = env_builder_args)
    eval_env_rec = build_jumping_leg_env.wrap_with_recorder(eval_env,
                                                        stepLength_sec=env_builder_args["stepLength_sec"],
                                                        log_folder=log_folder+"/eval",
                                                        video_save_freq=1)
    eval_env_rec_det = build_jumping_leg_env.wrap_with_recorder(eval_env,
                                                        stepLength_sec=env_builder_args["stepLength_sec"],
                                                        log_folder=log_folder+"/eval_deterministic",
                                                        video_save_freq=1)
    callbacks = []
    callbacks.append(EvalCallback(eval_env=eval_env_rec,
                                  model=model,
                                  n_eval_episodes=1,
                                  eval_freq_ep=10*num_envs,
                                  deterministic=False))
    callbacks.append(EvalCallback(eval_env=eval_env_rec_det,
                                  model=model,
                                  n_eval_episodes=1,
                                  eval_freq_ep=10*num_envs,
                                  deterministic=True))
    callbacks.append(CheckpointCallbackRB(save_path=log_folder+"/checkpoints",
                                          model=model,
                                          save_best=False,
                                          save_freq_ep=100*num_envs))
    try:
        train_off_policy(collector=collector,
            model = model,
            buffer = rb,
            total_timesteps=10_000_000,
            train_freq = hyperparams["train_freq"],
            learning_starts=500*num_envs*5,
            grad_steps=hyperparams["grad_steps"],
            batch_size=16384,
            log_freq=500,
            callbacks=callbacks)
    finally:
        collector.close()
    # writer.close()



if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from lr_gym.utils.session import launchRun

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
                start_lr_gym=False,
                pkgs_to_save=["lr_gym","jumping_leg"])