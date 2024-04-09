#!/usr/bin/env python3  

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch as th
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from lr_gym.utils.async_vector_env import AsyncVectorEnvShmem
import inspect
import lr_gym.utils.session
from lr_gym.envs.vector_env_logger import VectorEnvLogger
from lr_gym.utils.buffers import ThDReplayBuffer
import lr_gym.utils.sigint_handler
from jumping_leg.algorithms.sac import SAC, train
from jumping_leg.algorithms.collector import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector

def build_env(env_builder_args, log_folder, seed, num_envs):
    builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(num_envs)]
    envs = AsyncVectorEnvShmem(builders, context="forkserver")
    envs = VectorEnvLogger(env = envs)
    return envs

def build_sac(obs_space, act_space, hyperparams):
    return SAC(observation_space=obs_space,
                action_size=int(np.prod(act_space.shape)),
                q_network_arch=[512,256],
                q_lr=0.005,
                policy_arch=[512,256],
                action_min = th.as_tensor(act_space.low),
                action_max = th.as_tensor(act_space.high),
                torch_device=hyperparams["device"],
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=0.99,
                target_tau = 0.005,
                policy_update_freq=2,
                target_update_freq=1)

def main():


    seed = 0
    log_folder, session = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_comment="")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # env setup
    num_envs = 16
    env_builder_args = {
        "reward_contacts_weight" : 0.1,
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 10.0,
        "reward_torque_limit_weight" : 1.0,
        "reward_torque_weight" : 0.1,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.0,
        "th_device" : th.device("cpu"),
        "control_mode" : "torque",
        "video_save_freq" : 0,
        "stepLength_sec" : 0.01,
        "platform_randomization" : "single",
        "quiet" : False}

    hyperparams = {"device" : device,
                   "train_freq" : 50,
                   "grad_steps" : 25}
    vec_env_builder = lambda: build_env(env_builder_args=env_builder_args, log_folder=log_folder, seed=seed, num_envs=num_envs)
    collector = AsyncProcessExperienceCollector(vec_env_builder=vec_env_builder, 
                                         base_model_builder=lambda o,a: build_sac(o,a,hyperparams),
                                         storage_torch_device=device,
                                         buffer_size=hyperparams["train_freq"]*num_envs,
                                         session=session)
    observation_space = collector.observation_space()
    action_space = collector.action_space()
    model = build_sac(collector.observation_space(), collector.action_space(), hyperparams)

    # vec_env = build_env(env_builder_args=env_builder_args,
    #                     log_folder=log_folder,
    #                     seed=seed,
    #                     num_envs=num_envs)
    # model = build_sac(vec_env.single_observation_space, vec_env.single_action_space, hyperparams)
    # collector = AsyncThreadExperienceCollector(vec_env=vec_env,
    #                                base_model=model,
    #                                buffer_size=hyperparams["train_freq"]*num_envs,
    #                                storage_torch_device=device)
    # observation_space = vec_env.single_observation_space
    # action_space = vec_env.single_action_space

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


    train(collector=collector,
          model = model,
          buffer = rb,
          total_timesteps=10_000_000,
          train_freq = hyperparams["train_freq"],
          learning_starts=500*32,
          grad_steps=hyperparams["grad_steps"],
          batch_size=16384,
          log_freq=500)

    collector.close()
    # writer.close()


if __name__ == "__main__":
    main()
