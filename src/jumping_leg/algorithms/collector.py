
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import torch as th
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from lr_gym.utils.async_vector_env import AsyncVectorEnvShmem
import inspect
import lr_gym.utils.session
from lr_gym.envs.vector_env_logger import VectorEnvLogger
from lr_gym.utils.ThDictEpReplayBuffer import ThDictEpReplayBuffer
from lr_gym.utils.buffers import ThDReplayBuffer
from lr_gym.utils.ObsConverter import ObsConverter
from typing import List, Union, NamedTuple, Dict, Optional, Callable
from autoencoding_rl.utils import build_mlp_net
from lr_gym.utils.tensor_trees import map_tensor_tree
import lr_gym.utils.dbg.ggLog as ggLog
import copy
import threading
import lr_gym.utils.sigint_handler
from lr_gym.utils.buffers import BasicStorage
import torch.multiprocessing as mp
from lr_gym.utils.shared_env_data import SimpleCommander
import ctypes
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
import gymnasium as gym
import lr_gym.utils.mp_helper as mp_helper
import lr_gym.utils.session as session

class ExperienceCollector():
    def __init__(self, vec_env : gym.vector.VectorEnv):
        self._vec_env = vec_env
        self._last_obs = None

    def reset(self):
        if self._vec_env is not None:
            self._last_obs, info = self._vec_env.reset()

    def collect_experience(self, policy, vsteps_to_collect, global_vstep_count, random_vsteps, policy_device,
                           buffer):
        if  self._last_obs is None:
            raise RuntimeError(f"last_obs is not set. reset() should be called before running collect_experience the first time")
        obs = self._last_obs
        num_envs = self._vec_env.unwrapped.num_envs
        for step in range(vsteps_to_collect):
            if global_vstep_count < random_vsteps:
                actions = th.as_tensor(np.stack([self._vec_env.single_action_space.sample() for _ in range(num_envs)]))
            else:
                th_obs = map_tensor_tree(obs, lambda a: th.as_tensor(a, device = policy_device))
                actions = policy.sample_actions(th_obs)
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = self._vec_env.step(actions)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            buffer.add(obs=obs,
                        next_obs=real_next_obs,
                        action=actions,
                        reward=rewards,
                        terminated=terminations,
                        truncated=truncations)
            # ggLog.info(f"added step {step} to buffer")

            obs = next_obs
            self._last_obs = obs

class AsyncThreadExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        base_model,
                        buffer_size : int,
                        storage_torch_device):
        super().__init__(vec_env=vec_env)

        self._start_collect = threading.Event()
        self._collect_done = threading.Event()
        self._collector_model = copy.deepcopy(base_model)
        self._running = True
        self._buffer_size = buffer_size
        self._storage_torch_device = storage_torch_device
        self._buffer = BasicStorage(buffer_size = self._buffer_size,
                                    observation_space=self._vec_env.single_observation_space,
                                    action_space=self._vec_env.single_action_space,
                                    n_envs=self._vec_env.num_envs,
                                    storage_torch_device=self._storage_torch_device,
                                    share_mem=True,
                                    allow_rollover=False)
        self._collector_thread = threading.Thread(target=self._worker)
        self._collector_thread.start()

    def _worker(self):
        while self._running and not session.default_session.is_shutting_down():
            got_set = self._start_collect.wait(timeout=2)
            if got_set:
                self._start_collect.clear()
                t0 = time.monotonic()
                self.collect_experience(policy=self._collector_model,
                                        vsteps_to_collect=self._vsteps_to_collect,
                                        global_vstep_count=self._global_vstep_count,
                                        random_vsteps=self._random_vsteps,
                                        policy_device=self._collector_model.device,
                                        buffer=self._buffer)
                self._last_collection_duration = time.monotonic() - t0
                self._collect_done.set()

    def collect_experience_async(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._vsteps_to_collect, self._global_vstep_count, self._random_vsteps = vsteps_to_collect, global_vstep_count, random_vsteps
        self._buffer.clear()
        self._start_collect.set()

    def wait_collection(self, timeout = 10.0):
        got_set = self._collect_done.wait(timeout=timeout)
        if not got_set:
            raise TimeoutError(f"Collector timed out waiting for collect (timeout = {timeout})")
        self._collect_done.clear()
        return self._buffer

    def close(self):
        self._running = False
        self._collector_thread.join()

    def observation_space(self):
        return self._vec_env.single_observation_space
    
    def action_space(self):
        return self._vec_env.single_action_space
    
    def num_envs(self):
        return self._vec_env.num_envs
    
    def collection_duration(self):
        return self._last_collection_duration

class AsyncProcessExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env_builder,
                 base_model_builder : Callable[[gym.spaces.Space, gym.spaces.Space],th.nn.Module],
                 buffer_size, storage_torch_device, start_method = "forkserver",
                 session : session.Session = None):
        super().__init__(vec_env=None)
        self._buffer_size = buffer_size
        self._storage_torch_device = storage_torch_device
        self._vec_env_builder = CloudpickleWrapper(vec_env_builder)
        self._base_model_builder = CloudpickleWrapper(base_model_builder)
        ctx = mp_helper.get_context(method=start_method)
        self._commander = SimpleCommander(mp_context=ctx, n_envs=1, timeout_s=60)
        self._collect_args = ctx.Array(ctypes.c_uint64, 3, lock = False)
        self._running = ctx.Value(ctypes.c_bool)
        self._running.value = ctypes.c_bool(True)
        self._last_collect_wall_duration = ctx.Value(ctypes.c_float)
        self._last_collect_wall_duration.value = 0.0
        p1, p2 = ctx.Pipe()
        self._collector_process : mp.Process = ctx.Process(target = self._worker, args=(p2,session))
        self._collector_process.start()
        self._pipe = p1
        p2.close()

        time.sleep(5)
        ggLog.info(f"sending build req")
        self._commander.set_command("build")
        ggLog.info(f"waiting build")
        self._commander.wait_done(timeout=60)
        self._buffer, self._obs_space, self._action_space, self._num_envs, self._collector_model = self._pipe.recv()

    def observation_space(self):
        return self._obs_space
    
    def action_space(self):
        return self._action_space
    
    def num_envs(self):
        return self._num_envs

    def _worker(self, pipe, parent_session):
        ggLog.info(f"AsyncProcessExperienceCollector worker started with pid {os.getpid()}")
        session.default_session = parent_session
        session.default_session.reapply_globals()
        self._pipe = pipe
        while self._running.value:
            # ggLog.info(f"waiting command")
            cmd = self._commander.wait_command()
            # ggLog.info(f"got command {cmd}")
            if cmd == b"build":
                self._vec_env = self._vec_env_builder.var()
                self.reset()
                self._buffer = BasicStorage(buffer_size = self._buffer_size,
                                            observation_space=self._vec_env.single_observation_space,
                                            action_space=self._vec_env.single_action_space,
                                            n_envs=self._vec_env.num_envs,
                                            storage_torch_device=self._storage_torch_device,
                                            share_mem=True,
                                            allow_rollover=False)
                self._obs_space = self._vec_env.single_observation_space
                self._action_space = self._vec_env.single_action_space
                self._num_envs = self._vec_env.num_envs
                self._collector_model = self._base_model_builder.var(self._obs_space, self._action_space)
                self._pipe.send((self._buffer, 
                                 self._obs_space,
                                 self._action_space,
                                 self._num_envs,
                                 self._collector_model))
            elif cmd == b"collect":
                vsteps_to_collect, global_vstep_count, random_vsteps = self._collect_args
                self._buffer.clear()
                t0 = time.monotonic()
                self.collect_experience(policy=self._collector_model,
                                        vsteps_to_collect=vsteps_to_collect,
                                        global_vstep_count=global_vstep_count,
                                        random_vsteps=random_vsteps,
                                        policy_device=self._collector_model.device,
                                        buffer = self._buffer)
                self._last_collect_wall_duration.value = time.monotonic() - t0
            elif cmd == b"close":
                ggLog.warn(f"{type(self)}: closing")
                self._vec_env.close()
                self._running.value = ctypes.c_bool(False)
            elif cmd == None:
                pass # no command received, wait timed out
            else:
                ggLog.warn(f"{type(self)}: Unexpected command {cmd}")
            self._commander.mark_done()
            # ggLog.info(f" {cmd} done")
        ggLog.info(f"Collector worker terminating")

    def collect_experience_async(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._collect_args[:] = vsteps_to_collect, global_vstep_count, random_vsteps
        self._commander.set_command("collect")

    def wait_collection(self, timeout = 10.0):
        self._commander.wait_done(timeout=timeout)
        return self._buffer
    
    def collection_duration(self):
        return float(self._last_collect_wall_duration.value)
    
    def close(self):
        self._commander.set_command("close")
        self._collector_process.join(timeout=120)
        if self._collector_process.is_alive():
            self._collector_process.terminate()
            self._collector_process.join(timeout=30)
            if self._collector_process.is_alive():
                self._collector_process.kill()