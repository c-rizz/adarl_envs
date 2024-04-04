#!/usr/bin/env python3

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
from lr_gym.utils.buffers import ThDictReplayBuffer
from lr_gym.utils.ObsConverter import ObsConverter
from typing import List, Union, NamedTuple, Dict, Optional
from autoencoding_rl.utils import build_mlp_net
from lr_gym.utils.tensor_trees import map_tensor_tree
import lr_gym.utils.dbg.ggLog as ggLog
import copy
import threading
import lr_gym.utils.sigint_handler

class TransitionBatch(NamedTuple):
    observations : Union[th.Tensor, Dict]
    actions : th.Tensor
    next_observations : Union[th.Tensor, Dict]
    dones : th.Tensor
    rewards : th.Tensor

class QNetwork(nn.Module):
    def __init__(self, observation_space : gym.spaces.Space,
                 action_size : int,
                 q_network_arch : List[int],
                 torch_device : Union[str,th.device] = "cuda"):
        super().__init__()
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Not implemented yet")
        self._q_net = build_mlp_net(arch=q_network_arch,
                                     input_size=action_size + self._obs_converter.vector_part_size(),
                                     output_size=1).to(device=torch_device)

    def forward(self, observations, actions):
        observations = self._obs_converter.getVectorPart(observation_batch=observations)
        return self._q_net(torch.cat([observations, actions], 1))



class Actor(nn.Module):
    def __init__(self,  observation_space,
                        action_size,
                        policy_arch = [256,256],
                        action_max : Union[float, th.Tensor] = 1,
                        action_min : Union[float, th.Tensor] = -1,
                        log_std_max = 2,
                        log_std_min = -5,
                        torch_device : Union[str,th.device] = "cuda"):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Not implemented yet")
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.fc = build_mlp_net(arch=policy_arch[:-1],input_size=self._obs_converter.vector_part_size(), output_size=policy_arch[-1],
                                    last_activation_class=th.nn.LeakyReLU).to(device=torch_device)
        self.fc_mean = nn.Linear(policy_arch[-1], action_size, device=torch_device)
        self.fc_logstd = nn.Linear(policy_arch[-1], action_size, device=torch_device)

        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        # save action scaling factors as non-trained parameters
        self.register_buffer("action_scale", torch.as_tensor((action_max - action_min) / 2.0, dtype=torch.float32, device=torch_device))
        self.register_buffer("action_bias",  torch.as_tensor((action_max + action_min) / 2.0, dtype=torch.float32, device=torch_device))

    def forward(self, observations):
        observations = self._obs_converter.getVectorPart(observation_batch=observations)
        observations = self.fc(observations)
        mean = self.fc_mean(observations)
        log_std = self.fc_logstd(observations)
        log_std = (torch.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min
        return mean, log_std

    def sample_action(self, observations):
        mean, log_std = self(observations)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC(nn.Module):
    @dataclass
    class Hyperparams():
        q_lr : float
        policy_lr : float
        gamma : float
        auto_entropy_temperature : bool
        constant_entropy_temperature : Optional[float]
        action_size : int
        action_min : Union[float, th.Tensor]
        action_max : Union[float, th.Tensor]
        target_tau : float
        policy_update_freq : int
        targets_update_freq : int
        q_network_arch : List[int]
        policy_arch : List[int]
        torch_device : Union[str,th.device]
        target_entropy : Optional[float]

    def __init__(self,
                 observation_space,
                 action_size : int,
                 q_network_arch : List[int] = [256,256],
                 q_lr : float = 0.005,
                 policy_lr : float = 0.005,
                 policy_arch : List[int] = [256,256],
                 action_min : Union[float, th.Tensor] = -1.0,
                 action_max : Union[float, th.Tensor] = 1.0,
                 torch_device : Union[str,th.device] = "cuda",
                 auto_entropy_temperature : bool = True,
                 constant_entropy_temperature : Optional[float] = None,
                 target_entropy : Optional[float] = None,
                 gamma : float = 0.99,
                 target_tau = 0.005,
                 policy_update_freq = 2,
                 target_update_freq = 1):
        super().__init__()
        self._hp = SAC.Hyperparams(q_lr=q_lr,
                                   policy_lr = policy_lr,
                                   gamma=gamma,
                                   auto_entropy_temperature=auto_entropy_temperature,
                                   constant_entropy_temperature=constant_entropy_temperature,
                                   action_size=action_size,
                                   action_min = action_min,
                                   action_max = action_max,
                                   target_tau = target_tau,
                                   policy_update_freq=policy_update_freq,
                                   targets_update_freq=target_update_freq,
                                   q_network_arch = q_network_arch,
                                   policy_arch = policy_arch,
                                   torch_device = torch_device,
                                   target_entropy = target_entropy)
        self.device = torch_device
        self._value_func_updates = 0
        self._policy_updates = 0
        self._q_net1 = QNetwork(observation_space=observation_space,
                                action_size=self._hp.action_size,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device)
        self._q_net2 = QNetwork(observation_space=observation_space,
                                action_size=self._hp.action_size,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device)
        self._q_net1_target = QNetwork(observation_space=observation_space,
                                action_size=self._hp.action_size,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device)
        self._q_net2_target = QNetwork(observation_space=observation_space,
                                action_size=self._hp.action_size,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device)
        self._q_net1_target.load_state_dict(self._q_net1.state_dict())
        self._q_net2_target.load_state_dict(self._q_net2.state_dict())
        self._q_optimizer = optim.Adam(list(self._q_net1.parameters()) + list(self._q_net2.parameters()), lr=self._hp.q_lr)
        self._actor = Actor(observation_space = observation_space,
                            policy_arch=policy_arch,
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.torch_device)
        self._actor_optimizer = optim.Adam(list(self._actor.parameters()), lr=self._hp.policy_lr)
        if self._hp.auto_entropy_temperature:
            if self._hp.target_entropy is None:
                self._target_entropy = -self._hp.action_size
            else:
                self._target_entropy = self._hp.target_entropy
            self._log_alpha = torch.zeros(1, requires_grad=True, device=torch_device)
            self._alpha = self._log_alpha.exp().item()
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = constant_entropy_temperature

    def sample_actions(self, observation):
        action, log_prob, mean = self._actor.sample_action(observation)
        return action

    def _update_value_func(self, transitions : TransitionBatch):
        with torch.no_grad():
            # Compute next-values for TD
            next_state_actions, next_state_log_pi, _ = self._actor.sample_action(transitions.next_observations)
            q_next = torch.min( self._q_net1_target(transitions.next_observations, next_state_actions),
                                self._q_net1_target(transitions.next_observations, next_state_actions))
            soft_q_next = q_next - self._alpha * next_state_log_pi
            td_q_values = transitions.rewards.flatten() + (1 - transitions.dones.flatten()) * self._hp.gamma * (soft_q_next).view(-1)

        q_values1 = self._q_net1(transitions.observations, transitions.actions).view(-1)
        q_values2 = self._q_net2(transitions.observations, transitions.actions).view(-1)
        qf1_loss = F.mse_loss(q_values1, td_q_values)
        qf2_loss = F.mse_loss(q_values2, td_q_values)
        qf_loss = qf1_loss + qf2_loss

        self._q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self._q_optimizer.step()
        self._value_func_updates += 1

    def _update_policy(self, transitions : TransitionBatch):
        pi, log_pi, _ = self._actor.sample_action(transitions.observations)
        min_q_pi = torch.min(self._q_net1(transitions.observations, pi), self._q_net2(transitions.observations, pi))
        actor_loss = ((self._alpha * log_pi) - min_q_pi).mean()

        self._actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._actor_optimizer.step()

        if self._hp.auto_entropy_temperature:
            with torch.no_grad():
                _, log_pi, _ = self._actor.sample_action(transitions.observations)
            alpha_loss = (-self._log_alpha.exp() * (log_pi + self._target_entropy)).mean()

            self._alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._alpha = self._log_alpha.exp().item()
        self._policy_updates += 1

    @staticmethod
    def _target_update(param, target_param, tau):
            if tau == 1:
                target_param.data.copy_(param.data)
            else:
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
    def _update_target_nets(self):
        for param, target_param in zip(self._q_net1.parameters(), self._q_net1_target.parameters()):
            self._target_update(param, target_param, self._hp.target_tau)
        for param, target_param in zip(self._q_net2.parameters(), self._q_net2_target.parameters()):
            self._target_update(param, target_param, self._hp.target_tau)

    def update(self, transitions : TransitionBatch):
        self._update_value_func(transitions = transitions)
        if self._value_func_updates % self._hp.policy_update_freq == 0:
            for _ in range(self._hp.policy_update_freq):
                self._update_policy(transitions=transitions)
        if self._value_func_updates % self._hp.targets_update_freq == 0:
            self._update_target_nets()

class ExperienceCollector():
    def __init__(self, vec_env):
        self._vec_env = vec_env
        self._last_obs = None

    def reset(self):
        self._last_obs, info = self._vec_env.reset()

    def collect_experience(self, policy, vsteps_to_collect, global_vstep_count, random_vsteps, policy_device, buffer):
        if  self._last_obs is None:
            raise RuntimeError(f"last_obs is not set. reset() should be called before running collect_experience the first time")
        obs = self._last_obs
        num_envs = self._vec_env.unwrapped.num_envs
        for step in range(vsteps_to_collect):
            if global_vstep_count < random_vsteps:
                actions = np.array([self._vec_env.single_action_space.sample() for _ in range(num_envs)])
            else:
                th_obs = map_tensor_tree(obs, lambda a: th.as_tensor(a, device = policy_device))
                actions = policy.sample_actions(th_obs)
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = self._vec_env.step(actions)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

            obs = next_obs
            self._last_obs = obs



def train(vec_env : gym.vector.VectorEnv,
          model : SAC,
          buffer : ThDictEpReplayBuffer,
          total_timesteps : int,
          train_freq : int,
          learning_starts : int,
          grad_steps : int,
          batch_size : int,
          log_freq : int = -1,
          parallelize_collection : bool = True):
    if log_freq == -1: log_freq = train_freq
    num_envs = vec_env.num_envs
    collector = ExperienceCollector(vec_env)
    collector.reset()
    global_step = 0
    t_coll_sl = 0
    t_train_sl = 0
    t_tot_sl = 0
    if parallelize_collection:
        collection_model = copy.deepcopy(model)
    else:
        collection_model = model
    while global_step < total_timesteps:
        s0b = buffer.stored_frames()
        t0 = time.monotonic()
        steps_to_collect = train_freq*num_envs
        vsteps_to_collect = train_freq
        def collector_func():
            collector.collect_experience(collection_model,
                                     vsteps_to_collect=vsteps_to_collect,
                                     global_vstep_count=int(global_step/num_envs),
                                     random_vsteps=learning_starts,
                                     policy_device=model.device,
                                     buffer=buffer)
            t_coll = time.monotonic()
            nonlocal t_coll_sl
            t_coll_sl += t_coll - t0
        if parallelize_collection:
            collection_model.load_state_dict(model.state_dict())
            collector_thread = threading.Thread(target=collector_func)
            collector_thread.start()
        else:
            collector_func()

        t_before_train = time.monotonic()
        if global_step > learning_starts:
            for _ in range(grad_steps):
                data = buffer.sample(batch_size)
                model.update(transitions = data)
        
        if parallelize_collection:
            collector_thread.join()

        if buffer.stored_frames()-s0b != steps_to_collect and not buffer.full:
            raise RuntimeError(f"Expected to collect {steps_to_collect} but got {buffer.stored_frames()-s0b}")
        global_step += steps_to_collect

        tf = time.monotonic()
        t_train_sl += tf - t_before_train
        t_tot_sl += tf-t0
        if global_step/num_envs % log_freq == 0:
            ggLog.info(f"TRAIN: stps:{global_step} coll={t_coll_sl:.2f}s train={t_train_sl:.2f}s tot={t_tot_sl:.2f}")
            t_train_sl, t_coll_sl, t_tot_sl = 0,0,0
            lr_gym.utils.sigint_handler.haltOnSigintReceived()



def main():


    seed = 0
    log_folder = lr_gym.utils.session.lr_gym_startup(   __file__,
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
    builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(num_envs)]
    envs = AsyncVectorEnvShmem(builders, context="forkserver")
    envs = VectorEnvLogger(env = envs)

    model = SAC(observation_space=envs.single_observation_space,
                action_size=int(np.prod(envs.single_action_space.shape)),
                q_network_arch=[256,256],
                q_lr=0.005,
                policy_arch=[256,256],
                action_min = th.as_tensor(envs.single_action_space.low),
                action_max = th.as_tensor(envs.single_action_space.high),
                torch_device=device,
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=0.99,
                target_tau = 0.005,
                policy_update_freq=2,
                target_update_freq=1)

    compiled_model = th.compile(model)
    envs.single_observation_space.dtype = np.float32
    # rb = ThDictReplayBuffer(
    #     buffer_size=1000_000,
    #     observation_space=envs.single_observation_space,
    #     action_space=envs.single_action_space,
    #     device=device,
    #     storage_torch_device=device,
    #     handle_timeout_termination=False,
    #     n_envs=num_envs,
    #     disable_validation_set=True)
    rb = ThDictReplayBuffer(
        buffer_size=1000_000,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        storage_torch_device=device,
        handle_timeout_termination=False,
        n_envs=num_envs)
    start_time = time.time()


    train(vec_env=envs,
          model = compiled_model,
          buffer = rb,
          total_timesteps=10_000_000,
          train_freq = 50,
          learning_starts=500*32,
          grad_steps=50,
          batch_size=16384,
          log_freq=500,
          parallelize_collection = False)

    envs.close()
    # writer.close()


if __name__ == "__main__":
    main()
