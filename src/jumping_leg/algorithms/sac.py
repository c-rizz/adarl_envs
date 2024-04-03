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
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch as th
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from lr_gym.utils.async_vector_env import AsyncVectorEnvShmem
import inspect
import lr_gym.utils.session
from lr_gym.envs.vector_env_logger import VectorEnvLogger
from lr_gym.utils.sb3_buffers import ThDictReplayBuffer
from lr_gym.utils.ObsConverter import ObsConverter
from typing import List, Union, NamedTuple, Dict, Optional
from autoencoding_rl.utils import build_mlp_net



class TransitionBatch(NamedTuple):
    observations : Union[th.Tensor, Dict]
    actions : th.Tensor
    next_observations : Union[th.Tensor, Dict]
    dones : th.Tensor
    rewards : th.Tensor

class QNetwork(nn.Module):
    def __init__(self, observation_space : gym.spaces.Space,
                 q_network_arch : List[int]):
        super().__init__()
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Not implemented yet")
        self._q_net = build_mlp_net(arch=q_network_arch,
                                     input_size=self._action_size + self._obs_converter.vector_part_size(),
                                     output_size=1)

    def forward(self, observations, actions):
        observations = self._obs_converter.getVectorPart(observation_batch=observations)
        return self._q_net(torch.cat([observations, actions], 1))



class Actor(nn.Module):
    def __init__(self,  policy_input_size,
                        action_size,
                        policy_arch = [256,256],
                        action_max : Union[float, th.Tensor] = 1,
                        action_min : Union[float, th.Tensor] = -1,
                        log_std_max = 2,
                        log_std_min = -5):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Not implemented yet")
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.fc = build_mlp_net(arch=policy_arch[:-1],input_size=policy_input_size, output_size=policy_arch[-1],
                                    last_activation_class=th.nn.LeakyReLU)
        self.fc_mean = nn.Linear(policy_arch[-1], action_size)
        self.fc_logstd = nn.Linear(policy_arch[-1], action_size)

        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        # save action scaling factors as non-trained parameters
        self.register_buffer("action_scale", torch.as_tensor((action_max - action_min) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.as_tensor((action_max + action_min) / 2.0, dtype=torch.float32))

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

    def __init__(self,
                 observation_space,
                 action_size : int,
                 q_network_arch : List[int] = [256,256],
                 q_lr : float = 0.005,
                 policy_arch : List[int] = [256,256],
                 action_min : Union[float, th.Tensor] = -1.0,
                 action_max : Union[float, th.Tensor] = 1.0,
                 torch_device : Union[str,th.device] = "cuda",
                 auto_entropy_temperature : bool = True,
                 constant_entropy_temperature : Optional[float] = None,
                 gamma : float = 0.99,
                 target_tau = 0.005,
                 policy_update_freq = 2,
                 target_update_freq = 1):
        self._hp = SAC.Hyperparams(q_lr=q_lr,
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
                                   policy_arch = policy_arch)
        self._value_func_updates = 0
        self._policy_updates = 0
        self._q_net1 = QNetwork(observation_space=observation_space,
                                q_network_arch=q_network_arch)
        self._q_net2 = QNetwork(observation_space=observation_space,
                                q_network_arch=q_network_arch)
        self._q_net1_target = QNetwork(observation_space=observation_space,
                                q_network_arch=q_network_arch)
        self._q_net2_target = QNetwork(observation_space=observation_space,
                                q_network_arch=q_network_arch)
        self._q_net1_target.load_state_dict(self._q_net1.state_dict())
        self._q_net2_target.load_state_dict(self._q_net2.state_dict())
        self._q_optimizer = optim.Adam(list(self._q_net1.parameters()) + list(self._q_net2.parameters()), lr=self._hp.q_lr)
        self._actor = Actor(policy_input_size=self._obs_converter.vector_part_size(),
                            policy_arch=policy_arch,
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max)
        if self._hp.auto_entropy_temperature:
            self._target_entropy = -self._hp.action_size
            self._log_alpha = torch.zeros(1, requires_grad=True, device=torch_device)
            # self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = constant_entropy_temperature

    def sample_action(self, observation):
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
        actor_loss = ((alpha * log_pi) - min_q_pi).mean()

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
            alpha = self._log_alpha.exp().item()
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

    def train(self, transitions : TransitionBatch):
        self._update_value_func(transitions = transitions)
        if self._value_func_updates % self._hp.policy_update_freq == 0:
            for _ in range(self._hp.policy_update_freq):
                self._update_policy(transitions=transitions)
        if self._value_func_updates % self._hp.targets_update_freq == 0:
            self._update_target_nets()




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 16384
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 0.005
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.005
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
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
        "quiet" : False
                        }
    def thunk():
        if capture_video:
            raise NotImplementedError()
        env = env_builder(log_folder=f"lrg/cleanrl/sac/{run_name}",
                            seed=seed+idx,
                            env_builder_args = env_builder_args,
                            no_dict=True)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk



def main():


    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_folder = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=args.seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_comment="")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    num_envs = 16
    envs = AsyncVectorEnvShmem([make_env(None, seed, i, False, run_name) for i in range(num_envs)],
                                     context="forkserver")
    envs = VectorEnvLogger(env = envs)
    print(f"envs = {envs.single_observation_space}")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    model = SAC(observation_space=envs.single_observation_space,
                action_size=int(np.prod(envs.single_action_space.shape)),
                q_network_arch=[256,256],
                q_lr=0.005,
                policy_arch=[256,256],
                action_min = th.as_tensor(envs.single_action_space.low),
                action_max = th.as_tensor(envs.single_action_space.high),
                torch_device="cuda",
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=0.99,
                target_tau = 0.005,
                policy_update_freq=2,
                target_update_freq=1)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=num_envs
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(args.total_timesteps):

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = model.sample_action(obs)
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            model.train(transitions = data)

    envs.close()
    # writer.close()


if __name__ == "__main__":
    main()
