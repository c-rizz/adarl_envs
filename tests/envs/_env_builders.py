"""Builds the real LocomotionVecEnv and GraspVecEnv for the tests, and helpers to edit their states.

The configs are the ones of the scripts that train these envs (loco_kyon_ppo and solve_grasp_env),
captured by stubbing out their training call, so the tests follow whatever those scripts set. Only
two things are changed: what the MuJoCo CPU adapter does not implement is turned off (it has no
alter_model, so the model randomizations get no spread, and no link impulses), and the reward
weights a test asks for are set so that those terms show up in the sub-rewards.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable
from unittest import mock

import torch as th

# Turns off what MujocoAdapter does not implement: with no spread the randomization blocks are
# dropped, so nothing calls alter_model, and with no impulses nothing sets link impulses
MUJOCO_CPU_UNSUPPORTED = {
    "impulse_probability_per_sec" :                 0.0,
    "randomized_com_xyz_diff_distribution" :        ("normal", ([0.,0.,0.], [0.,0.,0.])),
    "randomized_dof_armature_ratios" :              ("uniform", [1.0, 1.0]),
    "randomized_dof_damping_ratios" :               ("uniform", [1.0, 1.0]),
    "randomized_dof_frictionloss_ratios" :          ("uniform", [1.0, 1.0]),
    "randomized_friction_slide_spin_roll_ratios" :  ("uniform", ([1.,1.,1.], [1.,1.,1.])),
    "randomized_mass_ratios" :                      ("normal", (1.0, 0.0)),
    "randomized_object_mass_ratios" :               None,
    "randomized_object_scale_ratios" :              None,
}


@dataclass
class BuiltEnv:
    runner : Any                            # the adarl EnvRunner
    env : Any                               # the adarl env under the runner
    reset_state : dict[str, th.Tensor]      # the state right after the first reset, never modified

    def fresh_state(self) -> dict[str, th.Tensor]:
        return {k: v.clone() for k, v in self.reset_state.items()}


def _capture_env_builder_args(run_function : Callable, train_module, train_function_name : str,
                              args : dict, folder : str) -> dict:
    captured = {}
    def fake_train(*a, **kwargs):
        captured.update(copy.deepcopy(kwargs["env_builder_args"]))
    with mock.patch.object(train_module, train_function_name, fake_train):
        run_function(0, folder, None, 0, args)
    return captured


def _disable_mujoco_cpu_unsupported(env_builder_args : dict):
    env_builder_args.update({k: v for k, v in MUJOCO_CPU_UNSUPPORTED.items() if k in env_builder_args})


def _reset_and_wrap(runner) -> BuiltEnv:
    import adarl.utils.session
    # Session.initialize creates it in a real run; LocomotionVecEnv.post_step writes into it
    adarl.utils.session.default_session.run_info.setdefault("extras", {})
    runner.reset()
    env = runner.get_base_env()
    return BuiltEnv(runner=runner, env=env, reset_state={k: v.clone() for k, v in env.get_states().items()})


def build_loco_env(folder : str, reward_weights : dict[str, float]) -> BuiltEnv:
    """The kyon LocomotionVecEnv of loco_kyon_ppo, single env, on the CPU."""
    import rreal.algorithms.ppo2 as ppo2
    from adarl_envs.experiments import loco_kyon_ppo
    from adarl_envs.experiments.loco_builder import robot_args_registry, loco_runner_builder
    args = _capture_env_builder_args(loco_kyon_ppo.runFunction, ppo2, "ppo_train",
                                     {"mode": "mujoco", "robot": "kyon", "algorithm": "ppo_small",
                                      "position_delta": False, "feet": False, "arms": False,
                                      "world": "flat_ground"},
                                     folder)
    full_args = robot_args_registry[args["robot_model"]](robot_options=args["robot_options"])
    full_args.update(args)
    full_args["th_device"] = th.device("cpu")
    _disable_mujoco_cpu_unsupported(full_args)
    full_args.update(reward_weights)
    return _reset_and_wrap(loco_runner_builder(seed=0, run_folder=folder, num_envs=1,
                                               env_builder_args=full_args, quiet=True))


def build_grasp_env(folder : str, reward_weights : dict[str, float]) -> BuiltEnv:
    """The kyon GraspVecEnv of solve_grasp_env, single env, no observation camera, on the CPU."""
    import rreal.algorithms.sac_helpers as sac_helpers
    from adarl_envs.experiments import solve_grasp_env
    from adarl_envs.experiments.grasp_builder import runner_builder
    args = _capture_env_builder_args(solve_grasp_env.runFunction, sac_helpers, "sac_train",
                                     {"mode": "mujoco", "robot": "kyon", "algorithm": "sac",
                                      "obs_cam": False, "no_wandb": True},
                                     folder)
    _disable_mujoco_cpu_unsupported(args)
    args.update(reward_weights)
    return _reset_and_wrap(runner_builder(seed=0, run_folder=folder, num_envs=1,
                                          env_builder_args=args, quiet=True))


def field_index(env, substate : str, field) -> int:
    """Index of a field in a substate, as the state helper lays it out (not the enum value)."""
    return int(env._state_helper.sub_helpers[substate].field_idx(field)[0])


def sub_rewards(env, state : dict[str, th.Tensor]) -> dict[str, float]:
    """The weighted sub-rewards of env 0 for this state."""
    sub = {}
    env.compute_rewards(state, sub_rewards_return=sub)
    return {k: float(v[0]) for k, v in sub.items()}
