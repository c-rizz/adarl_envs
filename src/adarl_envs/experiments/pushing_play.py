#!/usr/bin/env python3
"""Sanity-check script for the simplified pushing environment.

Builds the environment and drives it with either random actions or a scripted pushing policy,
printing per-step diagnostics. Useful to verify the scene, the end-effector position control and the
reward before hooking the environment up to a training run.

    python -m adarl_envs.experiments.pushing_play --envs 4 --steps 200 --policy scripted
"""
from __future__ import annotations

import argparse
import time
import torch as th

import adarl.utils.dbg.ggLog as ggLog
from adarl_envs.env.PushingVecEnv import PushingVecEnv
from adarl_envs.experiments.pushing_builder import runner_builder


def scripted_actions(env : PushingVecEnv, states : dict[str, th.Tensor]) -> th.Tensor:
    """A crude scripted pusher: go behind the cube (w.r.t. the goal), then push it towards the goal.

    It uses the privileged state, not the observation, so it works with the camera-based observation too.
    """
    idx = PushingVecEnv.VEC_STATE_IDX
    vec = states["vec"]
    tip_xy = vec[:,[idx.TIP_X, idx.TIP_Y]]
    cube_xy = vec[:,[idx.CUBE_X, idx.CUBE_Y]]
    goal_xy = vec[:,[idx.GOAL_X, idx.GOAL_Y]]
    cube2goal = goal_xy - cube_xy
    direction = cube2goal/th.clamp(th.linalg.vector_norm(cube2goal, dim=-1, keepdim=True), min=1e-6)
    standoff = env._cube_size*0.75 + env._ee_diameter/2
    approach_xy = cube_xy - direction*standoff
    behind_the_cube = th.linalg.vector_norm(tip_xy - approach_xy, dim=-1, keepdim=True) < 0.015
    target_xy = th.where(behind_the_cube, goal_xy, approach_xy)
    return th.clamp((target_xy - tip_xy)/env._max_position_change, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--policy", type=str, default="scripted", choices=["scripted","random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-camera", action="store_true", help="observe the cube pose instead of the camera image")
    parser.add_argument("--video", action="store_true", help="record a video of environment 0")
    parser.add_argument("--run-folder", type=str, default=f"./pushing_play/{int(time.time())}")
    args = parser.parse_args()

    th_device = th.device(args.device)
    env_builder_args = {"th_device" : th_device,
                        "observe_camera" : not args.no_camera,
                        "enable_rendering" : True,
                        "video_save_freq" : 1 if args.video else -1,
                        "record_video" : args.video}
    runner = runner_builder(seed=args.seed,
                            run_folder=args.run_folder,
                            num_envs=args.envs,
                            env_builder_args=env_builder_args,
                            autoreset=True,
                            quiet=False)
    env : PushingVecEnv = runner.get_base_env() # type: ignore

    runner.reset(seed=args.seed)
    episodes = 0
    successes = 0.0
    t0 = time.monotonic()
    for step in range(args.steps):
        if args.policy == "random":
            actions = (th.rand((args.envs, 2), device=th_device)*2-1)
        else:
            actions = scripted_actions(env, env.get_states())
        obs, next_obs, rewards, terminated, truncated, infos, next_infos, _ = runner.step(actions)
        # NOTE: the runner's 8th return value (reinit_done) is aliased to a tensor that reinit_envs()
        # zeroes in place, so it always reads back as all-False. Use terminated/truncated instead.
        ended = th.logical_or(terminated, truncated)
        if step % 10 == 0:
            ggLog.info(f"step {step:4d} "
                       f"reward = {rewards.mean().item(): .3f} "
                       f"cube2goal = {infos['cube2goal_dist'].mean().item():.3f} "
                       f"cube2tip = {infos['cube2tip_dist'].mean().item():.3f} "
                       f"ee_err = {infos['ee_tracking_error'].max().item():.4f}")
        if bool(ended.any()):
            episodes += int(ended.sum())
            successes += float(infos["success"][ended].sum())
    elapsed = time.monotonic()-t0
    ggLog.info(f"Ran {args.steps} steps on {args.envs} envs in {elapsed:.1f}s "
               f"({args.steps*args.envs/elapsed:.1f} env-steps/s)")
    ggLog.info(f"Completed {episodes} episodes, success rate = {successes/episodes:.2f}" if episodes > 0
               else "No episode completed")
    runner.close()


if __name__ == "__main__":
    main()
