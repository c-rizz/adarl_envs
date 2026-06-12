#!/usr/bin/env python3
from __future__ import annotations
import time
import inspect
import copy
from adarl.utils.buffers import BaseBuffer
import adarl.utils.dbg.dbg_img
from rreal.algorithms.rl_agent import load_agent
from rreal.algorithms.sac import SAC, compare_dicts
from rreal.algorithms.sac_helpers import EnvBuilderProtocol
from rreal.algorithms.ppo2 import PPO
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import numpy as np
import gymnasium as gym

import os
import torch as th
import adarl.utils.session
import adarl.utils.dbg.dbg_img as dbg_img
from adarl.utils.keyboard_listener import KeyboardListener
from adarl.utils.tensor_trees import map_tensor_tree, TensorTree
import adarl.utils.sigint_handler
from adarl_envs.experiments.grasp_builder import centgrasp_singleenv_builder
from adarl_envs.env.GraspVecEnv import GraspVecEnv
from rreal.algorithms.rl_agent import RLAgent, TransitionBatch
from adarl.utils.base_utils import record_time, clear_recorded_times, print_recorded_times, isinstance_noimport, set_disable_clear_recorded_times

import adarl.utils.dbg
from typing import Any
from ctypes.util import find_library
import readline
import math
from rreal.algorithms.hardcoded_policies import FixedPolicy, SinPolicy, RandPolicy


def get_action_len(env) -> int:
    base_env = env.get_runner().get_base_env()
    return base_env._action_helper.single_action_len()


def get_home_action(env, device):
    # In position_delta control mode a zero action means "hold the current pose".
    return th.zeros((get_action_len(env),), device=device)


def build_fixed_policy(env, scale : float = 0.0, device : th.device = th.device("cpu")):
    return FixedPolicy(  cmd = get_home_action(env, device))


def build_sin_policy(env, scale : float = 1.0, device = th.device("cpu")):
    home_action = get_home_action(env, device)
    action_len = len(home_action)
    act_range = th.ones((action_len,), device=device)
    return SinPolicy(  act_scale=act_range*scale,
                        act_offset=home_action,
                        act_speed=th.as_tensor([0.8], device = device),
                        action_size=action_len,
                        dt=0.05)


def build_rand_policy(env, scale : float = 1.0, device : th.device = th.device("cpu")):
    action_len = get_action_len(env)
    act_range = th.ones((action_len,), device=device)
    return RandPolicy( act_scale=act_range*scale,
                        action_size=action_len)


class KeyboardJointPolicy(RLAgent):
    """Lets the user drive the controlled joints from the keyboard.

    Number keys 1-7 increase joints 1-7, the letters just below (q w e r t y u)
    decrease them. The produced action is a per-joint position delta (clamped to
    the normalized [-1,1] range expected by the position_delta control mode).
    """
    INC_KEYS = ["1","2","3","4","5","6","7"]
    DEC_KEYS = ["q","w","e","r","t","y","u"]

    def __init__(self, action_size : int, step : float = 0.5, device : th.device = th.device("cpu")):
        self._action_size = action_size
        self._step = step
        self._device = device
        self._next_action = th.zeros((action_size,), device=device)

    def set_action_from_keys(self, keyboard_listener : KeyboardListener):
        action = th.zeros((self._action_size,), device=self._device)
        for i in range(min(self._action_size, len(self.INC_KEYS))):
            inc = keyboard_listener.get_key_press_count(self.INC_KEYS[i])
            dec = keyboard_listener.get_key_press_count(self.DEC_KEYS[i])
            action[i] = float(inc - dec) * self._step
        self._next_action = th.clamp(action, -1.0, 1.0)

    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict = None):
        return self._next_action.clone()

    def get_hidden_state(self):
        return None

    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        self._next_action = th.zeros((self._action_size,), device=self._device)

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass

    def load_(self, path: str):
        pass

    def input_device(self):
        return self._device


def build_keyboard_policy(env, step : float = 0.5, device : th.device = th.device("cpu")):
    return KeyboardJointPolicy(action_size=get_action_len(env), step=step, device=device)



from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
def find_recorder_wrapper(env) -> EnvRunnerRecorderWrapper | None:
    from adarl.envs.vec.Runner2GymWrapper import Runner2GymWrapper
    from adarl.envs.vec.EnvRunnerWrapper import EnvRunnerWrapper
    recorder = None
    if isinstance(env, Runner2GymWrapper):
        recorder = env
        while recorder is not None and not isinstance(recorder, EnvRunnerRecorderWrapper):
            if isinstance(recorder, EnvRunnerWrapper):
                recorder = recorder.get_base_runner()
            else:
                recorder = None
    return recorder




def grasp_builder_and_args():
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode = 500 #int(ep_duration_sec/step_length_sec)
    mode = args["mode"].lower()

    if mode == "pybullet":
        env_device = th.device("cpu", 0)
    elif mode == "mujoco":
        env_device = th.device("cpu", 0)
    elif mode == "mjx":
        env_device = th.device("cuda", 0)
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")

    height_pixels = args["resolution"]
    pixel_resolution = (height_pixels, int(height_pixels*16/9))
    record = args["record"] or args["verbose_recording"]
    skip_optionals = not args["verbose_recording"]

    r = 0.0  # randomization strength
    n = 0.0  # noise strength
    env_builder_args = {
        "reward_health_weight" : 0.0,

        "reward_joint_power_weight" : 0.0,
        "reward_joint_actacc_weight" :  0.0,
        "reward_joint_actdiff_weight" : 0.0,
        "reward_joint_torque_weight" :  0.0,
        "reward_joint_position_limit_weight" : 0.0,
        "reward_object_pose_weight" : 0.0,
        "reward_gripper_pose_weight" : 5.0,
        "reward_safety_weight" : 0.0,

        "target_object_link" : ("cube","cube"),
        "gripper_links" : [("centauro","dagana_1_fixed_palm_center"), ("centauro","dagana_1_claw_palm_center")],
        "observe_object_pose" : True,

        "noise_action_delay_mustd_std" : (0.008, 0.005*n, 0.0025*n),
        "noise_action_mustd" : (0.0,   0.0),
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position_delta",
        "control_mode_position_delta_max" : 0.05,
        "desired_foot_clearance" : 0.05,
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "enable_reference_filter" : True,
        "fail_on_safety" : False,
        "frame_stack_length" : 3,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.79,0.79],
        "goal_resampling_probability_per_sec" : 0.1,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "goal_yaw_vel_zero_ratio" : 0.25,
        "goal_yaw_vel_minmax" : (-1.0, 1.0),
        "held_joints_damping" : 20.0,
        "held_joints_stiffness" : 2000.0,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 0.7,
        "randomization_initial_height_range_meters" : 0.1,
        "randomization_initial_joint_pose_range" : 0.1,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 1.0, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "step_max_good_air_duration" : 0.5,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "step_min_good_air_duration" : 0.1,
        "mode" : mode,
        "noise_abs_obs_angvel_ep_mustd_step_std" :      [0.0, 0.02*n,  0.1*n],
        "noise_abs_obs_gravity_ep_mustd_step_std" :     [0.0, 0.0*n,   0.1*n],
        "noise_abs_obs_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.1*n],
        "noise_abs_obs_linacc_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_linvel_ep_mustd_step_std" :      [0.0, 0.0,     0.0],
        "noise_abs_obs_posz_ep_mustd_step_std" :        [0.0, 0.0,     0.0],
        "observe_full_robot_state" : False,
        "offset_envs_ep_starts" : False,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomized_dof_armature_ratios" :      0.05*r,
        "randomized_dof_frictionloss_ratios":   0.1*r,
        "randomized_dof_damping_ratios":        0.2*r,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_gains_damping_ratio_epstd"       : 0.2*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.2*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_reference_filter_distribution" : ("uniform", (20.0, 50.0)),
        "record_video" : True,
        "randomization_recycle_init_pose" : True,
        "robot_model" : args["robot"],
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : False,
        "stepLength_sec" : step_length_sec,
        "terminate_on_safety" : False,
        "terminate_on_crash" : True,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0,
        "minimal_infos" : True,
        "playground_style_reward" : False
    }

    env_builder_args.update({
        "enable_rendering" : True,
        "record_video" : True,
        "verbose_infos" : (not skip_optionals) or record,
        "minimal_infos" : skip_optionals or not record,
        "video_save_freq" : 1 if record else 0,
        "log_info_stats" : (not skip_optionals) or record,
        "noise_action_delay_mustd_std" : (0.0,0.0,0.0),
        "noise_action_mustd" : (0.0,0.0),
        "noise_abs_obs_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "noise_abs_obs_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "noise_abs_obs_linacc_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "noise_abs_obs_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "noise_abs_obs_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.0),
        "noise_abs_obs_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.0),
        "ui_camera_resolution_hw" : pixel_resolution,
        "randomization_initial_joint_pose_range" : 0.0,
        "randomization_initial_height_range_meters" : 0.0,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_dof_frictionloss_ratios"         : 0.0*r,
        "randomized_dof_damping_ratios"              : 0.0*r,
        "randomized_gains_damping_ratio_epstd"       : 0.0*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.0*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_dof_armature_ratios" : 0.0*r,
        "randomized_reference_filter_distribution" : ("uniform", (20.0, 50.0)),
        "impulse_probability_per_sec" : 0.0,
        "show_gui" : args["gui"],
        "goal_resampling_probability_per_sec" : 0.0,
        "walltime_factor" : args["rt_factor"],
    })
    return centgrasp_singleenv_builder, env_builder_args, step_length_sec




def runFunction(seed, folderName, resumeModelFile, run_id, args):
    builder, env_builder_args, step_length_sec = grasp_builder_and_args()

    return play(seed,
                folderName,
                run_id, args,
                env_builder = builder,
                env_builder_args = env_builder_args,
                step_length_sec = step_length_sec,
                render=args["publishimg"],
                deterministic = args["deterministic"])



def play(seed, folderName, run_id, args,
         env_builder : EnvBuilderProtocol,
         env_builder_args : dict[str,Any],
         step_length_sec : float, render : bool,
         deterministic : bool):

    ggLog.info(f"Starting run")
    if render:
        adarl.utils.dbg.dbg_img.helper.enable_web_dbg(True)
    log_folder, session = adarl.utils.session.adarl_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        folderName=folderName,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        use_wandb=False)

    if args["agent_device"].lower() == "cpu":
        device = th.device("cpu")
    elif args["agent_device"].lower() == "cuda":
        device = adarl.utils.utils.torch_selectBestGpu()
    else:
        device = th.device(args["agent_device"])
    env_device = env_builder_args["th_device"]
    ggLog.info("Building env...")

    env, fps = env_builder( log_folder=log_folder+"/eval",
                            seed=seed+100000000,
                            env_builder_args = env_builder_args,
                            is_eval=False)
    recorder = find_recorder_wrapper(env)
    ggLog.info("Built")
    control_mode = args["control"].lower().strip()
    manual = control_mode == "manual"
    if control_mode=="pretrained":
        model = load_agent(args["model"], device)
        if isinstance(model, SAC):
            trained_env_builder_args = model._init_args["init_hparams"].reference_init_args["env_builder_args"]
            try:
                equal, diffs = compare_dicts(env_builder_args, trained_env_builder_args)
                if not equal:
                    ggLog.warn(f"Loaded model was trained with different env args: \n{diffs}")
            except Exception as e:
                ggLog.warn(f"Could not compare env args with trained model: {type(e)}: {e}")
        elif isinstance(model, PPO):
            pass
    elif control_mode=="fixed":
        model = build_fixed_policy(env = env, device= env_device)
    elif control_mode=="random":
        model = build_rand_policy(env=env, scale=1.0, device = env_device)
    elif control_mode == "sine":
        model = build_sin_policy(env, scale = 1.0, device = env_device)
    elif control_mode == "manual":
        model = build_keyboard_policy(env, device = env_device)
    else:
        raise RuntimeError(f"Unknown control mode '{control_mode}'")

    play = True
    verbose = False
    interactive = False
    rewards = []
    durations = []

    keyboard_listener : KeyboardListener = None

    if render:
        img = env.render()
        dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
    try:
        while play:
            cmd = None
            options = {}
            if args["evaluate"] is None:
                while cmd != "c":
                    print(  f"\n"
                            f"Enter:\n"
                            f"  - 'c' to start an episode.\n"
                            f"  - 'interactive' to enter interactive control.\n"
                            f"  - 'quit' to quit:\n")
                    cmd = input(f" > ")
                    if cmd == "quit":
                        play = False
                        break
                    elif cmd == "interactive" and not manual:
                        print(f" Use WASD to move the goal in xy, RF to move it in z, IJKL/UO to move the camera, T to terminate.")
                        time.sleep(1)
                        interactive = True
                        options["max_ep_steps"] = 100_000
                        keyboard_listener = KeyboardListener()
                        cmd = 'c'
                if not play:
                    break
            else:
                if session.run_info["collected_episodes"].value >= args["evaluate"]:
                    break
            if not play:
                break
            if manual and keyboard_listener is None:
                print(f" Manual joint control:\n"
                      f"   1 2 3 4 5 6 7   : increase arm joints 1-7\n"
                      f"   q w e r t y u   : decrease arm joints 1-7\n"
                      f"   i/k pitch cam, j/l yaw cam, o/p cam distance, z terminate episode")
                time.sleep(1)
                keyboard_listener = KeyboardListener()
            obs : TensorTree[th.Tensor]

            obs, info = env.reset(options = options)  #type: ignore
            ggLog.info(f"env resetted")
            done = False
            ep_reward = 0
            step_count = 0
            step_wallduration = float("nan")
            full_step_wallduration = float("nan")
            ep_wall_duration = 0
            rt = args["rt_factor"]
            model.reset_hidden_state()
            if render:
                img = env.render()
                dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
                time.sleep(step_length_sec/rt)
            base_env : GraspVecEnv = env.get_runner().get_base_env()
            print(f"Env is of type {type(base_env)}")

            goal_pose = None
            if isinstance(base_env, GraspVecEnv):
                # goal is the target object pose, as a (xyz, xyzw) vector
                goal_pose = base_env.get_goals()[0].clone()
            while not done:
                t0 = time.monotonic()
                set_disable_clear_recorded_times(True)
                record_time("start_step")
                session.run_info["collected_steps"].value += 1
                t0_pred = time.monotonic()
                obs_batch = map_tensor_tree(obs,lambda t: th.unsqueeze(t,0).to(device))
                act_info = {}
                action = model.predict_action(obs_batch, deterministic = deterministic, extra_returns=act_info)
                if recorder is not None:
                    recorder.add_to_extra_info({"act_log_prob": act_info.get("log_prob", -1)})
                t0_step = time.monotonic()
                record_time("pre env step")
                obs, reward, terminated, truncated, info = env.step(action.detach().squeeze()) #type: ignore
                record_time("post env step")
                record_time("post print")
                t1_step = time.monotonic()
                if render:
                    img = env.render()
                    dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
                record_time("post render")
                if verbose:
                    print(f"obs = {obs}\n"+
                        f"rew = {reward}\n"+
                        f"terminated = {terminated}\n"+
                        f"truncated = {truncated}\n")
                if interactive:
                    goal_diff_xyz = [0.0, 0.0, 0.0]
                    if keyboard_listener.get_key_press_count("w")>0: goal_diff_xyz[0] +=  0.02
                    if keyboard_listener.get_key_press_count("s")>0: goal_diff_xyz[0] += -0.02
                    if keyboard_listener.get_key_press_count("a")>0: goal_diff_xyz[1] +=  0.02
                    if keyboard_listener.get_key_press_count("d")>0: goal_diff_xyz[1] += -0.02
                    if keyboard_listener.get_key_press_count("r")>0: goal_diff_xyz[2] +=  0.02
                    if keyboard_listener.get_key_press_count("f")>0: goal_diff_xyz[2] += -0.02

                    cam_dist_pitch_yaw_diff = [0.0,0.0,0.0]
                    if keyboard_listener.get_key_press_count("u")>0: cam_dist_pitch_yaw_diff[0] = -0.1
                    if keyboard_listener.get_key_press_count("o")>0: cam_dist_pitch_yaw_diff[0] =  0.1
                    if keyboard_listener.get_key_press_count("i")>0: cam_dist_pitch_yaw_diff[1] =  5*3.14159/180
                    if keyboard_listener.get_key_press_count("k")>0: cam_dist_pitch_yaw_diff[1] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("j")>0: cam_dist_pitch_yaw_diff[2] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("l")>0: cam_dist_pitch_yaw_diff[2] =  5*3.14159/180

                    if keyboard_listener.get_key_press_count("t")>0: truncated = True
                    keyboard_listener.reset_key_press_counters()

                    if isinstance(base_env, GraspVecEnv):
                        base_env.set_cam_pose(base_env.get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                        goal_pose[:3] = goal_pose[:3] + th.as_tensor(goal_diff_xyz, device=goal_pose.device)
                        base_env.set_goals(goal_pose.unsqueeze(0).clone())
                        base_env._set_goal_marker_pose(base_env._all_vecs)
                if manual:
                    # Read all keys, then reset once, so no presses are double-counted.
                    # The joint deltas drive the next predict_action() call.
                    model.set_action_from_keys(keyboard_listener)
                    cam_dist_pitch_yaw_diff = [0.0,0.0,0.0]
                    if keyboard_listener.get_key_press_count("o")>0: cam_dist_pitch_yaw_diff[0] = -0.1
                    if keyboard_listener.get_key_press_count("p")>0: cam_dist_pitch_yaw_diff[0] =  0.1
                    if keyboard_listener.get_key_press_count("i")>0: cam_dist_pitch_yaw_diff[1] =  5*3.14159/180
                    if keyboard_listener.get_key_press_count("k")>0: cam_dist_pitch_yaw_diff[1] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("j")>0: cam_dist_pitch_yaw_diff[2] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("l")>0: cam_dist_pitch_yaw_diff[2] =  5*3.14159/180
                    manual_stop = keyboard_listener.get_key_press_count("z")>0
                    keyboard_listener.reset_key_press_counters()
                    if isinstance(base_env, GraspVecEnv):
                        base_env.set_cam_pose(base_env.get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                if isinstance(base_env, GraspVecEnv):
                    goal_obj_pose = base_env.get_goals()[0].tolist()
                else:
                    goal_obj_pose = [0.0]*7
                obj2goal_dist = info.get("obj2goal_dist", None)
                obj2hand_dist = info.get("obj2hand_dist", None)
                if isinstance(obj2goal_dist, th.Tensor): obj2goal_dist = obj2goal_dist.flatten()[0].item()
                if isinstance(obj2hand_dist, th.Tensor): obj2hand_dist = obj2hand_dist.flatten()[0].item()
                step_count += 1

                done = terminated or truncated
                if manual:
                    # Manual control: ignore the env's normal termination/truncation and keep the
                    # episode running until the user presses 'z'. autoreset is off for the single-env
                    # runner, so the sim keeps stepping safely past a "done".
                    done = manual_stop or terminated
                ep_reward += reward
                step_wallduration = time.monotonic()-t0
                ep_wall_duration += step_wallduration

                record_time("pre sleep")
                time.sleep(max(0,step_length_sec/rt - step_wallduration))
                record_time("step end")
                full_step_wallduration = time.monotonic()-t0
                set_disable_clear_recorded_times(False)
                print_recorded_times()
                ggLog.info(f"step = {step_count: 3d} rtfactor = {step_length_sec/full_step_wallduration:.2f}"
                           f" max_rtfactor = {step_length_sec/step_wallduration:.2f} tpred={t0_step-t0_pred:1.4f}"
                           f" tstep={t1_step-t0_step:1.4f} \t"
                           f" goal_xyz=[{goal_obj_pose[0]:.3f}, {goal_obj_pose[1]:.3f}, {goal_obj_pose[2]:.3f}] \t"
                           f" obj2goal_dist={obj2goal_dist} \t"
                           f" obj2hand_dist={obj2hand_dist}")
            if step_count>0:
                rewards.append(th.as_tensor(ep_reward,device="cpu").sum().item())
                durations.append(th.as_tensor(step_count,device="cpu").item())
            with session.run_info["collected_episodes"].get_lock():
                session.run_info["collected_episodes"].value += 1
            ggLog.info("\n"
                    f"Episode reward =  {ep_reward}\n"
                    f"Wall duration =   {ep_wall_duration:.2f}s\n"
                    f"Sim  duration =   {step_count*step_length_sec:.2f}s\n"
                    f"Realtime factor = {step_count*step_length_sec/ep_wall_duration:.2f}\n")
            if interactive or manual:
                keyboard_listener.close()
                keyboard_listener = None
    finally:
        if keyboard_listener is not None:
            keyboard_listener.close()
    rewards = np.array(rewards)
    durations = np.array(durations)
    ggLog.info(f"Avg reward = {rewards.mean()}, {rewards.std()} std")
    ggLog.info(f"Avg durations = {durations.mean()}, {durations.std()} std")
    env.close()

if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--model", required = False, default=None, type=str, help="Model to load")
    ap.add_argument("--mode", default="mjx", type=str, help="Simulator to use ('mjx'/'pybullet')")
    ap.add_argument("--robot", default="centauro", type=str, help="Robot to be used ('centauro'/'franka')")
    ap.add_argument("--rt-factor", default=1.0, type=float, help="Tentative realtime factor")
    ap.add_argument("--evaluate", default=None, type=int, help="Evaluate the policy with this number of episodes")
    ap.add_argument("--gui", default=False, action='store_true', help="Start the gui, instead of streaming renderings")
    ap.add_argument("--record", default=False, action='store_true', help="Record episode videos")
    ap.add_argument("--control", default="sine", type=str, help="Controller to use [sine,fixed,random,pretrained,manual]")
    ap.add_argument("--resolution", default=240, type=int, help="Vertical video resolution")
    ap.add_argument("--deterministic", default=False, action='store_true', help="Force the policy to be deterministic")
    ap.add_argument("--publishimg", default=False, action='store_true', help="publish rendered images to the web dbg server")
    ap.add_argument("--verbose-recording", default=False, action='store_true', help="Print detailed infos about the env step timings and outputs")
    ap.add_argument("--agent-device", default="cpu", type=str, help="Device to load the model on (cpu or cuda)")

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
