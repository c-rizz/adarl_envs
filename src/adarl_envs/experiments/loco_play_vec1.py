#!/usr/bin/env python3
from __future__ import annotations
import time
import inspect
from adarl.utils.buffers import BaseBuffer
import adarl.utils.dbg.dbg_img
from rreal.algorithms.sac import SAC, compare_dicts
from rreal.algorithms.sac_helpers import EnvBuilderProtocol
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
from adarl_envs.experiments.loco_builder import named_loco_single_env_builder, get_quad_args, get_kyon_args, get_centauro_args
from adarl_envs.env.LocomotionVecEnv import LocomotionVecEnv
from rreal.algorithms.rl_agent import RLAgent, TransitionBatch
from adarl.utils.base_utils import record_time, clear_recorded_times, print_recorded_times

import adarl.utils.dbg
from typing import Any
from ctypes.util import find_library
import readline
import math

def load_model(model_path):
    return SAC.load(model_path)


class Fixedpolicy(RLAgent):
    def __init__(self, cmd : th.Tensor):
        self._cmd = cmd.detach().clone()

    def predict_action(self, observation_batch, deterministic = False, info_return : dict = None):
        return self._cmd.clone()
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        pass

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
        return self._a_offset.device

class SinPolicy(RLAgent):
    def __init__(self,  act_scale : th.Tensor,
                        act_offset : th.Tensor,
                        act_speed : th.Tensor,
                        action_size : int,
                        dt : float):
        self._t0 = 0.0
        self._t = 0.0
        self._dt = dt
        self._a_offset = act_offset
        self._a_speed = act_speed
        # self._t_off = th.asin(self._a_offset/act_range)
        self._a_scale = act_scale.expand((action_size,))

    def predict_action(self, observation_batch, deterministic = False, info_return : dict = None):
        theta = (self._t0-self._t)*self._a_speed
        a = th.sin(theta)*self._a_scale+self._a_offset
        print(f" theta = {theta} \n"
            #   f" _t_off = {self._t_off} \n"
              f" _t = {self._t} \n"
              f" _a_scale = {self._a_scale} \n"
              f" _a_offset = {self._a_offset} \n"
              f" a = {a}")
        self._t = self._t+self._dt
        return a
    
    def get_hidden_state(self):
        return self._t
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        self._t = self._t0

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
        return self._a_offset.device
    
class RandPolicy(RLAgent):
    def __init__(self,  act_scale : th.Tensor,
                        action_size : int):
        self._a_scale = act_scale.expand((action_size,))

    def predict_action(self, observation_batch, deterministic = False, info_return : dict = None):
        a = (th.rand_like(self._a_scale)*2-1)*self._a_scale
        return a
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        pass

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
        return self._a_scale.device
        
def build_sin_policy(env, robot : str, scale : float = 0.0, device = th.device("cpu")):
    if robot == "quad":
        home_jpose = get_quad_args()["homing_joint_pose"]
    elif robot == "kyon":
        home_jpose = get_kyon_args()["homing_joint_pose"]
    elif robot == "centauro":
        home_jpose = get_centauro_args()["homing_joint_pose"]
    else:
        raise RuntimeError(f"Unknown robot '{robot}")
    home_pvesd = {k:[v, 0.0, 0.0, 400, 10] for k,v in home_jpose.items()}
    home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(home_pvesd)
    if robot == "quad":
        act_range = th.as_tensor([0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2], device = device)
    elif robot == "kyon":
        act_range = th.as_tensor([   0.0, -0.1,  0.17,
                                    -0.0,  0.1, -0.17,
                                     0.0, -0.1,  0.17,
                                    -0.0,  0.1, -0.17],device = device)
        # act_range.view(4,3)[0] *= -1.0
        act_range.view(4,3)[1] *= -1.0
        # act_range.view(4,3)[2] *= -1.0
        act_range.view(4,3)[3] *= -1.0
    elif robot == "centauro":
        act_range = th.as_tensor([0.1], device = device)
    else:
        raise RuntimeError(f"Unknown robot '{robot}")
    model = SinPolicy(  act_scale=act_range*scale,
                        act_offset=home_action,
                        act_speed=th.as_tensor([0.8], device = device),
                        action_size=env.get_runner().get_base_env()._action_helper.single_action_len(),
                        dt=0.05)
    return model

def build_rand_policy(env, robot : str, scale : float = 0.0, device : th.device = th.device("cpu")):
    if robot == "quad":
        home_jpose = get_quad_args()["homing_joint_pose"]
    elif robot == "kyon":
        home_jpose = get_kyon_args()["homing_joint_pose"]
    elif robot == "centauro":
        home_jpose = get_centauro_args()["homing_joint_pose"]
    else:
        raise RuntimeError(f"Unknown robot '{robot}")
    home_pvesd = {k:[v, 0.0, 0.0, 400, 10] for k,v in home_jpose.items()}
    home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(home_pvesd)
    if robot == "quad":
        act_range = th.as_tensor([0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2], device = device)
    elif robot == "kyon":
        act_range = th.as_tensor([   0.0, -0.1,  0.17,
                                    -0.0,  0.1, -0.17,
                                     0.0, -0.1,  0.17,
                                    -0.0,  0.1, -0.17], device = device )
    elif robot == "centauro":
        act_range = th.as_tensor([0.1], device = device)
    else:
        raise RuntimeError(f"Unknown robot '{robot}")
    model = RandPolicy(  act_scale=act_range*scale,
                        action_size=env.get_runner().get_base_env()._action_helper.single_action_len())
    return model


def build_fixed_policy(env, robot : str, scale : float = 0.0):
    if robot == "quad":
        home_jpose = get_quad_args()["homing_joint_pose"]
        stiffness = 400
        damping = 10
    elif robot == "kyon":
        home_jpose = get_kyon_args()["homing_joint_pose"]
        stiffness = 400
        damping = 10
    elif robot == "centauro":
        home_jpose = get_centauro_args()["homing_joint_pose"]
        stiffness = 1000
        damping = 10
    else:
        raise RuntimeError(f"Unknown robot '{robot}")
    home_pvesd = {k:[v, 0.0, 0.0, stiffness, damping] for k,v in home_jpose.items()}
    home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(home_pvesd)
    model = Fixedpolicy(  cmd = home_action)
    return model

from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
def find_recorder_wrapper(env) -> EnvRunnerRecorderWrapper | None:
    from adarl.envs.vec.Runner2GymWrapper import Runner2GymWrapper
    from adarl.envs.vec.EnvRunnerWrapper import EnvRunnerWrapper
    if isinstance(env, Runner2GymWrapper):
        recorder = env
        while recorder is not None and not isinstance(recorder, EnvRunnerRecorderWrapper):
            if isinstance(recorder, EnvRunnerWrapper):
                recorder = recorder.get_base_runner()
            else:
                recorder = None
    # print(f"Recorder found: {recorder}")
    return recorder

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    step_length_sec = 20/1024 
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    mode = args["mode"]
    env_device = th.device("cuda") if mode == "mjx" else th.device("cpu")
    height_pixels = args["resolution"] #if mode == "mjx" else 720
    pixel_resolution = (height_pixels,int(height_pixels*16/9))
    
    r = 0.0
    n = 0.0
    p = 0.0
    eps= 1e-6
    env_builder_args = {
        "action_delay_mustd_std" : (0.003, 0.001*n, 0.0025*n),
        "action_noise_mustd" : (0.0,   0.0),
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position",
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "fail_on_safety" : False,
        "frame_stack_length" : 5,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : [0.47,0.47],
        "goal_resampling_probability_per_sec" : 0.1,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "held_joints_damping" : 10.0,
        "held_joints_stiffness" : 500.0,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 0.2,
        "initial_height_randomization_range_meters" : 0.0,
        "initial_joint_pose_randomization_range" : 0.5,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 0.05, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "max_good_step_duration" : 0.3,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "min_good_step_duration" : 0.1,
        "mode" : mode,
        "obs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.02*n, 0.05*n],
        "obs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.01*n, 0.02*n],
        "obs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.02*n],
        "obs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0,    0.0],
        "observe_full_robot_state" : False,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomized_armature_ratios" : 0.1*r,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_frictionloss_ratios"             : 0.2*r,
        "randomized_gains_damping_ratio_epstd"       : 0.2*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.2*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_reference_filter_distribution" : ("uniform", (20.0-15*r, 20.0+15*r)),
        "record_video" : True,
        "recycle_pose_randomization" : True,
        "reward_superweight_joint_penalties" : ["loguniform", (0.01, 1.0)],
        "reward_acceleration_weight" :        eps,
        "reward_actacc_weight" :              eps,
        "reward_actdiff_weight" :             eps,
        "reward_contacts_weight" :            eps,
        "reward_energy_weight" :              eps,
        "reward_failure_weight" :             eps,
        "reward_feet_air_time_weight" :       20.0,
        "reward_feet_ground_time_weight" :    eps,
        "reward_feet_on_ground_weight" :      eps,
        "reward_heading_velocity_weight" :    eps,
        "reward_heading_weight" :             0.1,
        "reward_health_weight" :              eps,
        "reward_height_position_weight" :     0.5,
        "reward_height_velocity_weight" :     eps,
        "reward_pitchnroll_velocity_weight" : 20.0,
        "reward_pitchnroll_weight" :          0.5,
        "reward_position_limit_weight" :      1.0,
        "reward_position_weight" :            0.1,
        "reward_posref_vel_weight" :          0.5,
        "reward_posref_acc_weight":           2.0,
        "reward_sensed_effort_weight" :       eps,
        "reward_slip_weight" :                eps,
        "reward_stand_position_weight" :      1.0,
        "reward_torque_limit_weight" :        eps,
        "reward_torque_weight" :              1.0,
        "reward_torquediff_weight" :          eps,
        "reward_torqueref_weight" :           eps,
        "reward_tracking_weight" :            2.0,
        "reward_velocity_limit_weight" :      eps,
        "reward_velocity_weight" :            eps,
        "reward_velref_weight" :              eps,
        "robot_model" : "kyon",
        "safe_damping" : 5,
        "safe_stiffness" : 600,
        "saturate_jimp_ref_limits" : False,
        "split_rewards" : True,
        "stepLength_sec" : step_length_sec,
        "stop_on_failure" : False,
        "terminate_on_body_contact" : False,
        "th_device" : env_device,
        "ui_camera_resolution_hw" : [144,256],
        "use_contacts" : False,
        "verbose_infos" : False,
        "video_save_freq" : 0,
        "walltime_factor" : 1.0,
        "minimal_infos" : False
    }

    skip_optionals = True
    env_builder_args.update({
        "enable_rendering" : True,
        "record_video" : args["mode"] not in ["xbot","xbot_zmq"],
        "verbose_infos" : (not skip_optionals) or args["record"],
        "video_save_freq" : True if args["record"] else 0,
        "action_delay_mustd_std" : (0.0,0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "obs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "obs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_linacc_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.0),
        "obs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.0),
        "ui_camera_resolution_hw" : pixel_resolution,
        "log_info_stats" : (not skip_optionals) or args["record"],
        "minimal_infos" : skip_optionals or not args["record"],
        "initial_joint_pose_randomization_range" : 0.0,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.0,0.0,0.0])),
        "randomized_friction_slide_spin_roll_ratios" : (0.0,0.0,0.0),
        "randomized_gains_damping_ratio_epstd" : 0.0,
        "randomized_gains_stiffness_ratio_epstd" : 0.0,
        "randomized_mass_ratios" : ("normal", (0.0, 0.0)),
        "impulse_probability_per_sec" : 0.0,
        "show_gui" : args["gui"],
        "just_health_reward" : skip_optionals,
        "goal_resampling_probability_per_sec" : 0.0,
        "walltime_factor" : args["rt_factor"]})
    return play(seed,
                folderName,
                run_id, args,
                env_builder = named_loco_single_env_builder,
                env_builder_args = env_builder_args,
                step_length_sec = step_length_sec,
                render=not args["gui"] and not args["norender"],
                robot = args["robot"],
                deterministic = args["deterministic"])



def play(seed, folderName, run_id, args, 
         env_builder : EnvBuilderProtocol, 
         env_builder_args : dict[str,Any], 
         step_length_sec : float, render : bool,
         robot : str,
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

    # th.cuda.set_sync_debug_mode("warn")
    device = adarl.utils.utils.torch_selectBestGpu()
    env_device = env_builder_args["th_device"]
    ggLog.info("Building env...")

    env, fps = env_builder( log_folder=log_folder+"/eval",
                            seed=seed+100000000,
                            env_builder_args = env_builder_args,
                            is_eval=False)
    recorder = find_recorder_wrapper(env)
    ggLog.info("Built")
    control_mode = args["control"].lower().strip()
    if control_mode=="pretrained":
        model = load_model(args["model"])
        trained_env_builder_args = model._init_args["init_hparams"].reference_init_args["env_builder_args"]
        try:
            equal, diffs = compare_dicts(env_builder_args, trained_env_builder_args)
            if not equal:
                ggLog.warn(f"Loaded model was trained with different env args: \n{diffs}")
        except Exception as e: 
            ggLog.warn(f"Could not compare env args with trained model: {type(e)}: {e}")
    elif control_mode=="fixed":
        model = build_fixed_policy(env = env, robot=robot)
    elif control_mode=="random":
        model = build_rand_policy(env=env, robot=robot, scale=1.0, device = env_device)
    elif control_mode == "sine":
        model = build_sin_policy(env, robot=robot, scale = 1.0, device = env_device)
    else:
        raise RuntimeError(f"Unknown control mode '{control_mode}'")

    play = True
    verbose = False
    interactive = False
    rewards = []
    durations = []
    avg10_dists = []

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
                    elif cmd == "interactive":
                        print(f" Use WASD to move the goal, IJKL to move the camera, T to terminate.")
                        time.sleep(1)
                        interactive = True
                        options["max_ep_steps"] = 100_000
                        options["goal_velocity_xy"] = [0.0, 0.0]
                        keyboard_listener = KeyboardListener()
                        cmd = 'c'   
                if not play:
                    break
            else:
                if session.run_info["collected_episodes"].value >= args["evaluate"]:
                    break
            if not play:
                break
            obs : TensorTree[th.Tensor]
            obs, info = env.reset(options = options)  #type: ignore
            ggLog.info(f"env resetted")
            # ggLog.info(f"ep_config = {info['ep_config']}")
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

            cmd_xys = [1.0,0.0,0.0]
            cmd_height = 0.45
            while not done:
                t0 = time.monotonic()
                record_time("start_step")
                session.run_info["collected_steps"].value += 1
                # ggLog.info(f"ep_config = {info['ep_config']}")
                t0_pred = time.monotonic()
                obs_batch = map_tensor_tree(obs,lambda t: th.unsqueeze(t,0).to(device))
                act_info = {}
                action = model.predict_action(obs_batch, deterministic = deterministic, info_return=act_info)
                if recorder is not None:
                    recorder.add_to_extra_info({"act_log_prob": act_info.get("log_prob", -1)})
                t0_step = time.monotonic()
                record_time("pre env step")
                obs, reward, terminated, truncated, info = env.step(action.detach().squeeze()) #type: ignore
                record_time("post env step")
                # print_recorded_times()
                clear_recorded_times()
                record_time("post print")
                t1_step = time.monotonic()
                if render:
                    img = env.render()
                    dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
                record_time("post render")
                # input("press enter")
                if verbose:
                    print(f"obs = {obs}\n"+
                        f"rew = {reward}\n"+
                        f"terminated = {terminated}\n"+
                        f"truncated = {truncated}\n")
                base_env : LocomotionVecEnv = env.get_runner().get_base_env()
                if interactive:
                    cmd_angle = np.arctan2(cmd_xys[1],cmd_xys[0])
                    if keyboard_listener.get_key_press_count("w")>0: cmd_xys[2] +=  0.05
                    if keyboard_listener.get_key_press_count("s")>0: cmd_xys[2] += -0.05
                    if keyboard_listener.get_key_press_count("a")>0: cmd_angle  +=  10*3.14159/180
                    if keyboard_listener.get_key_press_count("d")>0: cmd_angle  += -10*3.14159/180
                    if keyboard_listener.get_key_press_count("r")>0: cmd_height +=  0.005
                    if keyboard_listener.get_key_press_count("f")>0: cmd_height += -0.005
                    cmd_xys[0] = np.cos(cmd_angle)
                    cmd_xys[1] = np.sin(cmd_angle)
                    cmd_height = np.clip(cmd_height, 0.35, 0.57)
                    flip = keyboard_listener.get_key_press_count("x")>0
                    cam_dist_pitch_yaw_diff = [0.0,0.0,0.0]
                    if keyboard_listener.get_key_press_count("u")>0: cam_dist_pitch_yaw_diff[0] = -0.1
                    if keyboard_listener.get_key_press_count("o")>0: cam_dist_pitch_yaw_diff[0] =  0.1
                    if keyboard_listener.get_key_press_count("i")>0: cam_dist_pitch_yaw_diff[1] =  5*3.14159/180
                    if keyboard_listener.get_key_press_count("k")>0: cam_dist_pitch_yaw_diff[1] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("j")>0: cam_dist_pitch_yaw_diff[2] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("l")>0: cam_dist_pitch_yaw_diff[2] =  5*3.14159/180

                    if keyboard_listener.get_key_press_count("t")>0: truncated = True
                    base_env.set_cam_pose(base_env.get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                    if flip:
                        cmd_xys = [-cmd_xys[0],-cmd_xys[1],-cmd_xys[2]]
                    base_env.set_goal(goal_rel_linvel_xys = tuple(cmd_xys), goal_abs_height = cmd_height)
                    keyboard_listener.reset_key_press_counters()
                goals = base_env.get_goals()
                step_count += 1

                done = terminated or truncated
                ep_reward += reward
                step_wallduration = time.monotonic()-t0
                ep_wall_duration += step_wallduration

                record_time("pre sleep")
                if args["mode"] not in ["xbot","xbot_zmq"]:
                    time.sleep(max(0,step_length_sec/rt - step_wallduration))
                record_time("step end")
                full_step_wallduration = time.monotonic()-t0
                ggLog.info(f"step = {step_count: 3d} rtfactor = {step_length_sec/full_step_wallduration:.2f}"
                           f" max_rtfactor = {step_length_sec/step_wallduration:.2f} tpred={t0_step-t0_pred:1.4f}"
                           f" tstep={t1_step-t0_step:1.4f} \t"
                           f" rgoal_dir={goals['rel_linvel_xys'][0,:2].tolist()} \t"
                           f" rgoal_speed={goals['rel_linvel_xys'][0,2]} \t"
                           f" goal_height={goals['abs_height'][0].item()} \t")
                # print_recorded_times()
            if step_count>0:
                rewards.append(th.as_tensor(ep_reward,device="cpu").sum().item())
                durations.append(th.as_tensor(step_count,device="cpu").item())
                # avg10_dists.append(info["avg_vel_track_err"])
            with session.run_info["collected_episodes"].get_lock():
                session.run_info["collected_episodes"].value += 1
            ggLog.info("\n"
                    f"Episode reward =  {ep_reward}\n"
                    f"Wall duration =   {ep_wall_duration:.2f}s\n"
                    f"Sim  duration =   {step_count*step_length_sec:.2f}s\n"
                    f"Realtime factor = {step_count*step_length_sec/ep_wall_duration:.2f}\n")
            if interactive:
                keyboard_listener.close()
    finally:
        if keyboard_listener is not None:
            keyboard_listener.close()
    rewards = np.array(rewards)
    durations = np.array(durations)
    # avg10_dists = np.array(avg10_dists)
    ggLog.info(f"Avg reward = {rewards.mean()}, {rewards.std()} std")
    ggLog.info(f"Avg durations = {durations.mean()}, {durations.std()} std")
    # ggLog.info(f"Avg avg10_dists = {avg10_dists.mean()}, {avg10_dists.std()} std")
    # env.reset() # trigger video saving
    env.close()

if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--model", required = False, default=None, type=str, help="Model to load")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use [pybullet,xbot-gazebo,mjx]")
    ap.add_argument("--robot", default="quad", type=str, help="Robot to be used")
    ap.add_argument("--rt-factor", default=1.0, type=float, help="Tentative realtime factor")
    ap.add_argument("--evaluate", default=None, type=int, help="Evaluate the policy with this number of episodes")
    ap.add_argument("--gui", default=False, action='store_true', help="Start the gui, instead of streaming renderings")
    ap.add_argument("--record", default=False, action='store_true', help="Record episode videos")
    ap.add_argument("--control", default="sine", type=str, help="Controller to use [sine,fixed,random,pretrained]")
    ap.add_argument("--resolution", default=240, type=int, help="Vertical video resolution")
    ap.add_argument("--deterministic", default=False, action='store_true', help="Force the policy to be deterministic")
    ap.add_argument("--norender", default=False, action='store_true', help="Force disable the rendering")
    
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
