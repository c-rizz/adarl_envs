#!/usr/bin/env python3
from __future__ import annotations
import time
import inspect
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
from adarl_envs.experiments.loco_builder import named_loco_single_env_builder, get_quad_args, get_kyon_args, get_centauro_args, get_go1_args, robot_args_registry
from adarl_envs.env.LocomotionVecEnv import LocomotionVecEnv
from rreal.algorithms.rl_agent import RLAgent, TransitionBatch
from adarl.utils.base_utils import record_time, clear_recorded_times, print_recorded_times, isinstance_noimport, set_disable_clear_recorded_times

import adarl.utils.dbg
from typing import Any
from ctypes.util import find_library
import readline
import math
from rreal.algorithms.hardcoded_policies import FixedPolicy, SinPolicy, RandPolicy

def build_fixed_policy(env, robot : str, scale : float = 0.0, device : th.device = th.device("cpu")):
    
    base_env = env.get_runner().get_base_env()
    if isinstance_noimport(base_env, "RobotVecEnv"):
        home_jpose = robot_args_registry[robot]()["homing_joint_position_references"]
        stiffness = 400
        damping = 10
        home_pvesd = {k:[v, 0.0, 0.0, stiffness, damping] for k,v in home_jpose.items()}
        home_action = base_env._action_helper.pvesd_to_action(home_pvesd)
        action_len = base_env._action_helper.single_action_len()
    else:
        action_len = 12
        home_action = th.zeros((action_len,), device=device)

    model = FixedPolicy(  cmd = home_action)
    return model

def get_act_range(robot : str, device):
    if robot == "kyon_arms":
        hx, hy, ky = 0.0, 0.1, 0.17
        act_range = th.as_tensor([   hx, -hy,  ky,
                                     hx, -hy,  ky,
                                     hx, -hy,  ky,
                                     hx, -hy,  ky,
                                    0.5,  0.5,  0.5,
                                    0.0,  0.5,  0.5,
                                    0.0,  0.5,  0.5,
                                    0.0,  0.5,  0.5,
                                    ],device = device)
    elif robot == "centauro":
        act_range = th.as_tensor([0.1], device = device)
    else:
        if robot == "quad":
            hx, hy, ky = 0.0, 0.1, 0.2
        elif robot == "kyon":
            hx,hy,ky = 0.0, 0.1, 0.17
        elif robot == "spot":
            hx,hy,ky = 0.0, 1.0, 1.0
        elif robot == "go1":
            hx,hy,ky = 0.0, 0.4, 0.8
        else:
            raise RuntimeError(f"Unknown robot '{robot}'")
        act_range = th.as_tensor([   hx, -hy,  ky,
                                    hx, -hy,  ky,
                                    hx, -hy,  ky,
                                    hx, -hy,  ky],device = device)
    return act_range

def build_sin_policy(env, robot : str, scale : float = 0.0, device = th.device("cpu")):
    home_jpose = robot_args_registry[robot]()["homing_joint_position"]
    home_pvesd = {k:[v, 0.0, 0.0, 400, 10] for k,v in home_jpose.items()}
    base_env = env.get_runner().get_base_env()
    if isinstance_noimport(base_env, "RobotVecEnv"):
        home_action = base_env._action_helper.pvesd_to_action(home_pvesd)
        action_len = base_env._action_helper.single_action_len()
    else:
        action_len = 12
        home_action = th.zeros((action_len,), device=device)
    speed = 0.8 if robot!="go1" else 0.5
    act_range = get_act_range(robot, device)
    model = SinPolicy(  act_scale=act_range*scale,
                        act_offset=home_action,
                        act_speed=th.as_tensor([speed], device = device),
                        action_size=action_len,
                        dt=0.05)
    return model
    
    
def build_rand_policy(env, robot : str, scale : float = 0.0, device : th.device = th.device("cpu")):
    act_range = get_act_range(robot, device)
    model = RandPolicy( act_scale=act_range*scale,
                        action_size=env.get_runner().get_base_env()._action_helper.single_action_len())
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
    return recorder




robot_heights = {   "kyon" : [0.47,0.47],
                    "go1" : [0.30,0.30],
                    "centauro" : [0.79,0.79],
                    }



def adarl_builder_and_args():
    step_length_sec = 20/1024 
    max_steps_per_episode=500 #int(ep_duration_sec/step_length_sec)
    mode = args["mode"]
    env_device = th.device("cuda") if mode == "mjx" else th.device("cpu")
    height_pixels = args["resolution"] #if mode == "mjx" else 720
    pixel_resolution = (height_pixels,int(height_pixels*16/9))
    record = args["record"] or args["verbose_recording"]
    skip_optionals = not args["verbose_recording"]
    realworld = args["mode"] in ["xbot","xbot-zmq"]
    
    r = 0.0 # randomization strength
    n = 1.0 # noise strength
    p = 1.0 # penalties strength
    eps = 0 #1e-6 # For disabled things (but no zero, so I can still see how they would behave)
    env_builder_args = {
        "action_delay_mustd_std" : (0.0, 0.001*n, 0.005*n) if not realworld else (0.0, 0.0, 0.0),
        "action_noise_mustd" : (0.0,   0.0),
        "action_smoothing_halflife_sec" : 0.0,
        "control_mode" : "position",
        "enable_limits_safety" : True,
        "enable_posref_safety" : True,
        "enable_rendering" : False,
        "enable_reference_filter" : False,
        "fail_on_safety" : False,
        "frame_stack_length" : 3,
        "goal_err_smoothing_halflife_sec" : 0.05,
        "goal_height_minmax" : robot_heights.get(args["robot"].lower(), [0.5,0.5]),
        "goal_resampling_probability_per_sec" : 0.0,
        "goal_speed_minmax" : (0,1.0),
        "goal_yaw_minmax" : (-math.pi, math.pi),
        "goal_yaw_vel_zero_ratio" : 0.25,
        "goal_yaw_vel_minmax" : (-1.0, 1.0),
        "held_joints_damping" : 10.0,
        "held_joints_stiffness" : 500.0,
        "impulse_duration_minmax" : [0.01, 2.5],
        "impulse_mean_std" : [20.0,50.0],
        "impulse_probability_per_sec" : 0.0,
        "init_on_reset_ratio" : 1.0,
        "initial_height_randomization_range_meters" : 0.1,
        "initial_joint_pose_randomization_range" : 0.1,
        "just_health_reward" : False,
        "log_info_stats" : True,
        "longterm_states_decimation_time" : 1.0, # Averaging of the joint pose for the position reward
        "max_goal_height_pos_change_speed" : 0.1,
        "max_goal_height_speed" : 0.1,
        "max_good_step_duration" : 0.5,
        "max_steps_per_episode" : max_steps_per_episode,
        "merge_privileged" : False,
        "min_good_step_duration" : 0.1,
        "mode" : mode,
        "obs_abs_noise_angvel_ep_mustd_step_std" :      [0.0, 0.02*n, 0.1*n],
        "obs_abs_noise_gravity_ep_mustd_step_std" :     [0.0, 0.0*n, 0.0*n],
        "obs_abs_noise_joints_pve_ep_mustd_step_std" :  [0.0, 0.001*n, 0.1*n],
        "obs_abs_noise_linacc_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_abs_noise_linvel_ep_mustd_step_std" :      [0.0, 0.0,    0.0],
        "obs_abs_noise_posz_ep_mustd_step_std" :        [0.0, 0.0,    0.0],
        "observe_full_robot_state" : False,
        "offset_envs_ep_starts" : True,
        "posref_safety_period" : 0.02,
        "quiet" : False,
        "randomized_dof_armature_ratios" : 0.1*r,
        "randomized_dof_frictionloss_ratios"             : 0.0*r,
        "randomized_dof_damping_ratios":        0.2*r,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_gains_damping_ratio_epstd"       : 0.0*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.0*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_reference_filter_distribution" : ("uniform", (50.0, 50.0)),
        "record_video" : True,
        "recycle_pose_randomization" : True,
        "reward_superweight_joint_penalties" : 1.0,
        "reward_acceleration_weight" :        eps,
        "reward_acc_on_vel_weight" :          eps,
        "reward_actacc_weight" :              0.5*p,
        "reward_actdiff_weight" :             0.5*p,
        "reward_contacts_weight" :            eps,
        "reward_energy_weight" :              eps,
        "reward_power_weight" :               0.002*p,
        "reward_failure_weight" :             eps,
        "reward_feet_air_time_weight" :       10.0,
        "reward_feet_ground_time_weight" :    eps,
        "reward_feet_on_ground_weight" :      1.0,
        "reward_heading_velocity_weight" :    eps,
        "reward_heading_weight" :             eps,
        "reward_health_weight" :              eps,
        "reward_height_position_weight" :     0.5,
        "reward_height_velocity_weight" :     0.1,
        "reward_pitchnroll_velocity_weight" : 0.2,
        "reward_pitchnroll_weight" :          0.5,
        "reward_position_limit_weight" :      eps,
        "reward_position_weight" :            eps,
        "reward_posref_vel_weight" :          0.05,
        "reward_posref_acc_weight":           eps,
        "reward_scale_nolength":              0.01,
        "reward_sensed_effort_weight" :       eps,
        "reward_safety_triggered_weight" :    0.5,
        "reward_slip_weight" :                1.0,
        "reward_stand_position_weight" :      5.0,
        "reward_torque_limit_weight" :        eps,
        "reward_torque_weight" :              0.0001*p,
        "reward_torquediff_weight" :          0.0001*p,
        "reward_torqueref_weight" :           eps,
        "reward_tracking_weight" :            1.0,
        "reward_velocity_limit_weight" :      eps,
        "reward_velocity_weight" :            eps,
        "reward_velref_weight" :              eps,
        "reward_yaw_vel_tracking_weight" :    1.0,
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
        "offset_envs_ep_starts" : False,
        "enable_rendering" : True,
        "record_video" : not realworld,
        "verbose_infos" : (not skip_optionals) or record,
        "minimal_infos" : skip_optionals or not record,
        "video_save_freq" : 1 if record else 0,
        "action_delay_mustd_std" : (0.0,0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "obs_abs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "obs_abs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_abs_noise_linacc_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_abs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_abs_noise_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.0),
        "obs_abs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.0),
        "ui_camera_resolution_hw" : pixel_resolution,
        "log_info_stats" : (not skip_optionals) or record,
        # "minimal_infos" : skip_optionals or not record,
        "initial_joint_pose_randomization_range" : 0.0,
        "initial_height_randomization_range_meters" : 0.0,
        "randomized_com_xyz_diff_distribution" : ("normal",([0.,0.,0.],[0.10*r,0.02*r,0.02*r])),
        "randomized_friction_slide_spin_roll_ratios" : [0.2*r,0.2*r,0.2*r],
        "randomized_dof_frictionloss_ratios"         : 0.0*r,
        "randomized_dof_damping_ratios"              : 0.0*r,
        "randomized_gains_damping_ratio_epstd"       : 0.0*r,
        "randomized_gains_stiffness_ratio_epstd"     : 0.0*r,
        "randomized_mass_ratios" : ("normal", (0.0, 0.1*r)),
        "randomized_dof_armature_ratios" : 0.0*r,
        "randomized_reference_filter_distribution" : ("uniform", (50.0, 50.0)),
        "impulse_probability_per_sec" : 0.0,
        "show_gui" : args["gui"],
        "just_health_reward" : realworld,
        "goal_resampling_probability_per_sec" : 0.0,
        "walltime_factor" : args["rt_factor"],
        "record_whole_joint_trajectories" : False,
        # "mjx_opt_override" : {"noslip_iterations": 10, "impratio": 10.0, "iterations" : 20}
        })
    return named_loco_single_env_builder, env_builder_args, step_length_sec

def pg_builder_and_args():
    from adarl_envs.experiments.playground_builder import playground_single_env_builder
    
    step_length_sec = 20/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    
    env_builder_args = {
        "video_save_freq" : 1,
        "record_video" : True,
        "env_name" : "SpotFlatTerrainJoystick",
        "quiet" : False,
        "th_device" : th.device("cuda",0),
        "log_info_stats": True,
        "randomize_step_timeout_counters": True,
        "camera" : "track",
        "episode_length" : max_steps_per_episode,
        "autoreset" : True,
        "playground_config_overrides": {"reward_config.scales.feet_clearance":0.0,
                                        "reward_config.scales.feet_height":0.0,
                                        "obs_noise.scales.joint_pos": 0.0,
                                        "obs_noise.scales.gyro" : 0.0,
                                        "obs_noise.scales.gravity" : 0.0,
                                        "obs_noise.scales.feet_pos" : [0.0, 0.0, 0.0],
                                        }
    }
    return playground_single_env_builder, env_builder_args, step_length_sec













def runFunction(seed, folderName, resumeModelFile, run_id, args):

    if args["pg"]:
        builder, env_builder_args, step_length_sec = pg_builder_and_args()
    else:
        builder, env_builder_args, step_length_sec = adarl_builder_and_args()

    return play(seed,
                folderName,
                run_id, args,
                env_builder = builder,
                env_builder_args = env_builder_args,
                step_length_sec = step_length_sec,
                render=args["publishimg"],
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
        model = build_fixed_policy(env = env, robot=robot, device= env_device)
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

            minmax_height = robot_heights.get(args["robot"].lower(), [0.5,0.5])
            h = sum(minmax_height)/2

            if interactive:
                cmd_xys = [1.0,0.0,0.0]
                cmd_yawvel = 0.0
                cmd_height = h
            else:
                dir = np.random.rand()*2*3.14159
                speed = 0 #np.random.rand()
                cmd_xys = [np.cos(dir), np.sin(dir), speed]
                cmd_yawvel = 0 #(np.random.rand()*2-1)*0.5*(np.random.rand()>0.5)
                cmd_height = h
            # options["goal_velocity_xy"] = [cmd_xys[0]*cmd_xys[2], cmd_xys[1]*cmd_xys[2]]
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
            base_env : LocomotionVecEnv = env.get_runner().get_base_env()
            print(f"Env is of type {type(base_env)}")

            if isinstance(base_env, LocomotionVecEnv):
                # base_env.set_cam_pose((1.583, 0.201, 2.149))
                base_env.set_goal(
                                    goal_rel_linvel_xys = tuple(cmd_xys),
                                    goal_abs_height = cmd_height,
                                    goal_yaw_vel = cmd_yawvel,
                                    goal_heading_yaw = 0.0)
            while not done:
                t0 = time.monotonic()
                set_disable_clear_recorded_times(True)
                record_time("start_step")
                session.run_info["collected_steps"].value += 1
                # ggLog.info(f"ep_config = {info['ep_config']}")
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
                # input("press enter")
                if verbose:
                    print(f"obs = {obs}\n"+
                        f"rew = {reward}\n"+
                        f"terminated = {terminated}\n"+
                        f"truncated = {truncated}\n")
                if interactive:
                    cmd_angle = np.arctan2(cmd_xys[1],cmd_xys[0])
                    if keyboard_listener.get_key_press_count("w")>0: cmd_xys[2] +=  0.05
                    if keyboard_listener.get_key_press_count("s")>0: cmd_xys[2] += -0.05
                    if keyboard_listener.get_key_press_count("a")>0: cmd_angle  +=  10*3.14159/180
                    if keyboard_listener.get_key_press_count("d")>0: cmd_angle  += -10*3.14159/180
                    if keyboard_listener.get_key_press_count("r")>0: cmd_height +=  0.005
                    if keyboard_listener.get_key_press_count("f")>0: cmd_height += -0.005
                    if keyboard_listener.get_key_press_count("q")>0: cmd_yawvel += 0.05
                    if keyboard_listener.get_key_press_count("e")>0: cmd_yawvel += -0.05
                    if keyboard_listener.get_key_press_count("z")>0: 
                        cmd_xys[2] = 0.0
                        cmd_yawvel = 0.0
                    cmd_xys[0] = np.cos(cmd_angle)
                    cmd_xys[1] = np.sin(cmd_angle)
                    cmd_height = np.clip(cmd_height, minmax_height[0], minmax_height[1])
                    flip = keyboard_listener.get_key_press_count("x")>0
                    cam_dist_pitch_yaw_diff = [0.0,0.0,0.0]
                    if keyboard_listener.get_key_press_count("u")>0: cam_dist_pitch_yaw_diff[0] = -0.1
                    if keyboard_listener.get_key_press_count("o")>0: cam_dist_pitch_yaw_diff[0] =  0.1
                    if keyboard_listener.get_key_press_count("i")>0: cam_dist_pitch_yaw_diff[1] =  5*3.14159/180
                    if keyboard_listener.get_key_press_count("k")>0: cam_dist_pitch_yaw_diff[1] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("j")>0: cam_dist_pitch_yaw_diff[2] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("l")>0: cam_dist_pitch_yaw_diff[2] =  5*3.14159/180

                    if keyboard_listener.get_key_press_count("t")>0: truncated = True
                    if flip:
                        cmd_xys = [-cmd_xys[0],-cmd_xys[1],-cmd_xys[2]]
                    keyboard_listener.reset_key_press_counters()

                    if isinstance(base_env, LocomotionVecEnv):
                        base_env.set_cam_pose(base_env.get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                        base_env.set_goal(goal_rel_linvel_xys = tuple(cmd_xys), goal_abs_height = cmd_height,
                                          goal_yaw_vel = cmd_yawvel,
                                          goal_heading_yaw = 0.0)
                if isinstance(base_env, LocomotionVecEnv):
                    goals = base_env.get_goals()
                    goal_rel_linvel_xys = goals["rel_linvel_xys"][0].tolist()
                    goal_height = goals["abs_height"][0].item()
                else:
                    goal_rel_linvel_xys = [0.0, 0.0, 0.0]
                    goal_height = 0.0
                step_count += 1

                done = terminated or truncated
                ep_reward += reward
                step_wallduration = time.monotonic()-t0
                ep_wall_duration += step_wallduration

                record_time("pre sleep")
                if args["mode"] not in ["xbot","xbot-zmq"]:
                    time.sleep(max(0,step_length_sec/rt - step_wallduration))
                record_time("step end")
                full_step_wallduration = time.monotonic()-t0
                set_disable_clear_recorded_times(False)
                print_recorded_times()
                ggLog.info(f"step = {step_count: 3d} rtfactor = {step_length_sec/full_step_wallduration:.2f}"
                           f" max_rtfactor = {step_length_sec/step_wallduration:.2f} tpred={t0_step-t0_pred:1.4f}"
                           f" tstep={t1_step-t0_step:1.4f} \t"
                           f" cmd_reldir={np.arctan2(goal_rel_linvel_xys[1], goal_rel_linvel_xys[0])/3.14159*180} \t"
                           f" cmd_speed={goal_rel_linvel_xys[2]} \t"
                           f" cmd_height={goal_height} \t"
                           f" cmd_yaw_vel={cmd_yawvel}")
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
    ap.add_argument("--mode", default="mjx", type=str, help="Adapter to use [pybullet,xbot-gazebo,mjx]")
    ap.add_argument("--robot", default="kyon", type=str, help="Robot to be used")
    ap.add_argument("--rt-factor", default=1.0, type=float, help="Tentative realtime factor")
    ap.add_argument("--evaluate", default=None, type=int, help="Evaluate the policy with this number of episodes")
    ap.add_argument("--gui", default=False, action='store_true', help="Start the gui, instead of streaming renderings")
    ap.add_argument("--record", default=False, action='store_true', help="Record episode videos")
    ap.add_argument("--control", default="sine", type=str, help="Controller to use [sine,fixed,random,pretrained]")
    ap.add_argument("--resolution", default=240, type=int, help="Vertical video resolution")
    ap.add_argument("--deterministic", default=False, action='store_true', help="Force the policy to be deterministic")
    ap.add_argument("--publishimg", default=False, action='store_true', help="publish rendered images to the web dbg server")
    ap.add_argument("--pg", default=False, action='store_true', help="Use playground env")
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
