#!/usr/bin/env python3  
from __future__ import annotations



from __future__ import annotations
from adarl.envs.vec.GymVecRunnerWrapper import GymVecRunnerWrapper
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
import threading, os
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import typing 
from pathlib import Path
import adarl.utils.utils
from typing import Sequence, Any
from jumping_leg.env.LocomotionVecEnv import LocomotionVecEnv
from jumping_leg.env.RobotVecEnv import JOINT_FILTERS
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.GymRunnerWrapper import GymRunnerWrapper
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
import gymnasium as gym
import copy
from rreal.algorithms.sac_helpers import build_vec_env

def format_tensor(t, float_precision):
    t = t.squeeze().cpu().tolist()
    if not isinstance(t,list):
        t = [t]
    t = [f"{e: .{float_precision}f}" if isinstance(e,float) else str(e) for e in t]
    return f"[{', '.join(t)}]"

def loco_run_builder(   seed,
                        log_folder,
                        env_builder_args : dict,
                        num_envs : int,
                        mode : str,
                        quiet : bool,
                        autoreset : bool = True):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    ggLog.info(f"env_builder_args = {env_builder_args}")
    env_builder_args = copy.deepcopy(env_builder_args)
    stepLength_sec = env_builder_args.pop("stepLength_sec")
    video_save_freq = env_builder_args.pop("video_save_freq")
    th_device = env_builder_args.pop("th_device")
    # max_steps = 5/stepLength_sec
    max_steps = env_builder_args.pop("max_steps_per_episode")
    show_gui = env_builder_args.pop("show_gui",False)

    model_file = env_builder_args["model_file"]
    homing_joint_pose = env_builder_args["homing_joint_pose"]
    robot_name = env_builder_args["robot_name"]
    robot_main_body_link = env_builder_args["robot_main_body_link"]
    robot_root_link = env_builder_args["robot_root_link"]
    homing_body_pose_xyz_xyzw = env_builder_args["homing_body_pose_xyz_xyzw"]
    disallowed_contact_links = env_builder_args["disallowed_contact_links"]
    terminating_contact_pairs = env_builder_args["terminating_contact_pairs"]
    controlled_joints = env_builder_args["controlled_joints"]

    if mode == "gz":
        raise NotImplementedError()
    elif mode == "gazebo":
        raise NotImplementedError()
    elif mode == "xbot":
        raise NotImplementedError()
    elif mode == "xbot-gazebo":
        raise NotImplementedError()
    elif mode == "pybullet":
        from adarl.adapters.VecPyBulletJointImpedanceAdapter import VecPyBulletJointImpedanceAdapter
        env_controller = VecPyBulletJointImpedanceAdapter(stepLength_sec=stepLength_sec,
                                                       restore_on_reset=False,
                                                       debug_gui=show_gui,
                                                       simulation_step=1/1024,
                                                       enable_rendering=env_builder_args.pop("enable_rendering"),
                                                       global_max_torque_position_control = 100,
                                                       real_time_factor=None,
                                                       vec_size=num_envs,
                                                       th_device=th_device)
    elif mode == "mjx":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        import jax
        env_controller = MjxJointImpedanceAdapter(vec_size=num_envs,
                                                  enable_rendering=env_builder_args.pop("enable_rendering"),
                                                  jax_device=jax.devices("gpu")[0],
                                                  output_th_device = th_device,
                                                  sim_step_dt=2/1024,
                                                  step_length_sec=stepLength_sec,
                                                  realtime_factor=-1.0,
                                                  gui_env_index=0,
                                                  default_max_joint_impedance_ctrl_torque=100.0,
                                                  show_gui=False,
                                                  log_freq=100)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    time.sleep(1)

    
    urdf_string = adarl.utils.utils.compile_xacro_string(   model_definition_string=Path(model_file).read_text(),
                                                            model_kwargs={"use_cylinders" : "false"})

    lrenv = LocomotionVecEnv(action_delay_mustd = env_builder_args.pop("action_delay_mustd"),
                            action_noise_mustd = env_builder_args.pop("action_noise_mustd"), 
                            action_smoothing_halflife_sec=env_builder_args.pop("action_smoothing_halflife_sec"),
                            adapter=env_controller,
                            control_mode = env_builder_args.pop("control_mode"),
                            controlled_joints=controlled_joints,
                            goal_err_smoothing_halflife_sec = env_builder_args.pop("goal_err_smoothing_halflife_sec"),
                            maxStepsPerEpisode=max_steps,
                            minmax_damping=(1.0,30.0),
                            minmax_stiffness=(50.0,1000.0),
                            robot_main_body_link=robot_main_body_link,
                            robot_name=robot_name,
                            robot_root_link=robot_root_link,
                            robot_urdf_string=urdf_string,
                            safe_damping=env_builder_args.pop("safe_damping"),
                            safe_stiffness=env_builder_args.pop("safe_stiffness"),
                            safety_limits_factor=0.9,
                            seed=seed,
                            stepLength_sec=stepLength_sec,
                            step_precision_tolerance=0 if isinstance(env_controller, BaseSimulationAdapter) else 0.001,
                            stop_on_safety=env_builder_args.pop("stop_on_safety"),
                            th_device=th_device,
                            homing_joint_pose=homing_joint_pose,
                            frame_stack_length=env_builder_args.pop("frame_stack_length"),
                            observe_body_velocity=True,
                            homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                            control_limits_minmax_pve={},
                            verbose_infos=env_builder_args.pop("verbose_infos"),
                            quiet=quiet,
                            enable_dbg_checks=True,
                            initial_pose_randomization = env_builder_args.pop("initial_pose_randomization"),
                            init_on_reset_ratio = env_builder_args.pop("init_on_reset_ratio"),
                            obs_noise_joints_pve_ep_mustd_step_std = env_builder_args.pop("obs_noise_joints_pve_ep_mustd_step_std"),
                            obs_noise_linvel_ep_mustd_step_std = env_builder_args.pop("obs_noise_linvel_ep_mustd_step_std"),
                            obs_noise_angvel_ep_mustd_step_std = env_builder_args.pop("obs_noise_angvel_ep_mustd_step_std"),
                            obs_noise_posz_ep_mustd_step_std = env_builder_args.pop("obs_noise_posz_ep_mustd_step_std"),
                            obs_noise_gravity_ep_mustd_step_std = env_builder_args.pop("obs_noise_gravity_ep_mustd_step_std"),
                            ui_camera_resolution_hw=env_builder_args.pop("ui_camera_resolution_hw"),
                            reward_acceleration_weight = env_builder_args.pop("reward_acceleration_weight"),
                            reward_actdiff_weight = env_builder_args.pop("reward_actdiff_weight"),
                            reward_contacts_weight = env_builder_args.pop("reward_contacts_weight"),
                            reward_energy_weight = env_builder_args.pop("reward_energy_weight"),
                            reward_health_weight = env_builder_args.pop("reward_health_weight"),
                            reward_position_limit_weight = env_builder_args.pop("reward_position_limit_weight"),
                            reward_scale=1000/max_steps,
                            reward_torque_limit_weight = env_builder_args.pop("reward_torque_limit_weight"),
                            reward_torque_weight = env_builder_args.pop("reward_torque_weight"),
                            reward_torquediff_weight = env_builder_args.pop("reward_torquediff_weight"),
                            reward_tracking_weight = env_builder_args.pop("reward_tracking_weight"),
                            reward_velocity_limit_weight = env_builder_args.pop("reward_velocity_limit_weight"),
                            reward_velocity_weight = env_builder_args.pop("reward_velocity_weight"),
                            reward_height_weight=env_builder_args.pop("reward_height_weight"),
                            reward_pitchnroll_weight=env_builder_args.pop("reward_pitchnroll_weight"),
                            reward_position_weight=env_builder_args.pop("reward_position_weight"),
                            disallowed_contact_links = disallowed_contact_links,
                            goal_speed_minmax=env_builder_args.pop("goal_speed_minmax"),
                            use_contacts=env_builder_args.pop("use_contacts"),
                            terminating_contact_pairs=terminating_contact_pairs if env_builder_args.pop("terminate_on_body_contact") else [],
                            )
    # ggLog.info(f"state_space = {lrenv.state_space}")
    # ggLog.info(f"observation_space = {lrenv.observation_space}")
    # ggLog.info(f"action_space = {lrenv.action_space.shape}")
    vrunner = EnvRunner(env=lrenv, verbose=True, quiet=False, episodeInfoLogFile=log_folder+"/vec_runner.log",
                        render_envs=[0], autoreset=autoreset,
                        log_freq = max_steps)
    vrunner = EnvRunnerRecorderWrapper(vrunner,
                                    fps = 1/stepLength_sec,
                                    outFolder=log_folder+"/RunnerRecorder",
                                    env_index=0,
                                    saveFrequency_ep=video_save_freq,
                                    publish=False,
                                    stream=True,
                                    vec_obs_key="vec",
                                    overlay_text_xy=(0.025,0.025),
                                    overlay_text_height=0.035,
                                    overlay_text_color_rgb=(255,150,0),
                                    overlay_text_func=lambda vo, a, r, te, tr, info:   
                                            f"\n"
                                            f"Step    {info['ep_step_count']: .3f}\n"+
                                            f"body_abs_linvel       {format_tensor(info['state_extrinsic'][[LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X, LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y, LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z]], 3)}\n"
                                            f"goal_abs              {format_tensor(info['goal_abs_xyz_vec'], 3)}\n"
                                            f"goal_rel              {format_tensor(info['goal_rel_xyz_vec'], 3)}\n"
                                            f"smoothed_linvel_error {format_tensor(info['smoothed_linvel_error'], 3)}\n"
                                            f"linvel_error          {format_tensor(info['linvel_error'], 3)}\n"
                                            f"safety                {info['state_internal'][LocomotionVecEnv.INTERNAL_FIELDS.SAFETY_TRIGGERED]: .2f}\n")
    return vrunner


def loco_env_builder(   seed,
                        log_folder,
                        env_builder_args : dict,
                        num_envs : int):
    mode = env_builder_args["mode"].strip().lower()
    quiet = env_builder_args["quiet"]
    stepLength_sec = env_builder_args["stepLength_sec"]

    if mode == "pybullet":
        device = env_builder_args["th_device"]
        def single_env_builder(seed : int, log_folder : str, is_eval : bool, env_builder_args : dict[str, Any]):
            vrunner = loco_run_builder( seed = seed,
                                    log_folder = log_folder,
                                    env_builder_args = env_builder_args,
                                    num_envs = 1,
                                    mode = mode,
                                    quiet=quiet,
                                    autoreset = False)
            return GymRunnerWrapper(runner=vrunner, quiet=quiet), 1/stepLength_sec
        from jumping_leg.experiments.build_quad import quad_env_builder
        env = build_vec_env(env_builder=single_env_builder,
                            env_builder_args=env_builder_args,
                            log_folder=log_folder,
                            seed=seed,
                            num_envs=num_envs,
                            collector_device=device,
                            env_action_device = device)
    else:
        vrunner = loco_run_builder( seed = seed,
                                    log_folder = log_folder,
                                    env_builder_args = env_builder_args,
                                    num_envs = num_envs,
                                    mode = mode,
                                    quiet=quiet)
        env = GymVecRunnerWrapper(runner=vrunner, quiet=quiet)
    
    # if video_save_freq >0:
    #     env = wrap_with_recorder(env,
    #                              stepLength_sec=stepLength_sec,
    #                              log_folder=log_folder,
    #                              video_save_freq=video_save_freq)
    env.reset(seed=seed)
    if len(env_builder_args)>0:
        ggLog.warn(f"Unused env_builder_args: {env_builder_args}")
    return env, 1/stepLength_sec

video_recorder_kwargs : dict[str,typing.Any]  = {}

def wrap_with_recorder(env, stepLength_sec, log_folder, video_save_freq):
    return RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq,
                                **video_recorder_kwargs)

def quad_loco_env_builder(seed : int,
                    run_folder : str,
                    num_envs : int, 
                    env_builder_args : dict) -> gym.vector.VectorEnv:
    import adarl.utils.utils
    env_builder_args["model_file"] = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/quad_simple.urdf.xacro")
    env_builder_args["homing_joint_pose"]={ ("quad","hip_joint_x_back_left") : -3.14159*0.4,
                                            ("quad","hip_joint_x_back_right") : -3.14159*0.4,
                                            ("quad","hip_joint_x_front_left") : -3.14159*0.4,
                                            ("quad","hip_joint_x_front_right") : -3.14159*0.4,
                                            ("quad","hip_joint_y_back_left") : 0.75,
                                            ("quad","hip_joint_y_back_right") : 0.75,
                                            ("quad","hip_joint_y_front_left") : 0.75,
                                            ("quad","hip_joint_y_front_right") : 0.75,
                                            ("quad","knee_joint_back_left") : 1.8,
                                            ("quad","knee_joint_back_right") : 1.8,
                                            ("quad","knee_joint_front_left") : 1.8,
                                            ("quad","knee_joint_front_right") : 1.8}
    env_builder_args["robot_name"]="quad"
    env_builder_args["robot_main_body_link"]="body_link"
    env_builder_args["robot_root_link"]="body_link"
    env_builder_args["homing_body_pose_xyz_xyzw"]=(0.,0.,0.5,0.,0.,0.,1.)
    env_builder_args["disallowed_contact_links"] = [("quad","thigh_link_back_left"),
                                                    ("quad","shin_link_back_left"),
                                                    ("quad","thigh_link_back_right"),
                                                    ("quad","shin_link_back_right"),
                                                    ("quad","thigh_link_front_left"),
                                                    ("quad","shin_link_front_left"),
                                                    ("quad","thigh_link_front_right"),
                                                    ("quad","shin_link_front_right"),
                                                    ("quad","body_link")]
    env_builder_args["terminating_contact_pairs"]=[(("quad","body_link"),("ground_plane","planeLink"))]
    env_builder_args["controlled_joints"] = [JOINT_FILTERS.ALL_REVOLUTE]
    return loco_env_builder(seed = seed,
                            log_folder = run_folder,
                            env_builder_args = env_builder_args,
                            num_envs=num_envs)[0]

def jumping_leg_builder(seed : int,
                    run_folder : str,
                    num_envs : int, 
                    env_builder_args : dict) -> gym.vector.VectorEnv:
    import adarl.utils.utils
    env_builder_args["model_file"] = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")
    env_builder_args["homing_joint_pose"]={ ("leg","rail_joint") : 1.0,
                        ("leg","hip_joint_1") : 1.0,
                        ("leg","knee_joint_1") : 1.4}
    env_builder_args["robot_name"] = "leg"
    env_builder_args["robot_main_body_link"] = "slider_link"
    env_builder_args["robot_root_link"] = "slider_link"
    env_builder_args["homing_body_pose_xyz_xyzw"] = (0.,0.,0.0,0.,0.,0.,1.)
    env_builder_args["disallowed_contact_links"]  = []
    env_builder_args["terminating_contact_pairs"] = []
    env_builder_args["controlled_joints"] = [JOINT_FILTERS.ALL_REVOLUTE]

    return loco_env_builder(seed = seed,
                            log_folder = run_folder,
                            env_builder_args = env_builder_args,
                            num_envs=num_envs)[0]


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams
    import os
    
    step_length_sec = 52/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=100 #int(ep_duration_sec/step_length_sec)
    train_envs = 1
    env_device = th.device("cpu",0)
    eval_freq = 5
    env_builder_args = {
        "action_delay_mustd" : (0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "action_smoothing_halflife_sec" : 0.1,
        "control_mode" : "position",
        "enable_rendering" : False,
        "goal_err_smoothing_halflife_sec" : 0.2,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : "pybullet",
        "quiet" : True,
        "initial_pose_randomization" : 0.0,
        "reward_acceleration_weight" : 0.1,
        "reward_actdiff_weight" : 0.,
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_health_weight" : 0.0,
        "reward_position_limit_weight" : 0.5,
        "reward_torque_limit_weight" : 0.0,
        "reward_torque_weight" : 1.0,
        "reward_torquediff_weight" : 0.0,
        "reward_tracking_weight" : 2.0,
        "reward_velocity_limit_weight" : 0.5,
        "reward_velocity_weight" : 1.0,
        "reward_height_weight" : 1.0,
        "reward_pitchnroll_weight" : 1.0,
        "reward_position_weight" : 0.1,
        "safe_stiffness" : 400,
        "safe_damping" : 10,
        "stepLength_sec" : step_length_sec,
        "stop_on_safety" : False,
        "th_device" : env_device,
        "video_save_freq" : -1,
        "goal_speed_minmax" : (0,2),
        "use_contacts" : False,
        "frame_stack_length" : 1,
        "verbose_infos" : False,
        "terminate_on_body_contact" : False,
        "use_wandb" : False,
        "init_on_reset_ratio" : 0.8,
        "obs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "obs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.0),
        "obs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.0),
        "ui_camera_resolution_hw" : (144,256)
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
    video_eval_env_builder_args["verbose_infos"] = True
    video_eval_env_builder_args["video_save_freq"] = 1
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : eval_freq*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    # video_norand_eval_env_builder_args = copy.deepcopy(env_builder_args)
    # video_norand_eval_env_builder_args["enable_rendering"] = True
    # video_norand_eval_env_builder_args["verbose_infos"] = True
    # video_norand_eval_env_builder_args["video_save_freq"] = 1
    # video_norand_eval_env_builder_args["initial_pose_randomization"] = 0.0
    # eval_conf_video_norand_det = {
    #     "name" : "video_norand_stoch",
    #     "deterministic" : False,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_norand_eval_env_builder_args,
    #     "num_envs" : 1
    # }
    # run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    # run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    # run_1ms_env_builder_args["enable_rendering"] = True
    # run_1ms_env_builder_args["verbose_infos"] = True
    # run_1ms_env_builder_args["video_save_freq"] = 1
    # run_1ms_env_builder_args["initial_pose_randomization"] = 0.0
    # eval_conf_run_1ms = {
    #     "name" : "run_1ms",
    #     "deterministic" : False,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : run_1ms_env_builder_args,
    #     "num_envs" : 1
    # }
    # video_feasible_env_builder_args = copy.deepcopy(feasible_env_builder_args)
    # video_feasible_env_builder_args["enable_rendering"] = True
    # video_feasible_env_builder_args["video_save_freq"] = 1
    # video_feasible_env_builder_args["randomize_initial_pose"] = False
    # eval_conf_video_feasible = {
    #     "name" : "video_feasible",
    #     "deterministic" : True,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_feasible_env_builder_args,
    #     "num_envs" : 1
    # }
    # video_feasible_jump_env_builder_args = copy.deepcopy(video_feasible_env_builder_args)
    # video_feasible_jump_env_builder_args["leg_min_jump"] = 0.2
    # eval_conf_video_jump_feasible = {
    #     "name" : "video_jump_feasible",
    #     "deterministic" : True,
    #     "eval_freq_ep" : 10*train_envs,
    #     "eval_eps" : 1,
    #     "env_builder_args" : video_feasible_jump_env_builder_args,
    #     "num_envs" : 1
    # }
    eval_configuration = [  
                            eval_conf_video_det,
                            # eval_conf_video_stoch,
                            # eval_conf_run_1ms,
                            # eval_conf_video_norand_det,
                            # #  eval_conf_feasible,
                            # #  eval_conf_video_feasible,
                            # #  eval_conf_video_jump_feasible
                                ]
    sac_train(  seed,
                folderName,
                run_id,
                args,
                vec_env_builder = quad_loco_env_builder,
                env_builder = None,
                env_builder_args = env_builder_args,
                eval_configurations = eval_configuration,
                hyperparams = SAC_hyperparams(  device = "cuda",
                                                q_network_arch=[256,128],
                                                q_lr=0.001,
                                                policy_lr=0.001,
                                                policy_network_arch=[1024,512],
                                                gamma=0.99,
                                                target_tau = 0.005,
                                                batch_size=16384,
                                                buffer_size=1_000_000,
                                                total_steps=100_000_000,
                                                train_freq_vstep=5,
                                                grad_steps=10,
                                                learning_starts=max_steps_per_episode*max(train_envs*5, 100),
                                                parallel_envs=train_envs,
                                                log_freq_vstep=max_steps_per_episode,
                                                reference_init_args =  #{}
                                                                        {   "env_builder_args" : env_builder_args,
                                                                            "eval_configuration" : eval_configuration}
                                                ),
                checkpoint_freq=20,
                collector_device=env_device,
                debug_level=10,
                max_episode_duration=max_steps_per_episode,
                validation_buffer_size=100_000,
                validation_batch_size=250,
                validation_holdout_ratio=0.01,
                no_wandb=args["no_wandb"])



if __name__ == "__main__":

    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--no-wandb", default=False, action='store_true', help="Disable Weight&Biases")

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
                start_adarl=False,
                pkgs_to_save=["adarl","jumping_leg","rreal"])
