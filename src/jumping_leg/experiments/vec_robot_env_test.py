#!/usr/bin/env python3  
from __future__ import annotations



from __future__ import annotations
from jumping_leg.env.LocomotionEnv import LocomotionEnv
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
import threading, os
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import typing 
from pathlib import Path
import adarl.utils.utils
from typing import Sequence
from jumping_leg.env.RobotVecEnv import RobotVecEnv

def robot_env_builder( seed,
                            log_folder,
                            env_builder_args : dict,
                            model_file : str,
                            homing_joint_pose : dict[tuple[str,str],float],
                            robot_name : str,
                            robot_main_body_link : str,
                            robot_root_link : str,
                            homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                            controlled_joints : Sequence[str | RobotVecEnv.JOINT_FILTERS],
                            no_dict = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    ggLog.info(f"env_builder_args = {env_builder_args}")
    stepLength_sec = env_builder_args.pop("stepLength_sec")
    video_save_freq = env_builder_args.pop("video_save_freq")
    th_device = env_builder_args.pop("th_device")
    # max_steps = 5/stepLength_sec
    max_steps = env_builder_args.pop("max_steps_per_episode")
    quiet = env_builder_args.pop("quiet")
    show_gui = env_builder_args.pop("show_gui",False)

    mode = env_builder_args.pop("mode").strip().lower()
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
                                                       vec_size=1,
                                                       th_device=th.device("cpu"))
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    time.sleep(1)

    
    urdf_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(model_file).read_text())

    lrenv = RobotVecEnv(  action_delay_mustd = env_builder_args.pop("action_delay_mustd"),
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
                            ui_camera_resolution_hw=env_builder_args.pop("ui_camera_resolution_hw")
                            )
    # ggLog.info(f"state_space = {lrenv.state_space}")
    # ggLog.info(f"observation_space = {lrenv.observation_space}")
    # ggLog.info(f"action_space = {lrenv.action_space.shape}")


    if no_dict:
        from adarl.envs.lr_wrappers.ObsDict2FlatBox import ObsDict2FlatBox
        lrenv = ObsDict2FlatBox(lrenv, "vec")
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log",
                        quiet=quiet,
                        use_wandb=env_builder_args.pop("use_wandb"))
    
    if video_save_freq >0:
        env = wrap_with_recorder(env,
                                 stepLength_sec=stepLength_sec,
                                 log_folder=log_folder,
                                 video_save_freq=video_save_freq)
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

def quad_env_builder(seed,
                    log_folder,
                    is_eval,
                    env_builder_args : dict,
                    no_dict = False):
    import adarl.utils.utils
    model_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/quad_simple.urdf.xacro")
    homing_joint_pose={ ("quad","hip_joint_x_back_left") : -3.14159*0.4,
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
    robot_name="quad"
    robot_main_body_link="body_link"
    robot_root_link="body_link"
    homing_body_pose_xyz_xyzw=(0.,0.,0.5,0.,0.,0.,1.)

    return robot_env_builder(seed = seed,
                            log_folder = log_folder,
                            env_builder_args = env_builder_args,
                            model_file = model_file,
                            no_dict = no_dict,
                            homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                            homing_joint_pose=homing_joint_pose,
                            robot_name=robot_name,
                            robot_main_body_link=robot_main_body_link,
                            robot_root_link=robot_root_link,
                            controlled_joints=[RobotVecEnv.JOINT_FILTERS.ALL_REVOLUTE])


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams
    import os
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 100
    env_device = th.device("cpu")
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
        "initial_pose_randomization" : 0.25,
        "reward_acceleration_weight" : 0.1,
        "reward_actdiff_weight" : 0.1,
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
        "reward_position_weight" : 1.0,
        "safe_stiffness" : 400,
        "safe_damping" : 10,
        "stepLength_sec" : step_length_sec,
        "stop_on_safety" : False,
        "th_device" : env_device,
        "video_save_freq" : 0,
        "goal_speed_minmax" : (0,2),
        "use_contacts" : False,
        "frame_stack_length" : 1,
        "verbose_infos" : False,
        "terminate_on_body_contact" : False,
        "use_wandb" : False,
        "init_on_reset_ratio" : 0.9,
        "obs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.001),
        "obs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.001),
        "obs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.001),
        "obs_noise_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.001),
        "obs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.001),
        "ui_camera_resolution_hw" : (144,256)
    }
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["enable_rendering"] = True
    video_eval_env_builder_args["verbose_infos"] = True
    video_eval_env_builder_args["video_save_freq"] = 1
    eval_conf_video_det = {
        "name" : "video_det",
        "deterministic" : True,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1
    }
    video_norand_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_norand_eval_env_builder_args["enable_rendering"] = True
    video_norand_eval_env_builder_args["verbose_infos"] = True
    video_norand_eval_env_builder_args["video_save_freq"] = 1
    video_norand_eval_env_builder_args["initial_pose_randomization"] = 0.0
    eval_conf_video_norand_det = {
        "name" : "video_norand_stoch",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : video_norand_eval_env_builder_args,
        "num_envs" : 1
    }
    run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    run_1ms_env_builder_args["enable_rendering"] = True
    run_1ms_env_builder_args["verbose_infos"] = True
    run_1ms_env_builder_args["video_save_freq"] = 1
    run_1ms_env_builder_args["initial_pose_randomization"] = 0.0
    eval_conf_run_1ms = {
        "name" : "run_1ms",
        "deterministic" : False,
        "eval_freq_ep" : 10*train_envs,
        "eval_eps" : 1,
        "env_builder_args" : run_1ms_env_builder_args,
        "num_envs" : 1
    }
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
    eval_configuration = [  eval_conf_video_det,
                            eval_conf_video_stoch,
                            eval_conf_run_1ms,
                            eval_conf_video_norand_det,
                            #  eval_conf_feasible,
                            #  eval_conf_video_feasible,
                            #  eval_conf_video_jump_feasible
                                ]
    sac_train(  seed,
                folderName,
                run_id,
                args,
                env_builder = quad_env_builder,
                env_builder_args = env_builder_args,
                eval_configurations = eval_configuration,
                hyperparams = SAC_hyperparams(  device = "cuda",
                                                q_network_arch=[256,128],
                                                q_lr=0.001,
                                                policy_lr=0.0005,
                                                policy_network_arch=[1024,512],
                                                gamma=0.99,
                                                target_tau = 0.005,
                                                batch_size=16384,
                                                buffer_size=10_000_000,
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
                debug_level=0,
                max_episode_duration=max_steps_per_episode,
                validation_buffer_size=100_000,
                validation_batch_size=256,
                validation_holdout_ratio=0.01)



if __name__ == "__main__":

    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

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