
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
from jumping_leg.env.RobotEnv import RobotEnv

def locomotion_env_builder( seed,
                            log_folder,
                            is_eval,
                            env_builder_args : dict,
                            model_file : str,
                            homing_joint_pose : dict[tuple[str,str],float],
                            disallowed_contact_links : list[tuple[str,str]],
                            terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]],
                            robot_name : str,
                            robot_main_body_link : str,
                            robot_root_link : str,
                            homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                            controlled_joints : Sequence[str | RobotEnv.JOINT_FILTERS],
                            no_dict = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    ggLog.info(f"env_builder_args = {env_builder_args}")
    stepLength_sec = env_builder_args.pop("stepLength_sec")
    video_save_freq = env_builder_args.pop("video_save_freq")
    th_device = env_builder_args.pop("th_device")
    # max_steps = 5/stepLength_sec
    max_steps = env_builder_args.pop("max_steps_per_episode")
    quiet = env_builder_args.pop("quiet")

    mode = env_builder_args.pop("mode").strip().lower()
    if mode == "gz":
        from adarl_ros2.adapters.GzController import GzController
        env_controller = GzController(stepLength_sec=stepLength_sec)
    elif mode == "gazebo":
        from adarl_ros.adapters.GazeboAdapter import GazeboAdapter
        env_controller = GazeboAdapter(stepLength_sec=stepLength_sec)
    elif mode == "xbot":
        from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter
        env_controller = RosXbotAdapter(model_name = "leg",
                                        stepLength_sec = stepLength_sec,
                                        forced_ros_master_uri = None,
                                        maxObsDelay = float("+inf"),
                                        blocking_observation = False,
                                        is_floating_base = True,
                                        reference_frame = "base_link",
                                        torch_device = th.device("cpu"),
                                        fallback_cmd_stiffness = 200.0,
                                        fallback_cmd_damping = 10.0,
                                        allow_fallback = True,
                                        jpos_cmd_max_vel = {},
                                        jpos_cmd_max_vel_default = 1.0,
                                        jpos_cmd_max_acc = {},
                                        jpos_cmd_max_acc_default = 1.0)
    elif mode == "xbot-gazebo":
        from adarl_ros.adapters.RosXbotGazeboAdapter import RosXbotGazeboAdapter
        env_controller = RosXbotGazeboAdapter(model_name = "leg",
                                        stepLength_sec = stepLength_sec,
                                        forced_ros_master_uri = None,
                                        maxObsDelay = float("+inf"),
                                        blocking_observation = False,
                                        is_floating_base = True,
                                        reference_frame = "base_link",
                                        torch_device = th.device("cpu"),
                                        fallback_cmd_stiffness = 200.0,
                                        fallback_cmd_damping = 10.0,
                                        allow_fallback = True,
                                        jpos_cmd_max_vel = {},
                                        jpos_cmd_max_vel_default = 10.0,
                                        jpos_cmd_max_acc = {},
                                        jpos_cmd_max_acc_default = 10.0)
    elif mode == "pybullet":
        from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
        env_controller = PyBulletJointImpedanceAdapter(stepLength_sec=stepLength_sec,
                                                       restore_on_reset=False,
                                                       debug_gui=False,
                                                       simulation_step=1/1024,
                                                       enable_rendering=env_builder_args.pop("enable_rendering"),
                                                       global_max_torque_position_control = 100,
                                                       real_time_factor=None
                                                       )
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    time.sleep(1)

    
    urdf_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(model_file).read_text())

    lrenv = LocomotionEnv(  action_delay_mustd = env_builder_args.pop("action_delay_mustd"),
                            action_noise_mustd = env_builder_args.pop("action_noise_mustd"), 
                            action_smoothing_halflife_sec=env_builder_args.pop("action_smoothing_halflife_sec"),
                            adapter=env_controller,
                            control_mode = env_builder_args.pop("control_mode"),
                            controlled_joints=controlled_joints,
                            goal_err_smoothing_halflife_sec = env_builder_args.pop("goal_err_smoothing_halflife_sec"),
                            maxStepsPerEpisode=max_steps,
                            minmax_damping=(1.0,30.0),
                            minmax_stiffness=(50.0,1000.0),
                            obs_noise_ep_mustd=env_builder_args.pop("obs_noise_ep_mustd"),
                            obs_noise_step_std=env_builder_args.pop("obs_noise_step_std"),
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
                            disallowed_contact_links = disallowed_contact_links,
                            goal_speed_minmax=env_builder_args.pop("goal_speed_minmax"),
                            use_contacts=env_builder_args.pop("use_contacts"),
                            frame_stack_length=env_builder_args.pop("frame_stack_length"),
                            observe_body_velocity=True,
                            homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                            control_limits_minmax_pve={},
                            terminating_contact_pairs=terminating_contact_pairs if env_builder_args.pop("terminate_on_body_contact") else [],
                            verbose_infos=env_builder_args.pop("verbose_infos"),
                            quiet=quiet,
                            enable_dbg_checks=True,
                            randomize_initial_pose = env_builder_args.pop("randomize_initial_pose")
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

video_recorder_kwargs : dict[str,typing.Any]  = dict(vec_obs_key="vec",
                            overlay_text_xy=(0.025,0.025),
                            overlay_text_height=0.035,
                            overlay_text_func=lambda vo, a, r, te, tr, info:   
                                    f"\n"
                                    f"Step    {info['step_count']: .3f}\n"+
                                    # f"ImpSum  {info['impulses_sum']: .3f}\n"+
                                    # f"ExtWork {info['external_work']:+.3f}\n"+
                                    # f"TotEner {info['new_thigh_energy']+info['new_shin_energy']+info['new_slider_energy']:+.3f}\n"+
                                    # f"ThiWork {info['thigh_work']:+.3f}\n"+
                                    # f"ShiWork {info['shin_work']:+.3f}\n"+
                                    # f"SliWork {info['slider_work']:+.3f}\n"+
                                    # f"TotWork {info['slider_work']+info['shin_work']+info['thigh_work']:+.3f}\n"+
                                    # f"ThiJWor {info['thigh_joint_work']:+.3f}\n"+
                                    # f"ShiJWor {info['shin_joint_work']:+.3f}\n"+
                                    # f"ThiEner {info['new_thigh_energy']:+.3f}\n"+
                                    # f"ShiEner {info['new_shin_energy']:+.3f}\n"+
                                    # f"SliEner {info['new_slider_energy']:+.3f}\n"+
                                    # f"rContac {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT]:.2f}\n"+
                                    # f"rEnergy {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT]:.2f}\n"+
                                    # f"rImpThr {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD]:.2f}\n"+
                                    # f"rPosLim {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT]:.2f}\n"+
                                    # f"rTorLim {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT]:.2f}\n"+
                                    # f"rTorque {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT]:.2f}\n"+
                                    # f"rTrack  {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT]:.2f}\n"+
                                    # f"rVeloci {info['state'][LegJumpEnv.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT]:.2f}\n"
                                    f"goal_vel_rel       {info['state_loco'][[LocomotionEnv.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X]].cpu().item(): .3f}, " 
                                                     f"{info['state_loco'][[LocomotionEnv.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y]].cpu().item(): .3f}, "
                                                     f"{info['state_loco'][[LocomotionEnv.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z]].cpu().item(): .3f}\n"
                                    f"goal_vel       {info['goal_x']: .3f}, {info['goal_y']: .3f} \n"
                                    f"contacts_count {info['state_loco'][[LocomotionEnv.LOCOMOTION_FIELDS.COLLISON_COUNT]].cpu().item(): .3f}\n"
                                    f"body_vel_rel       {info['state_extrinsic'][[LocomotionEnv.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X]].cpu().item(): .3f}, "
                                                     f"{info['state_extrinsic'][[LocomotionEnv.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y]].cpu().item(): .3f}, "
                                                     f"{info['state_extrinsic'][[LocomotionEnv.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z]].cpu().item(): .3f}\n"
                                    f"track_err      {info['state_loco'][[LocomotionEnv.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]].cpu().item(): .3f}\n"
                                    f"safety         {info['state_internal'][LocomotionEnv.INTERNAL_FIELDS.SAFETY_TRIGGERED]: .2f}\n"
                                    # f"position {info['state_robot'][0]:.2f}, {info['state_robot'][8]:.2f}\n"
                                    # f"velocity {info['state_robot'][1]:.2f}, {info['state_robot'][8+1]:.2f}\n"
                                    # f"torque   {info['state_robot'][2]:.2f}, {info['state_robot'][8+2]:.2f}\n"
                                    # f"act     "+str([f"{ae:.3f}" for ae in a] if a is not None else None)
                                    )

def wrap_with_recorder(env, stepLength_sec, log_folder, video_save_freq):
    return RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq,
                                **video_recorder_kwargs)