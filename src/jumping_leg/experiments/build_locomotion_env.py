
from jumping_leg.env.LocomotionEnv import LocomotionEnv
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
from gymnasium.wrappers.normalize import NormalizeObservation
import threading, os
import traceback
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import typing 
from pathlib import Path
import adarl.utils.utils

def env_builder(seed,
                log_folder,
                is_eval,
                env_builder_args : dict,
                no_dict = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    ggLog.info(f"env_builder_args = {env_builder_args}")
    stepLength_sec = env_builder_args.pop("stepLength_sec")
    video_save_freq = env_builder_args.pop("video_save_freq")
    th_device = env_builder_args.pop("th_device")
    # max_steps = 5/stepLength_sec
    max_steps = env_builder_args.pop("max_steps_per_episode")

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
                                                       global_max_torque_position_control = 100)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    time.sleep(1)

    model_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/quad_simple.urdf.xacro")
    urdf_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(model_file).read_text())

    lrenv = LocomotionEnv(  action_delay_mustd = (0.0,0.0),
                            action_noise_mustd = (0.0,0.0), 
                            action_smoothing_halflife_sec=0.01,
                            adapter=env_controller,
                            control_mode = env_builder_args.pop("control_mode"),
                            controlled_joints=[LocomotionEnv.JOINT_FILTERS.ALL_REVOLUTE],
                            goal_err_smoothing_halflife_sec = env_builder_args.pop("goal_err_smoothing_halflife_sec"),
                            robot_main_body_link="body_link",
                            maxStepsPerEpisode=max_steps,
                            minmax_damping=(1.0,30.0),
                            minmax_stiffness=(50.0,1000.0),
                            obs_noise_ep_mustd=(0.0,0.0),
                            obs_noise_step_std=0.0,
                            reward_acceleration_weight = env_builder_args.pop("reward_acceleration_weight"),
                            reward_contacts_weight = env_builder_args.pop("reward_contacts_weight"),
                            reward_energy_weight = env_builder_args.pop("reward_energy_weight"),
                            reward_position_limit_weight = env_builder_args.pop("reward_position_limit_weight"),
                            reward_scale=500/max_steps,
                            reward_torque_limit_weight = env_builder_args.pop("reward_torque_limit_weight"),
                            reward_torque_weight = env_builder_args.pop("reward_torque_weight"),
                            reward_torquediff_weight = env_builder_args.pop("reward_torquediff_weight"),
                            reward_tracking_weight = env_builder_args.pop("reward_tracking_weight"),
                            reward_velocity_limit_weight = env_builder_args.pop("reward_velocity_limit_weight"),
                            reward_velocity_weight = env_builder_args.pop("reward_velocity_weight"),
                            robot_name="quadruped",
                            robot_urdf_string=urdf_string,
                            safety_limits_factor=0.9,
                            seed=seed,
                            stepLength_sec=stepLength_sec,
                            step_precision_tolerance=0 if isinstance(env_controller, BaseSimulationAdapter) else 0.001,
                            stop_on_safety=env_builder_args.pop("stop_on_safety"),
                            th_device=th_device,
                            safe_damping=env_builder_args.pop("safe_damping"),
                            safe_stiffness=env_builder_args.pop("safe_stiffness")
                            )
    if no_dict:
        from adarl.envs.lr_wrappers.ObsDict2FlatBox import ObsDict2FlatBox
        lrenv = ObsDict2FlatBox(lrenv, "vec")
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log",
                        quiet=env_builder_args.pop("quiet"),
                        use_wandb=False)
    
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
                                    f"goal_xy  {info['state_internal'][[LocomotionEnv.INTERNAL_FIELDS.GOAL_VELOCITY_X,LocomotionEnv.INTERNAL_FIELDS.GOAL_VELOCITY_Y]].cpu().tolist()}\n"
                                    f"safety   {info['state_internal'][LocomotionEnv.INTERNAL_FIELDS.SAFETY_TRIGGERED]:.2f}\n"
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