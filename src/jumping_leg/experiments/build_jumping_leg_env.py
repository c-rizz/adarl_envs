
from jumping_leg.env.leg_jump_env import LegJumpEnv
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

def env_builder(seed,
                log_folder,
                is_eval,
                env_builder_args,
                no_dict = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    stepLength_sec = env_builder_args["stepLength_sec"]
    video_save_freq = env_builder_args["video_save_freq"]
    th_device = env_builder_args["th_device"]
    # max_steps = 5/stepLength_sec
    max_steps = env_builder_args["max_steps_per_episode"]

    mode = env_builder_args["mode"].strip().lower()
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
                                        fallback_cmd_damping = 100.0,
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
                                        fallback_cmd_damping = 100.0,
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
                                                       enable_redering=is_eval)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    print(f"env_builder_args = {env_builder_args}")
    time.sleep(1)

    lrenv = LegJumpEnv( maxStepsPerEpisode=max_steps,
                        stepLength_sec=stepLength_sec,
                        environmentController=env_controller,
                        seed=seed,
                        obs_only_vec=env_builder_args["obs_only_vec"],
                        obs_only_img=False,
                        obs_img_height=64,
                        obs_img_width=64,
                        rgb=True,
                        th_device=th_device,
                        reward_torque_limit_weight = env_builder_args["reward_torque_limit_weight"],
                        reward_position_limit_weight = env_builder_args["reward_position_limit_weight"],
                        reward_velocity_weight = env_builder_args["reward_velocity_weight"],
                        reward_energy_weight = env_builder_args["reward_energy_weight"],
                        reward_tracking_weight = env_builder_args["reward_tracking_weight"],
                        reward_torque_weight = env_builder_args["reward_torque_weight"],
                        reward_contacts_weight = env_builder_args["reward_contacts_weight"],
                        control_mode = env_builder_args["control_mode"],
                        reward_scale=500/max_steps,
                        platform_randomization = env_builder_args["platform_randomization"],
                        use_contacts=env_builder_args["use_contacts"],
                        step_precision_tolerance=0 if isinstance(env_controller, BaseSimulationAdapter) else 0.001,
                        ep_obs_noise_mustd=env_builder_args["ep_obs_noise_mustd"],
                        step_obs_noise_std=env_builder_args["step_obs_noise_std"],
                        stop_on_safety=env_builder_args["stop_on_safety"],
                        leg_min_height = env_builder_args["leg_min_height"],
                        leg_max_height = env_builder_args["leg_max_height"],
                        leg_max_jump = env_builder_args["leg_max_jump"]) # scale it to be the same as if we have 500 steps (mostly so that we can compare easily)
    if no_dict:
        from adarl.envs.lr_wrappers.ObsDict2FlatBox import ObsDict2FlatBox
        lrenv = ObsDict2FlatBox(lrenv, "vec")
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log",
                        quiet=env_builder_args["quiet"])
    
    if video_save_freq >0:
        env = wrap_with_recorder(env,
                                 stepLength_sec=stepLength_sec,
                                 log_folder=log_folder,
                                 video_save_freq=video_save_freq)
    env.reset(seed=seed)
    return env, 1/stepLength_sec

video_recorder_kwargs : dict[str,typing.Any]  = dict(vec_obs_key="vec",
                            overlay_text_xy=(0.025,0.025),
                            overlay_text_height=0.035,
                            overlay_text_func=lambda vo, a, r, te, tr, info:   f"Step    {info['step_count']: .3f}\n"+
                                    f"ImpSum  {info['impulses_sum']: .3f}\n"+
                                    f"ExtWork {info['external_work']:+.3f}\n"+
                                    f"TotEner {info['new_thigh_energy']+info['new_shin_energy']+info['new_slider_energy']:+.3f}\n"+
                                    f"ThiWork {info['thigh_work']:+.3f}\n"+
                                    f"ShiWork {info['shin_work']:+.3f}\n"+
                                    f"SliWork {info['slider_work']:+.3f}\n"+
                                    f"TotWork {info['slider_work']+info['shin_work']+info['thigh_work']:+.3f}\n"+
                                    f"ThiJWor {info['thigh_joint_work']:+.3f}\n"+
                                    f"ShiJWor {info['shin_joint_work']:+.3f}\n"+
                                    f"ThiEner {info['new_thigh_energy']:+.3f}\n"+
                                    f"ShiEner {info['new_shin_energy']:+.3f}\n"+
                                    f"SliEner {info['new_slider_energy']:+.3f}\n"+
                                    f"rContac {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT]:.2f}\n"+
                                    f"rEnergy {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT]:.2f}\n"+
                                    f"rImpThr {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD]:.2f}\n"+
                                    f"rPosLim {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT]:.2f}\n"+
                                    f"rTorLim {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT]:.2f}\n"+
                                    f"rTorque {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT]:.2f}\n"+
                                    f"rTrack  {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT]:.2f}\n"+
                                    f"rVeloci {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT]:.2f}\n"
                                    f"goal_z   {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.HIP_GOAL_Z]:.2f}\n"
                                    f"hip_z    {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.HIP_POS_Z]:.2f}\n"
                                    f"torque   {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_EFFORT]:.2f}, {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]:.2f}\n"
                                    f"position {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_POS]:.2f}, {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_POS]:.2f}\n"
                                    f"velocity {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_VEL]:.2f}, {info['cbstate'][LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_VEL]:.2f}\n"
                                    f"act     "+str([f"{ae:.3f}" for ae in a] if a is not None else None)
                                    )
def wrap_with_recorder(env, stepLength_sec, log_folder, video_save_freq):
    return RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq,
                                **video_recorder_kwargs)