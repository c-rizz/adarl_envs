#!/usr/bin/env python3  
from __future__ import annotations
import random
from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
import threading, os
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from pathlib import Path
import adarl.utils.utils
from adarl_envs.env.LocomotionVecEnv import LocomotionVecEnv
from adarl_envs.env.RobotVecEnv import JOINT_FILTERS, LINK_FILTERS
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.Runner2GymWrapper import Runner2GymWrapper
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
import gymnasium as gym
import copy
from rreal.algorithms.sac_helpers import build_vec_env, VecEnvRunnerBuilderProtocol
from math import pi

def format_tensor(t, float_precision):
    if t is None:
        return "None"
    if isinstance(t, float) or isinstance(t, int):
        t = th.as_tensor(t)
    t = t.squeeze().cpu().tolist()
    if not isinstance(t,list):
        t = [t]
    t = [f"{e: .{float_precision}f}" if isinstance(e,float) else str(e) for e in t]
    return f"[{', '.join(t)}]"

def overlay_text_func(vo, a, r, te, tr, info, extra_info):   
    if 'state_extrinsic' in info:
        body_abs_linvel : th.Tensor = info['state_extrinsic'][[LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X, LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y, LocomotionVecEnv.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z]]
        body_abs_linvel_str = format_tensor(body_abs_linvel, 3)
    else:
        body_abs_linvel_str = 'N/A'
    if 'state_internal' in info:
        posref_safety_triggered = info['state_internal'][LocomotionVecEnv.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED] if 'state_internal' in info else 'N/A'
        limits_safety_triggered = info['state_internal'][LocomotionVecEnv.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED] if 'state_internal' in info else 'N/A'
    else:
        posref_safety_triggered = 'N/A'
        limits_safety_triggered = 'N/A'
    goal_abs_linvel_xyz = info.get('goal_abs_xyz_vec', None)
    vel_norm = f"{th.linalg.norm(goal_abs_linvel_xyz):.3f}" if goal_abs_linvel_xyz is not None else "N/A"
    return  (   f"\n"
                f"Step    {info['ep_step_count']: .3f}\n"+
                f"body_abs_linvel       {body_abs_linvel_str} ({th.linalg.norm(body_abs_linvel):.3f} m/s)\n"
                f"goal_vel_abs          {format_tensor(goal_abs_linvel_xyz, 3)} ({vel_norm} m/s)\n"
                f"goal_vel_rel          {format_tensor(info.get('goal_rel_xyz_vec',None), 3)}\n"
                f"smoothed_linvel_error {format_tensor(info.get('smoothed_linvel_error',None), 3)}\n"
                f"linvel_error          {format_tensor(info.get('linvel_error',None), 3)}\n"
                f"goal_height           {format_tensor(info.get('goal_height',None), 3)}\n"
                f"height_error          {format_tensor(info.get('height_err',None), 3)}\n"
                f"log_prob              {format_tensor(extra_info.get('act_log_prob',th.as_tensor(float('nan'))), 3)}\n"
                f"posref_safety         {posref_safety_triggered}\n"
                f"limits_safety         {limits_safety_triggered}\n"
                f"actacc_weight         {format_tensor(info.get('actacc_weight',float('nan')), 3)}\n"
                f"actdiff_weight        {format_tensor(info.get('actdiff_weight',float('nan')), 3)}\n"
                f"posrefvel_weight      {format_tensor(info.get('posref_vel_weight',float('nan')), 3)}\n"
                f"posrefacc_weight      {format_tensor(info.get('posref_acc_weight',float('nan')), 3)}\n")

def loco_runner_builder(seed,
                        run_folder,
                        num_envs : int,
                        env_builder_args : dict,
                        env_name : str = "",
                        autoreset : bool = True,
                        quiet : bool = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    ggLog.info(f"env_builder_args = {env_builder_args}")
    env_builder_args = copy.deepcopy(env_builder_args)
    stepLength_sec = env_builder_args.pop("stepLength_sec")
    th_device : th.device = env_builder_args["th_device"]
    th_device = th.device(th_device)
    if th_device.type == "cuda" and th_device.index is None:
        ggLog.info(f"Using generic torch device {th_device}")
        th_device = th.device("cuda", 0)
    ggLog.info(f"Using torch device {th_device}")
    show_gui = env_builder_args.pop("show_gui",False)
    robot_name = env_builder_args["robot_name"]
    max_steps = env_builder_args.pop("max_steps_per_episode")
    mode = env_builder_args["mode"]
    walltime_factor = env_builder_args.pop("walltime_factor")

    robot_urdf_string = adarl.utils.utils.compile_xacro_string(   model_definition_string=Path(env_builder_args.pop("model_file")).read_text(),
                                                            model_kwargs=env_builder_args.pop("model_kwargs"),
                                                            extra_pkg_paths=env_builder_args.pop("xacro_extra_pkg_paths"))
    
    if mode == "gz":
        raise NotImplementedError()
    elif mode == "gazebo":
        raise NotImplementedError()
    elif mode == "xbot":
        from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter
        from adarl_ros.adapters.VecRosXBotAdapterWrapper import VecRosXBotAdapterWrapper
        forced_ros_master_uri="http://127.0.0.1:11311"
        ground_link = ("ground_plane","ground_link")
        adapter = VecRosXBotAdapterWrapper(   adapter = RosXbotAdapter( model_name = robot_name,
                                                                        stepLength_sec = stepLength_sec,
                                                                        forced_ros_master_uri = forced_ros_master_uri,
                                                                        maxObsDelay = float("+inf"),
                                                                        blocking_observation = False,
                                                                        is_floating_base = True,
                                                                        reference_frame = "world",
                                                                        torch_device = th.device("cpu"),
                                                                        fallback_cmd_stiffness = 200.0,
                                                                        fallback_cmd_damping = 10.0,
                                                                        allow_fallback = False,
                                                                        jpos_cmd_max_vel = {},
                                                                        jpos_cmd_max_vel_default = 5.0,
                                                                        jpos_cmd_max_acc = {},
                                                                        jpos_cmd_max_acc_default = 5.0,
                                                                        enable_filters=True,
                                                                        position_commands_stiffness = 400.0,
                                                                        position_commands_damping = 10.0,
                                                                        walltime_factor=walltime_factor),
                                                vec_size = 1,
                                                th_device = th_device)
    elif mode == "xbot_zmq":
        from adarl.adapters.VecZmqXbotAdapter import VecZmqXbotAdapter
        from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter
        ground_link = ("ground_plane","ground_link") # Should not be used
        adapter = VecZmqXbotAdapter(   adapter = ZmqXbotAdapter( model_name = robot_name,
                                                                        stepLength_sec = stepLength_sec,
                                                                        is_floating_base = True,
                                                                        reference_frame = "world",
                                                                        torch_device = th.device("cpu"),
                                                                        allow_fallback = False,
                                                                        jpos_cmd_max_vel = {},
                                                                        jpos_cmd_max_vel_default = 5.0,
                                                                        jpos_cmd_max_acc = {},
                                                                        jpos_cmd_max_acc_default = 5.0,
                                                                        enable_filters = True,
                                                                        position_commands_stiffness = 400.0,
                                                                        position_commands_damping = 10.0,
                                                                        is_simulated = False,
                                                                        walltime_factor = 1.0,
                                                                        remote_ip = 'localhost',
                                                                        remote_port = 5557,
                                                                        remote_joint_state_port = 5556,
                                                                        remote_cmd_port = 5558,
                                                                        robot_urdf=robot_urdf_string),
                                                vec_size = 1,
                                                th_device = th_device)
    elif mode == "xbot-gazebo":
        from adarl_ros.adapters.RosXbotGazeboAdapter import RosXbotGazeboAdapter
        from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
        adapter = VecSimJointImpedanceAdapterWrapper(adapters = RosXbotGazeboAdapter(model_name = robot_name,
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
                                                                                    jpos_cmd_max_acc_default = 10.0),
                                                            th_device = th_device)
    elif mode == "pybullet":
        from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
        from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
        ground_link = ("ground_plane","ground_link")
        env_builder_args["enable_link_collisions"] = None
        adapter = VecSimJointImpedanceAdapterWrapper(adapters = PyBulletJointImpedanceAdapter(stepLength_sec=stepLength_sec,
                                                                            restore_on_reset=False,
                                                                            debug_gui=show_gui,
                                                                            simulation_step=1/1024,
                                                                            enable_rendering=env_builder_args.pop("enable_rendering"),
                                                                            global_max_torque_position_control = 100,
                                                                            real_time_factor=None,
                                                                            th_device=th_device),
                                                            th_device = th_device)
    elif mode == "mjx":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        import jax
        ground_link = ("ground","ground_link")
        robot_model = env_builder_args["robot_model"]
        sim_dt = 2/1024 if robot_model=="centauro" else 4/1024 
        iterations_per_ep = int(max_steps*stepLength_sec/sim_dt)
        opt_override = {}
        adapter = MjxJointImpedanceAdapter( vec_size=num_envs,
                                            enable_rendering=env_builder_args.pop("enable_rendering"),
                                            jax_device=jax.devices("gpu")[th_device.index] if th_device.type == "cuda" else jax.devices("cpu")[0],
                                            output_th_device = th_device,
                                            sim_step_dt=sim_dt,
                                            step_length_sec=stepLength_sec,
                                            realtime_factor=-1.0,
                                            gui_env_index=0,
                                            default_max_joint_impedance_ctrl_torque=100.0,
                                            show_gui=show_gui,
                                            log_freq=iterations_per_ep,
                                            record_whole_joint_trajectories = False,
                                            log_freq_joints_trajectories = iterations_per_ep,
                                            log_folder=run_folder,
                                            revolute_dof_armature_override=0.1,
                                            safe_revolute_dof_armature=0.1,
                                            opt_preset={"centauro":"fastest",
                                                        "kyon":"faster",
                                                        "quad":"fastest"}.get(robot_model, "faster"),
                                            opt_override=opt_override,
                                            reference_filter_cutoff_frequency=20.0)
    elif mode == "mujoco":
        from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
        from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
        ground_link = ("ground","ground_link")
        adapter = MujocoJointImpedanceAdapter(  step_length_sec=stepLength_sec,
                                                sim_step_dt=1/2048,
                                                output_th_device=th_device,
                                                reference_filter_cutoff_frequency=20.0)
    else:
        print(f"Requested unknown adapter '{mode}'")
        exit(0)

    time.sleep(1)

    
    lrenv = LocomotionVecEnv(action_delay_mustd_std = env_builder_args.pop("action_delay_mustd_std"),
                            action_noise_mustd = env_builder_args.pop("action_noise_mustd"), 
                            action_smoothing_halflife_sec=env_builder_args.pop("action_smoothing_halflife_sec"),
                            adapter=adapter,
                            control_mode = env_builder_args.pop("control_mode"),
                            controlled_joints=env_builder_args.pop("controlled_joints"),
                            disallowed_contact_links = env_builder_args.pop("disallowed_contact_links"),
                            enable_dbg_checks=True,
                            enable_limits_safety = env_builder_args.pop("enable_limits_safety"),
                            enable_link_collisions=env_builder_args.pop("enable_link_collisions"),
                            enable_posref_safety = env_builder_args.pop("enable_posref_safety"),
                            fail_on_safety=env_builder_args.pop("fail_on_safety"),
                            feet_links=env_builder_args.pop("feet_links"),
                            frame_stack_length=env_builder_args.pop("frame_stack_length"),
                            free_joints=[],
                            goal_err_smoothing_halflife_sec = env_builder_args.pop("goal_err_smoothing_halflife_sec"),
                            goal_height_minmax=env_builder_args.pop("goal_height_minmax"),
                            goal_resampling_probability_per_sec= env_builder_args.pop("goal_resampling_probability_per_sec"),
                            goal_speed_minmax=env_builder_args.pop("goal_speed_minmax"),
                            goal_yaw_minmax=env_builder_args.pop("goal_yaw_minmax"),
                            ground_link=ground_link,
                            held_joints_damping=env_builder_args.pop("held_joints_damping"),
                            held_joints_stiffness=env_builder_args.pop("held_joints_stiffness"),
                            homing_body_pose_xyz_xyzw=env_builder_args.pop("homing_body_pose_xyz_xyzw"),
                            homing_joint_pose=env_builder_args.pop("homing_joint_pose"),
                            impulse_duration_minmax=env_builder_args.pop("impulse_duration_minmax"),
                            impulse_mean_std=env_builder_args.pop("impulse_mean_std"),
                            impulse_probability_per_sec=env_builder_args.pop("impulse_probability_per_sec"),
                            init_on_reset_ratio = env_builder_args.pop("init_on_reset_ratio"),
                            initial_height_randomization_range_meters = env_builder_args.pop("initial_height_randomization_range_meters"),
                            initial_joint_pose_randomization_range = env_builder_args.pop("initial_joint_pose_randomization_range"),
                            just_health_reward = env_builder_args.pop("just_health_reward"),
                            longterm_states_decimation_time = env_builder_args.pop("longterm_states_decimation_time"),
                            max_goal_height_pos_change_speed=env_builder_args.pop("max_goal_height_pos_change_speed"),
                            maxStepsPerEpisode=max_steps,
                            max_good_step_duration=env_builder_args.pop("max_good_step_duration"),
                            merge_privileged = env_builder_args.pop("merge_privileged"),
                            min_good_step_duration=env_builder_args.pop("min_good_step_duration"),
                            minmax_damping=(1.0,30.0),
                            minmax_stiffness=(50.0,1000.0),
                            obs_noise_angvel_ep_mustd_step_std = env_builder_args.pop("obs_noise_angvel_ep_mustd_step_std"),
                            obs_noise_gravity_ep_mustd_step_std = env_builder_args.pop("obs_noise_gravity_ep_mustd_step_std"),
                            obs_noise_joints_pve_ep_mustd_step_std = env_builder_args.pop("obs_noise_joints_pve_ep_mustd_step_std"),
                            obs_noise_linacc_ep_mustd_step_std = env_builder_args.pop("obs_noise_linacc_ep_mustd_step_std"),
                            obs_noise_linvel_ep_mustd_step_std = env_builder_args.pop("obs_noise_linvel_ep_mustd_step_std"),
                            obs_noise_posz_ep_mustd_step_std = env_builder_args.pop("obs_noise_posz_ep_mustd_step_std"),
                            observe_full_robot_state = env_builder_args.pop("observe_full_robot_state"),
                            offset_envs_ep_starts = env_builder_args.pop("offset_envs_ep_starts"),
                            posref_safety_period = env_builder_args.pop("posref_safety_period"),
                            quiet=quiet,
                            randomized_armature_joints=env_builder_args.pop("randomized_armature_joints"),
                            randomized_armature_ratios= env_builder_args.pop("randomized_armature_ratios"),
                            randomized_com_links=env_builder_args.pop("randomized_com_links"),
                            randomized_com_xyz_diff_distribution=env_builder_args.pop("randomized_com_xyz_diff_distribution"),
                            randomized_friction_links=env_builder_args.pop("randomized_friction_links"),
                            randomized_friction_slide_spin_roll_ratios=env_builder_args.pop("randomized_friction_slide_spin_roll_ratios"),
                            randomized_frictionloss_joints=env_builder_args.pop("randomized_frictionloss_joints"),
                            randomized_frictionloss_ratios=env_builder_args.pop("randomized_frictionloss_ratios"),
                            randomized_gains_damping_ratio_epstd=env_builder_args.pop("randomized_gains_damping_ratio_epstd"),
                            randomized_gains_stiffness_ratio_epstd=env_builder_args.pop("randomized_gains_stiffness_ratio_epstd"),
                            randomized_mass_links=env_builder_args.pop("randomized_mass_links"),
                            randomized_mass_ratios_distr=env_builder_args.pop("randomized_mass_ratios"),
                            randomized_reference_filter_distribution=env_builder_args.pop("randomized_reference_filter_distribution"),
                            recycle_pose_randomization=env_builder_args.pop("recycle_pose_randomization"),
                            reward_superweight_joint_penalties = env_builder_args.pop("reward_superweight_joint_penalties"),    
                            reward_acceleration_weight = env_builder_args.pop("reward_acceleration_weight"),
                            reward_actacc_weight = env_builder_args.pop("reward_actacc_weight"),
                            reward_actdiff_weight = env_builder_args.pop("reward_actdiff_weight"),
                            reward_contacts_weight = env_builder_args.pop("reward_contacts_weight"),
                            reward_energy_weight = env_builder_args.pop("reward_energy_weight"),
                            reward_failure_weight = env_builder_args.pop("reward_failure_weight"),
                            reward_feet_air_time_weight = env_builder_args.pop("reward_feet_air_time_weight"),
                            reward_feet_ground_time_weight = env_builder_args.pop("reward_feet_ground_time_weight"),
                            reward_feet_on_ground_weight = env_builder_args.pop("reward_feet_on_ground_weight"),
                            reward_heading_weight = env_builder_args.pop("reward_heading_weight"),
                            reward_heading_velocity_weight = env_builder_args.pop("reward_heading_velocity_weight"),
                            reward_health_weight = env_builder_args.pop("reward_health_weight"),
                            reward_height_velocity_weight=env_builder_args.pop("reward_height_velocity_weight"),
                            reward_height_position_weight=env_builder_args.pop("reward_height_position_weight"),
                            reward_pitchnroll_weight=env_builder_args.pop("reward_pitchnroll_weight"),
                            reward_pitchnroll_velocity_weight=env_builder_args.pop("reward_pitchnroll_velocity_weight"),
                            reward_posref_vel_weight = env_builder_args.pop("reward_posref_vel_weight"),
                            reward_posref_acc_weight = env_builder_args.pop("reward_posref_acc_weight"),
                            reward_position_limit_weight = env_builder_args.pop("reward_position_limit_weight"),
                            reward_position_weight=env_builder_args.pop("reward_position_weight"),
                            reward_sensed_effort_weight = env_builder_args.pop("reward_sensed_effort_weight"),
                            reward_scale=1000/max_steps * env_builder_args.pop("reward_scale_nolength"),
                            reward_slip_weight = env_builder_args.pop("reward_slip_weight"),
                            reward_stand_position_weight = env_builder_args.pop("reward_stand_position_weight"),
                            reward_torque_limit_weight = env_builder_args.pop("reward_torque_limit_weight"),
                            reward_cmdtorque_weight = env_builder_args.pop("reward_torque_weight"),
                            reward_torquediff_weight = env_builder_args.pop("reward_torquediff_weight"),
                            reward_torqueref_weight = env_builder_args.pop("reward_torqueref_weight"),
                            reward_tracking_weight = env_builder_args.pop("reward_tracking_weight"),
                            reward_velocity_limit_weight = env_builder_args.pop("reward_velocity_limit_weight"),
                            reward_velocity_weight = env_builder_args.pop("reward_velocity_weight"),
                            reward_velref_weight = env_builder_args.pop("reward_velref_weight"),
                            robot_main_body_link=env_builder_args.pop("robot_main_body_link"),
                            robot_name=robot_name,
                            robot_root_link=env_builder_args.pop("robot_root_link"),
                            robot_urdf_string=robot_urdf_string,
                            safe_damping=env_builder_args.pop("safe_damping"),
                            control_limits_position_offset=env_builder_args.pop("control_limits_position_offset"),
                            safe_stiffness=env_builder_args.pop("safe_stiffness"),
                            safety_limits_ratios_minmax_pve=env_builder_args.pop("safety_limits_ratios_minmax_pve"),
                            control_limits_ratios_minmax_pve=env_builder_args.pop("control_limits_ratios_minmax_pve"),
                            saturate_jimp_ref_limits = env_builder_args.pop("saturate_jimp_ref_limits"),
                            seed=seed,
                            stepLength_sec=stepLength_sec,
                            step_precision_tolerance=0 if isinstance(adapter, BaseSimulationAdapter) else 0.001,
                            terminate_on_safety=env_builder_args.pop("terminate_on_safety"),
                            terminate_on_crash=env_builder_args.pop("terminate_on_crash"),
                            terminating_contact_pairs=env_builder_args.pop("terminating_contact_pairs") if env_builder_args.pop("terminate_on_body_contact") else [],
                            th_device=th_device,
                            ui_camera_resolution_hw=env_builder_args.pop("ui_camera_resolution_hw"),
                            use_contacts=env_builder_args.pop("use_contacts"),
                            verbose_infos=env_builder_args.pop("verbose_infos"),
                            split_rewards=env_builder_args.pop("split_rewards"),
                            minimal_infos=env_builder_args.pop("minimal_infos")
                            )
    # ggLog.info(f"state_space = {lrenv.state_space}")
    # ggLog.info(f"observation_space = {lrenv.observation_space}")
    # ggLog.info(f"action_space = {lrenv.action_space.shape}")
    vrunner = EnvRunner(env=lrenv, verbose=True, quiet=False, episodeInfoLogFile=run_folder+"/vec_runner.log",
                        ui_render_envs=[0], autoreset=autoreset,
                        log_freq = max_steps)
    if env_builder_args["video_save_freq"]>0:
        vrunner = EnvRunnerRecorderWrapper(vrunner,
                                        fps = 1/stepLength_sec,
                                        outFolder=run_folder+"/RunnerRecorder",
                                        env_index=0,
                                        saveFrequency_ep=env_builder_args.pop("video_save_freq"),
                                        publish=False,
                                        stream=True,
                                        vec_obs_key="base.vec", #TODO: somehow pass multiple keys and include privileged, or auto-detect which keys to save
                                        record_video=env_builder_args["record_video"],
                                        overlay_text_xy=(0.025,0.025),
                                        overlay_text_height=0.035,
                                        overlay_text_color_rgb=(255,150,0),
                                        overlay_text_func=overlay_text_func)
    return vrunner


def loco_env_builder(   seed : int,
                        log_folder : str,
                        is_eval : bool, 
                        env_builder_args : dict,
                        runner_builder : VecEnvRunnerBuilderProtocol):
    quiet = env_builder_args["quiet"]
    stepLength_sec = env_builder_args["stepLength_sec"]
    vrunner = runner_builder( seed = seed,
                                run_folder = log_folder,
                                env_builder_args = env_builder_args,
                                num_envs = 1,
                                quiet=quiet,
                                autoreset = False)
    return Runner2GymWrapper(runner=vrunner, quiet=quiet), 1/stepLength_sec
        

def loco_venv_builder(  seed,
                        log_folder,
                        env_builder_args : dict,
                        num_envs : int,
                        runner_builder : VecEnvRunnerBuilderProtocol):
    with th.no_grad():
        mode = env_builder_args["mode"].strip().lower()
        quiet = env_builder_args["quiet"]
        stepLength_sec = env_builder_args["stepLength_sec"]

        if mode == "pybullet":
            device = env_builder_args["th_device"]
            def env_builder(seed : int,
                            log_folder : str,
                            is_eval : bool, 
                            env_builder_args : dict):
                return loco_env_builder(seed=seed, log_folder=log_folder,is_eval=is_eval,env_builder_args=env_builder_args,runner_builder=runner_builder)
            env = build_vec_env(env_builder=env_builder,
                                env_builder_args=env_builder_args,
                                log_folder=log_folder,
                                seed=seed,
                                num_envs=num_envs,
                                collector_device=device,
                                env_action_device = device)
        else:
            vrunner = runner_builder( seed = seed,
                                        run_folder = log_folder,
                                        env_builder_args = env_builder_args,
                                        num_envs = num_envs,
                                        quiet=quiet)
            env = Runner2VecGymWrapper(runner=vrunner, quiet=quiet)
        
        # if video_save_freq >0:
        #     env = wrap_with_recorder(env,
        #                              stepLength_sec=stepLength_sec,
        #                              log_folder=log_folder,
        #                              video_save_freq=video_save_freq)
        env.reset(seed=seed)
        # if len(env_builder_args)>0:
        #     ggLog.warn(f"Unused env_builder_args: {env_builder_args}")
    return env, 1/stepLength_sec


def get_quad_args():
    homing = {  ("quad","hip_joint_x_back_left") : -3.14159*0.4,
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
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("adarl_envs","models/quad_simple.urdf.xacro"),
            "model_kwargs" : {  "use_cylinders" : "false",
                                "all_collisions" : "false"},
            "xacro_extra_pkg_paths" : {"adarl_envs" : adarl.utils.utils.pkgutil_get_path("adarl_envs")},
            "homing_joint_pose" : homing,
            "robot_name" : "quad",
            "robot_main_body_link" : "body_link",
            "robot_root_link" : "body_link",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.5,0.,0.,0.,1.),
            "disallowed_contact_links" : [  ("quad","thigh_link_back_left"),
                                            ("quad","shin_link_back_left"),
                                            ("quad","thigh_link_back_right"),
                                            ("quad","shin_link_back_right"),
                                            ("quad","thigh_link_front_left"),
                                            ("quad","shin_link_front_left"),
                                            ("quad","thigh_link_front_right"),
                                            ("quad","shin_link_front_right"),
                                            ("quad","body_link")],
            "terminating_contact_pairs" : [(("quad","body_link"),("ground_plane","planeLink"))],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.3, 0.9, 0.9],
                                                    [ 0.3, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_position_offset" : homing,
            "enable_link_collisions" : [    (('quad', 'foot_center_link_back_left'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_back_right'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_front_left'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_front_right'),[('ground','ground_link')])],
            "feet_links" : [('quad', 'foot_center_link_back_left'),
                            ('quad', 'foot_center_link_back_right'),
                            ('quad', 'foot_center_link_front_left'),
                            ('quad', 'foot_center_link_front_right')]
            }

def get_kyon_args(enable_arms : bool = False):
    hip_pitch = -0.8727 # = -50/180*3.14159
    hip_roll =   0.0349 # = 2/180*3.14159
    knee =      -1.5707 # = -90/180*3.14159
    homing = {  ("kyon","hip_roll_3") :  hip_roll,
                ("kyon","hip_roll_4") : -hip_roll,
                ("kyon","hip_roll_1") : -hip_roll,
                ("kyon","hip_roll_2") :  hip_roll,
                ("kyon","hip_pitch_3") :  hip_pitch,
                ("kyon","hip_pitch_4") : -hip_pitch,
                ("kyon","hip_pitch_1") :  hip_pitch,
                ("kyon","hip_pitch_2") : -hip_pitch,
                ("kyon","knee_pitch_3") : -knee,
                ("kyon","knee_pitch_4") :  knee,
                ("kyon","knee_pitch_1") : -knee,
                ("kyon","knee_pitch_2") :  knee}
    if enable_arms:
        homing.update({ ("kyon","shoulder_yaw_1") : 0.0,
                        ("kyon","shoulder_pitch_1") : 0.0,
                        ("kyon","elbow_pitch_1") : 0.0,
                        ("kyon","wrist_pitch_1") : 0.0,
                        ("kyon","wrist_yaw_1") : 0.0,
                        ("kyon","shoulder_yaw_2") : 0.0,
                        ("kyon","shoulder_pitch_2") : 0.0,
                        ("kyon","elbow_pitch_2") : 0.0,
                        ("kyon","wrist_pitch_2") : 0.0,
                        ("kyon","wrist_yaw_2") : 0.0,
                        ("kyon","dagana_1_clamp_joint") : 0.1,
                        ("kyon","dagana_2_clamp_joint") : 0.1})
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("pykyon", "iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf.xacro"),
            "model_kwargs" : {"upper_body" : f"{enable_arms}",
                              "footonly_collision" : "true",
                              "varta" : "true"},
            "xacro_extra_pkg_paths" : {"kyon_urdf" : adarl.utils.utils.pkgutil_get_path("pykyon", "iit-kyon-ros-pkg/kyon_urdf")},
            "homing_joint_pose" : homing,
            "robot_name" : "kyon",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.495,0.,0.,0.,1.),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_com_links" : [("kyon","pelvis")],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_ratios_minmax_pve" : {k:[[ 0.25, 0.9, 0.9],
                                                     [ 0.25, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_position_offset" : homing,
            "enable_link_collisions" : [    (('kyon', 'contact_1'),[('ground','ground_link')]),
                                            (('kyon', 'contact_2'),[('ground','ground_link')]),
                                            (('kyon', 'contact_3'),[('ground','ground_link')]),
                                            (('kyon', 'contact_4'),[('ground','ground_link')])],
            "feet_links" : [('kyon', 'contact_1'),
                            ('kyon', 'contact_2'),
                            ('kyon', 'contact_3'),
                            ('kyon', 'contact_4')]
        }

def get_pgspot_args():
    hip_pitch = -0.8727 # = -50/180*3.14159
    hip_roll =   0.0349 # = 2/180*3.14159
    knee =      -1.5707 # = -90/180*3.14159
    homing = {  ("kyon","hip_roll_3") :  hip_roll,
                ("kyon","hip_roll_4") : -hip_roll,
                ("kyon","hip_roll_1") : -hip_roll,
                ("kyon","hip_roll_2") :  hip_roll,
                ("kyon","hip_pitch_3") :  hip_pitch,
                ("kyon","hip_pitch_4") : -hip_pitch,
                ("kyon","hip_pitch_1") :  hip_pitch,
                ("kyon","hip_pitch_2") : -hip_pitch,
                ("kyon","knee_pitch_3") : -knee,
                ("kyon","knee_pitch_4") :  knee,
                ("kyon","knee_pitch_1") : -knee,
                ("kyon","knee_pitch_2") :  knee}
    enable_arms = False
    if enable_arms:
        homing.update({ ("kyon","shoulder_yaw_1") : 0.0,
                        ("kyon","shoulder_pitch_1") : 0.0,
                        ("kyon","elbow_pitch_1") : 0.0,
                        ("kyon","wrist_pitch_1") : 0.0,
                        ("kyon","wrist_yaw_1") : 0.0,
                        ("kyon","shoulder_yaw_2") : 0.0,
                        ("kyon","shoulder_pitch_2") : 0.0,
                        ("kyon","elbow_pitch_2") : 0.0,
                        ("kyon","wrist_pitch_2") : 0.0,
                        ("kyon","wrist_yaw_2") : 0.0,
                        ("kyon","dagana_1_clamp_joint") : 0.1,
                        ("kyon","dagana_2_clamp_joint") : 0.1})
    from mujoco_playground._src import mjx_env
    return {"model_file" : mjx_env.ROOT_PATH / "locomotion" / "spot" / "xmls" / "scene_mjx_feetonly_flat_terrain.xml",
            "model_kwargs" : {},
            "xacro_extra_pkg_paths" : {},
            "homing_joint_pose" : homing,
            "robot_name" : "kyon",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.495,0.,0.,0.,1.),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_com_links" : [("kyon","pelvis")],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_ratios_minmax_pve" : {k:[[ 0.25, 0.9, 0.9],
                                                     [ 0.25, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_position_offset" : homing,
            "enable_link_collisions" : [    (('kyon', 'contact_1'),[('ground','ground_link')]),
                                            (('kyon', 'contact_2'),[('ground','ground_link')]),
                                            (('kyon', 'contact_3'),[('ground','ground_link')]),
                                            (('kyon', 'contact_4'),[('ground','ground_link')])],
            "feet_links" : [('kyon', 'contact_1'),
                            ('kyon', 'contact_2'),
                            ('kyon', 'contact_3'),
                            ('kyon', 'contact_4')]
        }


def get_centauro_args():
    # # Standard homing
    # hip_yaw =      0.75
    # hip_pitch =    1.25
    # knee_pitch =   1.55
    # ankle_pitch =  0.30
    # ankle_yaw =   -0.75
    # Straight ankle homing:
    hip_yaw =      0.75
    hip_pitch =    1.25
    knee_pitch =   1.25
    ankle_pitch =  0.0
    ankle_yaw =   -0.75
    homing = {  ("centauro","hip_yaw_1") :      -hip_yaw,
                ("centauro","hip_pitch_1") :    -hip_pitch,
                ("centauro","knee_pitch_1") :   -knee_pitch,
                ("centauro","ankle_pitch_1") :  -ankle_pitch,
                ("centauro","ankle_yaw_1") :    -ankle_yaw,
                ("centauro","hip_yaw_2") :      hip_yaw,
                ("centauro","hip_pitch_2") :    hip_pitch,
                ("centauro","knee_pitch_2") :   knee_pitch,
                ("centauro","ankle_pitch_2") :  ankle_pitch,
                ("centauro","ankle_yaw_2") :    ankle_yaw,
                ("centauro","hip_yaw_3") :      hip_yaw,
                ("centauro","hip_pitch_3") :    hip_pitch,
                ("centauro","knee_pitch_3") :   knee_pitch,
                ("centauro","ankle_pitch_3") :  ankle_pitch,
                ("centauro","ankle_yaw_3") :    ankle_yaw,
                ("centauro","hip_yaw_4") :      -hip_yaw,
                ("centauro","hip_pitch_4") :    -hip_pitch,
                ("centauro","knee_pitch_4") :   -knee_pitch,
                ("centauro","ankle_pitch_4") :  -ankle_pitch,
                ("centauro","ankle_yaw_4") :    -ankle_yaw,
                ("centauro","torso_yaw") : 0.0,
                ("centauro","velodyne_joint") : 0,
                ("centauro","d435_head_joint") : 0,
                ("centauro","j_arm1_1") : 0.520149,
                ("centauro","j_arm1_2") : 0.320865,
                ("centauro","j_arm1_3") : 0.274669,
                ("centauro","j_arm1_4") : -2.23604,
                ("centauro","j_arm1_5") : 0.0500815,
                ("centauro","j_arm1_6") : -0.781461,
                ("centauro","j_arm2_1") : 0.520149,
                ("centauro","j_arm2_2") : -0.320865,
                ("centauro","j_arm2_3") : -0.274669,
                ("centauro","j_arm2_4") : -2.23604,
                ("centauro","j_arm2_5") : -0.0500815,
                ("centauro","j_arm2_6") : -0.781461,
                ("centauro","j_wheel_1") : 0.0,
                ("centauro","j_wheel_2") : 0.0,
                ("centauro","j_wheel_3") : 0.0,
                ("centauro","j_wheel_4") : 0.0
                # ("centauro","dagana_1_claw_joint") : 0,
                # ("centauro","dagana_2_claw_joint") : 0
                }
    legs = ["hip_yaw_1"
            ,"hip_pitch_1"
            ,"knee_pitch_1"
            ,"ankle_pitch_1"
            #,"ankle_yaw_1"
            ,"hip_yaw_2"
            ,"hip_pitch_2"
            ,"knee_pitch_2"
            ,"ankle_pitch_2"
            #,"ankle_yaw_2"
            ,"hip_yaw_3"
            ,"hip_pitch_3"
            ,"knee_pitch_3"
            ,"ankle_pitch_3"
            #,"ankle_yaw_3"
            ,"hip_yaw_4"
            ,"hip_pitch_4"
            ,"knee_pitch_4"
            ,"ankle_pitch_4"
            # ,"ankle_yaw_4"
            ]
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro"),
            "model_kwargs" : {  "realsense":"false",
                                "velodyne" :"false",
                                "floating_joint":"true",
                                "small_sphere_wheel_collision":"true"
                                },
            "xacro_extra_pkg_paths" : {"centauro_urdf" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf")},
            "homing_joint_pose" : homing,
            "robot_name" : "centauro",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.84,0.,0.,0.,1.),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : legs,
            "randomized_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_com_links" : [("centauro","pelvis")],
            "randomized_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.2, 0.9, 0.9],
                                                    [ 0.2, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_position_offset" : homing,
            "enable_link_collisions" : [    (('centauro', 'wheel_1'),[('ground','ground_link')]),
                                            (('centauro', 'wheel_2'),[('ground','ground_link')]),
                                            (('centauro', 'wheel_3'),[('ground','ground_link')]),
                                            (('centauro', 'wheel_4'),[('ground','ground_link')])],
            "feet_links" : [('centauro', 'wheel_1'),
                            ('centauro', 'wheel_2'),
                            ('centauro', 'wheel_3'),
                            ('centauro', 'wheel_4')]
        }




def named_loco_venv_builder(seed : int,
                    run_folder : str,
                    num_envs : int, 
                    env_builder_args : dict,
                    env_name : str = "") -> gym.vector.VectorEnv:
    robot_model = env_builder_args["robot_model"]
    if robot_model == "quad":
        env_builder_args.update(get_quad_args())
    elif robot_model == "kyon":
        env_builder_args.update(get_kyon_args())
    elif robot_model == "kyon_arms":
        env_builder_args.update(get_kyon_args(enable_arms=True))
    elif robot_model == "spot":
        env_builder_args.update(get_pgspot_args())
    elif robot_model == "centauro":
        env_builder_args.update(get_centauro_args())
    else:
        raise RuntimeError(f"Unknown robot_model {robot_model}")
    return loco_venv_builder(seed = seed,
                            log_folder = run_folder,
                            env_builder_args = env_builder_args,
                            num_envs=num_envs,
                            runner_builder=loco_runner_builder)[0]

def named_loco_single_env_builder(seed : int,
                    log_folder : str,
                    is_eval : bool, 
                    env_builder_args : dict) -> tuple[gym.Env,float]:
    robot_model = env_builder_args["robot_model"]
    if robot_model == "quad":
        env_builder_args.update(get_quad_args())
    elif robot_model == "kyon":
        env_builder_args.update(get_kyon_args())
    elif robot_model == "centauro":
        env_builder_args.update(get_centauro_args())
    else:
        raise RuntimeError(f"Unknown robot_model {robot_model}")
    return loco_env_builder(seed = seed,
                            log_folder = log_folder,
                            env_builder_args = env_builder_args,
                            is_eval=is_eval,
                            runner_builder=loco_runner_builder)

# def quad_loco_venv_builder(seed : int,
#                     run_folder : str,
#                     num_envs : int, 
#                     env_builder_args : dict,
#                     env_name : str = "") -> gym.vector.VectorEnv:
#     env_builder_args.update(get_quad_args())
#     return loco_venv_builder(seed = seed,
#                             log_folder = run_folder,
#                             env_builder_args = env_builder_args,
#                             num_envs=num_envs)[0]


# def quad_loco_env_builder(seed : int,
#                     log_folder : str,
#                     is_eval : bool, 
#                     env_builder_args : dict) -> tuple[gym.Env,float]:
#     env_builder_args.update(get_quad_args())
#     return loco_env_builder(seed = seed,
#                             log_folder = log_folder,
#                             is_eval=is_eval,
#                             env_builder_args = env_builder_args)




# def kyon_loco_env_builder(seed : int,
#                     log_folder : str,
#                     is_eval : bool, 
#                     env_builder_args : dict) -> tuple[gym.Env,float]:
#     env_builder_args.update(get_kyon_args())
#     return loco_env_builder(seed = seed,
#                             log_folder = log_folder,
#                             is_eval=is_eval,
#                             env_builder_args = env_builder_args)

# def kyon_loco_venv_builder(seed : int,
#                     run_folder : str,
#                     num_envs : int, 
#                     env_builder_args : dict) -> gym.vector.VectorEnv:
#     env_builder_args.update(get_kyon_args())
#     return loco_venv_builder(seed = seed,
#                             log_folder = run_folder,
#                             env_builder_args = env_builder_args,
#                             num_envs=num_envs)[0]


