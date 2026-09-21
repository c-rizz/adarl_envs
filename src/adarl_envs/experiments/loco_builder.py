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
from adarl_envs.env.LocomotionVecEnv import LocomotionVecEnv, LocomotionVecEnvInitArgs
from adarl_envs.env.RobotVecEnv import JOINT_FILTERS, LINK_FILTERS, RobotVecEnvInitArgs
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.Runner2GymWrapper import Runner2GymWrapper
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
import gymnasium as gym
import copy
from rreal.algorithms.sac_helpers import build_vec_env, VecEnvRunnerBuilderProtocol
from math import pi
import xml.etree.ElementTree as ET
import numpy as np
from typing import Literal

def set_asset_texture_paths(mjcf_string: str, meshdir: str, texturedir: str) -> str:
    """Set meshdir and texturedir in <compiler> to the given paths. If <compiler> doesn't exist, it is created."""
    root = ET.fromstring(mjcf_string)

    compiler = root.find(".//compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")

    if meshdir is not None:
        compiler.attrib["meshdir"] = meshdir
    if texturedir is not None:
        compiler.attrib["texturedir"] = texturedir

    return ET.tostring(root, encoding="unicode")

def make_asset_texture_paths_absolute(mjcf_string: str, model_file_path: str) -> str:
    """Set meshdir and texturedir in <compiler> to absolute paths based on the model file's directory.
    
    If <compiler> doesn't exist, it is created. Existing meshdir/texturedir are resolved
    relative to the model file's directory if they are not already absolute.
    """
    model_dir = str(Path(model_file_path).resolve().parent)
    root = ET.fromstring(mjcf_string)

    compiler = root.find(".//compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")

    meshdir = compiler.attrib.get("meshdir", ".")
    if not Path(meshdir).is_absolute():
        meshdir = str(Path(model_dir, meshdir).resolve())
    compiler.attrib["meshdir"] = meshdir

    texturedir = compiler.attrib.get("texturedir", ".")
    if not Path(texturedir).is_absolute():
        texturedir = str(Path(model_dir, texturedir).resolve())
    compiler.attrib["texturedir"] = texturedir

    return ET.tostring(root, encoding="unicode")

def format_tensor(t, float_precision):
    if t is None:
        return "None"
    if isinstance(t, str):
        return t
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
        body_abs_linvel_norm_str = f"{th.linalg.norm(body_abs_linvel):.3f}"
    else:
        body_abs_linvel_str = 'N/A'
        body_abs_linvel_norm_str = 'N/A'
    if 'state_internal' in info:
        posref_safety_triggered = info['state_internal'][LocomotionVecEnv.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED] if 'state_internal' in info else 'N/A'
        limits_safety_triggered = info['state_internal'][LocomotionVecEnv.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED] if 'state_internal' in info else 'N/A'
    else:
        posref_safety_triggered = 'N/A'
        limits_safety_triggered = 'N/A'
    goal_abs_linvel_xyz = info.get('goal_abs_xyz_vec', None)
    vel_norm = f"{th.linalg.norm(goal_abs_linvel_xyz):.3f}" if goal_abs_linvel_xyz is not None else "N/A"
    goal_rel_linvel_xyz = info.get('goal_rel_xyz_vec', None)
    rel_vel_norm = f"{th.linalg.norm(goal_rel_linvel_xyz):.3f}" if goal_rel_linvel_xyz is not None else "N/A"
    return  (   f"\n"
                f"Step                  {format_tensor(info.get('ep_step_count', -1), 0)}\n"+
                f"body_abs_linvel       {body_abs_linvel_str} ({body_abs_linvel_norm_str} m/s)\n"
                f"goal_vel_abs          {format_tensor(goal_abs_linvel_xyz, 3)} ({vel_norm} m/s)\n"
                f"goal_vel_rel          {format_tensor(goal_rel_linvel_xyz, 3)} ({rel_vel_norm} m/s)\n"
                f"smoothed_linvel_error {format_tensor(info.get('smoothed_linvel_error','N/A'), 3)}\n"
                f"linvel_error          {format_tensor(info.get('linvel_error','N/A'), 3)}\n"
                f"goal_yaw_vel          {format_tensor(info.get('goal_yaw_vel','N/A'), 3)}\n"
                f"smoothed_yawvel_error {format_tensor(info.get('smoothed_yawvel_error','N/A'), 3)}\n"
                f"goal_height           {format_tensor(info.get('goal_height','N/A'), 3)}\n"
                f"height_error          {format_tensor(info.get('height_err','N/A'), 3)}\n"
                f"log_prob              {format_tensor(extra_info.get('act_log_prob',th.as_tensor(float('nan'))), 3)}\n"
                f"posref_safety         {posref_safety_triggered}\n"
                f"limits_safety         {limits_safety_triggered}\n"
                f"actacc_weight         {format_tensor(info.get('actacc_weight',float('nan')), 3)}\n"
                f"actdiff_weight        {format_tensor(info.get('actdiff_weight',float('nan')), 3)}\n"
                f"posrefvel_weight      {format_tensor(info.get('posref_vel_weight',float('nan')), 3)}\n"
                f"posrefacc_weight      {format_tensor(info.get('posref_acc_weight',float('nan')), 3)}\n")

def get_robot_string_and_format(model_file_path : str,
                                robot_description_format : Literal["urdf", "xacro", "mjcf"],
                                robot_description_string : str,
                                model_kwargs : dict,
                                xacro_extra_pkg_paths : dict[str, str]) -> tuple[str, Literal["urdf", "sdf", "mjcf"]]:
    if model_file_path is None:
        if robot_description_string is None:
            raise ValueError("Either model_file or robot_description_string must be provided in env_builder_args")
    else:
        raw_model_string = Path(model_file_path).read_text()
        if robot_description_format in ["xacro"]:
            robot_description_string = adarl.utils.utils.compile_xacro_string(   model_definition_string=raw_model_string,
                                                                    model_kwargs=model_kwargs,
                                                                    extra_pkg_paths=xacro_extra_pkg_paths)
            robot_description_format = "urdf"
        elif robot_description_format == "mjcf":
            robot_description_string = make_asset_texture_paths_absolute(raw_model_string, model_file_path)
        else:
            robot_description_string = raw_model_string
    return robot_description_string, robot_description_format

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

    s,f = get_robot_string_and_format(model_file_path = env_builder_args.get("model_file", None),
                                        robot_description_format = env_builder_args["robot_description_format"],
                                        robot_description_string = env_builder_args.get("robot_description_string", None),
                                        model_kwargs = env_builder_args.get("model_kwargs"),
                                        xacro_extra_pkg_paths = env_builder_args.get("xacro_extra_pkg_paths"))
    robot_description_string = s
    robot_description_format = f


    if mode == "gz":
        raise NotImplementedError()
    elif mode == "gazebo":
        raise NotImplementedError()
    elif mode == "xbot-zmq":
        from adarl.adapters.VecZmqXbotAdapter import VecZmqXbotAdapter
        from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter
        ground_link = ("ground_plane","ground_link") # Should not be used

        xbotzmq_remote_ip = env_builder_args.get("xbotzmq_remote_ip", "127.0.0.1")
        zmq_protocol = env_builder_args.get("zmq_protocol", "tcp")

        adapter = VecZmqXbotAdapter(   adapter = ZmqXbotAdapter(model_name = robot_name,
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
                                                                remote_ip = xbotzmq_remote_ip,
                                                                comm_protocol =zmq_protocol, # or 'ipc'
                                                                ipc_pub_path ="/tmp/xbot2_zmq_pub.ipc",
                                                                ipc_cmd_path ="/tmp/xbot2_zmq_cmd.ipc",
                                                                ipc_service_path ="/tmp/xbot2_zmq_rep.ipc",
                                                                robot_urdf=robot_description_string,
                                                                tcp_service_port=env_builder_args.get("xbotzmq_tcp_service_port", 5557),
                                                                tcp_cmd_port=env_builder_args.get("xbotzmq_tcp_cmd_port", 5558),
                                                                tcp_state_port=env_builder_args.get("xbotzmq_tcp_state_port", 5559),
                                                                ),
                                                vec_size = 1,
                                                th_device = th_device)
    # elif mode == "pybullet":
    #     from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
    #     from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
    #     ground_link = ("ground_plane","ground_link")
    #     env_builder_args["enable_link_collisions"] = None
    #     adapter = VecSimJointImpedanceAdapterWrapper(adapters = PyBulletJointImpedanceAdapter(stepLength_sec=stepLength_sec,
    #                                                                         restore_on_reset=False,
    #                                                                         debug_gui=show_gui,
    #                                                                         simulation_step=1/1024,
    #                                                                         enable_rendering=env_builder_args.pop("enable_rendering"),
    #                                                                         global_max_torque_position_control = 100,
    #                                                                         real_time_factor=None,
    #                                                                         th_device=th_device),
    #                                                         th_device = th_device)
    elif mode == "mjx":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        import jax
        ground_link = ("ground","ground_link")
        robot_model = env_builder_args["robot_model"]
        sim_dt = {"centauro" : 2/1024, "spot" : 0.004}.get(robot_model, 2/1024)
        iterations_per_ep = int(max_steps*stepLength_sec/sim_dt)
        opt_override = {}
        opt_override.update(env_builder_args.pop("mjx_opt_override", {}))
        adapter = MjxJointImpedanceAdapter( vec_size=num_envs,
                                            enable_rendering=env_builder_args.pop("enable_rendering"),
                                            jax_device=jax.devices("gpu")[th_device.index] if th_device.type == "cuda" else jax.devices("cpu")[0],
                                            output_th_device = th_device,
                                            sim_step_dt=sim_dt,
                                            step_length_sec=stepLength_sec,
                                            realtime_factor=-1.0,
                                            gui_env_index=0,
                                            default_max_joint_impedance_ctrl_torque=env_builder_args.pop("default_max_joint_impedance_ctrl_torque", 100.0),
                                            max_joint_impedance_ctrl_torques=env_builder_args.pop("max_joint_impedance_ctrl_torques", {}),
                                            show_gui=show_gui,
                                            log_freq=iterations_per_ep,
                                            record_whole_joint_trajectories = env_builder_args.get("record_whole_joint_trajectories", False),
                                            log_freq_joints_trajectories = iterations_per_ep,
                                            log_folder=run_folder,
                                            revolute_dof_frictionloss_override  = env_builder_args.get("revolute_dof_frictionloss_override", 1.0),
                                            revolute_dof_armature_override      = env_builder_args.get("revolute_dof_armature_override", 0.1),
                                            revolute_dof_damping_override       = env_builder_args.get("revolute_dof_damping_override", 1.0),
                                            safe_revolute_dof_armature          = env_builder_args.get("safe_revolute_dof_armature", 0.1),
                                            opt_preset=env_builder_args.pop("mjx_opt_preset"),
                                            opt_override=opt_override,
                                            geom_overrides=env_builder_args.get("mjx_geom_overrides", None),
                                            reference_filter_cutoff_frequency=20.0,
                                            reference_filter_mode="second_order" if env_builder_args["enable_reference_filter"] else "none",
                                            mjx_impl="warp")
    elif mode == "mjx-act":
        from adarl.adapters.MjxActuatedAdapter import MjxActuatedAdapter
        import jax
        ground_link = ("ground","ground_link")
        robot_model = env_builder_args["robot_model"]
        sim_dt = 2/1024 if robot_model=="centauro" else 4/1024 
        iterations_per_ep = int(max_steps*stepLength_sec/sim_dt)
        opt_override = {}
        if env_builder_args.pop("enable_reference_filter", False):
            raise NotImplementedError("Reference filter not implemented for MjxActuatedAdapter yet")
        adapter = MjxActuatedAdapter(   vec_size=num_envs,
                                        enable_rendering=env_builder_args.pop("enable_rendering"),
                                        jax_device=jax.devices("gpu")[th_device.index] if th_device.type == "cuda" else jax.devices("cpu")[0],
                                        output_th_device = th_device,
                                        sim_step_dt=sim_dt,
                                        step_length_sec=stepLength_sec,
                                        realtime_factor=-1.0,
                                        gui_env_index=0,
                                        show_gui=show_gui,
                                        log_freq=iterations_per_ep,
                                        record_whole_joint_trajectories = env_builder_args.get("record_whole_joint_trajectories", False),
                                        log_freq_joints_trajectories = iterations_per_ep,
                                        log_folder=run_folder,
                                        revolute_dof_armature_override=0.1,
                                        safe_revolute_dof_armature=0.1,
                                        opt_preset=env_builder_args.pop("mjx_opt_preset"),
                                        opt_override=opt_override,
                                        default_actuator_kp=env_builder_args.get("ctrl_joints_stiffness", 100.0) if not isinstance(env_builder_args.get("ctrl_joints_stiffness", 100.0), dict) else env_builder_args.get("ctrl_joints_stiffness",{}).get("default", 100.0),
                                        default_actuator_kv=env_builder_args.get("ctrl_joints_damping", 10.0) if not isinstance(env_builder_args.get("ctrl_joints_damping", 10.0), dict) else env_builder_args.get("ctrl_joints_damping",{}).get("default", 10.0),
                                        default_max_actuator_force=env_builder_args.pop("default_max_joint_impedance_ctrl_torque", 100.0),
                                        max_actuator_forces=env_builder_args.get("max_joint_impedance_ctrl_torques", {}))
    elif mode == "mujoco":
        from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
        ground_link = ("ground","ground_link")
        robot_model = env_builder_args["robot_model"]
        sim_dt = {"centauro" : 2/1024, "spot" : 0.004}.get(robot_model, 2/1024)
        opt_override = {}
        opt_override.update(env_builder_args.pop("mjx_opt_override", {}))
        adapter = MujocoJointImpedanceAdapter(  step_length_sec=stepLength_sec,
                                                sim_step_dt=sim_dt,
                                                output_th_device=th_device,
                                                log_folder=run_folder,
                                                show_gui=show_gui,
                                                default_max_joint_impedance_ctrl_torque=env_builder_args.pop("default_max_joint_impedance_ctrl_torque", 100.0),
                                                max_joint_impedance_ctrl_torques=env_builder_args.pop("max_joint_impedance_ctrl_torques", {}),
                                                revolute_dof_frictionloss_override = env_builder_args.get("revolute_dof_frictionloss_override", 1.0),
                                                revolute_dof_armature_override     = env_builder_args.get("revolute_dof_armature_override", 0.1),
                                                revolute_dof_damping_override      = env_builder_args.get("revolute_dof_damping_override", 1.0),
                                                safe_revolute_dof_armature         = env_builder_args.get("safe_revolute_dof_armature", 0.1),
                                                opt_preset=env_builder_args.get("mjx_opt_preset", "faster"),
                                                opt_override=opt_override,
                                                reference_filter_cutoff_frequency=20.0,
                                                reference_filter_mode="second_order" if env_builder_args["enable_reference_filter"] else "none",                                                
                                                geom_overrides=env_builder_args.get("mjx_geom_overrides", None))
    elif mode == "genesis":
        from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
        ground_link = ("ground","ground_link")
        robot_model = env_builder_args["robot_model"]
        sim_dt = {"centauro" : 2/1024, "spot" : 0.004}.get(robot_model, 2/1024)
        # collision-pair filtering (set_body_collisions) is not implemented in GenesisAdapter
        env_builder_args["enable_link_collisions"] = None
        # the ui camera ("simple_camera") is parsed automatically from the camera model spawned by the env
        adapter = GenesisJointImpedanceAdapter( vec_size=num_envs,
                                                step_length_sec=stepLength_sec,
                                                sim_step_dt=sim_dt,
                                                output_th_device=th_device,
                                                log_folder=run_folder,
                                                enable_rendering=env_builder_args.pop("enable_rendering"),
                                                show_gui=show_gui,
                                                default_max_joint_impedance_ctrl_torque=env_builder_args.pop("default_max_joint_impedance_ctrl_torque", 100.0),
                                                max_joint_impedance_ctrl_torques=env_builder_args.pop("max_joint_impedance_ctrl_torques", {}),
                                                reference_filter_cutoff_frequency=20.0,
                                                reference_filter_mode="second_order" if env_builder_args["enable_reference_filter"] else "none")
    else:
        print(f"Requested unknown adapter '{mode}'")
        exit(0)

    time.sleep(1)

    homing_body_pose_xyz_xyzw = env_builder_args.pop("homing_body_pose_xyz_xyzw")
    randomized_homing_body_position_minmax_xyz = env_builder_args.pop("randomized_homing_body_position_minmax_xyz")
    if randomized_homing_body_position_minmax_xyz is None:
        # No spawn box: keep the robot at its homing position (zero-width range = no randomization).
        randomized_homing_body_position_minmax_xyz = (tuple(homing_body_pose_xyz_xyzw[:3]), tuple(homing_body_pose_xyz_xyzw[:3]))

    lrenv = LocomotionVecEnv(LocomotionVecEnvInitArgs(
                                robot_init_args = RobotVecEnvInitArgs(
                                    noise_action_delay_mustd_std = env_builder_args.pop("noise_action_delay_mustd_std"),
                                    noise_action_mustd = env_builder_args.pop("noise_action_mustd"), 
                                    action_smoothing_halflife_sec=env_builder_args.pop("action_smoothing_halflife_sec"),
                                    adapter=adapter,
                                    control_mode = env_builder_args.pop("control_mode"),
                                    controlled_joints=env_builder_args.pop("controlled_joints"),
                                    enable_dbg_checks=True,
                                    enable_limits_safety = env_builder_args.pop("enable_limits_safety"),
                                    enable_link_collisions=env_builder_args.pop("enable_link_collisions"),
                                    enable_posref_safety = env_builder_args.pop("enable_posref_safety"),
                                    fail_on_safety=env_builder_args.pop("fail_on_safety"),
                                    frame_stack_length=env_builder_args.pop("frame_stack_length"),
                                    free_joints=[],
                                    goal_err_smoothing_halflife_sec = env_builder_args.pop("goal_err_smoothing_halflife_sec"),
                                    ground_link=ground_link,
                                    held_joints_damping=env_builder_args.pop("held_joints_damping"),
                                    held_joints_stiffness=env_builder_args.pop("held_joints_stiffness"),
                                    homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                                    randomized_homing_body_position_minmax_xyz=randomized_homing_body_position_minmax_xyz,
                                    homing_joint_position=env_builder_args.pop("homing_joint_position"),
                                    homing_joint_position_references=env_builder_args.pop("homing_joint_position_references"),
                                    impulse_duration_minmax=env_builder_args.pop("impulse_duration_minmax"),
                                    impulse_mean_std=env_builder_args.pop("impulse_mean_std"),
                                    impulse_probability_per_sec=env_builder_args.pop("impulse_probability_per_sec"),
                                    init_on_reset_ratio = env_builder_args.pop("init_on_reset_ratio"),
                                    randomized_initial_joint_pose_range = env_builder_args.pop("randomized_initial_joint_pose_range"),
                                    just_health_reward = env_builder_args.pop("just_health_reward"),
                                    longterm_states_decimation_time = env_builder_args.pop("longterm_states_decimation_time"),
                                    maxStepsPerEpisode=max_steps,
                                    merge_privileged = env_builder_args.pop("merge_privileged"),
                                    minmax_damping=env_builder_args.pop("minmax_ctrl_damping",(0.0,30.0)),
                                    minmax_stiffness=env_builder_args.pop("minmax_ctrl_stiffness",(0.0,1000.0)),
                                    noise_abs_obs_angvel_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_angvel_ep_mustd_step_std"),
                                    noise_abs_obs_gravity_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_gravity_ep_mustd_step_std"),
                                    noise_abs_obs_joints_pve_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_joints_pve_ep_mustd_step_std"),
                                    noise_abs_obs_linacc_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_linacc_ep_mustd_step_std"),
                                    noise_abs_obs_linvel_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_linvel_ep_mustd_step_std"),
                                    noise_abs_obs_posz_ep_mustd_step_std = env_builder_args.pop("noise_abs_obs_posz_ep_mustd_step_std"),
                                    observe_full_robot_state = env_builder_args.pop("observe_full_robot_state"),
                                    offset_envs_ep_starts = env_builder_args.pop("offset_envs_ep_starts"),
                                    posref_safety_period = env_builder_args.pop("posref_safety_period"),
                                    quiet=quiet,
                                    randomized_dof_armature_joints=env_builder_args.pop("randomized_dof_armature_joints"),
                                    randomized_dof_armature_ratios= env_builder_args.pop("randomized_dof_armature_ratios"),
                                    randomized_dof_damping_joints=env_builder_args.pop("randomized_dof_damping_joints"),
                                    randomized_dof_damping_ratios=env_builder_args.pop("randomized_dof_damping_ratios"),
                                    randomized_dof_frictionloss_joints=env_builder_args.pop("randomized_dof_frictionloss_joints"),
                                    randomized_dof_frictionloss_ratios=env_builder_args.pop("randomized_dof_frictionloss_ratios"),
                                    randomized_com_links=env_builder_args.pop("randomized_com_links"),
                                    randomized_com_xyz_diff_distribution=env_builder_args.pop("randomized_com_xyz_diff_distribution"),
                                    randomized_friction_links=env_builder_args.pop("randomized_friction_links"),
                                    randomized_friction_slide_spin_roll_ratios=env_builder_args.pop("randomized_friction_slide_spin_roll_ratios"),
                                    randomized_gains_damping_ratio_epstd=env_builder_args.pop("randomized_gains_damping_ratio_epstd"),
                                    randomized_gains_stiffness_ratio_epstd=env_builder_args.pop("randomized_gains_stiffness_ratio_epstd"),
                                    randomized_mass_links=env_builder_args.pop("randomized_mass_links"),
                                    randomized_mass_ratios_distr=env_builder_args.pop("randomized_mass_ratios"),
                                    randomized_reference_filter_distribution=env_builder_args.pop("randomized_reference_filter_distribution"),
                                    randomization_recycle_init_pose=env_builder_args.pop("randomization_recycle_init_pose"),
                                    robot_main_body_link=env_builder_args.pop("robot_main_body_link"),
                                    robot_name=robot_name,
                                    robot_root_link=env_builder_args.pop("robot_root_link"),
                                    main_body_gait_frame_quat_xyzw=env_builder_args.pop("main_body_gait_frame_quat_xyzw", (0.,0.,0.,1.)),
                                    robot_description_string=robot_description_string,
                                    robot_description_format=robot_description_format,
                                    ctrl_joints_damping=env_builder_args.pop("ctrl_joints_damping"),
                                    control_limits_center=env_builder_args.pop("control_limits_center"),
                                    ctrl_joints_stiffness=env_builder_args.pop("ctrl_joints_stiffness"),
                                    safety_limits_ratios_minmax_pve=env_builder_args.pop("safety_limits_ratios_minmax_pve"),
                                    control_limits_ratios_minmax_pve=env_builder_args.pop("control_limits_ratios_minmax_pve"),
                                    control_limits_minmax_pve=env_builder_args.pop("control_limits_minmax_pve"),
                                    saturate_jimp_posref_limits = env_builder_args.pop("saturate_jimp_ref_limits"),
                                    seed=seed,
                                    stepLength_sec=stepLength_sec,
                                    step_precision_tolerance=0 if isinstance(adapter, BaseSimulationAdapter) else 0.001,
                                    terminate_on_safety=env_builder_args.pop("terminate_on_safety"),
                                    th_device=th_device,
                                    ui_camera_resolution_hw=env_builder_args.pop("ui_camera_resolution_hw"),
                                    verbose_infos=env_builder_args.pop("verbose_infos"),
                                    minimal_infos=env_builder_args.pop("minimal_infos"),
                                    no_infos=env_builder_args.pop("no_infos"),
                                    extrinsics_only_privileged=env_builder_args.pop("extrinsics_only_privileged",False),
                                    history_length_action_raw=env_builder_args.pop("history_length_action_raw"),
                                    history_length_action_smoothed=env_builder_args.pop("history_length_action_smoothed"),
                                    observe_actor_safety_state=env_builder_args.pop("observe_actor_safety_state"),
                                    posref_err_history_length=env_builder_args.pop("posref_err_history_length"),
                                    observe_linvel_nonprivileged=env_builder_args.pop("observe_linvel_nonprivileged",False),
                                    randomization_recycle_model_alterations=env_builder_args.pop("randomization_recycle_model_alterations",False),
                                    world_description_format=env_builder_args.pop("world_description_format","predefined"),
                                    world_description_string=env_builder_args.pop("world_description_string","flat_ground"),
                                    ),
                                desired_foot_clearance = env_builder_args.pop("desired_foot_clearance"),
                                disallowed_contact_links = env_builder_args.pop("disallowed_contact_links"),
                                feet_contact_links=env_builder_args.pop("feet_contact_links"),
                                feet_bottom_links=env_builder_args.pop("feet_bottom_links"),
                                goal_height_minmax=env_builder_args.pop("goal_height_minmax"),
                                goal_resampling_probability_per_sec= env_builder_args.pop("goal_resampling_probability_per_sec"),
                                goal_speed_minmax=env_builder_args.pop("goal_speed_minmax"),
                                goal_yaw_minmax=env_builder_args.pop("goal_yaw_minmax"),
                                max_goal_height_pos_change_speed=env_builder_args.pop("max_goal_height_pos_change_speed"),
                                max_good_step_air_duration=env_builder_args.pop("step_max_good_air_duration"),
                                min_good_step_air_duration=env_builder_args.pop("step_min_good_air_duration"),
                                step_max_good_ground_duration=env_builder_args.pop("step_max_good_ground_duration"),
                                step_min_good_ground_duration=env_builder_args.pop("step_min_good_ground_duration"),
                                pitchnroll_reward_settle_point=env_builder_args.pop("pitchnroll_reward_settle_point"),
                                reward_superweight_joint_penalties = env_builder_args.pop("reward_superweight_joint_penalties"),    
                                reward_contacts_weight = env_builder_args.pop("reward_contacts_weight"),
                                reward_failure_weight = env_builder_args.pop("reward_failure_weight"),
                                reward_feet_air_time_weight = env_builder_args.pop("reward_feet_air_time_weight"),
                                reward_feet_ground_time_weight = env_builder_args.pop("reward_feet_ground_time_weight"),
                                reward_feet_on_ground_weight = env_builder_args.pop("reward_feet_on_ground_weight"),
                                reward_feet_step_height_weight=env_builder_args.pop("reward_feet_step_height_weight"),
                                reward_heading_velocity_weight = env_builder_args.pop("reward_heading_velocity_weight"),
                                reward_heading_weight = env_builder_args.pop("reward_heading_weight"),
                                reward_health_weight = env_builder_args.pop("reward_health_weight"),
                                reward_height_position_weight=env_builder_args.pop("reward_height_position_weight"),
                                reward_height_velocity_weight=env_builder_args.pop("reward_height_velocity_weight"),
                                reward_joint_acc_on_vel_weight = env_builder_args.pop("reward_joint_acc_on_vel_weight"),
                                reward_joint_acceleration_weight = env_builder_args.pop("reward_joint_acceleration_weight"),
                                reward_joint_actacc_weight = env_builder_args.pop("reward_joint_actacc_weight"),
                                reward_joint_actdiff_weight = env_builder_args.pop("reward_joint_actdiff_weight"),
                                reward_joint_cmdtorque_weight = env_builder_args.pop("reward_joint_torque_weight"),
                                reward_joint_energy_weight = env_builder_args.pop("reward_joint_energy_weight"),
                                reward_joint_position_limit_weight = env_builder_args.pop("reward_joint_position_limit_weight"),
                                reward_joint_position_weight=env_builder_args.pop("reward_joint_position_weight"),
                                reward_joint_posref_acc_weight = env_builder_args.pop("reward_joint_posref_acc_weight"),
                                reward_joint_posref_vel_weight = env_builder_args.pop("reward_joint_posref_vel_weight"),
                                reward_joint_power_weight = env_builder_args.pop("reward_joint_power_weight"),
                                reward_joint_sensed_effort_weight = env_builder_args.pop("reward_joint_sensed_effort_weight"),
                                reward_joint_stand_position_weight = env_builder_args.pop("reward_joint_stand_position_weight"),
                                reward_joint_stand_velocity_weight = env_builder_args.pop("reward_joint_stand_velocity_weight"),
                                reward_joint_torque_limit_weight = env_builder_args.pop("reward_joint_torque_limit_weight"),
                                reward_joint_torquediff_weight = env_builder_args.pop("reward_joint_torquediff_weight"),
                                reward_joint_torqueref_weight = env_builder_args.pop("reward_joint_torqueref_weight"),
                                reward_joint_velocity_limit_weight = env_builder_args.pop("reward_joint_velocity_limit_weight"),
                                reward_joint_velocity_weight = env_builder_args.pop("reward_joint_velocity_weight"),
                                reward_joint_velref_weight = env_builder_args.pop("reward_joint_velref_weight"),
                                reward_pitchnroll_velocity_weight=env_builder_args.pop("reward_pitchnroll_velocity_weight"),
                                reward_pitchnroll_weight=env_builder_args.pop("reward_pitchnroll_weight"),
                                reward_safety_triggered_weight = env_builder_args.pop("reward_safety_triggered_weight"),
                                reward_scale=1000/max_steps * env_builder_args.pop("reward_scale_nolength"),
                                reward_slip_weight = env_builder_args.pop("reward_slip_weight"),
                                reward_tracking_weight = env_builder_args.pop("reward_tracking_weight"),
                                reward_yaw_vel_tracking_weight=env_builder_args.pop("reward_yaw_vel_tracking_weight"),
                                terminate_on_crash=env_builder_args.pop("terminate_on_crash"),
                                terminating_contact_pairs=env_builder_args.pop("terminating_contact_pairs") if env_builder_args.pop("terminate_on_body_contact") else [],
                                use_contacts=env_builder_args.pop("use_contacts"),
                                split_rewards=env_builder_args.pop("split_rewards"),
                                goal_yaw_vel_minmax=env_builder_args.pop("goal_yaw_vel_minmax"),
                                goal_yaw_vel_zero_ratio=env_builder_args.pop("goal_yaw_vel_zero_ratio"),
                                playground_style_reward=env_builder_args.pop("playground_style_reward"),
                                terminal_gravity_angle = env_builder_args.pop("terminal_gravity_angle", 30*pi/180.0))
                            )
    # ggLog.info(f"state_space = {lrenv.state_space}")
    # ggLog.info(f"observation_space = {lrenv.observation_space}")
    # ggLog.info(f"action_space = {lrenv.action_space.shape}")
    vrunner = EnvRunner(env=lrenv, verbose=True, quiet=False, episodeInfoLogFile=run_folder+"/vec_runner.log",
                        ui_render_envs=[0], autoreset=autoreset,
                        log_freq = max_steps,
                        sync_on_reinit = env_builder_args.pop("sync_on_reinit", False))
    if env_builder_args["video_save_freq"]>0:
        vrunner = EnvRunnerRecorderWrapper(vrunner,
                                        fps = 1/stepLength_sec,
                                        outFolder=run_folder+"/RunnerRecorder",
                                        env_index=0,
                                        saveFrequency_ep=env_builder_args.pop("video_save_freq"),
                                        publish=False,
                                        stream=True,
                                        vec_obs_keys=["base.vec","privileged.vec"], #TODO: somehow pass multiple keys and include privileged, or auto-detect which keys to save
                                        record_video=env_builder_args["record_video"],
                                        overlay_text_xy=(0.025,0.025),
                                        overlay_text_height=0.035,
                                        overlay_text_color_rgb=(255,150,0),
                                        overlay_text_func=overlay_text_func)
    return vrunner


def single_env_builder(   seed : int,
                        log_folder : str,
                        is_eval : bool, 
                        env_builder_args : dict,
                        runner_builder : VecEnvRunnerBuilderProtocol):
    quiet = env_builder_args["quiet"]
    stepLength_sec = env_builder_args["stepLength_sec"]
    autoreset = env_builder_args.get("autoreset", False)
    vrunner = runner_builder( seed = seed,
                                run_folder = log_folder,
                                env_builder_args = env_builder_args,
                                num_envs = 1,
                                quiet=quiet,
                                autoreset = autoreset)
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
                return single_env_builder(seed=seed, log_folder=log_folder,is_eval=is_eval,env_builder_args=env_builder_args,runner_builder=runner_builder)
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



robot_args_registry = {}

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
    feet_links = [  ('quad', 'foot_center_link_back_left'),
                    ('quad', 'foot_center_link_back_right'),
                    ('quad', 'foot_center_link_front_left'),
                    ('quad', 'foot_center_link_front_right')]
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("adarl_envs","models/quad_simple.urdf.xacro"),
            "model_kwargs" : {  "use_cylinders" : "false",
                                "all_collisions" : "false"},
            "xacro_extra_pkg_paths" : {"adarl_envs" : adarl.utils.utils.pkgutil_get_path("adarl_envs")},
            "homing_joint_position" : homing,
            "homing_joint_position_references" : None,
            "robot_name" : "quad",
            "robot_main_body_link" : "body_link",
            "robot_root_link" : "body_link",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.5,0.,0.,0.,1.),
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,0.5-0.1),(0.,0.,0.5+0.1)),
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
            "control_limits_center" : homing,
            "enable_link_collisions" : [    (('quad', 'foot_center_link_back_left'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_back_right'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_front_left'),[('ground','ground_link')]),
                                            (('quad', 'foot_center_link_front_right'),[('ground','ground_link')])],
            "feet_contact_links" : feet_links,
            "feet_bottom_links" : feet_links
            }
robot_args_registry["quad"] = get_quad_args


def get_kyon_args(robot_options : dict = {}):
    enable_arms = robot_options.get("enable_arms", False)
    # Flat feet instead of the contact spheres: the wheeled leg with a triangular foot plate where the
    # wheel goes, touching the ground with two spheres, and the wheel motor driving an ankle pitch joint.
    # NOTE: the quadruped homing below was measured with the contact-sphere legs and is not retuned for
    # this (the legs get 0.068m longer), the humanoid config is the one that uses it.
    feet = robot_options.get("feet", False)
    # hip_pitch = -0.8727 # = -50/180*3.14159
    # hip_roll =   0.0349 # = 2/180*3.14159
    # knee =      -1.5707 # = -90/180*3.14159
    # height = 0.47
    # # These are correct for stiffness=500
    # homing = {  ("kyon","hip_roll_3") :    0.0930,
    #             ("kyon","hip_roll_4") :   -0.0930,
    #             ("kyon","hip_roll_1") :   -0.1115,
    #             ("kyon","hip_roll_2") :    0.1115,
    #             ("kyon","hip_pitch_3") :  -0.8795,
    #             ("kyon","hip_pitch_4") :   0.8795,
    #             ("kyon","hip_pitch_1") :  -0.8840,
    #             ("kyon","hip_pitch_2") :   0.8840,
    #             ("kyon","knee_pitch_3") :  1.6330,
    #             ("kyon","knee_pitch_4") : -1.6330,
    #             ("kyon","knee_pitch_1") :  1.6495,
    #             ("kyon","knee_pitch_2") : -1.6495}
    
    hip_pitch = -0.70 # = -50/180*3.14159
    hip_roll =   0.02 # = 2/180*3.14159
    knee =      -1.4 # = -90/180*3.14159
    height = 0.493
    # These are correct for stiffness=500
    homing = {  ("kyon","hip_roll_3") :    0.0955,
                ("kyon","hip_roll_4") :   -0.0953,
                ("kyon","hip_roll_1") :   -0.0919,
                ("kyon","hip_roll_2") :    0.0917,
                ("kyon","hip_pitch_3") :  -0.6916,
                ("kyon","hip_pitch_4") :   0.6917,
                ("kyon","hip_pitch_1") :  -0.6918,
                ("kyon","hip_pitch_2") :   0.6919,
                ("kyon","knee_pitch_3") :  1.4750,
                ("kyon","knee_pitch_4") : -1.4741,
                ("kyon","knee_pitch_1") :  1.4715,
                ("kyon","knee_pitch_2") : -1.4707}
    
    # hip_pitch = -0.91 # = -50/180*3.14159
    # hip_roll =  -0.10 # = 2/180*3.14159
    # knee =      -1.62 # = -90/180*3.14159
    # height = 0.47
    # # These are correct for stiffness=500
    # homing = {  ("kyon","hip_roll_3") :   -0.054,
    #             ("kyon","hip_roll_4") :    0.054,
    #             ("kyon","hip_roll_1") :    0.0360,
    #             ("kyon","hip_roll_2") :   -0.0360,
    #             ("kyon","hip_pitch_3") :  -0.918,
    #             ("kyon","hip_pitch_4") :   0.917,
    #             ("kyon","hip_pitch_1") :  -0.923,
    #             ("kyon","hip_pitch_2") :   0.923,
    #             ("kyon","knee_pitch_3") :  1.683,
    #             ("kyon","knee_pitch_4") : -1.682,
    #             ("kyon","knee_pitch_1") :  1.701,
    #             ("kyon","knee_pitch_2") : -1.700}
    

    homing_ref = {  
                    ("kyon","hip_roll_1") :     -hip_roll,
                    ("kyon","hip_pitch_1") :     hip_pitch,
                    ("kyon","knee_pitch_1") :   -knee,
                    ("kyon","hip_roll_2") :      hip_roll,
                    ("kyon","hip_pitch_2") :    -hip_pitch,
                    ("kyon","knee_pitch_2") :    knee,
                    ("kyon","hip_roll_3") :      hip_roll,
                    ("kyon","hip_pitch_3") :     hip_pitch,
                    ("kyon","knee_pitch_3") :   -knee,
                    ("kyon","hip_roll_4") :     -hip_roll,
                    ("kyon","hip_pitch_4") :    -hip_pitch,
                    ("kyon","knee_pitch_4") :    knee,
                    }
    j_vel_phys_lim = 7.6
    j_eff_phys_lim = 185
    j_pos_ctrl_range = 0.5
    j_vel_ctrl_lim = j_vel_phys_lim*0.9
    j_eff_ctrl_lim = j_eff_phys_lim*0.9
    # The ankles are driven by the wheel motor, which is weaker and faster than the leg ones
    ankle_vel_ctrl_lim = 30.0*0.9
    ankle_eff_ctrl_lim = 100.0*0.9
    ankle_joints = [("kyon",f"ankle_pitch_{i}") for i in range(1,5)]

    if enable_arms:
        generic_arm_joint_names = ["shoulder_yaw_", "shoulder_pitch_", "elbow_pitch_", "wrist_pitch_", "wrist_yaw_"]
        left_arm_joints =  [("kyon", f"{jn}1") for jn in generic_arm_joint_names]
        right_arm_joints = [("kyon", f"{jn}2") for jn in generic_arm_joint_names]
        left_arm_homing =  [0.0,  1,  2,  1, 0.0]  # shoulder_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_yaw
        right_arm_homing = [0.0, -1, -2, -1, 0.0]  # shoulder_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_yaw
        left_arm_pos = {j: v for j,v in zip(left_arm_joints, left_arm_homing)}
        dagana_left_pos = {("kyon","dagana_1_clamp_joint") : 0.1}
        right_arm_pos = {j: v for j,v in zip(right_arm_joints, right_arm_homing)}
        dagana_right_pos = {("kyon","dagana_2_clamp_joint") : 0.1}

        homing_ref.update(left_arm_pos)
        homing_ref.update(right_arm_pos)
        homing_ref.update(dagana_left_pos)
        homing_ref.update(dagana_right_pos)

        homing.update(left_arm_pos)
        homing.update(right_arm_pos)
        homing.update(dagana_left_pos)
        homing.update(dagana_right_pos)

    if feet:
        ankle_homing = {jn:0.0 for jn in ankle_joints}
        homing.update(ankle_homing)
        homing_ref.update(ankle_homing)

    j_pos_ctrl_lims = {k:np.array([-1.0,1.0])*j_pos_ctrl_range+homing[k] for k in homing.keys()}
    j_vel_ctrl_lims = {k:(ankle_vel_ctrl_lim if k in ankle_joints else j_vel_ctrl_lim) for k in homing.keys()}
    j_eff_ctrl_lims = {k:(ankle_eff_ctrl_lim if k in ankle_joints else j_eff_ctrl_lim) for k in homing.keys()}
    file = adarl.utils.utils.pkgutil_get_path("pykyon", "iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf.xacro")
    format = "xacro"
    # file = adarl.utils.utils.pkgutil_get_path("pykyon", "iit-kyon-ros-pkg/kyon_mjx/kyon_mjx.xml")
    # format = "mjcf"
    if feet:
        # the foot plate is what collides, the contact link is the frame at the center of the sole
        feet_links         = [('kyon', f'foot_{i}') for i in range(1,5)]
        feet_bottom_links  = [('kyon', f'contact_{i}') for i in range(1,5)]
    else:
        feet_links  = [ ('kyon', 'contact_1'),
                        ('kyon', 'contact_2'),
                        ('kyon', 'contact_3'),
                        ('kyon', 'contact_4')]
        feet_bottom_links = feet_links
    return {"model_file" : file,
            "robot_description_format" : format,
            "model_kwargs" : {"upper_body" : f"{enable_arms}",
                              "footonly_collision" : "true",
                              "feet" : f"{feet}",
                              "varta" : "true"},
            "xacro_extra_pkg_paths" : {"kyon_urdf" : adarl.utils.utils.pkgutil_get_path("pykyon", "iit-kyon-ros-pkg/kyon_urdf")},
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing_ref,
            "robot_name" : "kyon",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,height,0.,0.,0.,1.),
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,height-0.1),(0.,0.,height+0.1)),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_damping_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_com_links" : [("kyon","pelvis")],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k,v in homing_ref.items()},
            "control_limits_ratios_minmax_pve" : None,
                                                # {k:[[ joint_ranges[k], 0.9, 0.9],
                                                #     [ joint_ranges[k], 0.9, 0.9]] for k,v in homing_ref.items()},
            "control_limits_minmax_pve" : {k:th.as_tensor(
                                             [[ j_pos_ctrl_lims[k][0], -j_vel_ctrl_lims[k], -j_eff_ctrl_lims[k]],
                                              [ j_pos_ctrl_lims[k][1],  j_vel_ctrl_lims[k],  j_eff_ctrl_lims[k]]]) for k,v in homing_ref.items()},
            "control_limits_center" : None, #homing_ref,
            "enable_link_collisions" : [(fl,[('ground','ground_link'),'world']) for fl in feet_links],
            "feet_contact_links" : feet_links,
            "feet_bottom_links" : feet_bottom_links,
            "ctrl_joints_stiffness" :500.0,
            "ctrl_joints_damping" :20.0,
            "default_max_joint_impedance_ctrl_torque" : 150.0,
            "max_joint_impedance_ctrl_torques" : {jn:ankle_eff_ctrl_lim for jn in ankle_joints} if feet else {},
            "revolute_dof_damping_override" : 1.0,
            "mjx_opt_preset" : "faster",
            "mjx_opt_override" : {  
                                    #"noslip_iterations" : 10,
                                    #   "impratio" : 10.0,
                                    "cone" : 1},
            "revolute_dof_frictionloss_override" : 4.68,
            "revolute_dof_damping_override" : 1.7,
            "revolute_dof_armature_override" : 0.234
            # "mjx_geom_overrides" : {"kyon#contact_1" : {"solimp" : np.array([0.9, 0.95, 0.036, 0.5, 2.0]),
            #                                             "friction" : np.array([0.8, 0.005, 0.0001])},
            #                         "kyon#contact_2" : {"solimp" : np.array([0.9, 0.95, 0.036, 0.5, 2.0]),
            #                                             "friction" : np.array([0.8, 0.005, 0.0001])},
            #                         "kyon#contact_3" : {"solimp" : np.array([0.9, 0.95, 0.036, 0.5, 2.0]),
            #                                             "friction" : np.array([0.8, 0.005, 0.0001])},
            #                         "kyon#contact_4" : {"solimp" : np.array([0.9, 0.95, 0.036, 0.5, 2.0]),
            #                                             "friction" : np.array([0.8, 0.005, 0.0001])}}
        }
robot_args_registry["kyon"] = get_kyon_args
robot_args_registry["kyon_arms"] = lambda robot_options = {}: get_kyon_args(robot_options={**robot_options, "enable_arms": True})


def get_kyon_humanoid_args(robot_options : dict = {}):
    """Kyon treated as a humanoid: it stands upright on its rear legs (3 and 4), with the front legs
    (1 and 2) hanging in front of it as arms.

    The pelvis is spawned pitched by 90 degrees, so that the pelvis +x axis (the direction of the front
    hips) points up and the pelvis +z axis (the belly side) points forward. All four legs stay
    agent-controlled, only the rear ones are treated as feet.

    By default it stands on the flat feet of the `feet` model option (see get_kyon_args), so that each
    foot has a +-0.10m support base along the walking direction, worth +-73Nm of pitch authority.
    Passing robot_options['feet'] = False falls back to the contact-sphere legs, where the two feet only
    give a support *line*, with no resistance to pitch at all: the robot can then only stay up by
    actively balancing fore-aft. Each of the two has its own homing pose and standing height.

    The joint homing is the IK solution that puts the rear feet (soles flat on the ground, when there
    are feet) right below the robot center of mass, with the pelvis at `height` and at least 0.55rad of
    margin on every joint limit (so that the +-0.5rad position control range around the homing stays
    inside the physical limits). Note that in this configuration the rear knees bend towards the back of
    the robot, i.e. away from the walking direction: the solutions that bend them the other way sit
    within 0.25rad of the hip_pitch limits.
    """
    robot_options = {"feet": True, **robot_options}
    args = get_kyon_args(robot_options=robot_options)
    feet = robot_options["feet"]

    # Pelvis pitched by 90deg: pelvis +x -> world up, pelvis +z (belly) -> world forward.
    upright_body_quat_xyzw = (0.70710678, 0.0, 0.70710678, 0.0)
    # The gait frame (z-up, x-forward while standing) expressed in the pelvis frame is the conjugate of
    # the upright spawn orientation, so that at spawn the gait frame is aligned with the world.
    gait_frame_quat_xyzw = (-0.70710678, 0.0, -0.70710678, 0.0)
    if feet:
        # Pelvis height with the rear soles flat on the ground in the homing pose. This is 88% of the
        # kinematic maximum (1.15m): standing more crouched costs a lot of torque just to hold the pose
        # (77Nm at the knee at 1.00m, against a 150Nm actuator limit, i.e. 3.2cm of sag at stiffness 500)
        # and needs a bigger ankle angle to keep the sole flat, standing more extended runs out of
        # vertical authority (6.2rad/m of joint motion per meter of squat here, against 23rad/m at
        # 1.14m) without buying any meaningful fall time, since that only scales with sqrt(h/g).
        height = 1.055
        # Rear soles flat on the ground at (-1.06, +-0.33, 0.02) in the pelvis frame (1.06m below the
        # pelvis, 0.66m apart, right below the center of mass), hands at (-0.10, +-0.31, 0.25) (0.1m
        # below and 0.25m in front of the pelvis) with the ankle straight.
        upright_legs = { ("kyon","hip_roll_1")    : -0.0838,
                         ("kyon","hip_pitch_1")   : -1.1508,
                         ("kyon","knee_pitch_1")  : -1.5496,
                         ("kyon","ankle_pitch_1") :  0.0,
                         ("kyon","hip_roll_2")    :  0.0838,
                         ("kyon","hip_pitch_2")   :  1.1508,
                         ("kyon","knee_pitch_2")  :  1.5496,
                         ("kyon","ankle_pitch_2") :  0.0,
                         ("kyon","hip_roll_3")    :  0.0,
                         ("kyon","hip_pitch_3")   : -1.1196,
                         ("kyon","knee_pitch_3")  : -1.0511,
                         ("kyon","ankle_pitch_3") :  0.5998,
                         ("kyon","hip_roll_4")    :  0.0,
                         ("kyon","hip_pitch_4")   :  1.1196,
                         ("kyon","knee_pitch_4")  :  1.0511,
                         ("kyon","ankle_pitch_4") : -0.5998}
    else:
        # Same pose on the contact-sphere legs, which are 0.068m shorter and have no ankle. 87% of the
        # kinematic maximum of that leg (1.049m), by the same reasoning as above.
        height = 0.955
        # Feet at (-0.96, +-0.30, 0.01) in the pelvis frame, hands at (-0.10, +-0.31, 0.25).
        upright_legs = { ("kyon","hip_roll_1")    :  0.0389,
                         ("kyon","hip_pitch_1")   : -1.5211,
                         ("kyon","knee_pitch_1")  : -1.1356,
                         ("kyon","hip_roll_2")    : -0.0389,
                         ("kyon","hip_pitch_2")   :  1.5211,
                         ("kyon","knee_pitch_2")  :  1.1356,
                         ("kyon","hip_roll_3")    :  0.0,
                         ("kyon","hip_pitch_3")   : -0.9456,
                         ("kyon","knee_pitch_3")  : -1.3744,
                         ("kyon","hip_roll_4")    :  0.0,
                         ("kyon","hip_pitch_4")   :  0.9456,
                         ("kyon","knee_pitch_4")  :  1.3744}
    # Only the legs are moved, so that the upper body joints keep the homing of the quadruped config
    # when it is enabled. References and initial positions are the same: unlike the quadruped homing
    # these were not measured after letting the robot settle, so at the episode start the robot sags a
    # bit under gravity.
    homing = dict(args["homing_joint_position"])
    homing.update(upright_legs)
    homing_ref = dict(args["homing_joint_position_references"])
    homing_ref.update(upright_legs)

    j_pos_ctrl_range = 0.5
    ankle_joints = [("kyon",f"ankle_pitch_{i}") for i in range(1,5)]
    j_pos_ctrl_lims = {k:np.array([-1.0,1.0])*j_pos_ctrl_range+homing[k] for k in homing.keys()}
    j_vel_ctrl_lims = {k:(30.0 if k in ankle_joints else 7.6)*0.9 for k in homing.keys()}
    j_eff_ctrl_lims = {k:(100.0 if k in ankle_joints else 185.0)*0.9 for k in homing.keys()}

    # Only the rear legs are feet. With the plates, they are what collides and the contact links are the
    # sole centers; with the contact spheres the same link is both.
    feet_links        = [('kyon', f'foot_{i}' if feet else f'contact_{i}') for i in (3,4)]
    feet_bottom_links = [('kyon', f'contact_{i}') for i in (3,4)]
    hand_links        = [('kyon', f'foot_{i}' if feet else f'contact_{i}') for i in (1,2)]

    args.update({
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing_ref,
            "homing_body_pose_xyz_xyzw" : (0.,0.,height)+upright_body_quat_xyzw,
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,height-0.1),(0.,0.,height+0.1)),
            "main_body_gait_frame_quat_xyzw" : gait_frame_quat_xyzw,
            "control_limits_minmax_pve" : {k:th.as_tensor(
                                             [[ j_pos_ctrl_lims[k][0], -j_vel_ctrl_lims[k], -j_eff_ctrl_lims[k]],
                                              [ j_pos_ctrl_lims[k][1],  j_vel_ctrl_lims[k],  j_eff_ctrl_lims[k]]]) for k in homing.keys()},
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k in homing.keys()},
            # The hands are not feet, but they must still collide with the ground
            "feet_contact_links" : feet_links,
            "feet_bottom_links" : feet_bottom_links,
            "enable_link_collisions" : [(l,[('ground','ground_link'),'world']) for l in feet_links+hand_links],
            "goal_height_minmax" : [height, height],
            # A biped tips over much more easily than a quadruped, give it some room before calling it a crash
            "terminal_gravity_angle" : 40*pi/180,
        })
    if feet:
        # Standing on flat feet, the body leans by deflecting the ankles, so a soft ankle makes the
        # upright pose an unstable equilibrium no matter how stiff the rest of the leg is: the restoring
        # torque of the ankle spring has to beat gravity's m*g*h_com ~= 740Nm/rad. At the default
        # 500Nm/rad the two cancel almost exactly and the robot slowly topples; 1200Nm/rad gives 1.6x of
        # margin, and saturates the 90Nm ankle at 0.075rad, well inside the +-0.1rad the sole can
        # support before the CoP runs off the toe.
        args.update({
            "ctrl_joints_stiffness" : {"default" : 500.0, **{jn:1200.0 for jn in ankle_joints}},
            "ctrl_joints_damping" :   {"default" :  20.0, **{jn:  30.0 for jn in ankle_joints}},
            "minmax_ctrl_stiffness" : (0.0, 2000.0),
        })
    return args
robot_args_registry["kyon_humanoid"] = get_kyon_humanoid_args

def get_go1_args():

    # physical limits are:
    #  hip:   -0.863 +0.863 (midpoint = 0.0)
    #  thigh: -0.686 +4.501 (midpoint = 1.9075) (0.0 points straight down, positive points backward)
    #  knee:  -2.818 -0.888 (midpoint = -1.853)
    rname = "go1"
    hip_roll =   0.0 
    hip_pitch =   1.0 # 1.0
    knee =       -1.8 #-1.84
    homing = {  (rname,"RL_hip_joint") :  hip_roll,
                (rname,"RR_hip_joint") :  hip_roll,
                (rname,"FL_hip_joint") :  hip_roll,
                (rname,"FR_hip_joint") :  hip_roll,
                (rname,"RL_thigh_joint") :  hip_pitch,
                (rname,"RR_thigh_joint") :  hip_pitch,
                (rname,"FL_thigh_joint") :  hip_pitch,
                (rname,"FR_thigh_joint") :  hip_pitch,
                (rname,"RL_calf_joint") :  knee,
                (rname,"RR_calf_joint") :  knee,
                (rname,"FL_calf_joint") :  knee,
                (rname,"FR_calf_joint") :  knee}
    max_calf_torque = 35.55
    max_thigh_torque = 23.7
    max_hip_torque = 23.7

    calf_ctrl_limits = [[ 0.8, 0.9, 0.9],
                        [ 0.8, 0.9, 0.9]]
    thigh_ctrl_limits = [[ 0.5, 0.9, 0.9],
                        [ 0.5, 0.9, 0.9]]
    hip_ctrl_limits = [[ 0.5, 0.9, 0.9],
                        [ 0.5, 0.9, 0.9]]
    ctrl_lims = {  (rname,"RL_hip_joint") : hip_ctrl_limits,
                    (rname,"RR_hip_joint") : hip_ctrl_limits,
                    (rname,"FL_hip_joint") : hip_ctrl_limits,
                    (rname,"FR_hip_joint") : hip_ctrl_limits,
                    (rname,"RL_thigh_joint") : thigh_ctrl_limits,
                    (rname,"RR_thigh_joint") : thigh_ctrl_limits,
                    (rname,"FL_thigh_joint") : thigh_ctrl_limits,
                    (rname,"FR_thigh_joint") : thigh_ctrl_limits,
                    (rname,"RL_calf_joint") : calf_ctrl_limits,
                    (rname,"RR_calf_joint") : calf_ctrl_limits,
                    (rname,"FL_calf_joint") : calf_ctrl_limits,
                    (rname,"FR_calf_joint") : calf_ctrl_limits}
    # from robot_descriptions import go1_mj_description
    # go1_description_path = "unitree_ros/robots/go1_description"
    # go1_file_path = adarl.utils.utils.pkgutil_get_path("pyunitree_ros", go1_description_path+"/xacro/robot.xacro")
    # xacro_extr_pkg_paths = {"go1_description" : adarl.utils.utils.pkgutil_get_path("pyunitree_ros", go1_description_path)}
    
    # go1_file_path = go1_mj_description.MJCF_PATH
    # xacro_extr_pkg_paths = {}

    go1_file_path = adarl.utils.utils.pkgutil_get_path("mujoco_playground","_src/locomotion/go1/xmls/go1_mjx.xml")
    xacro_extra_pkg_paths = {}
    raw_model_string = Path(go1_file_path).read_text()

    # Need to have mujoco_playground with the already downloaded menagerie inside it
    menagerie_go1_assets_folder = adarl.utils.utils.pkgutil_get_path("mujoco_playground", "external_deps/mujoco_menagerie/unitree_go1/assets")
    go1_string = set_asset_texture_paths(raw_model_string,
                            menagerie_go1_assets_folder,
                            menagerie_go1_assets_folder)
    feet_links = [  (rname, 'FL'),
                    (rname, 'FR'),
                    (rname, 'RL'),
                    (rname, 'RR')]
    ggLog.info(f"Using go1 model string: \n{go1_string}")
    return {"robot_description_string" : go1_string,
            "robot_description_format" : "mjcf",
            "model_kwargs" : {},
            "xacro_extra_pkg_paths" : xacro_extra_pkg_paths,
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing,
            "robot_name" : rname,
            "robot_main_body_link" : "trunk",
            "robot_root_link" : "trunk",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.4,0.,0.,0.,1.),
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,0.4-0.1),(0.,0.,0.4+0.1)),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_com_links" : [(rname,"trunk")],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_dof_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_ratios_minmax_pve" : ctrl_lims,
            "control_limits_center" : homing,
            "enable_link_collisions" : [    ((rname, 'FL_calf'),[('ground','ground_link')]),
                                            ((rname, 'FR_calf'),[('ground','ground_link')]),
                                            ((rname, 'RL_calf'),[('ground','ground_link')]),
                                            ((rname, 'RR_calf'),[('ground','ground_link')])],
            "max_joint_impedance_ctrl_torques" : {  (rname,"RL_hip_joint") :  max_hip_torque,
                                                    (rname,"RR_hip_joint") :  max_hip_torque,
                                                    (rname,"FL_hip_joint") :  max_hip_torque,
                                                    (rname,"FR_hip_joint") :  max_hip_torque,
                                                    (rname,"RL_thigh_joint") :  max_thigh_torque,
                                                    (rname,"RR_thigh_joint") :  max_thigh_torque,
                                                    (rname,"FL_thigh_joint") :  max_thigh_torque,
                                                    (rname,"FR_thigh_joint") :  max_thigh_torque,
                                                    (rname,"RL_calf_joint") :  max_calf_torque,
                                                    (rname,"RR_calf_joint") :  max_calf_torque,
                                                    (rname,"FL_calf_joint") :  max_calf_torque,
                                                    (rname,"FR_calf_joint") :  max_calf_torque},
            "feet_contact_links" : feet_links,
            "feet_bottom_links" : feet_links,
            "ctrl_joints_stiffness" :50.0,
            "ctrl_joints_damping" :2.5
        }
robot_args_registry["go1"] = get_go1_args

def get_spot_args():
    # physical limits are (on the xml actually they are not all the same, these are the tighter ones):
    #  hipx:   -0.78 +0.78  (midpoint = 0.0, range 1.56)
    #  hipy:   -0.89 +2.24  (midpoint = 0.675, range 3.13) (0.0 points straight down? positive points backward?)
    #  knee:   -2.79 -0.245 (midpoint = -1.5175, range 2.545)
    hipx_lims = (-0.78, 0.78)
    hipy_lims = (-0.89, 2.24)
    knee_lims = (-2.79, -0.245)

    rname = "spot"
    hip_roll =    0.0 
    hip_pitch =   1.04 # 1.0
    knee =       -1.8  #-1.84
    homing = {  (rname,"fl_hx") :  hip_roll,
                (rname,"fr_hx") :  hip_roll,
                (rname,"hl_hx") :  hip_roll,
                (rname,"hr_hx") :  hip_roll,
                (rname,"fl_hy") :  hip_pitch,
                (rname,"fr_hy") :  hip_pitch,
                (rname,"hl_hy") :  hip_pitch,
                (rname,"hr_hy") :  hip_pitch,
                (rname,"fr_kn") :  knee,
                (rname,"fl_kn") :  knee,
                (rname,"hl_kn") :  knee,
                (rname,"hr_kn") :  knee}
    max_knee_torque = 100
    max_hipy_torque = 100
    max_hipx_torque = 100

    # Ratios set so that the range is 0.3 radians from homing
    hxr = 0.3/((hipx_lims[1]-hipx_lims[0])/2)
    hyr = 0.3/((hipy_lims[1]-hipy_lims[0])/2)
    knr = 0.3/((knee_lims[1]-knee_lims[0])/2)

    hip_ctrl_limits =   [[ hxr, 0.9, 0.9],
                         [ hxr, 0.9, 0.9]]
    thigh_ctrl_limits = [[ hyr, 0.9, 0.9],
                         [ hyr, 0.9, 0.9]]
    calf_ctrl_limits =  [[ knr, 0.9, 0.9],
                         [ knr, 0.9, 0.9]]
    ctrl_lims = {   (rname,"fl_hx") : hip_ctrl_limits,
                    (rname,"fr_hx") : hip_ctrl_limits,
                    (rname,"hl_hx") : hip_ctrl_limits,
                    (rname,"hr_hx") : hip_ctrl_limits,
                    
                    (rname,"fl_hy") : thigh_ctrl_limits,
                    (rname,"fr_hy") : thigh_ctrl_limits,
                    (rname,"hl_hy") : thigh_ctrl_limits,
                    (rname,"hr_hy") : thigh_ctrl_limits,
                    
                    (rname,"fl_kn") : calf_ctrl_limits,
                    (rname,"fr_kn") : calf_ctrl_limits,
                    (rname,"hl_kn") : calf_ctrl_limits,
                    (rname,"hr_kn") : calf_ctrl_limits}
    # from robot_descriptions import go1_mj_description
    # go1_description_path = "unitree_ros/robots/go1_description"
    # go1_file_path = adarl.utils.utils.pkgutil_get_path("pyunitree_ros", go1_description_path+"/xacro/robot.xacro")
    # xacro_extr_pkg_paths = {"go1_description" : adarl.utils.utils.pkgutil_get_path("pyunitree_ros", go1_description_path)}
    
    # go1_file_path = go1_mj_description.MJCF_PATH
    # xacro_extr_pkg_paths = {}

    spot_file_path = adarl.utils.utils.pkgutil_get_path("mujoco_playground","_src/locomotion/spot/xmls/spot_mjx_feetonly.xml")
    xacro_extra_pkg_paths = {}
    raw_model_string = Path(spot_file_path).read_text()

    # Need to have mujoco_playground with the already downloaded menagerie inside it
    os.environ["MUJOCO_GL"] = "egl" # Need to set this before importing mujoco
    from mujoco_playground._src.mjx_env import ensure_menagerie_exists
    ensure_menagerie_exists()
    menagerie_spot_assets_folder = adarl.utils.utils.pkgutil_get_path("mujoco_playground", "external_deps/mujoco_menagerie/boston_dynamics_spot/assets")
    spot_string = set_asset_texture_paths(raw_model_string,
                            menagerie_spot_assets_folder,
                            menagerie_spot_assets_folder)
    feet_links = [  (rname, 'fl_lleg'),
                    (rname, 'fr_lleg'),
                    (rname, 'hl_lleg'),
                    (rname, 'hr_lleg')]
    ggLog.info(f"Using spot model string: \n{spot_string}")
    return {"robot_description_string" : spot_string,
            "robot_description_format" : "mjcf",
            "model_kwargs" : {},
            "xacro_extra_pkg_paths" : xacro_extra_pkg_paths,
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing,
            "robot_name" : rname,
            "robot_main_body_link" : "body",
            "robot_root_link" : "body",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.460915,0.,0.,0.,1.),
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,0.460915-0.1),(0.,0.,0.460915+0.1)),
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_com_links" : [(rname,"body")],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_dof_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 1.0, 0.9, 0.9],
                                                    [ 1.0, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_ratios_minmax_pve" : ctrl_lims,
            "control_limits_center" : homing,
            "enable_link_collisions" : [    ((rname, 'fr_lleg'),[('ground','ground_link')]),
                                            ((rname, 'fl_lleg'),[('ground','ground_link')]),
                                            ((rname, 'hr_lleg'),[('ground','ground_link')]),
                                            ((rname, 'hl_lleg'),[('ground','ground_link')])],
            "max_joint_impedance_ctrl_torques" : {  (rname,"fl_hx") :  max_hipx_torque,
                                                    (rname,"fr_hx") :  max_hipx_torque,
                                                    (rname,"hr_hx") :  max_hipx_torque,
                                                    (rname,"hl_hx") :  max_hipx_torque,
                                                    (rname,"fl_hy") :  max_hipy_torque,
                                                    (rname,"fr_hy") :  max_hipy_torque,
                                                    (rname,"hr_hy") :  max_hipy_torque,
                                                    (rname,"hl_hy") :  max_hipy_torque,
                                                    (rname,"fl_kn") :  max_knee_torque,
                                                    (rname,"fr_kn") :  max_knee_torque,
                                                    (rname,"hr_kn") :  max_knee_torque,
                                                    (rname,"hl_kn") :  max_knee_torque},
            "default_max_joint_impedance_ctrl_torque" : 100.0,
            "feet_links" : feet_links,
            "feet_bottom_links" : feet_links,
            "ctrl_joints_stiffness" :300.0,
            "ctrl_joints_damping" :20.0,
            "mjx_opt_override" : {"impratio" : 1.0,
                                  "iterations" : 1,
                                  "ls_iterations" : 5,
                                  "noslip_iterations" : 0},
            "revolute_dof_damping_override" : 1.0
        }
robot_args_registry["spot"] = get_spot_args

def union(dicts : list[dict]):
    out = {}
    for d in dicts:
        out.update(d)
    return out


def get_centauro_args(control_arms=False, robot_options : dict | None = None):
    add_twisting_pelvis = robot_options is not None and robot_options.get("add_twisting_pelvis", False)
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
                ("centauro","j_arm1_1") : 0.52,
                ("centauro","j_arm1_2") : 0.40,
                ("centauro","j_arm1_3") : 0.27,
                ("centauro","j_arm1_4") : -2.00,
                ("centauro","j_arm1_5") : 0.05,
                ("centauro","j_arm1_6") : -0.78,
                ("centauro","j_arm2_1") : 0.52,
                ("centauro","j_arm2_2") : -0.40,
                ("centauro","j_arm2_3") : -0.27,
                ("centauro","j_arm2_4") : -2.00,
                ("centauro","j_arm2_5") : -0.05,
                ("centauro","j_arm2_6") : -0.78,
                ("centauro","j_wheel_1") : 0.0,
                ("centauro","j_wheel_2") : 0.0,
                ("centauro","j_wheel_3") : 0.0,
                ("centauro","j_wheel_4") : 0.0
                # ("centauro","dagana_1_claw_joint") : 0.3,
                # ("centauro","dagana_2_claw_joint") : 0
                }
    if add_twisting_pelvis:
        homing[("centauro","twisting_pelvis_joint")] = 0.0
    homing_ref = homing.copy()
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
    arms = [    "j_arm1_1",
                "j_arm1_2",
                "j_arm1_3",
                "j_arm1_4",
                "j_arm1_5",
                "j_arm1_6",
                "j_arm2_1",
                "j_arm2_2",
                "j_arm2_3",
                "j_arm2_4",
                "j_arm2_5",
                "j_arm2_6",
                ]
    controlled_joints = legs
    if control_arms:
        controlled_joints += arms

    use_contact_lnks = True
    feet_bottom_links = [   ('centauro', 'wheel_contact_1'),
                            ('centauro', 'wheel_contact_2'),
                            ('centauro', 'wheel_contact_3'),
                            ('centauro', 'wheel_contact_4')]
    feet_contact_links = [  ('centauro', 'wheel_1'),
                            ('centauro', 'wheel_2'),
                            ('centauro', 'wheel_3'),
                            ('centauro', 'wheel_4')]

    j_vel_ctrl_lim = 7.0
    j_eff_ctrl_lim_leg_a = 200.0
    j_eff_ctrl_lim_leg_b = 100.0
    j_eff_ctrl_lim_leg_c = 35.0
    j_eff_ctrl_lim_arm_a = 140.0
    j_eff_ctrl_lim_arm_b = 55.0
    j_eff_ctrl_lims =union([{   ("centauro",f"hip_yaw_{i}") :      j_eff_ctrl_lim_leg_a,
                                ("centauro",f"hip_pitch_{i}") :    j_eff_ctrl_lim_leg_a,
                                ("centauro",f"knee_pitch_{i}") :   j_eff_ctrl_lim_leg_a,
                                ("centauro",f"ankle_pitch_{i}") :  j_eff_ctrl_lim_leg_b,
                                ("centauro",f"ankle_yaw_{i}") :    j_eff_ctrl_lim_leg_c} for i in range(1,5)]+
                            [{  ("centauro",f"j_arm{i}_1") : j_eff_ctrl_lim_arm_a,
                                ("centauro",f"j_arm{i}_2") : j_eff_ctrl_lim_arm_a,
                                ("centauro",f"j_arm{i}_3") : j_eff_ctrl_lim_arm_a,
                                ("centauro",f"j_arm{i}_4") : j_eff_ctrl_lim_arm_a,
                                ("centauro",f"j_arm{i}_5") : j_eff_ctrl_lim_arm_b,
                                ("centauro",f"j_arm{i}_6") : j_eff_ctrl_lim_arm_b} for i in range(1,3)]+
                            [{  ("centauro",f"j_wheel_{i}") : j_eff_ctrl_lim_leg_c} for i in range(1,5)]+
                            [{  ("centauro","torso_yaw") : 140.0,
                                ("centauro","velodyne_joint") : 35.0,
                                ("centauro","d435_head_joint") : 35.0
                                }])
    if add_twisting_pelvis:
        j_eff_ctrl_lims[("centauro","twisting_pelvis_joint")] = 10000.0
    j_pos_range = 0.5
    j_pos_ctrl_lims = {k:np.array([-1.0,1.0])*j_pos_range+homing[k] for k in homing.keys()}
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro"),
            "model_kwargs" : {  "realsense":"false",
                                "velodyne" :"false",
                                "floating_joint":"true",
                                "sphere_or_ellipsoid_wheel_collision":use_contact_lnks,
                                "add_twisting_pelvis" : "true" if add_twisting_pelvis else "false"
                                },
            "robot_description_format" : "xacro",
            "xacro_extra_pkg_paths" : {"centauro_urdf" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf")},
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing_ref,
            "robot_name" : "centauro",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.84,0.,0.,0.,1.),
            # spawn box: x,y fixed at the homing spot, z a clearance above the local terrain
            "randomized_homing_body_position_minmax_xyz" : ((0.,0.,0.84-0.1),(0.,0.,0.84+0.1)),
            "default_max_joint_impedance_ctrl_torque" : 100.0,
            "max_joint_impedance_ctrl_torques" : j_eff_ctrl_lims,
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : controlled_joints,
            "randomized_dof_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_com_links" : [("centauro","pelvis")],
            "randomized_dof_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_damping_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {k:[[ 0.9, 0.9, 0.9],
                                                    [ 0.9, 0.9, 0.9]] for k,v in homing.items()},
            "control_limits_center" : None,
            "control_limits_ratios_minmax_pve" : None,
            "control_limits_minmax_pve" : {k:th.as_tensor(
                                             [[ j_pos_ctrl_lims[k][0], -j_vel_ctrl_lim, -j_eff_ctrl_lims[k]],
                                              [ j_pos_ctrl_lims[k][1],  j_vel_ctrl_lim,  j_eff_ctrl_lims[k]]]) for k in homing.keys()},            
            "enable_link_collisions" : [(fl,[('ground','ground_link')]) for fl in feet_contact_links],
            "feet_contact_links" : feet_contact_links,
            "feet_bottom_links" : feet_bottom_links,
            "ctrl_joints_stiffness" :600.0,
            "ctrl_joints_damping" :20.0,
            "mjx_opt_preset" : "faster",
            "revolute_dof_frictionloss_override" : 4.68,
            "revolute_dof_damping_override" :  1.7,
            "revolute_dof_armature_override" : 0.234
        }
robot_args_registry["centauro"] = get_centauro_args
robot_args_registry["centauro_legs_arms"] = lambda robot_options = {}: get_centauro_args(control_arms=True, robot_options=robot_options)

def named_loco_venv_builder(seed : int,
                    run_folder : str,
                    num_envs : int, 
                    env_builder_args : dict,
                    env_name : str = "") -> gym.vector.VectorEnv:
    full_args = robot_args_registry[env_builder_args["robot_model"]](robot_options=env_builder_args["robot_options"])
    full_args.update(env_builder_args)
    return loco_venv_builder(seed = seed,
                            log_folder = run_folder,
                            env_builder_args = full_args,
                            num_envs=num_envs,
                            runner_builder=loco_runner_builder)[0]

def named_loco_single_env_builder(seed : int,
                    log_folder : str,
                    is_eval : bool, 
                    env_builder_args : dict) -> tuple[gym.Env,float]:
    full_args = robot_args_registry[env_builder_args["robot_model"]](robot_options=env_builder_args["robot_options"])
    full_args.update(env_builder_args)
    return single_env_builder(seed = seed,
                            log_folder = log_folder,
                            env_builder_args = full_args,
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


