#!/usr/bin/env python3  
from __future__ import annotations
from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
import threading, os
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from pathlib import Path
import adarl.utils.utils
from adarl_envs.env.GraspVecEnv import GraspVecEnv, GrapVecEnvInitArgs, RobotVecEnvInitArgs
from adarl_envs.env.RobotVecEnv import JOINT_FILTERS, LINK_FILTERS
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.Runner2GymWrapper import Runner2GymWrapper
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
import gymnasium as gym
import copy
from rreal.algorithms.sac_helpers import build_vec_env, VecEnvRunnerBuilderProtocol
from math import pi
from rreal.algorithms.sac_helpers import VecEnvBuilderProtocol, EnvBuilderProtocol
from adarl_envs.experiments.loco_builder import get_robot_string_and_format, set_asset_texture_paths

def format_tensor(t, float_precision):
    t = t.squeeze().cpu().tolist()
    if not isinstance(t,list):
        t = [t]
    t = [f"{e: .{float_precision}f}" if isinstance(e,float) else str(e) for e in t]
    return f"[{', '.join(t)}]"

def overlay_text_func(vo, a, r, te, tr, info, extra_info):
    return  (   f"\n"
                f"Step    {info['ep_step_count']: .3f}\n"+
                f"obj2hand_dist          {format_tensor(info.get('obj2hand_dist',None), 3)}\n"
                f"obj2goal_dist          {format_tensor(info.get('obj2goal_dist',None), 3)}\n")

def runner_builder(seed,
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
    th_device = env_builder_args["th_device"]
    show_gui = env_builder_args.pop("show_gui",False)
    env_builder_args.update({"centauro" : get_centauro_args,
                             "franka" : get_franka_args,
                             }[env_builder_args["robot_model"].lower()]())
    robot_name = env_builder_args["robot_name"]
    max_steps = env_builder_args.pop("max_steps_per_episode")

    mode = env_builder_args["mode"]
    if mode == "gz":
        raise NotImplementedError()
    elif mode == "gazebo":
        raise NotImplementedError()
    elif mode == "mjx":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        import jax
        ground_link = ("ground","ground_link")
        sim_dt = 1/1024
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
                                            mjx_impl="jax")
    elif mode == "mujoco":
        from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
        if num_envs != 1:
            raise RuntimeError(f"The 'mujoco' adapter (mujoco classic) only supports a single environment, but got num_envs={num_envs}")
        ground_link = ("ground","ground_link")
        # MujocoJointImpedanceAdapter always renders and runs on cpu; it does not take enable_rendering/show_gui.
        env_builder_args.pop("enable_rendering", None)
        adapter = MujocoJointImpedanceAdapter(  vec_size=num_envs,
                                                sim_step_dt=1/2048,
                                                step_length_sec=stepLength_sec,
                                                output_th_device=th_device,
                                                log_folder=run_folder,
                                                show_gui=show_gui,
                                                default_max_joint_impedance_ctrl_torque=env_builder_args.get("default_max_joint_impedance_ctrl_torque", 100.0),
                                                max_joint_impedance_ctrl_torques=env_builder_args.get("max_joint_impedance_ctrl_torques", {}),
                                                reference_filter_cutoff_frequency=20.0,
                                                reference_filter_mode="second_order" if env_builder_args.get("enable_reference_filter", True) else "none")
    elif mode == "genesis":
        from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
        ground_link = ("ground","ground_link")
        sim_dt = 1/1024
        # collision-pair filtering (set_body_collisions) is not implemented in GenesisAdapter
        env_builder_args["enable_link_collisions"] = None
        # the ui camera ("simple_camera") is parsed automatically from the camera model spawned by the env
        adapter = GenesisJointImpedanceAdapter( vec_size=num_envs,
                                                output_th_device=th_device,
                                                sim_step_dt=sim_dt,
                                                step_length_sec=stepLength_sec,
                                                enable_rendering=env_builder_args.pop("enable_rendering", False),
                                                show_gui=show_gui,
                                                log_folder=run_folder,
                                                default_max_joint_impedance_ctrl_torque=env_builder_args.pop("default_max_joint_impedance_ctrl_torque", 100.0),
                                                max_joint_impedance_ctrl_torques=env_builder_args.pop("max_joint_impedance_ctrl_torques", {}),
                                                reference_filter_cutoff_frequency=20.0,
                                                reference_filter_mode="second_order" if env_builder_args["enable_reference_filter"] else "none")
    else:
        print(f"Requested unknown adapter '{mode}'")
        exit(0)

    time.sleep(1)

    s,f = get_robot_string_and_format(model_file_path = env_builder_args.get("model_file", None),
                                        robot_description_format = env_builder_args["robot_description_format"],
                                        robot_description_string = env_builder_args.get("robot_description_string", None),
                                        model_kwargs = env_builder_args.get("model_kwargs"),
                                        xacro_extra_pkg_paths = env_builder_args.get("xacro_extra_pkg_paths"))
    robot_description_string = s
    robot_description_format = f
    
    lrenv = GraspVecEnv(grasp_init_args = GrapVecEnvInitArgs(
                                                robot_init_args = RobotVecEnvInitArgs(
                                                    noise_action_delay_mustd_std = env_builder_args.pop("noise_action_delay_mustd_std"),
                                                    noise_action_mustd = env_builder_args.pop("noise_action_mustd"), 
                                                    action_smoothing_halflife_sec=env_builder_args.pop("action_smoothing_halflife_sec"),
                                                    adapter=adapter,
                                                    control_mode = env_builder_args.pop("control_mode"),
                                                    control_mode_position_delta_max = env_builder_args.pop("control_mode_position_delta_max"),
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
                                                    homing_body_pose_xyz_xyzw=env_builder_args.pop("homing_body_pose_xyz_xyzw"),
                                                    homing_joint_position=env_builder_args.pop("homing_joint_position"),
                                                    homing_joint_position_references=env_builder_args.pop("homing_joint_position_references"),
                                                    impulse_duration_minmax=env_builder_args.pop("impulse_duration_minmax"),
                                                    impulse_mean_std=env_builder_args.pop("impulse_mean_std"),
                                                    impulse_probability_per_sec=env_builder_args.pop("impulse_probability_per_sec"),
                                                    init_on_reset_ratio = env_builder_args.pop("init_on_reset_ratio"),
                                                    randomization_initial_height_range_meters = env_builder_args.pop("randomization_initial_height_range_meters"),
                                                    randomization_initial_joint_pose_range = env_builder_args.pop("randomization_initial_joint_pose_range"),
                                                    just_health_reward = env_builder_args.pop("just_health_reward"),
                                                    longterm_states_decimation_time = env_builder_args.pop("longterm_states_decimation_time"),
                                                    maxStepsPerEpisode=max_steps,
                                                    merge_privileged = env_builder_args.pop("merge_privileged"),
                                                    minmax_damping=(0.0,30.0),
                                                    minmax_stiffness=(0.0,1000.0),
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
                                                    robot_description_string=robot_description_string,
                                                    robot_description_format=robot_description_format,
                                                    safe_damping=env_builder_args.pop("safe_damping"),
                                                    control_limits_center=env_builder_args.pop("control_limits_center"),
                                                    safe_stiffness=env_builder_args.pop("safe_stiffness"),
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
                                                    history_length_action_raw=env_builder_args.pop("history_length_action_raw"),
                                                    history_length_action_smoothed=env_builder_args.pop("history_length_action_smoothed"),
                                                    extrinsics_only_privileged=env_builder_args.pop("extrinsics_only_privileged"),
                                                    posref_err_history_length=env_builder_args.pop("posref_err_history_length"),
                                                    observe_actor_safety_state=env_builder_args.pop("observe_actor_safety_state")),
                                                reward_gripper_pose_weight = env_builder_args.pop("reward_gripper_pose_weight"),
                                                reward_health_weight = env_builder_args.pop("reward_health_weight"),
                                                reward_joint_actacc_weight = env_builder_args.pop("reward_joint_actacc_weight"),
                                                reward_joint_actdiff_weight = env_builder_args.pop("reward_joint_actdiff_weight"),
                                                reward_joint_position_weight = env_builder_args.pop("reward_joint_position_weight"),
                                                reward_joint_position_limit_weight = env_builder_args.pop("reward_joint_position_limit_weight"),
                                                reward_joint_power_weight = env_builder_args.pop("reward_joint_power_weight"),
                                                reward_joint_torque_weight = env_builder_args.pop("reward_joint_torque_weight"),
                                                reward_object_pose_weight = env_builder_args.pop("reward_object_pose_weight"),
                                                reward_safety_weight = env_builder_args.pop("reward_safety_weight"),
                                                reward_scale=1000/max_steps,
                                                target_object_link=env_builder_args.pop("target_object_link"),
                                                gripper_links=env_builder_args.pop("gripper_links"),
                                                observe_object_pose=env_builder_args.pop("observe_object_pose"),
                                                manipulator_links=env_builder_args.pop("manipulator_links")))
    vrunner = EnvRunner(env=lrenv, verbose=True, quiet=False, episodeInfoLogFile=run_folder+"/vec_runner.log",
                        ui_render_envs=[0], autoreset=autoreset,
                        log_freq = max_steps)
    vrunner = EnvRunnerRecorderWrapper(vrunner,
                                    fps = 1/stepLength_sec,
                                    outFolder=run_folder+"/RunnerRecorder",
                                    env_index=0,
                                    saveFrequency_ep=env_builder_args.pop("video_save_freq"),
                                    publish=False,
                                    stream=True,
                                    vec_obs_keys=["base.vec","privileged.vec"],
                                    overlay_text_xy=(0.025,0.025),
                                    overlay_text_height=0.035,
                                    overlay_text_color_rgb=(255,150,0),
                                    overlay_text_func=overlay_text_func)
    return vrunner

from adarl_envs.experiments.loco_builder import union
import numpy as np

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
    fullhoming = {  
                ("centauro","hip_yaw_1") :      -hip_yaw,
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
                ("centauro","d435_head_joint") : -0.8,
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
                ("centauro","j_wheel_4") : 0.0,
                ("centauro","dagana_1_claw_joint") : 0.3,
                # ("centauro","dagana_2_claw_joint") : 0
                }
    arm1 = ["j_arm1_1",
            "j_arm1_2",
            "j_arm1_3",
            "j_arm1_4",
            "j_arm1_5",
            "j_arm1_6",
            "dagana_1_claw_joint"]
    arm2 = ["j_arm2_1",
            "j_arm2_2",
            "j_arm2_3",
            "j_arm2_4",
            "j_arm2_5",
            "j_arm2_6",
            "dagana_2_claw_joint"]
    legs = [
        "hip_yaw_1",
        "hip_pitch_1",
        "knee_pitch_1",
        "ankle_pitch_1",
        "ankle_yaw_1",
        "hip_yaw_2"
        "hip_pitch_2"
        "knee_pitch_2"
        "ankle_pitch_2"
        "ankle_yaw_2"
        "hip_yaw_3"
        "hip_pitch_3"
        "knee_pitch_3"
        "ankle_pitch_3"
        "ankle_yaw_3"
        "hip_yaw_4",
        "hip_pitch_4",
        "knee_pitch_4",
        "ankle_pitch_4",
        "ankle_yaw_4",
    ]
    torso = ["torso_yaw"]
    cams = ["velodyne_joint","d435_head_joint"]
    
    controlled_joints = arm1
    spawn_legs = False
    if spawn_legs:
        present_joints = arm1 + arm2 + legs + torso + cams
    else:
        present_joints = arm1 + torso + cams
    homing = {k:v for k,v in fullhoming.items() if k[1] in present_joints}
    homing_ref = homing.copy()
    j_vel_ctrl_lim = 7.0
    j_eff_ctrl_lim_leg_a = 200.0
    j_eff_ctrl_lim_leg_b = 100.0
    j_eff_ctrl_lim_leg_c = 35.0
    j_eff_ctrl_lim_arm_a = 140.0
    j_eff_ctrl_lim_arm_b = 55.0
    j_eff_ctrl_lim_dagana = 100.0
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
                                ("centauro","d435_head_joint") : 35.0,
                                ("centauro","dagana_1_claw_joint") : j_eff_ctrl_lim_dagana,
                                }])        
    j_pos_range = 0.8
    j_pos_ctrl_lims = {k:np.array([1.0,1.0])*j_pos_range for k in fullhoming.keys()}
    
    return {"model_file" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro"),
            "model_kwargs" : {  "realsense":"false",
                                "velodyne" :"false",
                                "floating_joint":f"{spawn_legs}".lower(),
                                "sphere_or_ellipsoid_wheel_collision":"true",
                                "end_effector_left":"dagana",
                                "fixed_base_joint":"true",
                                "legs":f"{spawn_legs}".lower(),
                                "dagana_claws_type":"centauro_claws_boxycollision"
                                },
            "robot_description_format" : "xacro",
            "xacro_extra_pkg_paths" : {"centauro_urdf" : adarl.utils.utils.pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf"),
                                       "dagana_urdf" : adarl.utils.utils.pkgutil_get_path("pydagana","iit-dagana-ros-pkg/dagana_urdf")},
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing_ref,
            "robot_name" : "centauro",
            "robot_main_body_link" : "pelvis",
            "robot_root_link" : "pelvis",
            "homing_body_pose_xyz_xyzw" : (0.,0.,0.8,0.,0.,0.,1.),
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
                                                    [ 0.9, 0.9, 0.9]] for k,v in fullhoming.items()},
            "control_limits_center" : fullhoming,
            "control_limits_ratios_minmax_pve" : {k:th.as_tensor(
                                                    [[ j_pos_ctrl_lims[k][0],  0.9,  0.9],
                                                    [  j_pos_ctrl_lims[k][1],  0.9,  0.9]]) for k in fullhoming.keys()},            
            "control_limits_minmax_pve" : None,
            "enable_link_collisions" : [    
                                        # (('centauro', 'wheel_1'),[('ground','ground_link')]),
                                        # (('centauro', 'wheel_2'),[('ground','ground_link')]),
                                        # (('centauro', 'wheel_3'),[('ground','ground_link')]),
                                        # (('centauro', 'wheel_4'),[('ground','ground_link')])
                                        ],
            "manipulator_links" : [('centauro', 'dagana_1_top_link'),('centauro', 'dagana_1_bottom_link')],
            "gripper_links" : [("centauro","dagana_1_fixed_palm_center"), ("centauro","dagana_1_claw_palm_center")],
            "feet_links" : [('centauro', 'wheel_contact_1'),
                            ('centauro', 'wheel_contact_2'),
                            ('centauro', 'wheel_contact_3'),
                            ('centauro', 'wheel_contact_4')],
                            "safe_stiffness" : 600.0,
            "safe_damping" : 20.0,
            "mjx_opt_preset" : "faster",
            "revolute_dof_frictionloss_override" : 4.68,
            "revolute_dof_damping_override" : 1.7,
            "revolute_dof_armature_override" : 0.234
        }

       

def get_franka_args():
    """Franka Emika Panda (7-dof arm + parallel gripper), loaded from mujoco_menagerie
    (downloaded by mujoco_playground), the same way the spot configuration is done in loco_builder."""
    rname = "franka"
    arm_joints = [f"joint{i}" for i in range(1, 8)]
    finger_joints = ["finger_joint1", "finger_joint2"]
    present_joints = arm_joints + finger_joints
    # The two fingers are coupled by an equality constraint in the model; we still command both
    # (with the same reference) so neither is a "held" joint fighting that constraint.
    controlled_joints = present_joints

    home_vals = {   "joint1":  0.0,
                    "joint2":  0.0,
                    "joint3":  0.0,
                    "joint4": -1.57079,
                    "joint5":  0.0,
                    "joint6":  1.57079,
                    "joint7": -0.7853,
                    "finger_joint1": 0.02,
                    "finger_joint2": 0.02}
    pos_lims = {    "joint1": (-2.8973,  2.8973),
                    "joint2": (-1.7628,  1.7628),
                    "joint3": (-2.8973,  2.8973),
                    "joint4": (-3.0718, -0.0698),
                    "joint5": (-2.8973,  2.8973),
                    "joint6": (-0.0175,  3.7525),
                    "joint7": (-2.8973,  2.8973),
                    "finger_joint1": (0.0, 0.05),
                    "finger_joint2": (0.0, 0.05)}
    eff_lims = {    "joint1": 87.0, "joint2": 87.0, "joint3": 87.0, "joint4": 87.0,
                    "joint5": 12.0, "joint6": 12.0, "joint7": 12.0,
                    "finger_joint1": 100.0, "finger_joint2": 100.0}
    vel_lim = 2.5

    homing = {(rname, j): home_vals[j] for j in present_joints}

    # Load the Panda MJCF from mujoco_menagerie (same approach as get_spot_args in loco_builder).
    os.environ["MUJOCO_GL"] = "egl"  # must be set before importing mujoco
    from mujoco_playground._src.mjx_env import ensure_menagerie_exists
    ensure_menagerie_exists()
    franka_dir = adarl.utils.utils.pkgutil_get_path("mujoco_playground",
                                                    "external_deps/mujoco_menagerie/franka_emika_panda")
    raw_model_string = Path(franka_dir, "mjx_panda.xml").read_text()
    assets_folder = str(Path(franka_dir, "assets"))
    franka_string = set_asset_texture_paths(raw_model_string, assets_folder, assets_folder)
    # The Panda's base (link0) is welded to the world at the origin, and homing_body_pose only
    # repositions floating bases - so to mount the arm higher (to reach the table-height object) we
    # offset link0 directly in the model. Raise/lower base_height to move the whole arm.
    # The cube spawns on the table (table_height=0.8) at z~0.83 and x~0.5-0.6, so mounting the base
    # at table height puts the object comfortably inside the Panda's ~0.85m reach.
    base_height = 0.8
    franka_string = franka_string.replace('<body name="link0" childclass="panda">',
                                          f'<body name="link0" childclass="panda" pos="0 0 {base_height}">')

    return {"robot_description_string" : franka_string,
            "robot_description_format" : "mjcf",
            "model_kwargs" : {},
            "xacro_extra_pkg_paths" : {},
            "homing_joint_position" : homing,
            "homing_joint_position_references" : homing,
            "robot_name" : rname,
            "robot_main_body_link" : "link0",
            "robot_root_link" : "link0",
            "homing_body_pose_xyz_xyzw" : (0., 0., 0.0, 0., 0., 0., 1.),
            "default_max_joint_impedance_ctrl_torque" : 87.0,
            "max_joint_impedance_ctrl_torques" : {(rname, j): eff_lims[j] for j in present_joints},
            "disallowed_contact_links" : [ ],
            "terminating_contact_pairs" : [ ],
            "controlled_joints" : controlled_joints,
            "randomized_dof_armature_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_mass_links" : [LINK_FILTERS.ALL_ROBOT],
            "randomized_friction_links" : [LINK_FILTERS.ALL],
            "randomized_com_links" : [(rname, "link0")],
            "randomized_dof_frictionloss_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "randomized_dof_damping_joints" : [JOINT_FILTERS.ALL_REVOLUTE],
            "safety_limits_ratios_minmax_pve" : {(rname, j): [[0.95, 0.95, 0.95],
                                                              [0.95, 0.95, 0.95]] for j in present_joints},
            "control_limits_center" : None,
            "control_limits_ratios_minmax_pve" : None,
            "control_limits_minmax_pve" : {(rname, j): th.as_tensor(
                                             [[ pos_lims[j][0], -vel_lim, -eff_lims[j]],
                                              [ pos_lims[j][1],  vel_lim,  eff_lims[j]]]) for j in present_joints},
            "enable_link_collisions" : [ ],
            "manipulator_links" : [(rname, "left_finger"), (rname, "right_finger")],
            "gripper_links" : [(rname, "left_finger"), (rname, "right_finger")],
            "feet_links" : [ ],
            "safe_stiffness" : 600.0,
            "safe_damping" : 20.0,
            "mjx_opt_preset" : "faster",
            # Keep the menagerie model's tuned dof properties (None disables the override).
            "revolute_dof_frictionloss_override" : None,
            "revolute_dof_damping_override" : None,
            "revolute_dof_armature_override" : None,
        }


def runner_builder_to_vecenv(runner_builder: VecEnvRunnerBuilderProtocol) -> VecEnvBuilderProtocol:
    def builder(seed,
                run_folder,
                num_envs : int,
                env_builder_args : dict,
                env_name : str = ""):
        mode = env_builder_args["mode"].strip().lower()
        quiet = env_builder_args["quiet"]
        stepLength_sec = env_builder_args["stepLength_sec"]

        if mode == "pybullet":
            device = env_builder_args["th_device"]
            def single_env_builder(seed : int,
                            log_folder : str,
                            is_eval : bool, 
                            env_builder_args : dict):
                return env_builder(seed=seed, log_folder=log_folder,is_eval=is_eval,env_builder_args=env_builder_args,runner_builder=runner_builder)
            env = build_vec_env(env_builder=single_env_builder,
                                env_builder_args=env_builder_args,
                                log_folder=run_folder,
                                seed=seed,
                                num_envs=num_envs,
                                collector_device=device,
                                env_action_device = device)
        else:
            vrunner = runner_builder( seed = seed,
                                        run_folder = run_folder,
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
        return env
    return builder


def runner_builder_to_singleenv(runner_builder: VecEnvRunnerBuilderProtocol) -> EnvBuilderProtocol:
    def builder(seed : int,
                log_folder : str,
                is_eval : bool, 
                env_builder_args : dict):
        quiet = env_builder_args["quiet"]
        stepLength_sec = env_builder_args["stepLength_sec"]
        vrunner = runner_builder( seed = seed,
                                    run_folder = log_folder,
                                    env_builder_args = env_builder_args,
                                    num_envs = 1,
                                    quiet=quiet,
                                    autoreset = False)
        return Runner2GymWrapper(runner=vrunner, quiet=quiet), 1/stepLength_sec
    return builder

centgrasp_vecenv_builder = runner_builder_to_vecenv(runner_builder)
centgrasp_singleenv_builder = runner_builder_to_singleenv(runner_builder)