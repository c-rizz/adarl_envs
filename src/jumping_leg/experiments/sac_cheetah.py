#!/usr/bin/env python3  
from __future__ import annotations

def env_builder(seed,
                    log_folder,
                    is_eval,
                    env_builder_args : dict,
                    no_dict = False):
    import adarl.utils.utils
    from jumping_leg.experiments.build_locomotion_env import locomotion_env_builder
    from jumping_leg.env.LocomotionEnv import LocomotionEnv

    model_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/cheetah.urdf.xacro")
    homing_joint_pose={ ("cheetah","torso_z_slider"):0.8,
                        ("cheetah","torso_x_slider"):0.0,
                        ("cheetah","torso_pitch_joint"):0.0,
                        ("cheetah","hip_joint_y_front"):0.785,
                        ("cheetah","knee_joint_front"):1.57,
                        ("cheetah","hip_joint_y_back"):0.785,
                        ("cheetah","knee_joint_back"):1.57}
    disallowed_contact_links = [("quad","thigh_link_back_left"),
                                                        ("quad","shin_link_back_left"),
                                                        ("quad","thigh_link_back_right"),
                                                        ("quad","shin_link_back_right"),
                                                        ("quad","thigh_link_front_left"),
                                                        ("quad","shin_link_front_left"),
                                                        ("quad","thigh_link_front_right"),
                                                        ("quad","shin_link_front_right"),
                                                        ("quad","body_link")]
    terminating_contact_pairs=[(("cheetah","body_link"),("ground_plane","planeLink"))]
    robot_name="cheetah"
    robot_main_body_link="body_link"
    homing_body_pose_xyz_xyzw=None
    controlled_joints=[LocomotionEnv.JOINT_FILTERS.ALL_REVOLUTE]
    return locomotion_env_builder(seed = seed,
                                    log_folder = log_folder,
                                    is_eval = is_eval,
                                    env_builder_args = env_builder_args,
                                    model_file = model_file,
                                    no_dict = no_dict,
                                    homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                                    homing_joint_pose=homing_joint_pose,
                                    disallowed_contact_links=disallowed_contact_links,
                                    terminating_contact_pairs=terminating_contact_pairs,
                                    robot_name=robot_name,
                                    robot_main_body_link=robot_main_body_link,
                                    controlled_joints=controlled_joints)

def runFunction(seed, folderName, resumeModelFile, run_id, args):

    import copy
    import torch as th
    from rreal.examples.solve_sac import sac_train, SAC_hyperparams
    
    step_length_sec = 50/1024  # use multiples of 1/1024 to keep it representable in binary (so we can step precisely)
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    train_envs = 1
    env_device = th.device("cpu")
    env_builder_args = {
        "action_delay_mustd" : (0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "action_smoothing_halflife_sec" : 0.1,
        "control_mode" : "position",
        "enable_rendering" : False,
        "goal_err_smoothing_halflife_sec" : 0.0,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : "pybullet",
        "quiet" : True,
        "reward_acceleration_weight" : 0.0,
        "reward_actdiff_weight" : 0.0,
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_health_weight" : 0.0,
        "reward_position_limit_weight" : 0.0,
        "reward_torque_limit_weight" : 0.0,
        "reward_torque_weight" : 0.0,
        "reward_torquediff_weight" : 0.0,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_limit_weight" : 0.0,
        "reward_velocity_weight" : 0.0,
        "reward_height_weight" : 0.0,
        "reward_pitchnroll_weight" : 0.01,
        "safe_stiffness" : 400,
        "safe_damping" : 10,
        "stepLength_sec" : step_length_sec,
        "obs_noise_step_std" : 0.01,
        "obs_noise_ep_mustd" : (0.0, 0.0),
        "stop_on_safety" : False,
        "th_device" : env_device,
        "video_save_freq" : 0,
        "goal_speed_minmax" : (0,2),
        "use_contacts" : False,
        "frame_stack_length" : 1,
        "verbose_infos" : False,
        "terminate_on_body_contact" : False,
        "use_wandb" : False}
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
    run_1ms_env_builder_args = copy.deepcopy(env_builder_args)
    run_1ms_env_builder_args["goal_speed_minmax"] = (1,1)
    run_1ms_env_builder_args["enable_rendering"] = True
    run_1ms_env_builder_args["verbose_infos"] = True
    run_1ms_env_builder_args["video_save_freq"] = 1
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
    
    sac_train(  seed,
                folderName,
                run_id,
                args,
                env_builder = env_builder,
                env_builder_args = env_builder_args,
                eval_env_builder_args = [
                                        eval_conf_video_det,
                                        eval_conf_video_stoch,
                                        eval_conf_run_1ms,
                                        #  eval_conf_feasible,
                                        #  eval_conf_video_feasible,
                                        #  eval_conf_video_jump_feasible
                                         ],
                hyperparams = SAC_hyperparams(  device = "cuda",
                                                q_network_arch=[256,128],
                                                q_lr=0.001,
                                                policy_lr=0.0001,
                                                policy_network_arch=[256,256],
                                                gamma=0.99,
                                                target_tau = 0.005,
                                                batch_size=8192,
                                                buffer_size=10_000_000,
                                                total_steps=100_000_000,
                                                train_freq_vstep=5,
                                                grad_steps=10,
                                                learning_starts=max_steps_per_episode*max(train_envs*5, 100),
                                                parallel_envs=train_envs,
                                                log_freq_vstep=max_steps_per_episode),
                checkpoint_freq=100,
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