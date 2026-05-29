#!/usr/bin/env python3  

from rreal.algorithms.sac_helpers import sac_train, SAC_init_hparams, gym_builder



from adarl_envs.env.leg_reach import LegReachEnv
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import adarl.utils.dbg.ggLog as ggLog
import torch as th
from gymnasium.wrappers.normalize import NormalizeObservation
import threading, os
import traceback
import time
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter

def env_builder(seed, log_folder, env_builder_args, no_dict = False):
    ggLog.info(f"Building env: thread={threading.current_thread()}, pid={os.getpid()}")
    stepLength_sec = env_builder_args["stepLength_sec"]
    video_save_freq = env_builder_args["video_save_freq"]
    th_device = env_builder_args["th_device"]
    max_steps = 5/stepLength_sec

    mode = env_builder_args["mode"].strip().lower()
    if mode == "xbot":
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
                                                       simulation_step=1/1024)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)
    obs_only_vec = True

    print(f"env_builder_args = {env_builder_args}")
    time.sleep(1)

    lrenv = LegReachEnv( maxStepsPerEpisode=max_steps,
                        stepLength_sec=stepLength_sec,
                        environmentController=env_controller,
                        seed=seed,
                        th_device=th_device,
                        reward_joint_torque_limit_weight = env_builder_args["reward_joint_torque_limit_weight"],
                        reward_joint_position_limit_weight = env_builder_args["reward_joint_position_limit_weight"],
                        reward_joint_velocity_weight = env_builder_args["reward_joint_velocity_weight"],
                        reward_tracking_weight = env_builder_args["reward_tracking_weight"],
                        reward_joint_torque_weight = env_builder_args["reward_joint_torque_weight"],
                        reward_scale=500/max_steps,
                        step_precision_tolerance=0 if isinstance(env_controller, BaseSimulationAdapter) else 0.001,
                        stop_on_safety=env_builder_args["stop_on_safety"]) # scale it to be the same as if we have 500 steps (mostly so that we can compare easily)
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
    return env

def wrap_with_recorder(env, stepLength_sec, log_folder, video_save_freq):
    return RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq,
                                vec_obs_key="vec",
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
                                        f"rPosLim {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.reward_joint_position_limit_weight]:.2f}\n"+
                                        f"rTorLim {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.reward_joint_torque_limit_weight]:.2f}\n"+
                                        f"rTorque {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.reward_joint_torque_weight]:.2f}\n"+
                                        f"rTrack  {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT]:.2f}\n"+
                                        f"rVeloci {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.reward_joint_velocity_weight]:.2f}\n"
                                        f"goal_z   {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.HIP_GOAL_Z]:.2f}\n"
                                        f"hip_z    {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.HIP_POS_Z]:.2f}\n"
                                        f"torque   {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_EFFORT]:.2f}, {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]:.2f}\n"
                                        f"position {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_POS]:.2f}, {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_POS]:.2f}\n"
                                        f"velocity {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_VEL]:.2f}, {info['cbstate'][LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_VEL]:.2f}\n"
                                        f"act     "+str([f"{ae:.3f}" for ae in a] if a is not None else None)
                                        )




def runFunction(seed, folderName, resumeModelFile, run_id, args):
    sac_train(seed, folderName, run_id, args,
              env_builder=gym_builder,
              env_builder_args = {  "env_name" : "HalfCheetah-v4",
                                    "forward_reward_weight" : 1.0,
                                    "ctrl_cost_weight" : 0.1,
                                    "reset_noise_scale" : 0.1,
                                    "exclude_current_positions_from_observation" : True,
                                    "max_episode_steps" : 1000},
              hyperparams = SAC_init_hparams(train_freq_vstep=25,
                                  grad_steps=50,
                                  q_lr=0.005,
                                  policy_lr=0.0005,
                                  model_th_device = "cuda",
                                  gamma = 0.99,
                                  target_tau=0.005,
                                  buffer_size=1_000_000,
                                  total_steps = 10_000_000,
                                  batch_size=16384,
                                  q_network_arch=[64,64],
                                  policy_arch=[64,64],
                                  learning_starts=5000,
                                  parallel_envs = 16,
                                  log_freq_vstep = 1000,
                                  eval_freq_ep=10))

if __name__ == "__main__":

    import argparse
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","rreal"])