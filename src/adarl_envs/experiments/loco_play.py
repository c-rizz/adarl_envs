#!/usr/bin/env python3

import time
import inspect
import adarl.utils.dbg.dbg_img
from adarl_envs.utils.modded_sac import SAC as SB3_SAC
from rreal.algorithms.sac import SAC
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import numpy as np

import os
import torch as th
import adarl.utils.session
import adarl.utils.dbg.dbg_img as dbg_img 
from adarl.utils.keyboard_listener import KeyboardListener
from adarl.utils.tensor_trees import map_tensor_tree, TensorTree
import adarl.utils.sigint_handler
from adarl_envs.experiments.build_quad import quad_env_builder

import adarl.utils.dbg
from ctypes.util import find_library
import readline

def load_model(model_path):
    try:
        return SB3_SAC.load(model_path)
    except:
        return SAC.load(model_path)



def runFunction(seed, folderName, resumeModelFile, run_id, args):

    step_length_sec = 50/1024 
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    env_device = th.device("cpu")
    env_builder_args = {
        "action_delay_mustd" : (0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "action_smoothing_halflife_sec" : 0.1,
        "control_mode" : "position",
        "enable_rendering" : not args["gui"],
        "goal_err_smoothing_halflife_sec" : 0.2,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : args["mode"],
        "quiet" : True,
        "initial_pose_randomization" : 0.25,
        "reward_acceleration_weight" : 0.1,
        "reward_actdiff_weight" : 0.1,
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_health_weight" : 0.0,
        "reward_position_limit_weight" : 0.5,
        "reward_torque_limit_weight" : 0.0,
        "reward_torque_weight" : 0.1,
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
        "video_save_freq" : 1 if args["record"] else -1,
        "goal_speed_minmax" : (0,2),
        "use_contacts" : False,
        "frame_stack_length" : 1,
        "verbose_infos" : True,
        "terminate_on_body_contact" : False,
        "use_wandb" : False,
        "init_on_reset_ratio" : 1.0,
        "obs_noise_joints_pve_ep_mustd_step_std" :  (0.0, 0.0, 0.0),
        "obs_noise_linvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_angvel_ep_mustd_step_std" :      (0.0, 0.0, 0.0),
        "obs_noise_posz_ep_mustd_step_std" :        (0.0, 0.0, 0.0),
        "obs_noise_gravity_ep_mustd_step_std" :     (0.0, 0.0, 0.0),
        "show_gui" : args["gui"],
        "ui_camera_resolution_hw" : (720,int(720*16/9))
    }

    return play(seed,
                folderName,
                run_id, args,
                env_builder = quad_env_builder,
                env_builder_args = env_builder_args,
                step_length_sec = step_length_sec,
                render=not args["gui"])

def play(seed, folderName, run_id, args, env_builder, env_builder_args, step_length_sec : float, render : bool):
    
    ggLog.info(f"Starting run")
    if render:
        adarl.utils.dbg.dbg_img.helper.enable_web_dbg(True)
    log_folder, session = adarl.utils.session.adarl_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        folderName=folderName,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        use_wandb=False)

    # th.cuda.set_sync_debug_mode("warn")
    device = adarl.utils.utils.torch_selectBestGpu()
    ggLog.info("Building env...")

    env, fps = env_builder(log_folder=log_folder+"/eval",
                                    seed=seed+100000000,
                                    env_builder_args = env_builder_args,
                                    is_eval=True)
    ggLog.info("Built")
    model = load_model(args["pretrained"])

    play = True
    verbose = False
    interactive = False
    rewards = []
    durations = []
    avg10_dists = []

    keyboard_listener : KeyboardListener = None

    try:
        while play:
            cmd = None
            options = {}
            if args["evaluate"] is None:
                while cmd != "c":
                    cmd = input("Enter 'c' to continue. Type 'quit' to quit:\n > ")
                    if cmd == "quit":
                        play = False
                        break
                    elif cmd == "interactive":
                        print(f" Use WASD to move the goal, IJKL to move the camera, T to terminate.")
                        time.sleep(1)
                        interactive = True
                        options["max_ep_steps"] = 100_000
                        options["goal_velocity_xy"] = [0.0, 0.0]
                        keyboard_listener = KeyboardListener()
                        cmd = 'c'   
                if not play:
                    break
            else:
                if session.run_info["collected_episodes"].value >= args["evaluate"]:
                    break
            obs : TensorTree[th.Tensor]
            obs, info = env.reset(options = options)  #type: ignore
            # ggLog.info(f"ep_config = {info['ep_config']}")
            done = False
            ep_reward = 0
            step_count = 0
            step_wallduration = float("nan")
            full_step_wallduration = float("nan")
            ep_wall_duration = 0
            while not done:
                t0 = time.monotonic()
                session.run_info["collected_steps"].value += 1
                goal_velocity_xy = env.getBaseEnv().get_goal()
                ggLog.info(f"step = {step_count} rtfactor = {step_length_sec/full_step_wallduration:.2f} max_rtfactor = {step_length_sec/step_wallduration:.2f} \t goal_velocity_xy={goal_velocity_xy}")
                # ggLog.info(f"ep_config = {info['ep_config']}")
                obs_batch = map_tensor_tree(obs,lambda t: th.unsqueeze(t,0).to(device))
                action, hidden_state = model.predict(obs_batch, deterministic = True)
                obs, reward, terminated, truncated, info = env.step(action.detach().squeeze().cpu().numpy()) #type: ignore
                if render:
                    img = env.render()
                    dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
                if verbose:
                    print(f"obs = {obs}\n"+
                        f"rew = {reward}\n"+
                        f"terminated = {terminated}\n"+
                        f"truncated = {truncated}\n")
                if interactive:
                    speed_yaw_diff = [0.0,0.0]
                    if keyboard_listener.get_key_press_count("w")>0: speed_yaw_diff[0] =  0.05
                    if keyboard_listener.get_key_press_count("s")>0: speed_yaw_diff[0] = -0.05
                    if keyboard_listener.get_key_press_count("a")>0: speed_yaw_diff[1] =  10*3.14159/180
                    if keyboard_listener.get_key_press_count("d")>0: speed_yaw_diff[1] = -10*3.14159/180
                    flip = keyboard_listener.get_key_press_count("v")>0
                    cam_dist_pitch_yaw_diff = [0.0,0.0,0.0]
                    if keyboard_listener.get_key_press_count("u")>0: cam_dist_pitch_yaw_diff[0] = -0.1
                    if keyboard_listener.get_key_press_count("o")>0: cam_dist_pitch_yaw_diff[0] =  0.1
                    if keyboard_listener.get_key_press_count("i")>0: cam_dist_pitch_yaw_diff[1] =  5*3.14159/180
                    if keyboard_listener.get_key_press_count("k")>0: cam_dist_pitch_yaw_diff[1] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("j")>0: cam_dist_pitch_yaw_diff[2] = -5*3.14159/180
                    if keyboard_listener.get_key_press_count("l")>0: cam_dist_pitch_yaw_diff[2] =  5*3.14159/180

                    if keyboard_listener.get_key_press_count("t")>0: truncated = True
                    env.getBaseEnv().set_cam_pose(env.getBaseEnv().get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                    if flip:
                        env.getBaseEnv().set_goal(-env.getBaseEnv().get_goal()[:2])
                    else:
                        env.getBaseEnv().set_goal(goal_velocity_diff_speed_yaw = speed_yaw_diff)
                    keyboard_listener.reset_key_press_counters()
                step_count += 1

                done = terminated or truncated
                # def f(): 
                #     nonlocal done
                #     done = True
                # adarl.utils.sigint_handler.run_on_sigint_received(f)
                ep_reward += reward
                step_wallduration = time.monotonic()-t0
                ep_wall_duration += step_wallduration
                time.sleep(max(0,step_length_sec - step_wallduration))
                full_step_wallduration = time.monotonic()-t0
            if step_count>0:
                rewards.append(ep_reward)
                durations.append(step_count)
                avg10_dists.append(info["avg_vel_track_err"])
            with session.run_info["collected_episodes"].get_lock():
                session.run_info["collected_episodes"].value += 1
            ggLog.info("\n"
                    f"Episode reward =  {ep_reward}\n"
                    f"Wall duration =   {ep_wall_duration:.2f}s\n"
                    f"Sim  duration =   {step_count*step_length_sec:.2f}s\n"
                    f"Realtime factor = {step_count*step_length_sec/ep_wall_duration:.2f}\n")
            if interactive:
                keyboard_listener.close()
    finally:
        if keyboard_listener is not None:
            keyboard_listener.close()
    rewards = np.array(rewards)
    durations = np.array(durations)
    avg10_dists = np.array(avg10_dists)
    ggLog.info(f"Avg reward = {rewards.mean()}, {rewards.std()} std")
    ggLog.info(f"Avg durations = {durations.mean()}, {durations.std()} std")
    ggLog.info(f"Avg avg10_dists = {avg10_dists.mean()}, {avg10_dists.std()} std")
    env.reset() # trigger video saving
    env.close()

if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--pretrained", required = True, type=str, help="Model to load")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use [pybullet,xbot-gazebo]")
    ap.add_argument("--evaluate", default=None, type=int, help="Evaluate the policy with this number of episodes")
    ap.add_argument("--gui", default=False, action='store_true', help="Do not start the gui, instead stream renderings")
    ap.add_argument("--record", default=False, action='store_true', help="Record episode videos")
    
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=1,
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","adarl_envs"])
