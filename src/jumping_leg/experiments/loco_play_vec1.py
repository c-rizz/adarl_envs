#!/usr/bin/env python3

import time
import inspect
from adarl.utils.buffers import BaseBuffer
import adarl.utils.dbg.dbg_img
from jumping_leg.utils.modded_sac import SAC as SB3_SAC
from rreal.algorithms.sac import SAC
from rreal.algorithms.sac_helpers import EnvBuilderProtocol
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import numpy as np
import gymnasium as gym

import os
import torch as th
import adarl.utils.session
import adarl.utils.dbg.dbg_img as dbg_img 
from adarl.utils.keyboard_listener import KeyboardListener
from adarl.utils.tensor_trees import map_tensor_tree, TensorTree
import adarl.utils.sigint_handler
from jumping_leg.experiments.vec_loco_env_test import quad_loco_env_builder, kyon_loco_env_builder
from jumping_leg.env.LocomotionVecEnv import LocomotionVecEnv
from rreal.algorithms.rl_agent import RLAgent, TransitionBatch

import adarl.utils.dbg
from typing import Any
from ctypes.util import find_library
import readline

def load_model(model_path):
    return SAC.load(model_path)


class Fixedpolicy(RLAgent):
    def __init__(self, cmd : th.Tensor):
        self._cmd = cmd.detach().clone()

    def predict_action(self, observation_batch, deterministic = False):
        return self._cmd.clone()
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        pass

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass
    
    def load_(self, path: str):
        pass
    
    def input_device(self):
        return self._a_offset.device

class SinPolicy(RLAgent):
    def __init__(self,  act_scale : th.Tensor,
                        act_offset : th.Tensor,
                        act_speed : th.Tensor,
                        action_size : int,
                        dt : float):
        self._t0 = 0.0
        self._t = 0.0
        self._dt = dt
        self._a_offset = act_offset
        self._a_speed = act_speed
        # self._t_off = th.asin(self._a_offset/act_range)
        self._a_scale = act_scale.expand((action_size,))

    def predict_action(self, observation_batch, deterministic = False):
        theta = (self._t0-self._t)*self._a_speed
        a = th.sin(theta)*self._a_scale+self._a_offset
        print(f" theta = {theta} \n"
            #   f" _t_off = {self._t_off} \n"
              f" _t = {self._t} \n"
              f" _a_scale = {self._a_scale} \n"
              f" _a_offset = {self._a_offset} \n"
              f" a = {a}")
        self._t = self._t+self._dt
        return a
    
    def get_hidden_state(self):
        return self._t
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        self._t = self._t0

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass
    
    def load_(self, path: str):
        pass
    
    def input_device(self):
        return self._a_offset.device
        
kyon_homing = { ("kyon","hip_roll_3") : [-3.14159*0.05, 0, 0, 400, 10],
                ("kyon","hip_roll_4") : [ 3.14159*0.05, 0, 0, 400, 10],
                ("kyon","hip_roll_1") : [ 3.14159*0.05, 0, 0, 400, 10],
                ("kyon","hip_roll_2") : [-3.14159*0.05, 0, 0, 400, 10],
                ("kyon","hip_pitch_3") : [0.75, 0, 0, 400, 10],
                ("kyon","hip_pitch_4") : [-0.75, 0, 0, 400, 10],
                ("kyon","hip_pitch_1") : [0.75, 0, 0, 400, 10],
                ("kyon","hip_pitch_2") : [-0.75, 0, 0, 400, 10],
                ("kyon","knee_pitch_3") : [-1.8, 0, 0, 400, 10],
                ("kyon","knee_pitch_4") : [ 1.8, 0, 0, 400, 10],
                ("kyon","knee_pitch_1") : [-1.8, 0, 0, 400, 10],
                ("kyon","knee_pitch_2") : [ 1.8, 0, 0, 400, 10]}
quad_homing = { ("quad","hip_joint_x_back_left") : [-3.14159*0.4, 0, 0, 400, 10],
                ("quad","hip_joint_x_back_right") : [-3.14159*0.4, 0, 0, 400, 10],
                ("quad","hip_joint_x_front_left") : [-3.14159*0.4, 0, 0, 400, 10],
                ("quad","hip_joint_x_front_right") : [-3.14159*0.4, 0, 0, 400, 10],
                ("quad","hip_joint_y_back_left") : [0.75, 0, 0, 400, 10],
                ("quad","hip_joint_y_back_right") : [0.75, 0, 0, 400, 10],
                ("quad","hip_joint_y_front_left") : [0.75, 0, 0, 400, 10],
                ("quad","hip_joint_y_front_right") : [0.75, 0, 0, 400, 10],
                ("quad","knee_joint_back_left") : [1.8, 0, 0, 400, 10],
                ("quad","knee_joint_back_right") : [1.8, 0, 0, 400, 10],
                ("quad","knee_joint_front_left") : [1.8, 0, 0, 400, 10],
                ("quad","knee_joint_front_right") : [1.8, 0, 0, 400, 10]}
def build_sin_policy(env, robot : str, scale : float = 0.0):
    if robot == "quad":
        home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(quad_homing)
        act_range = th.as_tensor([0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2,
                                  0.0, 0.1, 0.2])
    elif robot == "kyon":
        home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(kyon_homing)
        act_range = th.as_tensor([0.0, 0.1, 0.2,
                                -0.0, -0.1, -0.2,
                                0.0, 0.1, 0.2,
                                -0.0, -0.1, -0.2])
    else:
        RuntimeError(f"Unknown robot '{robot}")
    model = SinPolicy(  act_scale=act_range*scale,
                        act_offset=home_action,
                        act_speed=th.as_tensor([0.8]),
                        action_size=12,
                        dt=0.05)
    return model


def build_fixed_policy(env, robot : str, scale : float = 0.0):
    if robot == "quad":
        home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(quad_homing)
    elif robot == "kyon":
        home_action = env.get_runner().get_base_env()._action_helper.pvesd_to_action(kyon_homing)
    else:
        RuntimeError(f"Unknown robot '{robot}")
    model = Fixedpolicy(  cmd = home_action)
    return model


def runFunction(seed, folderName, resumeModelFile, run_id, args):

    step_length_sec = 50/1024 
    max_steps_per_episode=250 #int(ep_duration_sec/step_length_sec)
    env_device = th.device("cpu")
    pixel_resolution = 360 if args["mode"] == "mjx" else 720
    env_builder_args = {
        "action_delay_mustd" : (0.0,0.0),
        "action_noise_mustd" : (0.0,0.0),
        "action_smoothing_halflife_sec" : 0.1,
        "control_mode" : "position",
        "enable_rendering" : not args["gui"] or args["record"],
        "goal_err_smoothing_halflife_sec" : 0.2,
        "max_steps_per_episode" : max_steps_per_episode,
        "mode" : args["mode"],
        "quiet" : False,
        "initial_pose_randomization" : 0.0,
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
        "ui_camera_resolution_hw" : (pixel_resolution,int(pixel_resolution*16/9))
    }

    robot = args["robot"]
    if robot == "quad":
        builder = quad_loco_env_builder
    elif robot == "kyon":
        builder = kyon_loco_env_builder
    else:
        raise RuntimeError(f"Unknown robot '{robot}'")

    return play(seed,
                folderName,
                run_id, args,
                env_builder = builder,
                env_builder_args = env_builder_args,
                step_length_sec = step_length_sec,
                render=not args["gui"],
                robot = robot)



def play(seed, folderName, run_id, args, 
         env_builder : EnvBuilderProtocol, 
         env_builder_args : dict[str,Any], 
         step_length_sec : float, render : bool,
         robot : str):
    
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

    env, fps = env_builder( log_folder=log_folder+"/eval",
                            seed=seed+100000000,
                            env_builder_args = env_builder_args,
                            is_eval=False)
    ggLog.info("Built")
    if args["pretrained"] is not None:
        model = load_model(args["pretrained"])
    else:
        # model = build_fixed_policy(env = env, robot=robot)
        model = build_sin_policy(env, robot=robot, scale = 1.0)

    play = True
    verbose = False
    interactive = False
    rewards = []
    durations = []
    avg10_dists = []

    keyboard_listener : KeyboardListener = None

    if render:
        img = env.render()
        dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)

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
            rt = 1.0
            model.reset_hidden_state()
            if render:
                img = env.render()
                dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
                time.sleep(step_length_sec/rt)
            while not done:
                t0 = time.monotonic()
                session.run_info["collected_steps"].value += 1
                base_env : LocomotionVecEnv = env.get_runner().get_base_env()
                goal_velocity_xy = base_env.get_goals()[0]
                ggLog.info(f"step = {step_count} rtfactor = {step_length_sec/full_step_wallduration:.2f} max_rtfactor = {step_length_sec/step_wallduration:.2f} \t goal_velocity_xy={goal_velocity_xy}")
                # ggLog.info(f"ep_config = {info['ep_config']}")
                obs_batch = map_tensor_tree(obs,lambda t: th.unsqueeze(t,0).to(device))
                action, hidden_state = model.predict(obs_batch, deterministic = True)
                obs, reward, terminated, truncated, info = env.step(action.detach().squeeze()) #type: ignore
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
                    base_env.set_cam_pose(base_env.get_cam_pose() + th.as_tensor(cam_dist_pitch_yaw_diff))
                    if flip:
                        base_env.set_goal(-base_env.get_goals()[0,:2])
                    else:
                        base_env.set_goal(goal_velocity_diff_speed_yaw = tuple(speed_yaw_diff))
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
                time.sleep(max(0,step_length_sec/rt - step_wallduration))
                full_step_wallduration = time.monotonic()-t0
            if step_count>0:
                rewards.append(ep_reward)
                durations.append(step_count)
                # avg10_dists.append(info["avg_vel_track_err"])
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
    # avg10_dists = np.array(avg10_dists)
    ggLog.info(f"Avg reward = {rewards.mean()}, {rewards.std()} std")
    ggLog.info(f"Avg durations = {durations.mean()}, {durations.std()} std")
    # ggLog.info(f"Avg avg10_dists = {avg10_dists.mean()}, {avg10_dists.std()} std")
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
    ap.add_argument("--pretrained", required = False, default=None, type=str, help="Model to load")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use [pybullet,xbot-gazebo,mjx]")
    ap.add_argument("--robot", default="quad", type=str, help="Robot to be used")
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
                pkgs_to_save=["adarl","jumping_leg"])
