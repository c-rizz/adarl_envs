#!/usr/bin/env python3

import time
import inspect
from jumping_leg.utils.modded_sac import SAC as SB3_SAC
from jumping_leg.algorithms.sac import SAC
from stable_baselines3.ppo import PPO
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils

import os
from lr_gym.utils.buffers import ThDReplayBuffer
import torch as th
import lr_gym.utils.session
from jumping_leg.experiments.build_jumping_leg_env import env_builder
from wandb.integration.sb3 import WandbCallback
# from stable_baselines3.common.vec_env import SubprocVecEnv
from lr_gym.utils.subproc_vec_env import SubprocVecEnv
from lr_gym.envs.VecEnvLogger import VecEnvLogger
from lr_gym.utils.sb3_callbacks import EvalCallback_ep, SigintHaltCallback, PrintLrRunInfo, CheckpointCallbackRB
import stable_baselines3.common.base_class
import stable_baselines3.common.on_policy_algorithm
import lr_gym.utils.dbg.dbg_img as dbg_img 
from lr_gym.utils.keyboard_listener import KeyboardListener
from lr_gym.utils.tensor_trees import map_tensor_tree, TensorTree

def load_model(model_path):
    try:
        return SB3_SAC.load(model_path)
    except:
        return SAC.load(model_path)



def runFunction(seed, folderName, resumeModelFile, run_id, args):

    env_builder_args = {
        "reward_contacts_weight" : 0.0,
        "reward_energy_weight" : 0.0,
        "reward_position_limit_weight" : 10.0,
        "reward_torque_limit_weight" : 1.0,
        "reward_torque_weight" : 0.1,
        "reward_tracking_weight" : 1.0,
        "reward_velocity_weight" : 0.0,
        "th_device" : th.device("cpu"),
        "control_mode" : "position_and_gains",
        "video_save_freq" : 1,
        "stepLength_sec" : 20/1024, # about 50Hz
        "platform_randomization" : "single",
        "quiet" : False,
        "mode" : args["mode"],
        "use_contacts" : args["mode"] == "pybullet"}

    return play(seed, folderName, run_id, args, env_builder_args)

def play(seed, folderName, run_id, args, env_builder_args):
    
    ggLog.info(f"Starting run")
    log_folder, session = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        folderName=folderName,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        use_wandb=False)

    # th.cuda.set_sync_debug_mode("warn")
    device = lr_gym.utils.utils.torch_selectBestGpu()
    ggLog.info("Building env...")

    env = env_builder(log_folder=log_folder+"/eval_env", seed = seed+100000000, 
                         env_builder_args = env_builder_args)

    ggLog.info("Built")
#    env.action_space.seed(seed)

    ep_duration = 5/env_builder_args["stepLength_sec"]

    model = load_model(args["pretrained"])

    play = True
    verbose = False
    interactive = False

    while play:
        cmd = None
        options = {}
        while cmd != "c":
            cmd = input("Enter 'c' to continue. Type 'quit' to quit:\n > ")
            if cmd == "quit":
                play = False
                break
            elif cmd.startswith("s1x "):
                options["support1_pos_x"] = float(cmd.split(" ")[1])
            elif cmd.startswith("s1z "):
                options["support1_pos_z"] = float(cmd.split(" ")[1])
            elif cmd.startswith("s2x "):
                options["support2_pos_x"] = float(cmd.split(" ")[1])
            elif cmd.startswith("s2z "):
                options["support2_pos_z"] = float(cmd.split(" ")[1])
            elif cmd.startswith("contact "):
                options["reward_contacts_weights"] = float(cmd.split(" ")[1])
            elif cmd.startswith("hz "):
                options["hip_goal_z"] = float(cmd.split(" ")[1])
            elif cmd == "interactive":
                print(f" Use WASD to move the platform, LP to move the goal, T to terminate.")
                interactive = True
                options["max_ep_steps"] = 100000
                keyboard_listener = KeyboardListener()
                cmd = 'c'   
        if not play:
            break
        obs : TensorTree[th.Tensor]
        obs, info = env.reset(options = options)  #type: ignore
        # ggLog.info(f"ep_config = {info['ep_config']}")
        done = False
        ep_reward = 0
        step_count = 0
        while not done:
            t0 = time.monotonic()
            ggLog.info(f"step = {step_count}")
            # ggLog.info(f"ep_config = {info['ep_config']}")
            obs_batch = map_tensor_tree(obs,lambda t: th.unsqueeze(t,0).to(device))
            action, hidden_state = model.predict(obs_batch, deterministic = True)
            obs, reward, terminated, truncated, info = env.step(action.detach().squeeze().cpu().numpy()) #type: ignore
            img = env.render()
            dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
            if verbose:
                print(f"obs = {obs}\n"+
                    f"rew = {reward}\n"+
                    f"terminated = {terminated}\n"+
                    f"truncated = {truncated}\n")
            if interactive:
                keys = keyboard_listener.get_pressed_keys()
                dx, dz = 0,0
                if 'w' in keys: dz +=  0.005
                if 's' in keys: dz += -0.005
                if 'a' in keys: dx +=  0.005
                if 'd' in keys: dx += -0.005
                gdz = 0
                if 'l' in keys: gdz += -0.005
                if 'p' in keys: gdz += +0.005
                if 't' in keys: truncated = True
                env.getBaseEnv()._current_episode_config.support2_pos_x = info["ep_config"]["support2_pos_x"]+dx
                env.getBaseEnv()._current_episode_config.support2_pos_z = info["ep_config"]["support2_pos_z"]+dz
                env.getBaseEnv()._current_episode_config.hip_goal_z = info["ep_config"]["hip_goal_z"]+gdz
                env.getBaseEnv()._place_objects()
            step_count += 1

            done = terminated or truncated
            ep_reward += reward
            time.sleep(max(0,env_builder_args["stepLength_sec"] - (time.monotonic()-t0)))
        ggLog.info(f"Episode reward = {ep_reward}")


if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from lr_gym.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--pretrained", required = True, type=str, help="Model to load")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use")

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
                start_lr_gym=False,
                pkgs_to_save=["lr_gym","jumping_leg"])
