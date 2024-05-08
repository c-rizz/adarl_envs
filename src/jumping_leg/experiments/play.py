#!/usr/bin/env python3

import lr_gym.utils.session
import inspect
import os
import datetime
import torch as th
import lr_gym.utils.utils
import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
import random
import numpy as np
import torch as th

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
        "video_save_freq" : 0,
        "stepLength_sec" : 1/128,
        "platform_randomization" : "single",
        "quiet" : False,
        "mode" : args["mode"],
        "use_contacts" : args["mode"].lower().strip() != "xbot"}

    hyperparams = {"train_freq" : 50,
                   "grad_steps" : 25,
                   "q_lr" : 0.005,
                   "policy_lr" : 0.0005}
    main(seed, folderName, run_id, args, env_builder_args, hyperparams)

# import traceback
# import threading
# t = threading.Thread.__init__
# def threadwrapper(self : threading.Thread, *args, **kwargs):
#     t(self, *args, **kwargs)
#     print(f" created thread {self.name}")
#     traceback.print_stack()

direction = 1
def oscillate_policy(obs):
    global direction
    hz = 100 # expected call freq
    hip_pos = obs["vec"][0]
    # knee_pos = obs["vec"][3]
    # print(f"hip_pos = {hip_pos:.3f} kpos = {knee_pos:.3f}")

    speed = 10
    href = hip_pos + 1/hz*speed*direction
    kref = href*2
    if href > 0.5:
        direction = -1
    if href < -0.5:
        direction = 1
    # print(f"d = {direction} href = {href:.3f} kref {kref:.3f}")


    return th.tensor([href,kref,1,1]), None

def main(seed, folderName, run_id, args, env_builder_args, hyperparams):

    # print(f"active threads = {threading.enumerate()}")
    # threading.Thread.__init__ = threadwrapper
    # torchexplorer.setup()
    log_folder, session = lr_gym.utils.session.lr_gym_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        use_wandb=False)

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    hyperparams["device"] = device
    # env setup
    with build_jumping_leg_env.env_builder(log_folder=log_folder,
                                        seed=seed,
                                        env_builder_args = env_builder_args) as env:
        action_size = env.action_space.shape[0]

        def zero(obs):
            return th.zeros(size=(action_size,)), None
        
        res = lr_gym.utils.utils.evaluatePolicy(env = env, model = None, episodes = 5, predict_func=oscillate_policy,
                                                images_return = None, obs_return=None)
        print(f"evaluation returned {res}")



if __name__ == "__main__":

    import os
    import argparse
    from lr_gym.utils.session import launchRun

    ap = argparse.ArgumentParser()

    ap.set_defaults(feature=True)
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use")
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=1,
                seedsOffset=0,
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_lr_gym=False,
                pkgs_to_save=["lr_gym","jumping_leg"])