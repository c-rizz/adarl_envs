#!/usr/bin/env python3

import adarl.utils.session
import inspect
import os
import datetime
import torch as th
import adarl.utils.utils
import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
import random
import numpy as np
import torch as th
import math
import adarl

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
        "stepLength_sec" : 1/1024,
        "platform_randomization" : "no_platforms",
        "quiet" : False,
        "mode" : args["mode"],
        "use_contacts" : args["mode"].lower().strip() == "pybullet"}

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

policy_time = 0
hdirection = 1
kdirection = 1
def oscillate_policy(obs):
    hz = 1000 # expected call freq
    hip_pos = obs["vec"][0]
    knee_pos = obs["vec"][3]

    rate = 0.5
    hip_range = 0.4
    knee_range = 0.8
    global policy_time
    if policy_time == 0:
        policy_time = (math.asin(hip_pos/hip_range)+2*3.14159)/(rate*2*3.14159)
    t = policy_time
    policy_time+=1/hz

    href = hip_range*math.sin(t*(2*3.14159)*rate)
    kref = knee_range*math.sin(t*(2*3.14159)*rate)

    # print(f"t = {t} href = {href:.3f}={href*2.4:.3f} kref {kref:.3f}={kref*2.4:.3f}")
    return th.tensor([href,kref,1,1]), None

def ep_done_cb(episodeReward, steps, episode):
    global policy_time
    policy_time = 0
    adarl.utils.session.default_session.run_info["collected_episodes"].value += 1
    adarl.utils.session.default_session.run_info["collected_steps"].value += steps

keep_h = 0
keep_k = 0
def keep(obs):
    global policy_time, keep_h, keep_k
    hz = 100 # expected call freq
    hip_pos = obs["vec"][0]
    knee_pos = obs["vec"][3]
    if policy_time == 0:
        keep_h = hip_pos
        keep_k = knee_pos
    policy_time += 1/hz
    return th.tensor([keep_h,keep_k,1,1]), None


def normal(obs):
    return th.randn(size=(4,), dtype=th.float32), None

def main(seed, folderName, run_id, args, env_builder_args, hyperparams):

    # print(f"active threads = {threading.enumerate()}")
    # threading.Thread.__init__ = threadwrapper
    # torchexplorer.setup()
    log_folder, session = adarl.utils.session.adarl_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        use_wandb=False)

    policy_name = args["policy"].lower().strip()
    if policy_name == "keep":
        policy = keep
    elif policy_name == "oscillate":
        policy = oscillate_policy
    elif policy_name == "random":
        policy = normal
    else:
        raise RuntimeError(f"Invalid policy '{policy_name}'")
        
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
        res = adarl.utils.utils.evaluatePolicy(env = env, model = None, episodes = 5, predict_func=policy,
                                                images_return = None, obs_return=None,
                                                on_ep_done_callback=ep_done_cb)
        print(f"evaluation returned {res}")



if __name__ == "__main__":

    import os
    import argparse
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()

    ap.set_defaults(feature=True)
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use")
    ap.add_argument("--policy", default="oscillate", type=str, help="Policy to use (oscillate, keep, random)")
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=1,
                seedsOffset=0,
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","jumping_leg"])