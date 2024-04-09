#!/usr/bin/env python3

import lr_gym.utils.session
import inspect
import os
import datetime
import torch as th
import lr_gym.utils.utils
from jumping_leg.experiments.build_jumping_leg_env import build_env

def main():
    logFolder, session = lr_gym.utils.session.lr_gym_startup(__file__,
                                                    inspect.currentframe(),
                                                    folderName = os.path.basename(__file__)+f"/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                                    seed = 0,
                                                    experiment_name = None,
                                                    run_id = None)
    
    env = build_env(logFolder, video_save_freq=1)
    def zero(obs):
        return th.tensor([0.0,0.0]), None


    res = lr_gym.utils.utils.evaluatePolicy(env = env, model = None, episodes = 5, predict_func=zero,
                                            images_return = None, obs_return=None)
    print(f"evaluation returned {res}")

if __name__ == "__main__":
    main()