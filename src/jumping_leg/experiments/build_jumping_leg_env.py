
from jumping_leg.env.leg_jump_env import LegJumpEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import torch as th

def env_builder(seed, log_folder, env_builder_args):
    stepLength_sec = 0.01
    video_save_freq = env_builder_args["video_save_freq"]
    th_device = env_builder_args["th_device"]

    mode = "PyBulletController"
    if mode == "GzController":
        from lr_gym_ros2.env_controllers.GzController import GzController
        env_controller = GzController(stepLength_sec=stepLength_sec)
    elif mode == "GazeboController":
        from lr_gym_ros.envControllers.GazeboController import GazeboController
        env_controller = GazeboController(stepLength_sec=stepLength_sec)
    elif mode == "PyBulletController":
        from lr_gym.env_controllers.PyBulletController import PyBulletController
        env_controller = PyBulletController(stepLength_sec=stepLength_sec, restore_on_reset=False)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)

    lrenv = LegJumpEnv(maxStepsPerEpisode=500,
                       stepLength_sec=stepLength_sec,
                       environmentController=env_controller,
                       seed=seed,
                       obs_only_vec=True,
                       obs_only_img=False,
                       obs_img_height=64,
                       obs_img_width=64,
                       rgb=True,
                       th_device=th_device)
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log")
    if video_save_freq >0:
        env = RecorderGymWrapper(env=env, fps = 1/stepLength_sec, outFolder=log_folder+"/videos/RecorderGymWrapper", saveFrequency_ep=video_save_freq, vec_obs_key="vec")
    env.reset(seed=seed)
    return env