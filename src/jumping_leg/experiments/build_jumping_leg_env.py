
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
        env_controller = PyBulletController(stepLength_sec=stepLength_sec, restore_on_reset=False, debug_gui=False)
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
                       th_device=th_device,
                        reward_torque_limit_weight = env_builder_args["reward_torque_limit_weight"],
                        reward_position_limit_weight = env_builder_args["reward_position_limit_weight"],
                        reward_velocity_weight = env_builder_args["reward_velocity_weight"],
                        reward_energy_weight = env_builder_args["reward_energy_weight"],
                        reward_tracking_weight = env_builder_args["reward_tracking_weight"],
                        reward_torque_weight = env_builder_args["reward_torque_weight"],
                        reward_contacts_weight = env_builder_args["reward_contacts_weight"],
                        use_velocity_control = env_builder_args["use_velocity_control"])
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log")
    if video_save_freq >0:
        env = RecorderGymWrapper(env=env,
                                 fps = 1/stepLength_sec,
                                 outFolder=log_folder+"/videos/RecorderGymWrapper",
                                 saveFrequency_ep=video_save_freq,
                                 vec_obs_key="vec",
                                 overlay_text_func=lambda vo, a, r, te, tr, info:   f"S   {info['step_count']: .3f}\n"+
                                                                                    f"CF  {info['impulses_sum']: .3f}\n"+
                                                                                    f"ExW {info['external_work']:+.3f}\n"+
                                                                                    f"ToE {info['new_thigh_energy']+info['new_shin_energy']+info['new_slider_energy']:+.3f}\n"+
                                                                                    f"TW  {info['thigh_work']:+.3f}\n"+
                                                                                    f"SW  {info['shin_work']:+.3f}\n"+
                                                                                    f"SlW {info['slider_work']:+.3f}\n"+
                                                                                    f"ToW {info['slider_work']+info['shin_work']+info['thigh_work']:+.3f}\n"+
                                                                                    f"TJW {info['thigh_joint_work']:+.3f}\n"+
                                                                                    f"SJW {info['shin_joint_work']:+.3f}\n"+
                                                                                    f"TE  {info['new_thigh_energy']:+.3f}\n"+
                                                                                    f"SE  {info['new_shin_energy']:+.3f}\n"+
                                                                                    f"SlE {info['new_slider_energy']:+.3f}")
    env.reset(seed=seed)
    return env