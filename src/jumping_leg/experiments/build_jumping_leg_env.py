
from jumping_leg.env.leg_jump_env import LegJumpEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import torch as th
from gymnasium.wrappers.normalize import NormalizeObservation

def env_builder(seed, log_folder, env_builder_args):
    stepLength_sec = env_builder_args["stepLength_sec"]
    video_save_freq = env_builder_args["video_save_freq"]
    th_device = env_builder_args["th_device"]
    max_steps = 5/stepLength_sec

    mode = "PyBulletController"
    if mode == "GzController":
        from lr_gym_ros2.env_controllers.GzController import GzController
        env_controller = GzController(stepLength_sec=stepLength_sec)
    elif mode == "GazeboController":
        from lr_gym_ros.envControllers.GazeboController import GazeboController
        env_controller = GazeboController(stepLength_sec=stepLength_sec)
    elif mode == "PyBulletController":
        from lr_gym.env_controllers.PyBulletJointImpedanceController import PyBulletJointImpedanceController
        env_controller = PyBulletJointImpedanceController(stepLength_sec=stepLength_sec, restore_on_reset=False, debug_gui=False)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)
    obs_only_vec = True

    lrenv = LegJumpEnv(maxStepsPerEpisode=max_steps,
                       stepLength_sec=stepLength_sec,
                       environmentController=env_controller,
                       seed=seed,
                       obs_only_vec=obs_only_vec,
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
                        control_mode = env_builder_args["control_mode"],
                        reward_scale=500/max_steps,
                        platform_randomization = env_builder_args["platform_randomization"]) # scale it to be the same as if we have 500 steps (mostly so that we can compare easily)
    env = GymEnvWrapper(env=lrenv, episodeInfoLogFile=log_folder+f"/GymEnvWrapperLog.{seed}.log",
                        quiet=True)
    
    if video_save_freq >0:
        env = RecorderGymWrapper(env=env,
                                 fps = 1/stepLength_sec,
                                 outFolder=log_folder+"/videos/RecorderGymWrapper",
                                 saveFrequency_ep=video_save_freq,
                                 vec_obs_key="vec",
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
                                                                                    f"rContac {info['vstate'][LegJumpEnv.STATE.REWARD_CONTACTS_WEIGHT]:.2f}\n"+
                                                                                    f"rEnergy {info['vstate'][LegJumpEnv.STATE.REWARD_ENERGY_WEIGHT]:.2f}\n"+
                                                                                    f"rImpThr {info['vstate'][LegJumpEnv.STATE.REWARD_IMPULSE_THRESHOLD]:.2f}\n"+
                                                                                    f"rPosLim {info['vstate'][LegJumpEnv.STATE.REWARD_POSITION_LIMIT_WEIGHT]:.2f}\n"+
                                                                                    f"rTorLim {info['vstate'][LegJumpEnv.STATE.REWARD_TORQUE_LIMIT_WEIGHT]:.2f}\n"+
                                                                                    f"rTorque {info['vstate'][LegJumpEnv.STATE.REWARD_TORQUE_WEIGHT]:.2f}\n"+
                                                                                    f"rTrack  {info['vstate'][LegJumpEnv.STATE.REWARD_TRACKING_WEIGHT]:.2f}\n"+
                                                                                    f"rVeloci {info['vstate'][LegJumpEnv.STATE.REWARD_VELOCITY_WEIGHT]:.2f}\n"
                                                                                    f"torqHip {info['vstate'][LegJumpEnv.STATE.HIP_JOINT_EFFORT]:.2f}\n"
                                                                                    f"torqKne {info['vstate'][LegJumpEnv.STATE.KNEE_JOINT_EFFORT]:.2f}\n"
                                                                                    f"posiHip {info['vstate'][LegJumpEnv.STATE.HIP_JOINT_POS]:.2f}\n"
                                                                                    f"posiKne {info['vstate'][LegJumpEnv.STATE.KNEE_JOINT_POS]:.2f}\n"
                                                                                    )
    env.reset(seed=seed)
    return env