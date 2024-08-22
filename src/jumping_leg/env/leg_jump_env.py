#!/usr/bin/env python3
from __future__ import annotations
import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List, Literal, TypeVar, SupportsFloat
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces

from adarl.envs.ControlledEnv import ControlledEnv
import adarl
from adarl.utils.utils import build_pose, JointState, LinkState, quat_swing_twist_decomposition, quat_angle, MoveFailError, exc_to_str
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import torch as th
import adarl.utils.utils
from enum import IntEnum
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
import dataclasses
from dataclasses import dataclass
from adarl.utils.robot_helpers import Robot
from pathlib import Path
import time
import os

_T = TypeVar('_T', float, th.Tensor)




def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1
@dataclass
class RobotStateHelper:
    joint_state_pve : th.Tensor
    control_state_pvesd : th.Tensor
    joint_limits_pve : th.Tensor
    control_limits_pvesd : th.Tensor
    def __init__(self, joints : list[str], obs_dtype : th.dtype, th_device : th.device,
                        joint_limits_minmax_pve : dict[str,th.Tensor], control_limits_minmax_pvesd : dict[str,th.Tensor],
                        joint_state_history_length : int = 1, control_state_history_len : int = 1):
        joints_num = len(joints)
        self._joint_names = joints
        self.joint_state_pve     = th.zeros(size=(joint_state_history_length, joints_num, 3,),
                                            dtype=obs_dtype, device=th_device)
        self.control_state_pvesd = th.zeros(size=(control_state_history_len, joints_num, 5,),
                                            dtype=obs_dtype, device=th_device)
        self.joint_state_name = "jstate"
        self.control_state_name = "ctrlstate"
        
        self.joint_limits_pve = th.stack([joint_limits_minmax_pve[jn] for jn in joints])
        self.control_limits_pvesd = th.stack([control_limits_minmax_pvesd[jn] for jn in joints])

        if self.joint_limits_pve.size() != (2, joints_num,3):
            raise RuntimeError(f"Joint limits shape does not match shape, should be {(2, joints_num,3)}, but it's {self.joint_limits_pve.size()}")
        if self.control_limits_pvesd.size() != (2, joints_num,5):
            raise RuntimeError(f"Joint limits shape does not match shape, should be {(2, joints_num,5)}, but it's {self.control_limits_pvesd.size()}")
        
       
        
    def update(self, joint_state_pve : th.Tensor, control_state_pvesd : th.Tensor):
        for i in range(1,self.joint_state_pve.size()[0]):
            self.joint_state_pve[i] = self.joint_state_pve[i-1]
        for i in range(1,self.control_state_pvesd.size()[0]):
            self.control_state_pvesd[i] = self.control_state_pvesd[i-1]
        self.joint_state_pve[0] = joint_state_pve
        self.control_state_pvesd[0] = control_state_pvesd

    def get_state_normalized(self):
        return {self.joint_state_name   : normalize(self.joint_state_pve, self.joint_limits_pve[0], self.joint_limits_pve[1]),
                self.control_state_name : normalize(self.control_state_pvesd, self.control_limits_pvesd[0], self.control_limits_pvesd[1]),}

    def get_space(self):
        return spaces.gym_spaces.Dict(spaces = {self.joint_state_name : spaces.ThBox(low = th.full_like(self.joint_state_pve, fill_value=-1.0),
                                                                                     high= th.full_like(self.joint_state_pve, fill_value=1.0)),
                                                self.control_state_name : spaces.ThBox(low = th.full_like(self.control_state_pvesd, fill_value=-1.0),
                                                                                       high= th.full_like(self.control_state_pvesd, fill_value=1.0))})


class LegJumpEnv(ControlledEnv):

    metadata = {'render.modes': ['rgb_array']}
    STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT = "a" # component of the state that is the last performed action (it has a different size depending onthe configuration)
    STATE_IMG = "i" # componento f thestate that contains the rendered image

    BASE_STATE_IDXS = IntEnum("BASE_STATE", [
                            "HIP_POS_Z",
                            "HIP_VEL_Z",
                            "SUPPORT1_X",
                            "SUPPORT1_Z",
                            "SUPPORT2_X",
                            "SUPPORT2_Z",
                            "HIP_GOAL_Z",
                            "REWARD_TORQUE_LIMIT_WEIGHT",
                            "REWARD_POSITION_LIMIT_WEIGHT",
                            "REWARD_VELOCITY_WEIGHT",
                            "REWARD_ENERGY_WEIGHT",
                            "REWARD_TRACKING_WEIGHT",
                            "REWARD_TORQUE_WEIGHT",
                            "REWARD_CONTACTS_WEIGHT",
                            "REWARD_IMPULSE_THRESHOLD",
                            "KNEE_TORQUE_CMD_SCALE",
                            "HIP_TORQUE_CMD_SCALE",
                            "HIP_JOINT_POS",
                            "HIP_JOINT_VEL",
                            "HIP_JOINT_EFFORT",
                            "KNEE_JOINT_POS",
                            "KNEE_JOINT_VEL",
                            "KNEE_JOINT_EFFORT",
                            "HIP_POS_REF",
                            "HIP_VEL_REF",
                            "HIP_EFFORT_REF",
                            "HIP_STIFFNESS",
                            "HIP_DAMPING",
                            "KNEE_POS_REF",
                            "KNEE_VEL_REF",
                            "KNEE_EFFORT_REF",
                            "KNEE_STIFFNESS",
                            "KNEE_DAMPING",

                            "THIGH_VEL_X",
                            "THIGH_VEL_Y",
                            "THIGH_VEL_Z",
                            "THIGH_ANG_VEL_X",
                            "THIGH_ANG_VEL_Y",
                            "THIGH_ANG_VEL_Z",
                            "SHIN_VEL_X",
                            "SHIN_VEL_Y",
                            "SHIN_VEL_Z",
                            "SHIN_ANG_VEL_X",
                            "SHIN_ANG_VEL_Y",
                            "SHIN_ANG_VEL_Z",
                            # "THIGH_POS_X",
                            # "THIGH_POS_Y",
                            # "THIGH_POS_Z",
                            # "THIGH_ANG_POS_X",
                            # "THIGH_ANG_POS_Y",
                            # "THIGH_ANG_POS_Z",
                            # "SHIN_POS_X",
                            # "SHIN_POS_Y",
                            # "SHIN_POS_Z",
                            # "SHIN_ANG_POS_X",
                            # "SHIN_ANG_POS_Y",
                            # "SHIN_ANG_POS_Z",
                            "IMPULSES_SUM",
                            "FORCES_SUM",
                            "FORCES_NUM",
                            "IMPULSES_SUM_AVG",
                            "SAFETY_TRIGGERED",
                            "SMOOTHED_GOAL_DIST"], start=0)
    
    # NullState = {
    #     EXTRINSIC_STATE : th.zeros(size=(len(EXTRINSIC_STATES),), dtype=obs_dtype),
    #     JOINT_STATE : th.zeros(size=(joints_num*JOINT_STATE_FIELDS,), dtype=obs_dtype),
    #     CONTROL_STATE : th.zeros(size=(joints_num*JOINT_CTRL_FIELDS,), dtype=obs_dtype),
    #     ACTION_HISTORY : th.zeros(size=(action_len*action_history_len,), dtype=obs_dtype),
    #     CURRENT_IMAGE : th.zeros(size=img_size_chw, dtype=obs_dtype)
    # }
    
    CONTROL_MODES = IntEnum("CONTROL_MODES", [  "VELOCITY",
                                                "TORQUE",
                                                "POSITION",
                                                "IMPEDANCE",
                                                "IMPEDANCE_NO_GAINS",
                                                "POSITION_AND_TORQUES",
                                                "POSITION_AND_GAINS"], start=0)
    action_lengths = {
        CONTROL_MODES.IMPEDANCE: 10 ,
        CONTROL_MODES.IMPEDANCE_NO_GAINS: 6,
        CONTROL_MODES.POSITION_AND_TORQUES: 4,
        CONTROL_MODES.POSITION_AND_GAINS: 4,
        CONTROL_MODES.TORQUE: 2,
        CONTROL_MODES.VELOCITY: 2,
        CONTROL_MODES.POSITION: 2,
        }

    @dataclass
    class EpisodeConfiguration:
        hip_goal_z : th.Tensor
        support1_pos_x : th.Tensor
        support1_pos_z : th.Tensor
        support2_pos_x : th.Tensor
        support2_pos_z : th.Tensor
        reward_contacts_weights : th.Tensor
        obs_noise_mustd : th.Tensor
        initial_joint_pose_rhk : th.Tensor
        max_ep_steps : th.Tensor

    @dataclass
    class EnvConfiguration:
        action_delay_mustd : th.Tensor
        action_exp_smoothing_1s : float
        action_len : int
        action_noise_mustd : th.Tensor
        bstate_minmax : th.Tensor
        control_mode : LegJumpEnv.CONTROL_MODES
        ep_obs_noise_mustd : th.Tensor
        goal_dist_exp_smoothing_1s : float
        impulses_avg_alpha : float
        leg_max_height : float
        leg_max_jump : float
        leg_min_height : float
        max_damping : float
        max_stiffness : float
        min_damping : float
        min_stiffness : float
        obs_dtype : th.dtype
        obs_img_height : int
        obs_img_width : int
        obs_only_img : bool
        obs_only_vec : bool
        original_max_epsteps : int
        platform_randomization : str
        position_cmd_limits_hip : Tuple[float,float]
        position_cmd_limits_knee : Tuple[float,float]
        position_phys_limits_hip : Tuple[float,float]
        position_phys_limits_knee : Tuple[float,float]
        position_safety_limits_hip : Tuple[float,float]
        position_safety_limits_knee : Tuple[float,float]
        rail_initial_position_limits : tuple[float,float]
        randomize_initial_pose : bool
        real : bool
        rendering_enabled :bool
        reward_contacts_weight : float
        reward_energy_weight : float
        reward_max_impulse : float
        reward_position_limit_weight : float
        reward_scale : float
        reward_torque_limit_weight : float
        reward_torque_weight : float
        reward_tracking_weight : float
        reward_velocity_weight : float
        safe_damping : float
        safe_stiffness : float
        show_goal : bool
        start_height : float
        stepLength_sec : float
        step_obs_noise_std : th.Tensor
        stop_on_safety : bool
        th_device : th.device
        th_device : th.device
        torque_command_scale_hip : float
        torque_command_scale_knee : float
        torque_phys_limits_hip : Tuple[float,float]
        torque_phys_limits_knee : Tuple[float,float]
        torque_safety_limits_hip : Tuple[float,float]
        torque_safety_limits_knee : Tuple[float,float]
        use_contacts : bool
        use_threnderer : bool
        velocity_command_scale_hip : float
        velocity_command_scale_knee : float
        velocity_phys_limits_hip : Tuple[float,float]
        velocity_phys_limits_knee : Tuple[float,float]
        velocity_safety_limits_hip : Tuple[float,float]
        velocity_safety_limits_knee : Tuple[float,float]
        wall_sim_speed : bool
        history_length : int
        frame_stack_length : int
        

    @dataclass
    class Statistics:
        dists_to_goal : th.Tensor
        cumulative_dist_to_goal : th.Tensor = th.tensor(0.0)
        cumulative_knee_torque : th.Tensor = th.tensor(0.0)
        cumulative_hip_torque : th.Tensor = th.tensor(0.0)
        max_knee_torque : th.Tensor = th.tensor(0.0)
        max_hip_torque : th.Tensor = th.tensor(0.0)
        cumulated_abs_impulses : float = 0.0
        last_abs_impulses_sum : float = 0.0
        ep_max_abs_impulse : float = 0.0
        ep_max_abs_impulses_sum : float = 0.0
        ep_max_abs_contact : float = 0.0
        ep_max_abs_contacts_sum : float = 0.0
        last_external_work : float = 0.0
        last_step_got_state : int = -1
        last_abs_impulses_sum_avg : float = 0.0

    @staticmethod
    def _sample(value_or_dist : Union[float,Tuple[str,float,float]], generator, device) -> th.Tensor:
        if type(value_or_dist) == tuple:
            if value_or_dist[0] == "uniform":
                traj_len = th.rand((1,), generator=generator, device=device)*(value_or_dist[2]-value_or_dist[1])+value_or_dist[1]
            else:
                raise RuntimeError(f"Unexpected distribution name {value_or_dist[0]}")
        elif type(value_or_dist) == float:
            traj_len = value_or_dist
        else:
            raise RuntimeError(f"Unexpected value_or_dist={value_or_dist}")
        return th.as_tensor(traj_len, device=device)








    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    stepLength_sec : float = 0.01,
                    environmentController : BaseJointImpedanceAdapter = None,
                    startSimulation : bool = True,
                    wall_sim_speed = False,
                    seed = 0,
                    obs_only_vec = False,
                    obs_only_img = False,
                    obs_img_height = 64,
                    obs_img_width = 64,
                    rgb = True,
                    th_device = th.device("cpu"),
                    reward_torque_limit_weight = 0.0,
                    reward_position_limit_weight = 1.0,
                    reward_velocity_weight = 0.0,
                    reward_energy_weight = 0.01,
                    reward_tracking_weight = 1.0,
                    reward_torque_weight = 0.0,
                    reward_contacts_weight = 0.0,
                    control_mode = "torque",
                    reward_scale = 1.0,
                    platform_randomization : Literal["fixed","single","double","no_platforms"] = "fixed",
                    use_contacts : bool = True,
                    real : bool = False,
                    step_precision_tolerance : float = 0.0,
                    ep_obs_noise_mustd : list[float] | th.Tensor = [0.0,0.0], 
                    step_obs_noise_std : list[float] | float | th.Tensor = 0.0,
                    stop_on_safety = True,
                    action_delay_mustd : tuple[float,float] = (0.0,0.0),
                    action_smoothing_halflife_sec = 0.05,
                    leg_min_height : float = 0.4,
                    leg_max_height : float = 0.65,
                    leg_max_jump : float = 0.6,
                    goal_dist_smoothing_halflife_sec = 0.5,
                    randomize_initial_pose : bool = False):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        environmentController : BaseAdapter
            Specifies which simulator controller to use. By default it connects to Gazebo


        """



        control_mode = self.CONTROL_MODES[control_mode.upper()]
        if use_contacts == False and reward_contacts_weight!=0:
            raise RuntimeError(f"use_contacts is False but reward_contacts_weight is not zero")

        self._knee_joint = ("leg","knee_joint_1")
        self._hip_joint = ("leg","hip_joint_1")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","foot_link")
        self._thigh_base_link = ("leg", "thigh_link_base")
        self._shin_base_link = ("leg", "shin_link_base")
        self._thigh_com_link = ("leg", "thigh_link_com")
        self._shin_com_link = ("leg", "shin_link_com")
        self._rendering_cam_name = "simple_camera"        
        # max_dact_dt = 100 #max change in action, i.e. da/dt
        # self._max_act_change = th.tensor(max_dact_dt*stepLength_sec,dtype=th.float32, device=self._th_device)
        # self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)
        self._last_out_action = th.empty((0,))
        self._spawned = False

        self._rng = th.Generator(device=th_device)
        self._stats = self.Statistics(dists_to_goal=th.empty((0,)))
        self._reward_dbg_info = {   "external_work" : 0,
                                    "thigh_work" : 0,
                                    "shin_work" : 0,
                                    "thigh_joint_work" : 0,
                                    "shin_joint_work" : 0,
                                    "new_thigh_energy" : 0,
                                    "new_shin_energy" : 0,
                                    "new_slider_energy" : 0,
                                    "slider_work" : 0}
        

        # self._hip_torque_scale = 100
        # self._knee_torque_scale = 100
        # self._velocity_scale = {self._knee_joint : 1,
        #                         self._hip_joint  : 1}
        # self._configuration = dict( reward_torque_limit_weight = reward_torque_limit_weight,
        #                             reward_position_limit_weight = reward_position_limit_weight,
        #                             reward_velocity_weight = reward_velocity_weight,
        #                             reward_energy_weight = reward_energy_weight,
        #                             reward_tracking_weight = reward_tracking_weight,
        #                             reward_torque_weight = reward_torque_weight,
        #                             hip_torque_scale = self._hip_torque_scale,
        #                             knee_torque_scale = self._knee_torque_scale,
        #                             reward_contacts_weight = reward_contacts_weight,
        #                             max_impulse = 10,
        #                             velocity_scale = {  self._knee_joint : 1,
        #                                                 self._hip_joint  : 1})
        # self._torque_limits =   {self._knee_joint : [-112,112],
        #                          self._hip_joint : [-112,112]}
        # self._velocity_limits = {self._knee_joint : [-20,20],
        #                          self._hip_joint : [-20,20]}
        # self._position_limits = {self._knee_joint : [-2.4,2.4],
        #                          self._hip_joint : [-2.4,2.4]}
        
        # alpha = r^(1/t) where r is the residual value and t is the elapsed time. 
        # So if we want a transition from 1 to 0 to be at 0.05 after 0.1 seconds
        #   we get alpha = 0.05^(1/0.1) = 9.76e-14
        # If we want to use the halving time as a parameter we can say:
        #   alpha = 0.5^(1/halving_time)
        # E.g. with a halving time of 0.01s:
        #   alpha = 0.5^(1/0.01) = 7.88e-31
        action_exp_smoothing_1s = 0.5**(1/action_smoothing_halflife_sec) if action_smoothing_halflife_sec>0 else 0.0
        goal_dist_exp_smoothing_1s = 0.5**(1/goal_dist_smoothing_halflife_sec) if goal_dist_smoothing_halflife_sec>0 else 0.0
        obs_dtype = th.float32
        self._configuration = LegJumpEnv.EnvConfiguration(  action_delay_mustd = th.as_tensor(action_delay_mustd,
                                                                                              dtype=obs_dtype,
                                                                                              device=th_device),
                                                            action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                            action_len = LegJumpEnv.action_lengths[control_mode],
                                                            action_noise_mustd=th.empty((0,)),
                                                            bstate_minmax = th.empty((0,)),
                                                            control_mode=control_mode,
                                                            ep_obs_noise_mustd=th.empty((0,)),
                                                            goal_dist_exp_smoothing_1s = goal_dist_exp_smoothing_1s,
                                                            impulses_avg_alpha = 0.5,
                                                            leg_max_height = leg_max_height,
                                                            leg_max_jump = leg_max_jump,
                                                            leg_min_height = leg_min_height,
                                                            max_damping=200,
                                                            max_stiffness=1000,
                                                            min_damping=10,
                                                            min_stiffness=10,
                                                            obs_img_height=obs_img_height,
                                                            obs_img_width=obs_img_width,
                                                            obs_only_img=obs_only_img,
                                                            obs_only_vec=obs_only_vec,
                                                            position_cmd_limits_hip =  (-2.2, 2.2),
                                                            position_cmd_limits_knee = (-2.2, 2.2),
                                                            position_phys_limits_hip =  (-2.4, 2.4),
                                                            position_phys_limits_knee = (-2.4, 2.4),
                                                            position_safety_limits_hip =  (-2.3, 2.3),
                                                            position_safety_limits_knee = (-2.3, 2.3),
                                                            rail_initial_position_limits = (0.1,1.5),
                                                            randomize_initial_pose = randomize_initial_pose,
                                                            reward_contacts_weight = reward_contacts_weight,
                                                            reward_energy_weight = reward_energy_weight,
                                                            reward_max_impulse = 10,
                                                            reward_position_limit_weight = reward_position_limit_weight,
                                                            reward_scale = reward_scale,
                                                            reward_torque_limit_weight = reward_torque_limit_weight,
                                                            reward_torque_weight = reward_torque_weight,
                                                            reward_tracking_weight = reward_tracking_weight,
                                                            reward_velocity_weight = reward_velocity_weight,
                                                            safe_damping = 50,
                                                            safe_stiffness = 200,
                                                            stepLength_sec=stepLength_sec,
                                                            step_obs_noise_std=th.empty((0,)),
                                                            stop_on_safety = stop_on_safety,
                                                            torque_command_scale_hip = 100,
                                                            torque_command_scale_knee = 100,
                                                            torque_phys_limits_hip =  (-112, 112),
                                                            torque_phys_limits_knee = (-112, 112),
                                                            torque_safety_limits_hip =  (-105, 105),
                                                            torque_safety_limits_knee = (-105, 105),
                                                            velocity_command_scale_hip = 20,
                                                            velocity_command_scale_knee = 20,
                                                            velocity_phys_limits_hip =  (-20, 20),
                                                            velocity_phys_limits_knee = (-20, 20),
                                                            velocity_safety_limits_hip =  (-18, 18),
                                                            velocity_safety_limits_knee = (-18, 18),
                                                            use_threnderer = False,
                                                            platform_randomization = platform_randomization,
                                                            th_device = th_device,
                                                            rendering_enabled = True,
                                                            start_height = 0.9,
                                                            show_goal = True,
                                                            wall_sim_speed = wall_sim_speed,
                                                            original_max_epsteps = maxStepsPerEpisode,
                                                            use_contacts = use_contacts,
                                                            real = real,
                                                            obs_dtype = obs_dtype,
                                                            history_length = 4,
                                                            frame_stack_length = 3
                                                            )
        
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=th.tensor(0, device=self._configuration.th_device),
                                                                       support1_pos_x=th.tensor(0, device=self._configuration.th_device),
                                                                       support1_pos_z=th.tensor(0, device=self._configuration.th_device),
                                                                       support2_pos_x=th.tensor(0, device=self._configuration.th_device),
                                                                       support2_pos_z=th.tensor(0, device=self._configuration.th_device),
                                                                       reward_contacts_weights=th.tensor(0, device=self._configuration.th_device),
                                                                       obs_noise_mustd=th.tensor(0, device=self._configuration.th_device),
                                                                       initial_joint_pose_rhk=th.tensor([1.0,1.0,1.0], device=self._configuration.th_device),
                                                                       max_ep_steps=th.tensor(self._configuration.original_max_epsteps, device=self._configuration.th_device))
        

        self._robot_state_helper = RobotStateHelper(joints=["knee","hip"],
                                                    obs_dtype=obs_dtype,
                                                    th_device=th_device,
                                                    joint_limits_minmax_pve={"knee" : th.as_tensor([self._configuration.position_phys_limits_knee,
                                                                                                     self._configuration.velocity_phys_limits_knee,
                                                                                                     self._configuration.torque_phys_limits_knee]).permute(1,0),
                                                                             "hip" : th.as_tensor([ self._configuration.position_phys_limits_hip,
                                                                                                    self._configuration.velocity_phys_limits_hip,
                                                                                                    self._configuration.torque_phys_limits_hip]).permute(1,0)},
                                                    control_limits_minmax_pvesd={"knee" : th.as_tensor([self._configuration.position_phys_limits_knee,
                                                                                                     self._configuration.velocity_phys_limits_knee,
                                                                                                     self._configuration.torque_phys_limits_knee,
                                                                                                     self._configuration.min_stiffness,
                                                                                                     self._configuration.min_damping]).permute(1,0),
                                                                             "hip" : th.as_tensor([ self._configuration.position_phys_limits_hip,
                                                                                                    self._configuration.velocity_phys_limits_hip,
                                                                                                    self._configuration.torque_phys_limits_hip,
                                                                                                    self._configuration.min_stiffness,
                                                                                                    self._configuration.min_damping]).permute(1,0)},
                                                    joint_state_history_length=self._configuration.frame_stack_length,
                                                    control_state_history_len=self._configuration.frame_stack_length)
        vstate_min_max = {  self.BASE_STATE_IDXS.HIP_JOINT_POS : self._configuration.position_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_JOINT_VEL : self._configuration.velocity_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_JOINT_EFFORT : self._configuration.torque_phys_limits_hip,
                            self.BASE_STATE_IDXS.KNEE_JOINT_POS : self._configuration.position_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_JOINT_VEL : self._configuration.velocity_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT : self._configuration.torque_phys_limits_knee,
                            self.BASE_STATE_IDXS.HIP_POS_Z : [0,3],
                            self.BASE_STATE_IDXS.HIP_VEL_Z : [-100,100],
                            self.BASE_STATE_IDXS.SUPPORT1_X : [-2,2],
                            self.BASE_STATE_IDXS.SUPPORT1_Z : [0,2],
                            self.BASE_STATE_IDXS.SUPPORT2_X : [-2,2],
                            self.BASE_STATE_IDXS.SUPPORT2_Z : [0,2],
                            self.BASE_STATE_IDXS.HIP_GOAL_Z : [0,2],
                            self.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD : [0,10],
                            self.BASE_STATE_IDXS.KNEE_TORQUE_CMD_SCALE : [0,150],
                            self.BASE_STATE_IDXS.HIP_TORQUE_CMD_SCALE : [0,150],
                            self.BASE_STATE_IDXS.HIP_POS_REF : self._configuration.position_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_VEL_REF : self._configuration.velocity_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_EFFORT_REF : self._configuration.torque_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_STIFFNESS : [self._configuration.min_stiffness,self._configuration.max_stiffness],
                            self.BASE_STATE_IDXS.HIP_DAMPING : [self._configuration.min_damping,self._configuration.max_damping],
                            self.BASE_STATE_IDXS.KNEE_POS_REF : self._configuration.position_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_VEL_REF : self._configuration.velocity_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_EFFORT_REF : self._configuration.torque_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_STIFFNESS : [self._configuration.min_stiffness,self._configuration.max_stiffness],
                            self.BASE_STATE_IDXS.KNEE_DAMPING : [self._configuration.min_damping,self._configuration.max_damping],

                            self.BASE_STATE_IDXS.THIGH_VEL_X : [-100,100],
                            self.BASE_STATE_IDXS.THIGH_VEL_Y : [-100,100],
                            self.BASE_STATE_IDXS.THIGH_VEL_Z : [-100,100],
                            self.BASE_STATE_IDXS.THIGH_ANG_VEL_X : [-100,100],
                            self.BASE_STATE_IDXS.THIGH_ANG_VEL_Y : [-100,100],
                            self.BASE_STATE_IDXS.THIGH_ANG_VEL_Z : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_VEL_X : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_VEL_Y : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_VEL_Z : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_ANG_VEL_X : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_ANG_VEL_Y : [-100,100],
                            self.BASE_STATE_IDXS.SHIN_ANG_VEL_Z : [-100,100],
                            # self.BASE_STATE_IDXS.THIGH_POS_X : [-2,2],
                            # self.BASE_STATE_IDXS.THIGH_POS_Y : [-2,2],
                            # self.BASE_STATE_IDXS.THIGH_POS_Z : [-2,2],
                            # self.BASE_STATE_IDXS.THIGH_ANG_POS_X : [-100,100],
                            # self.BASE_STATE_IDXS.THIGH_ANG_POS_Y : [-100,100],
                            # self.BASE_STATE_IDXS.THIGH_ANG_POS_Z : [-100,100],
                            # self.BASE_STATE_IDXS.SHIN_POS_X : [-2,2],
                            # self.BASE_STATE_IDXS.SHIN_POS_Y : [-2,2],
                            # self.BASE_STATE_IDXS.SHIN_POS_Z : [-2,2],
                            # self.BASE_STATE_IDXS.SHIN_ANG_POS_X : [-100,100],
                            # self.BASE_STATE_IDXS.SHIN_ANG_POS_Y : [-100,100],
                            # self.BASE_STATE_IDXS.SHIN_ANG_POS_Z : [-100,100],
                            self.BASE_STATE_IDXS.IMPULSES_SUM : [0,100],
                            self.BASE_STATE_IDXS.FORCES_SUM : [0,1000],
                            self.BASE_STATE_IDXS.FORCES_NUM : [0,1000],
                            self.BASE_STATE_IDXS.IMPULSES_SUM_AVG : [0,100],
                            self.BASE_STATE_IDXS.SAFETY_TRIGGERED : [0,1],
                            self.BASE_STATE_IDXS.SMOOTHED_GOAL_DIST : [0,10]}        
        self._configuration.bstate_minmax = th.tensor([vstate_min_max[k] for k in self.BASE_STATE_IDXS], device = self._configuration.th_device)

        # Part of the BASE_STATE that gets stacked
        if self._configuration.obs_only_vec:
            self._stacked_obs_part = th.as_tensor([ self.BASE_STATE_IDXS.HIP_POS_Z,
                                                    self.BASE_STATE_IDXS.HIP_VEL_Z,
                                                    self.BASE_STATE_IDXS.SUPPORT1_X,
                                                    self.BASE_STATE_IDXS.SUPPORT1_Z,
                                                    self.BASE_STATE_IDXS.SUPPORT2_X,
                                                    self.BASE_STATE_IDXS.SUPPORT2_Z,
                                                    self.BASE_STATE_IDXS.HIP_JOINT_POS,
                                                    self.BASE_STATE_IDXS.HIP_JOINT_VEL,
                                                    self.BASE_STATE_IDXS.HIP_JOINT_EFFORT,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_POS,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_VEL,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT,
                                                    self.BASE_STATE_IDXS.HIP_POS_REF,
                                                    self.BASE_STATE_IDXS.HIP_VEL_REF,
                                                    self.BASE_STATE_IDXS.HIP_EFFORT_REF,
                                                    self.BASE_STATE_IDXS.HIP_STIFFNESS,
                                                    self.BASE_STATE_IDXS.HIP_DAMPING,
                                                    self.BASE_STATE_IDXS.KNEE_POS_REF,
                                                    self.BASE_STATE_IDXS.KNEE_VEL_REF,
                                                    self.BASE_STATE_IDXS.KNEE_EFFORT_REF,
                                                    self.BASE_STATE_IDXS.KNEE_STIFFNESS,
                                                    self.BASE_STATE_IDXS.KNEE_DAMPING], device=self._configuration.th_device)
        else:
            self._stacked_obs_part = th.as_tensor([ self.BASE_STATE_IDXS.HIP_JOINT_POS,
                                                    self.BASE_STATE_IDXS.HIP_JOINT_VEL,
                                                    self.BASE_STATE_IDXS.HIP_JOINT_EFFORT,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_POS,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_VEL,
                                                    self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT,
                                                    self.BASE_STATE_IDXS.HIP_POS_REF,
                                                    self.BASE_STATE_IDXS.HIP_VEL_REF,
                                                    self.BASE_STATE_IDXS.HIP_EFFORT_REF,
                                                    self.BASE_STATE_IDXS.HIP_STIFFNESS,
                                                    self.BASE_STATE_IDXS.HIP_DAMPING,
                                                    self.BASE_STATE_IDXS.KNEE_POS_REF,
                                                    self.BASE_STATE_IDXS.KNEE_VEL_REF,
                                                    self.BASE_STATE_IDXS.KNEE_EFFORT_REF,
                                                    self.BASE_STATE_IDXS.KNEE_STIFFNESS,
                                                    self.BASE_STATE_IDXS.KNEE_DAMPING], device=self._configuration.th_device)

        # Part of the BASE_STATE that does not get stacked
        self._constant_obs_part = th.as_tensor([self.BASE_STATE_IDXS.HIP_GOAL_Z,
                                                self.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT,
                                                self.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD], device=self._configuration.th_device)
        vec_obs_size = self._stacked_obs_part.size()[0]*self._configuration.frame_stack_length + self._constant_obs_part.size()[0]
        vec_obs_space_high = np.array( [1.0]*vec_obs_size)
        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        
        self._img_shape_chw = (3 if rgb else 1,self._configuration.obs_img_height,self._configuration.obs_img_width)
        img_observation_space = spaces.ThBox(low=0, high=255, shape=self._img_shape_chw, dtype=np.uint8, torch_device=self._configuration.th_device)

        state_space = spaces.gym_spaces.Dict({  self.STATE_BASE: spaces.ThBox(low=-float("inf"), high=float("inf"), shape=(self._configuration.history_length,len(LegJumpEnv.BASE_STATE_IDXS),), torch_device=self._configuration.th_device),
                                                self.STATE_ACT: spaces.ThBox(low=-float("inf"), high=float("inf"), shape=(self._configuration.history_length,self._configuration.action_len,), torch_device=self._configuration.th_device),
                                                self.STATE_IMG: img_observation_space})
        
        if self._configuration.obs_only_vec:
            observation_space = spaces.gym_spaces.Dict({ self.STATE_BASE : vec_obs_space})     
        elif self._configuration.obs_only_img:
            observation_space = spaces.gym_spaces.Dict({ self.STATE_IMG  : img_observation_space})
        else:
            observation_space = spaces.gym_spaces.Dict({ self.STATE_BASE : vec_obs_space,
                                                            self.STATE_IMG  : img_observation_space})
            
        action_space_high = np.array([1]*self._configuration.action_len)
        action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high, seed=seed)

        step_obs_noise_std = th.tensor(step_obs_noise_std, device=self._configuration.th_device)
        step_obs_noise_std = step_obs_noise_std.expand((len(self._stacked_obs_part),))
        ep_obs_noise_mustd = th.tensor(ep_obs_noise_mustd, device=self._configuration.th_device)
        if ep_obs_noise_mustd.dim() == 1: ep_obs_noise_mustd = ep_obs_noise_mustd.unsqueeze(1)
        ep_obs_noise_mustd = ep_obs_noise_mustd.expand((2, len(self._stacked_obs_part)))

        self._configuration.ep_obs_noise_mustd = th.tensor(ep_obs_noise_mustd, dtype=self._configuration.obs_dtype, device=self._configuration.th_device)
        self._configuration.step_obs_noise_std = th.tensor(step_obs_noise_std, dtype=self._configuration.obs_dtype, device=self._configuration.th_device)
        self._configuration.action_noise_mustd = 0.0 * th.ones(size=(self._configuration.action_len,), dtype=th.float32, device=self._configuration.th_device)
        
        # # delay noises will actually be discretized by the step_length
        # action_delay_noise_mustd = th.tensor([0.01,0.01], dtype=th.float32, device=self._th_device)
        # obs_delay_noise_mustd = th.tensor([0.01,0.01], dtype=th.float32, device=self._th_device)

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = environmentController,
                         startSimulation = startSimulation,
                         observation_space=observation_space,
                         action_space = action_space,
                         state_space=state_space,
                         step_precision_tolerance=step_precision_tolerance)
        self._environmentController = environmentController
        if not isinstance(self._environmentController , BaseJointImpedanceAdapter):
            raise RuntimeError()
        self.seed(seed)
        self._environmentController.set_monitored_joints([self._knee_joint,self._hip_joint, self._rail_joint])
        self._environmentController.set_monitored_links([self._foot_link, self._shin_com_link, self._thigh_com_link, self._shin_base_link, self._thigh_base_link])
        self._environmentController.set_monitored_cameras([self._rendering_cam_name])
        if self._configuration.use_contacts:
            if not isinstance(self._environmentController, PyBulletAdapter):
                raise RuntimeError(f"Required to use contacts, but environment adapter does not support contacts")
            self._environmentController.monitor_contacts([("leg",None,None,None)]) # Monitor the contacts between the leg and all the environment

        self._environmentController.startup()








    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    def _pvesd_to_action(self, cmds_pvesd):
        
        hp_lim = self._configuration.position_phys_limits_hip
        hv_lim = self._configuration.velocity_command_scale_hip
        he_lim = self._configuration.torque_command_scale_hip
        kp_lim = self._configuration.position_phys_limits_knee
        kv_lim = self._configuration.velocity_command_scale_knee
        ke_lim = self._configuration.torque_command_scale_knee
        max_stiffness = self._configuration.max_stiffness
        max_damping = self._configuration.max_damping
        min_stiffness = self._configuration.min_stiffness
        min_damping = self._configuration.min_damping
        minmax_hipknee_pvesd = th.tensor([[[hp_lim[0], -hv_lim, -he_lim, min_stiffness, min_damping],
                                           [kp_lim[0], -kv_lim, -ke_lim, min_stiffness, min_damping]],

                                          [[hp_lim[1], hv_lim, he_lim, max_stiffness, max_damping],
                                           [kp_lim[1], kv_lim, ke_lim, max_stiffness, max_damping]]], device=self._configuration.th_device)
        act_pvesd_interleaved = th.tensor([ cmds_pvesd[self._hip_joint][0],
                                            cmds_pvesd[self._knee_joint][0],
                                            cmds_pvesd[self._hip_joint][1],
                                            cmds_pvesd[self._knee_joint][1],
                                            cmds_pvesd[self._hip_joint][2],
                                            cmds_pvesd[self._knee_joint][2],
                                            cmds_pvesd[self._hip_joint][3],
                                            cmds_pvesd[self._knee_joint][3],
                                            cmds_pvesd[self._hip_joint][4],
                                            cmds_pvesd[self._knee_joint][4]], device=self._configuration.th_device)
        act_pvesd_interleaved = normalize(act_pvesd_interleaved,
                                                minmax_hipknee_pvesd[[0,0],[0,1]].flatten(),
                                                minmax_hipknee_pvesd[[1,1],[1,0]].flatten())
        if self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE:
            act = act_pvesd_interleaved
        elif self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            act = act_pvesd_interleaved[[0,1,2,3,4,5]]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            act = act_pvesd_interleaved[[0,1,4,5]]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_GAINS:
            act = act_pvesd_interleaved[[0,1,6,7]]
        elif self._configuration.control_mode == self.CONTROL_MODES.TORQUE:
            act = act_pvesd_interleaved[[4,5]]
        elif self._configuration.control_mode == self.CONTROL_MODES.VELOCITY:
            act = act_pvesd_interleaved[[2,3]]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION:
            act = act_pvesd_interleaved[[0,1]]
        else:
            raise RuntimeError(f"invalid control mode {self._configuration.control_mode}")
        return act
    
    def _action_to_pvesd(self, action) -> dict[tuple[str,str],tuple[float,float,float,float,float]]:

        hp_lim = self._configuration.position_cmd_limits_hip
        hv_lim = self._configuration.velocity_command_scale_hip
        he_lim = self._configuration.torque_command_scale_hip
        kp_lim = self._configuration.position_cmd_limits_knee
        kv_lim = self._configuration.velocity_command_scale_knee
        ke_lim = self._configuration.torque_command_scale_knee
        max_stiffness = self._configuration.max_stiffness
        max_damping = self._configuration.max_damping
        min_stiffness = self._configuration.min_stiffness
        min_damping = self._configuration.min_damping
        minmax_hipknee_pvesd = th.tensor([[[hp_lim[0], -hv_lim, -he_lim, min_stiffness, min_damping],
                                           [kp_lim[0], -kv_lim, -ke_lim, min_stiffness, min_damping]],

                                          [[hp_lim[1], hv_lim, he_lim, max_stiffness, max_damping],
                                           [kp_lim[1], kv_lim, ke_lim, max_stiffness, max_damping]]], device=self._configuration.th_device)
        s = normalize(self._configuration.safe_stiffness,min= self._configuration.min_stiffness,max=self._configuration.max_stiffness)
        d = normalize(self._configuration.safe_damping,min= self._configuration.min_damping,max=self._configuration.max_damping)
        hip_pvesd  = th.tensor([0,0,0,s,d], dtype=self._configuration.obs_dtype, device=self._configuration.th_device)
        knee_pvesd = th.tensor([0,0,0,s,d], dtype=self._configuration.obs_dtype, device=self._configuration.th_device)
        
        if self._configuration.control_mode == self.CONTROL_MODES.VELOCITY:
            hip_pvesd[1] = action[0]
            knee_pvesd[1] = action[1]
            hip_pvesd[3] = -1
            knee_pvesd[3] = -1
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION:
            hip_pvesd[0] = action[0]
            knee_pvesd[0] = action[1]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            hip_pvesd[0] = action[0]
            knee_pvesd[0] = action[1]
            hip_pvesd[2] = action[2]
            knee_pvesd[2] = action[3]
        elif self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            hip_pvesd[0] = action[0]
            knee_pvesd[0] = action[1]
            hip_pvesd[1] = action[2]
            knee_pvesd[1] = action[3]
            hip_pvesd[2] = action[4]
            knee_pvesd[2] = action[5]
        elif self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE:
            hip_pvesd[0]  = action[0]
            knee_pvesd[0] = action[1]
            hip_pvesd[1]  = action[2]
            knee_pvesd[1] = action[3]
            hip_pvesd[2]  = action[4]
            knee_pvesd[2] = action[5]
            hip_pvesd[3]  = action[6]
            knee_pvesd[3] = action[7]
            hip_pvesd[4]  = action[8]
            knee_pvesd[4] = action[9]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_GAINS:
            hip_pvesd[0]  = action[0]
            knee_pvesd[0] = action[1]
            hip_pvesd[3]  = action[2]
            knee_pvesd[3] = action[3]
        elif self._configuration.control_mode == self.CONTROL_MODES.TORQUE:
            hip_pvesd[2]  = action[0]
            knee_pvesd[2] = action[1]
            hip_pvesd[3]  = -1
            knee_pvesd[3] = -1
            hip_pvesd[4]  = -1
            knee_pvesd[4] = -1
        else:
            raise RuntimeError(f"Invalid control mode {self._configuration.control_mode}")
        
        hip_pvesd = unnormalize(hip_pvesd, min=minmax_hipknee_pvesd[0][0], max=minmax_hipknee_pvesd[1][0])
        knee_pvesd = unnormalize(knee_pvesd, min=minmax_hipknee_pvesd[0][1], max=minmax_hipknee_pvesd[1][1])
        return {self._hip_joint :  tuple(hip_pvesd.tolist()),
                self._knee_joint:  tuple(knee_pvesd.tolist())}

    def submitAction(self, action : th.Tensor) -> None:
        with th.no_grad():
            # ggLog.info(f"Submitting action {action}")
            action = th.as_tensor(action).detach().cpu()
            super().submitAction(action)
            dt = self._configuration.stepLength_sec
            alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
            prev_action = self._current_state[self.STATE_ACT][0].detach().cpu()
            if self._actionsCounter != 0:
                action = action*(1-alpha) + prev_action*alpha
            action = th.clamp(action, min=-1, max=1)
            jimp_pvesd = self._action_to_pvesd(action)
            self._last_out_action = action
            self._last_sent_pvesd = jimp_pvesd
            n = th.randn(size=(1,),
                        generator=self._rng,
                        dtype=self._configuration.obs_dtype,
                        device=self._configuration.th_device)
            action_delay = self._configuration.action_delay_mustd[0] + self._configuration.action_delay_mustd[1]*n
            action_delay = th.clamp(action_delay, min = 0.0)
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_pvesd,
                                                                delay_sec=action_delay.item())
            






















    # --------------------------------------------------------------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------------------------------------------------------------


    @staticmethod
    def _kinetic_energy_2d(mass, inertia_moment, vel_x, vel_z, ang_vel_y):
        return 0.5*mass*(vel_x**2 + vel_z**2) + 0.5*inertia_moment*ang_vel_y**2


    @staticmethod
    def _compute_mechanical_energies(vstate_unnorm):
        thigh_mass = 3.37
        thigh_length = 0.3
        shin_mass = 1.3
        shin_length = 0.45
        slider_mass = 8
        g = 9.8
        
        thigh_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = thigh_mass,
                                                        inertia_moment=1/12*thigh_mass*thigh_length**2,
                                                        vel_x=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_VEL_X],
                                                        vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_VEL_Z],
                                                        ang_vel_y=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_VEL_Y])
        shin_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = shin_mass,
                                                        inertia_moment=1/12*shin_mass*shin_length**2,
                                                        vel_x=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_VEL_X],
                                                        vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_VEL_Z],
                                                        ang_vel_y=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_VEL_Y])
        slider_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = slider_mass,
                                                        inertia_moment=0,
                                                        vel_x=0,
                                                        vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.HIP_VEL_Z],
                                                        ang_vel_y=0)
        thigh_pot_energy = 0 #thigh_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_POS_Z]
        shin_pot_energy = 0 #shin_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_POS_Z]
        slider_pot_energy = slider_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.HIP_POS_Z]
        return thigh_kin_energy+thigh_pot_energy, shin_kin_energy+shin_pot_energy, slider_kin_energy+slider_pot_energy

    @staticmethod
    def computeReward(previousState : Dict[str,th.Tensor],
                      state : Dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : Dict[str,th.Tensor] = {}, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        vstate_norm = state[LegJumpEnv.STATE_BASE][0]
        pvstate_norm = state[LegJumpEnv.STATE_BASE][-1]

        normtorques = vstate_norm[[LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_EFFORT,LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]]
        normvelocities = vstate_norm[[LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_VEL,LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_VEL]]
        normpositions = vstate_norm[[LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_POS,LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_POS]]
        # ntorques =       [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_EFFORT,LegJumpEnv.STATE.KNEE_JOINT_EFFORT]]
        # nvelocities =    [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_VEL,LegJumpEnv.STATE.KNEE_JOINT_VEL]]
        # npositions =     [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_POS,LegJumpEnv.STATE.KNEE_JOINT_POS]]
        max_r = 100
        torque_reward = - th.clamp(th.mean(th.pow(normtorques,4)),-max_r,max_r)
        torque_limit_reward = - th.clamp(th.mean(th.pow(normtorques,50)),-max_r,max_r)
        velocity_reward = - th.clamp(th.mean(th.pow(normvelocities,2)),-max_r,max_r)
        position_limit_reward = - th.clamp(th.mean(th.pow(normpositions,50)),-max_r,max_r)
        # torque_reward : th.Tensor = -(sum([t**4 for t in ntorques])/len(ntorques)) # type: ignore
        # torque_limit_reward : th.Tensor =   -(sum([t**50 for t in ntorques])/len(ntorques)) # type: ignore # 0.0769 at 0.95
        # velocity_reward : th.Tensor =       -(sum([t**2  for t in nvelocities])/len(nvelocities)) # type: ignore
        # position_limit_reward : th.Tensor = -(sum([t**50 for t in npositions])/len(npositions)) # type: ignore # 0.0769 at 0.95

        # ggLog.info(f"normtorques = {normtorques}")
        # ggLog.info(f"torque_limit_reward = {torque_limit_reward}")

        vstate_un = LegJumpEnv._unnormalize(vstate_norm,env_conf["bstate_minmax"][:,0],env_conf["bstate_minmax"][:,1])
        pvstate_un = LegJumpEnv._unnormalize(pvstate_norm,env_conf["bstate_minmax"][:,0],env_conf["bstate_minmax"][:,1])

        goal_dist = vstate_un[LegJumpEnv.BASE_STATE_IDXS.SMOOTHED_GOAL_DIST]
        # tracking_reward = 1 - goal_dist
        # tracking_reward = 1/(1+goal_dist/0.05)       # 0.50 at 0.05m, 0.35 at 0.10m, 0.2 at 0.2
        tracking_reward = 1/(1+(goal_dist/0.1)**2) # 0.75 at 0.05m, 0.50 at 0.10m, 0.2 at 0.2
        impulse_threshold = pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD]
        contacts_reward = th.clamp(-(vstate_un[LegJumpEnv.BASE_STATE_IDXS.IMPULSES_SUM_AVG]/impulse_threshold)**4, min = -1)


        ktorque = vstate_un[LegJumpEnv.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]
        htorque = vstate_un[LegJumpEnv.BASE_STATE_IDXS.HIP_JOINT_EFFORT]
        # shin_rotation = vstate_un[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_POS_Y] - pvstate_un[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_POS_Y]
        # thigh_rotation = vstate_un[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_POS_Y] - pvstate_un[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_POS_Y]
        shin_rotation = 0
        thigh_rotation = 0


        new_thigh_energy, new_shin_energy, new_slider_energy = LegJumpEnv._compute_mechanical_energies(vstate_un)
        old_thigh_energy, old_shin_energy, old_slider_energy = LegJumpEnv._compute_mechanical_energies(pvstate_un)
        # Overall energy diff on the links, due to all sources: anelastic collisions, joint torques and joint constrain forces
        thigh_work = new_thigh_energy-old_thigh_energy
        shin_work = new_shin_energy-old_shin_energy
        slider_work = new_slider_energy-old_slider_energy
        # work done by the joints on the degree of freedom in the fixed reference frame
        shin_joint_work = ktorque*shin_rotation
        thigh_joint_work = htorque*thigh_rotation + (-ktorque)*thigh_rotation # can we simply use the joint displacement and work on one side only?
        # perlink_energy_reward = - (thigh_work**2 + shin_work**2)
        external_work = slider_work + thigh_work + shin_work - shin_joint_work -thigh_joint_work # this should more or less be the work coming from outside forces (constrain forces should cancel each other out)
        global_energy_reward = -(external_work**2) # minimize the energy exchanged by the whole robot to the outside world

        # ggLog.info(f"new_thigh_energy={new_thigh_energy}, old_thigh_energy={old_thigh_energy}, new_shin_energy={new_shin_energy}, old_shin_energy={old_shin_energy}")
        # ggLog.info(f"knee_work={knee_work}\t hip_work={hip_work}\t thigh_work={thigh_work}\t shin_work={shin_work}")

        sub_rewards["reward_tracking"] = tracking_reward
        sub_rewards["reward_torque"] = torque_reward
        sub_rewards["reward_torque_limit"] = torque_limit_reward
        sub_rewards["reward_velocity"] = velocity_reward
        sub_rewards["reward_position_limit"] = position_limit_reward
        sub_rewards["reward_energy"] = global_energy_reward
        sub_rewards["reward_contacts"] = contacts_reward
        sub_rewards["reward_health"] = th.tensor(1, device=vstate_norm.device)
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards.items()}

        weights = { "reward_tracking" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT],
                    "reward_torque" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT],
                    "reward_torque_limit" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "reward_velocity" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT],
                    "reward_position_limit" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT],
                    "reward_energy" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT],
                    "reward_contacts" : pvstate_un[LegJumpEnv.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT],
                    "reward_health" : 1}
        for k in sub_rewards:
            sub_rewards[k] = sub_rewards[k]*env_conf["reward_scale"]*weights[k]
        reward = th.as_tensor(sum(list(sub_rewards.values())))

        if dbg_info is not None:
            sub_rewards_scaled = {f"{k}_scaled":v for k,v in sub_rewards.items()}
            sub_rewards_scaled_agg_names = [k for k in sub_rewards_scaled.keys()]
            sub_rewards_scaled_agg = th.stack([sub_rewards_scaled[k] for k in sub_rewards_scaled_agg_names])
            sub_rewards_scaled_agg_names = th.as_tensor([list(n.encode("utf-8").ljust(16)[:16]) for n in sub_rewards_scaled_agg_names], dtype=th.uint8)
            sub_rewards_unscaled_agg_names = [k for k in sub_rewards_unscaled.keys()]
            sub_rewards_unscaled_agg = th.stack([sub_rewards_unscaled[k] for k in sub_rewards_unscaled_agg_names])
            sub_rewards_unscaled_agg_names = th.as_tensor([list(n.encode("utf-8").ljust(16)[:16]) for n in sub_rewards_unscaled_agg_names], dtype=th.uint8)
            dbg_info["external_work"] = external_work
            dbg_info["thigh_work"] = thigh_work
            dbg_info["shin_work"] = shin_work
            dbg_info["slider_work"] = slider_work
            dbg_info["thigh_joint_work"] = thigh_joint_work
            dbg_info["shin_joint_work"] = shin_joint_work
            dbg_info["new_thigh_energy"] = new_thigh_energy
            dbg_info["new_shin_energy"] = new_shin_energy
            dbg_info["new_slider_energy"] = new_slider_energy
            dbg_info["sub_rewards_unscaled"] = sub_rewards_unscaled_agg
            dbg_info["sub_rewards_unscaled_labels"] = sub_rewards_unscaled_agg_names
            dbg_info["sub_rewards_scaled"] = sub_rewards_scaled_agg
            dbg_info["sub_rewards_scaled_labels"] = sub_rewards_scaled_agg_names
            dbg_info.update({k:r.cpu().item() if isinstance(r,th.Tensor) else r for k,r in sub_rewards.items()})
            dbg_info["reward"] = reward
        if sub_rewards["reward_contacts"] != 0: raise RuntimeError(f"reward_contacts is {sub_rewards['reward_contacts']}, weights = {weights}, bstate = {state[LegJumpEnv.STATE_BASE]}")
        return reward
























    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------

    def initializeEpisode(self, options = {}) -> None:

        
        self._current_state = {self.STATE_BASE   : th.full((self._configuration.history_length, len(self.BASE_STATE_IDXS)), fill_value=-1, dtype=th.float32, device=self._configuration.th_device),
                            self.STATE_ACT    : th.zeros((self._configuration.history_length, self._configuration.action_len), dtype=th.float32, device=self._configuration.th_device),
                            self.STATE_IMG    : th.zeros(self._img_shape_chw, dtype = th.uint8, device = self._configuration.th_device)}
        
        if not self._spawned and isinstance(self._environmentController, BaseSimulationAdapter):
            leg_pose = build_pose(0,0,0,0,0,0,1)
            camera_pose = build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707)
            support1_pose = build_pose(-0.5, 0.2, 0.2, 0,0,0,1)
            support2_pose = build_pose(-0.5, 0.2, 0.4, 0,0,0,1)
            red_ball_pose = leg_pose
            self._spawned = True
            leg_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")
            support_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro")
            camera_file = adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro")
            supports_xacro_args = { "add_world_link":str(isinstance(self._environmentController, PyBulletAdapter)),
                                    "size_x":0.2,
                                    "size_y":0.2,
                                    "size_z":0.005}
            if isinstance(self._environmentController, PyBulletAdapter):
                self._environmentController.spawn_model(model_file=leg_file,
                                                        model_name="leg",
                                                        pose=leg_pose,
                                                        model_format="urdf.xacro")
            self._environmentController.spawn_model(model_file=camera_file,
                                                    model_name="camera",
                                                    pose=camera_pose,
                                                    model_format="sdf.xacro",
                                                    model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._intendedStepLength_sec})
            self._environmentController.spawn_model(model_file=support_file,
                                                    model_name="support1",
                                                    pose=support1_pose,
                                                    model_format="urdf.xacro",
                                                    model_kwargs=supports_xacro_args)
            self._environmentController.spawn_model(model_file=support_file,
                                                    model_name="support2",
                                                    pose=support2_pose,
                                                    model_format="urdf.xacro",
                                                    model_kwargs=supports_xacro_args)
            if self._configuration.show_goal:
                self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/red_intangible_ball.urdf.xacro"),
                                                        model_name="red_ball",
                                                        pose=red_ball_pose,
                                                        model_format="urdf.xacro",
                                                        model_kwargs={"add_world_link":str(isinstance(self._environmentController, PyBulletAdapter))})
            self._robot_model = Robot(adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(leg_file).read_text()))
            self._robot_model.disable_tree_self_collisions("rail_joint")
            self._robot_model.remove_collision_pairs([("rail_link_0","slider_link_0")])
            self._support1_co_id = self._robot_model.add_collision_box( pose_xyz_xyzw=support1_pose.array_xyz_xyzw(type=np.ndarray),
                                                                        collision_box_size_xyz=(supports_xacro_args["size_x"],
                                                                                                supports_xacro_args["size_y"],
                                                                                                supports_xacro_args["size_z"]),
                                                                        collision_obj_id="support1_collision")
            self._support2_co_id = self._robot_model.add_collision_box( pose_xyz_xyzw=support2_pose.array_xyz_xyzw(type=np.ndarray),
                                                                        collision_box_size_xyz=(supports_xacro_args["size_x"],
                                                                                                supports_xacro_args["size_y"],
                                                                                                supports_xacro_args["size_z"]),
                                                                        collision_obj_id="support2_collision")
            self._ground_co_id = self._robot_model.add_collision_box( pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
                                                                    collision_box_size_xyz=(10,10,1),
                                                                    collision_obj_id="ground_collision")
            self._robot_model.remove_collision_pairs([("support1_collision","ground_collision")])
            self._robot_model.remove_collision_pairs([("support1_collision","support2_collision")])
            self._robot_model.remove_collision_pairs([("support2_collision","ground_collision")])
            self._robot_model.remove_collision_pairs([("rail_link_0","ground_collision")])
        

        self._set_current_ep_config(reset_options = options)

        initial_dist = th.abs(self._current_episode_config.hip_goal_z-self._current_episode_config.initial_joint_pose_rhk[0]).cpu().item()
        self._stats = self.Statistics(dists_to_goal=th.full(size=(int(self._maxStepsPerEpisode/10),),
                                                    fill_value=initial_dist,
                                                    dtype=th.float32, device=self._configuration.th_device))
        
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            self._realworld_initialization()
        self._last_out_action = self._pvesd_to_action(self._last_sent_pvesd)        
        self._update_state()
        self._update_dbg_info()

    def _set_current_ep_config(self, reset_options : dict):
        if reset_options is not None:
            reset_options = {}
        # Having conservative values here will not make the policy learn to behave nice in unfeasible cases
        # Having too broad value will have unfeasible cases in training
        min_goal_z = self._configuration.leg_min_height
        max_goal_z = self._configuration.leg_max_jump + self._configuration.leg_max_height
        
        hip_goal_z = reset_options.get("hip_goal_z",
                                        min_goal_z + th.rand(size=(1,), generator=self._rng, device=self._configuration.th_device)*(max_goal_z-min_goal_z)) # uniform(0.4,1.2)
        
        s1_xz,s2_xz = self._choose_platforms_positions(hip_goal_z)
        if "support1_pos_x" in reset_options: s1_xz[0] = reset_options["support1_pos_x"]
        if "support1_pos_z" in reset_options: s1_xz[1] = reset_options["support1_pos_z"]
        if "support2_pos_x" in reset_options: s2_xz[0] = reset_options["support2_pos_x"]
        if "support2_pos_z" in reset_options: s2_xz[1] = reset_options["support2_pos_z"]

        reward_contacts_weights = reset_options.get("reward_contacts_weights",
                                                    self._sample(self._configuration.reward_contacts_weight,
                                                    self._rng,
                                                    self._configuration.th_device))
        maxStepsPerEpisode = reset_options.get("max_ep_steps", self._configuration.original_max_epsteps)
           
        obs_noise_mu = self._configuration.ep_obs_noise_mustd[0] + self._configuration.ep_obs_noise_mustd[1]*th.randn(size=(len(self._stacked_obs_part),),
                                                                                                                      generator=self._rng,
                                                                                                                      dtype=self._configuration.obs_dtype,
                                                                                                                      device=self._configuration.th_device)
        found_good_configuration = False
        if self._configuration.randomize_initial_pose:
            collisions = []
            for i in range(10000):
                rail_hip_knee_pos = th.rand(size=(3,), device=self._configuration.th_device, dtype = th.float32)*2-1
                rail_hip_knee_pos = unnormalize(rail_hip_knee_pos,
                                        min=th.tensor([ self._configuration.rail_initial_position_limits[0],
                                                        self._configuration.position_cmd_limits_hip[0],
                                                        self._configuration.position_cmd_limits_knee[0]]),
                                        max=th.tensor([self._configuration.rail_initial_position_limits[1],
                                                    self._configuration.position_cmd_limits_hip[1],
                                                    self._configuration.position_cmd_limits_knee[1]]))
                self._robot_model.set_joint_pose(rail_hip_knee_pos.cpu().numpy())
                self._robot_model.move_collision_object(self._support1_co_id, np.array((s1_xz[0], 0.3, s1_xz[1], 0.0,0.0,0.0,1.0)))
                self._robot_model.move_collision_object(self._support2_co_id, np.array((s2_xz[0], 0.3, s2_xz[1], 0.0,0.0,0.0,1.0)))
                collisions = self._robot_model.get_all_collisions()
                if len(collisions) == 0:
                    found_good_configuration = True
                    break
            if not found_good_configuration:
                import cv2
                imgfile = f"failed_spawn_{int(time.time()*1000_000)}.png"
                ggLog.error(f"Couldn't sample initial pose, last pos= {rail_hip_knee_pos.cpu().tolist()}, s1_xz={s1_xz}, s2_xz={s2_xz}, last collisions = {collisions}")
                # ggLog.error(f"saving image at {imgfile}")
                # cv2.imwrite(imgfile, self._robot_model.get_dbg_image())

            link_poses = self._robot_model.get_frame_poses()
            if link_poses["foot_link"][0][2] <0:
                ggLog.error("foot is under the ground! link_poses = "+"\n".join([f"{n}:{p}" for n,p in link_poses.items()]))
                ggLog.error(f"checked collisions = {self._robot_model._collision_pairs}")
                ggLog.error(f"collisions = {collisions}")

        if not found_good_configuration:
            rail_pos, hip_pos, knee_pos = 0.65, 3.1459/4,  3.14159/2
            if s2_xz[0] < 0:
                hip_pos, knee_pos = -hip_pos, -knee_pos
            rail_hip_knee_pos = th.tensor((rail_pos, hip_pos, knee_pos), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
        
        # ggLog.info(f"{os.getpid()} init: chosen jpos= {rail_hip_knee_pos.cpu().tolist()}")        
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=hip_goal_z,
                                                                       support1_pos_x=s1_xz[0],
                                                                       support1_pos_z=s1_xz[1],
                                                                       support2_pos_x=s2_xz[0],
                                                                       support2_pos_z=s2_xz[1],
                                                                       reward_contacts_weights=reward_contacts_weights,
                                                                       obs_noise_mustd=th.stack([obs_noise_mu, self._configuration.step_obs_noise_std]),
                                                                       initial_joint_pose_rhk = rail_hip_knee_pos,
                                                                       max_ep_steps = maxStepsPerEpisode)
        self.set_max_episode_steps(self._current_episode_config.max_ep_steps)

    def _choose_platforms_positions(self, hip_goal_z):
        # Having conservative values here will not make the policy learn to behave nice in unfeasible cases
        # Having too broad value will have unfeasible cases in training
        min_stretch_z = self._configuration.leg_min_height
        max_stretch_z = self._configuration.leg_max_height
        max_jump_z = self._configuration.leg_max_jump
        min_plat_z = hip_goal_z-max_stretch_z
        max_plat_z = th.min(th.as_tensor([max_jump_z, hip_goal_z-min_stretch_z]))
        if self._configuration.platform_randomization == "double":
            raise NotImplementedError()
            # s1_area = th.tensor([[0.20, 0.30],  # minx, maxx
            #                      [0.05, 0.40]], # miny, maxy
            #                     device=self._th_device)
            # s1_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            # s1_xz = s1_xz*(s1_area[:,1]-s1_area[:,0])+s1_area[:,0]
            # # s1_pos = th.tensor([0.0,0.0])


            # s2_area = th.tensor([[0.05, 0.30], # minx, maxx
            #                     [0.05, 0.45]], device=self._th_device) # miny, maxy
            # s2_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            # s2_xz = s1_xz + s2_xz*(s2_area[:,1]-s2_area[:,0])+s2_area[:,0]

            # s1_xz[0] = s1_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
            # s2_xz[0] = s2_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
        elif self._configuration.platform_randomization == "single":
            s1_xz = th.tensor([-0.1-0.125, -0.3], device = self._configuration.th_device) # hide platform 
            s2_area = th.tensor([[0.20, 0.30],  # minx, maxx
                                 [min_plat_z, max_plat_z]], # miny, maxy
                                device=self._configuration.th_device)
            s2_xz = th.rand(size=(2,), generator=self._rng, device=self._configuration.th_device)
            s2_xz = s2_xz*(s2_area[:,1]-s2_area[:,0])+s2_area[:,0]
            s2_xz[0] = s2_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._configuration.th_device)-0.5)
        elif self._configuration.platform_randomization == "fixed":
            s1_xz = th.tensor([-0.1-0.125, 0.3], device=self._configuration.th_device)
            s2_xz = th.tensor([-0.15-0.125, 0.6], device=self._configuration.th_device)
        elif self._configuration.platform_randomization == "no_platforms":
            s1_xz = th.tensor([10, 10], device=self._configuration.th_device)
            s2_xz = th.tensor([10, 11], device=self._configuration.th_device)
        elif self._configuration.platform_randomization == "single_full":
            min_pos = th.as_tensor([-0.5,-0.1])
            max_pos = th.as_tensor([0.5,1.0])
            s1_xz = th.rand(size=(2,), device=self._configuration.th_device)*(max_pos-min_pos) + min_pos
            s2_xz = th.tensor([10, 11], device=self._configuration.th_device)
        else:
            raise RuntimeError(f"Invalid platform_randomization mode '{self._configuration.platform_randomization}'")
        return s1_xz, s2_xz

    def _realworld_initialization(self):
        rail_pos, hip_pos, knee_pos = self._current_episode_config.initial_joint_pose_rhk.cpu().tolist()
        moved = False
        while not moved:
            ggLog.info(f"Cannot automatically initialize episode with non-simulated adapter. Lift up the robot and press ENTER. Be safe :)")
            input()
            if isinstance(self._environmentController, BaseJointPositionAdapter):
                try:
                    self._environmentController.moveToJointPoseSync({self._hip_joint:  hip_pos,
                                                                    self._knee_joint: knee_pos})
                except MoveFailError as e:
                    ggLog.warn(f"Failed to move to joint position. Error = {exc_to_str(e)}")
            start_jimp : dict[tuple[str,str], tuple] = {self._hip_joint: (hip_pos,0,0,200,50),
                                                        self._knee_joint:(knee_pos,0,0,200,50)}       
            self._environmentController.setJointsImpedanceCommand(start_jimp)
            self._environmentController.apply_joint_impedances(start_jimp)
            self._last_sent_pvesd = start_jimp

    def _simulation_initialization(self):
        rail_pos, hip_pos, knee_pos = self._current_episode_config.initial_joint_pose_rhk
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._place_objects(support1_xz=(10,10),
                                support2_xz=(10,10),
                                goal_z=10)
            # ggLog.info(f"Directly setting jpos = {rail_pos, hip_pos, knee_pos}")
            self._environmentController.setJointsStateDirect({  self._rail_joint: JointState(position = rail_pos, rate=0, effort=0),
                                                                self._hip_joint:  JointState(position = hip_pos, rate=0, effort=0),
                                                                self._knee_joint: JointState(position = knee_pos, rate=0, effort=0)})
            start_jimp : dict[tuple[str,str], tuple] = {self._hip_joint: (hip_pos,0,0,200,50),
                                                        self._knee_joint:(knee_pos,0,0,200,50)}         
            self._environmentController.setJointsImpedanceCommand(start_jimp)
            self._environmentController.apply_joint_impedances(start_jimp)
            self._last_sent_pvesd = start_jimp
            # if self._environmentController.__class__.__name__== "RosXbotGazeboAdapter":
            # self._environmentController.run(3.0) # let the leg fall
            # ggLog.info(f"jpos set")
            self._place_objects(support1_xz=(self._current_episode_config.support1_pos_x,self._current_episode_config.support1_pos_z),
                                support2_xz=(self._current_episode_config.support2_pos_x, self._current_episode_config.support2_pos_z),
                                goal_z=self._current_episode_config.hip_goal_z)
            # jpos = {k:v.position for k,v in self._environmentController.getJointsState(requestedJoints=[self._rail_joint, self._hip_joint, self._knee_joint]).items()}
            # ggLog.info(f"Init: current jpos = {jpos}")
        else:
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")

    def _place_objects(self, support1_xz=None, support2_xz=None, goal_z=None):
        if not isinstance(self._environmentController, BaseSimulationAdapter):
            raise RuntimeError("Cannot place objects in the real")
        # ggLog.info(f"Placing supports")
        # ggLog.info(f"placing: _current_episode_config {self._current_episode_config}")
        if support1_xz is None:
            support1_xz = self._current_episode_config.support1_pos_x,self._current_episode_config.support1_pos_z
        if support2_xz is None:
            support2_xz = self._current_episode_config.support2_pos_x,self._current_episode_config.support2_pos_z 
        if goal_z is None:
            goal_z = self._current_episode_config.hip_goal_z
        self._environmentController.setLinksStateDirect({self._support1_base : 
                                                        LinkState( position_xyz = th.tensor((support1_xz[0],
                                                                                             0.3,
                                                                                             support1_xz[1]), device=self._configuration.th_device),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0), device=self._configuration.th_device),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
        self._environmentController.setLinksStateDirect({self._support2_base :
                                                        LinkState(position_xyz = th.tensor((support2_xz[0],
                                                                                            0.3,
                                                                                            support2_xz[1]), device=self._configuration.th_device),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0), device=self._configuration.th_device),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
        if self._configuration.show_goal:
            self._environmentController.setLinksStateDirect({self._red_ball_base :
                                                            LinkState( position_xyz = th.tensor((0.,
                                                                                                0.2,
                                                                                                goal_z), device=self._configuration.th_device),
                                                                        orientation_xyzw = th.tensor((0.,0.,0.,1.0), device=self._configuration.th_device),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})


    def buildSimulation(self):
        # ggLog.info("Building env")
        envCtrlName = type(self._environmentController).__name__
        self._knee_joint = ("leg","knee_joint_1")
        self._hip_joint = ("leg","hip_joint_1")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","tip1")
        self._thigh_base_link = ("leg", "thigh_link1")
        self._shin_base_link = ("leg", "shin_link1")
        self._thigh_com_link = ("leg", "thigh_link1_com")
        self._shin_com_link = ("leg", "shin_link1_com")
        self._rendering_cam_name = "simple_camera"
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._environmentController.build_scenario(None)
            self._support1_base = ("support1","world")
            self._support2_base = ("support2","world")
            self._red_ball_base = ("red_ball","world")
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._configuration.real:
                raise NotImplementedError()
            else:
                self._environmentController.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path("jumping_leg",
                                                                                                                          "gazebo/all_gazebo_xbot.launch"),
                                                           launch_file_args={"gui":"false"})
                self._support1_base = ("support1","plate")
                self._support2_base = ("support2","plate")
                self._red_ball_base = ("red_ball","sphere_link")
        else:
            raise NotImplementedError("environmentController "+envCtrlName+" not supported")

    def _destroySimulation(self):
        self._environmentController.destroy_scenario()





















    # --------------------------------------------------------------------------------------------------------------------
    # State & Observation
    # --------------------------------------------------------------------------------------------------------------------

    def _render_image(self, bstate, th_render = None):
        with th.no_grad():
            if not self._configuration.use_threnderer or (th_render is not None and not th_render):
                import torchvision.transforms.functional # only import it if needed
                img, time = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
                img = th.tensor(img, dtype = th.uint8, device = self._configuration.th_device)
                img = th.permute(img,(2,0,1)) # hwc to chw
                if img.size()[0] == self._img_shape_chw[0]:
                    pass
                elif img.size()[0] == 3 and self._img_shape_chw[0] == 1:
                    img = torchvision.transforms.functional.rgb_to_grayscale(img)
                else:
                    raise RuntimeError(f"Cannot adapt image shape {img.size()} to required {self._img_shape_chw}")
                img = torchvision.transforms.functional.resized_crop(img,
                                                                    top=0,
                                                                    left=int(0.29*img.size()[2]),
                                                                    height=img.size()[1],
                                                                    width=int(0.4*img.size()[2]),
                                                                    size = list(self._img_shape_chw[1:]))
                img = img.permute(1,2,0)
                img = img.cpu().numpy()
                # ggLog.info(f"img = {img.shape}")
                if img is None:
                    time = -1
                return img, time
            else:
                import adarl.utils.simple_threndering as simple_threndering

                draw_device = "cuda"
                bstate = unnormalize(bstate,self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1]).to(device=draw_device)
                image_chw = th.zeros(size=self._img_shape_chw, device=draw_device, dtype=th.uint8)
                # ggLog.info(f"image_chw = {image_chw.size()}")
                body_size = 0.1
                thigh_width = 0.08
                thigh_length = 0.3
                shin_width = 0.07
                shin_length = 0.45
                support_height = 0.02
                support_width = 0.2
                draw_yoffset = -0.8
                
                #to make things more efficient all these should be preallocated
                body_shape = simple_threndering.build_rectangle_hw(body_size, body_size).to(device=draw_device)
                thigh_shape = simple_threndering.build_rectangle_hw(thigh_width, thigh_length+thigh_width, -thigh_length/2).to(device=draw_device)
                shin_shape = simple_threndering.build_rectangle_hw(shin_width, shin_length+shin_width, -shin_length/2).to(device=draw_device)
                shapes_Nxy = [body_shape, thigh_shape, shin_shape]
                body_color = th.as_tensor([[128,128,128]], dtype=th.uint8, device=draw_device)
                thigh_color = th.as_tensor([[192,192,192]], dtype=th.uint8, device=draw_device)
                shin_color = th.as_tensor([[255,255,255]], dtype=th.uint8, device=draw_device)
                support_shape = simple_threndering.build_rectangle_hw(support_height,support_width).to(device=draw_device)
                support_color = th.as_tensor([[255,0,0]], dtype=th.uint8, device=draw_device)

                # rpos, hpos, kpos = 0.9, -3.14159/8, 3.14159/16
                rpos, hpos, kpos = bstate[0][self.BASE_STATE_IDXS.HIP_POS_Z, self.BASE_STATE_IDXS.HIP_JOINT_POS, self.BASE_STATE_IDXS.KNEE_JOINT_POS]
                s1x, s1y, s2x, s2y = bstate[0][self.BASE_STATE_IDXS.SUPPORT1_X,
                                            self.BASE_STATE_IDXS.SUPPORT1_Z,
                                            self.BASE_STATE_IDXS.SUPPORT2_X,
                                            self.BASE_STATE_IDXS.SUPPORT2_Z]
                
                # ggLog.info(f"rpos, kpos, hpos = {rpos, kpos, hpos}")
                image_chw = simple_threndering.draw_chain(  images_bchw = image_chw.unsqueeze(0), 
                                                shapes_BNxy=[s.unsqueeze(0) for s in shapes_Nxy],
                                                joints_BNax=th.as_tensor( [[[0,              rpos],
                                                                            [ hpos-3.14159,  thigh_length],
                                                                            [ -kpos,          shin_length] ]], device=draw_device),
                                                chain_colors_Brgb=[body_color,
                                                                thigh_color,
                                                                shin_color],
                                                scale=1,
                                                origin_xya=th.as_tensor([0,draw_yoffset, 3.14159/2], device=draw_device)).squeeze()
                image_chw = simple_threndering.draw_shapes(images_bchw = image_chw.unsqueeze(0),
                                                        shapes_BNxy=support_shape.unsqueeze(0),
                                                            transform_Bxya=th.as_tensor([-s1x,s1y+draw_yoffset,0.], device=draw_device).unsqueeze(0),
                                                            color_Brgb=support_color).squeeze()
                image_chw = simple_threndering.draw_shapes(images_bchw = image_chw.unsqueeze(0),
                                                        shapes_BNxy=support_shape.unsqueeze(0),
                                                            transform_Bxya=th.as_tensor([-s2x,s2y+draw_yoffset,0.], device=draw_device).unsqueeze(0),
                                                            color_Brgb=support_color).squeeze()
                # ggLog.info(f"image_chw 2 = {image_chw.size()}")
                img = image_chw.permute(1,2,0)
                # ggLog.info(f"img = {img}")
                # ggLog.info(f"nonzero = {img.count_nonzero()}")
                img = img.cpu().numpy()
                return img, self._last_step_simtime


    def getUiRendering(self) -> Tuple[Union[np.ndarray, th.Tensor, None], float]:
        try:
            if not self._configuration.use_threnderer:
                return self._render_image(self._current_state[self.STATE_BASE])            
            else:
                thrender_img, t = self._render_image(self._current_state[self.STATE_BASE])
                real_img, t = self._render_image(self._current_state[self.STATE_BASE], th_render=False)
                return np.hstack([thrender_img,real_img]), t
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return None, -1


    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        stacked_part =  state[self.STATE_BASE][:self._configuration.frame_stack_length,self._stacked_obs_part].detach().clone()
        stacked_part = stacked_part.flatten()
        constant_part = state[self.STATE_BASE][0,self._constant_obs_part]
        vec_obs = th.cat([stacked_part,constant_part])
        img_obs = state[self.STATE_IMG]
        if self._configuration.obs_only_vec:
            return {self.STATE_BASE : vec_obs}
        elif self._configuration.obs_only_img:
            return {self.STATE_IMG : img_obs}
        else:
            return {self.STATE_IMG : img_obs,
                    self.STATE_BASE : vec_obs}
            
    def _update_state(self):
        # ggLog.info(f"_stepCounter = {self._stepCounter}")
        self._stats.last_step_got_state = self._stepCounter
        
        jstates = self._environmentController.getJointsState(requestedJoints=[self._rail_joint, self._knee_joint, self._hip_joint])
        lstates : Dict[Tuple[str,str],LinkState] = self._environmentController.getLinksState(requestedLinks = [self._thigh_com_link,
                                                                            self._shin_com_link,
                                                                            self._thigh_base_link])
        hip_height = lstates[self._thigh_base_link].pose.position[2]
        hip_vel_z = lstates[self._thigh_base_link].pos_velocity_xyz[2]

        # n = '\n'
        # ggLog.info(f"\nlstates = \n{n.join([str(i) for i in lstates.items()])}")
        # ggLog.info(f"contacts == {n.join([str(c) for c in contacts])}")
        # thigh_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
        # thigh_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
        # thigh_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([0.0,0.0,1.0], device=self._th_device))[1])
        # shin_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
        # shin_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
        # shin_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]].to(self._th_device),
        #                                                                     th.tensor([0.0,0.0,1.0], device=self._th_device))[1])

        if self._configuration.use_contacts:
            if not isinstance(self._environmentController, PyBulletAdapter):
                raise RuntimeError(f"Required to use contacts, but environment adapter does not support contacts")
            contacts = self._environmentController.get_contacts()
            impulses = []
            forces = []
            clean_contacts = []
            for simstep_contacts in contacts:
                nonzero_contacts = [contact for contact in simstep_contacts if contact[3]!=0]
                if len(nonzero_contacts)>0:
                    clean_contacts.append(nonzero_contacts)
            contacts = clean_contacts
            for simsteps_contacts in contacts:
                forces   += [contact[3] for contact in simsteps_contacts]
                impulses += [contact[3]*contact[4] for contact in simsteps_contacts]
            abs_impulses = [abs(i) for i in impulses]
            abs_forces = [abs(i) for i in forces]
            abs_forces_sum = sum(abs_forces)
            abs_forces_num = len(abs_forces)
            abs_impulses_sum = sum(abs_impulses)
            abs_impulses_sum_avg = self._configuration.impulses_avg_alpha*self._stats.last_abs_impulses_sum_avg + self._configuration.impulses_avg_alpha*abs_impulses_sum
        else:
            abs_impulses_sum = -1
            abs_forces_sum = -1
            abs_forces_num = -1
            abs_impulses_sum_avg = -1
        # ggLog.info(f"jstates = {jstates}")

        bstate_history = self._current_state[self.STATE_BASE]
        prev_bstate = unnormalize(bstate_history[0],self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])                
        if len(self._current_state)!=0 and prev_bstate[self.BASE_STATE_IDXS.SAFETY_TRIGGERED] > 0:
            safety_triggered = True
        else:
            jstate_th = th.as_tensor([jstates[self._hip_joint].position[0],
                                    jstates[self._hip_joint].rate[0],
                                    jstates[self._hip_joint].effort[0],
                                    jstates[self._knee_joint].position[0],
                                    jstates[self._knee_joint].rate[0],
                                    jstates[self._knee_joint].effort[0]])
            j_safety_lims = th.as_tensor([self._configuration.position_safety_limits_hip,
                                        self._configuration.velocity_safety_limits_hip,
                                        self._configuration.torque_safety_limits_hip,
                                        self._configuration.position_safety_limits_knee,
                                        self._configuration.velocity_safety_limits_knee,
                                        self._configuration.torque_safety_limits_knee])
            triggered_limits = th.logical_or(jstate_th < j_safety_lims[:,0], jstate_th > j_safety_lims[:,1])
            safety_triggered = th.any(triggered_limits)
            if safety_triggered:       
                elements = ["hip_pos","hip_vel","hip_eff","knee_pos","knee_vel","knee_eff"]
                triggered = []
                for i in range(len(elements)):
                    if triggered_limits[i]:
                        triggered.append(elements[i])
                ggLog.info(f"SAFETY TRIGGERED:\n"
                            f"    triggered      = {triggered}\n"
                            f"    jstate_th      = {jstate_th}\n"
                            f"    j_safety_lims  = {j_safety_lims} ")




        goal_dist = abs(hip_height - self._current_episode_config.hip_goal_z)
        prev_goal_dist = prev_bstate[self.BASE_STATE_IDXS.SMOOTHED_GOAL_DIST]
        alpha = self._configuration.goal_dist_exp_smoothing_1s**(self._configuration.stepLength_sec)
        # ggLog.info(f"prev_goal_dist = {prev_goal_dist}")
        # ggLog.info(f"alpha = {alpha}")
        if prev_goal_dist > 0:
            smoothed_goal_dist = goal_dist*(1-alpha) + prev_goal_dist*alpha
        else:
            smoothed_goal_dist = 1
        # ggLog.info(f"smoothed_goal_dist = {smoothed_goal_dist}")

        for i in range(bstate_history.size()[0]-2):
            bstate_history[i+1] = bstate_history[i]
        # probably slower but safer assignment
        dictstate : dict[int,SupportsFloat] = {   self.BASE_STATE_IDXS.HIP_JOINT_POS : jstates[self._hip_joint].position[0],
                        self.BASE_STATE_IDXS.HIP_JOINT_VEL : jstates[self._hip_joint].rate[0],
                        self.BASE_STATE_IDXS.HIP_JOINT_EFFORT : jstates[self._hip_joint].effort[0],
                        self.BASE_STATE_IDXS.KNEE_JOINT_POS : jstates[self._knee_joint].position[0],
                        self.BASE_STATE_IDXS.KNEE_JOINT_VEL : jstates[self._knee_joint].rate[0],
                        self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT : jstates[self._knee_joint].effort[0],
                        self.BASE_STATE_IDXS.HIP_POS_Z : hip_height,
                        self.BASE_STATE_IDXS.HIP_VEL_Z : hip_vel_z,
                        self.BASE_STATE_IDXS.SUPPORT1_X : self._current_episode_config.support1_pos_x,
                        self.BASE_STATE_IDXS.SUPPORT1_Z : self._current_episode_config.support1_pos_z,
                        self.BASE_STATE_IDXS.SUPPORT2_X : self._current_episode_config.support2_pos_x,
                        self.BASE_STATE_IDXS.SUPPORT2_Z : self._current_episode_config.support2_pos_z,
                        self.BASE_STATE_IDXS.HIP_GOAL_Z : self._current_episode_config.hip_goal_z,
                        self.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT : self._configuration.reward_torque_limit_weight,
                        self.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT : self._configuration.reward_position_limit_weight,
                        self.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT : self._configuration.reward_velocity_weight,
                        self.BASE_STATE_IDXS.REWARD_ENERGY_WEIGHT : self._configuration.reward_energy_weight,
                        self.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT : self._configuration.reward_tracking_weight,
                        self.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT : self._configuration.reward_torque_weight,
                        self.BASE_STATE_IDXS.REWARD_CONTACTS_WEIGHT : self._current_episode_config.reward_contacts_weights,
                        self.BASE_STATE_IDXS.REWARD_IMPULSE_THRESHOLD : self._configuration.reward_max_impulse,
                        self.BASE_STATE_IDXS.KNEE_TORQUE_CMD_SCALE : self._configuration.torque_command_scale_knee,
                        self.BASE_STATE_IDXS.HIP_TORQUE_CMD_SCALE : self._configuration.torque_command_scale_hip,
                        self.BASE_STATE_IDXS.HIP_POS_REF : self._last_sent_pvesd[self._hip_joint][0],
                        self.BASE_STATE_IDXS.HIP_VEL_REF : self._last_sent_pvesd[self._hip_joint][1],
                        self.BASE_STATE_IDXS.HIP_EFFORT_REF : self._last_sent_pvesd[self._hip_joint][2],
                        self.BASE_STATE_IDXS.HIP_STIFFNESS : self._last_sent_pvesd[self._hip_joint][3],
                        self.BASE_STATE_IDXS.HIP_DAMPING : self._last_sent_pvesd[self._hip_joint][4],
                        self.BASE_STATE_IDXS.KNEE_POS_REF : self._last_sent_pvesd[self._knee_joint][0],
                        self.BASE_STATE_IDXS.KNEE_VEL_REF : self._last_sent_pvesd[self._knee_joint][1],
                        self.BASE_STATE_IDXS.KNEE_EFFORT_REF : self._last_sent_pvesd[self._knee_joint][2],
                        self.BASE_STATE_IDXS.KNEE_STIFFNESS : self._last_sent_pvesd[self._knee_joint][3],
                        self.BASE_STATE_IDXS.KNEE_DAMPING : self._last_sent_pvesd[self._knee_joint][4],
                        self.BASE_STATE_IDXS.THIGH_VEL_X : lstates[self._thigh_com_link].pos_velocity_xyz[0],
                        self.BASE_STATE_IDXS.THIGH_VEL_Y : lstates[self._thigh_com_link].pos_velocity_xyz[1],
                        self.BASE_STATE_IDXS.THIGH_VEL_Z : lstates[self._thigh_com_link].pos_velocity_xyz[2],
                        self.BASE_STATE_IDXS.THIGH_ANG_VEL_X : lstates[self._thigh_com_link].ang_velocity_xyz[0],
                        self.BASE_STATE_IDXS.THIGH_ANG_VEL_Y : lstates[self._thigh_com_link].ang_velocity_xyz[1],
                        self.BASE_STATE_IDXS.THIGH_ANG_VEL_Z : lstates[self._thigh_com_link].ang_velocity_xyz[2],
                        self.BASE_STATE_IDXS.SHIN_VEL_X : lstates[self._shin_com_link].pos_velocity_xyz[0],
                        self.BASE_STATE_IDXS.SHIN_VEL_Y : lstates[self._shin_com_link].pos_velocity_xyz[1],
                        self.BASE_STATE_IDXS.SHIN_VEL_Z : lstates[self._shin_com_link].pos_velocity_xyz[2],
                        self.BASE_STATE_IDXS.SHIN_ANG_VEL_X : lstates[self._shin_com_link].ang_velocity_xyz[0],
                        self.BASE_STATE_IDXS.SHIN_ANG_VEL_Y : lstates[self._shin_com_link].ang_velocity_xyz[1],
                        self.BASE_STATE_IDXS.SHIN_ANG_VEL_Z : lstates[self._shin_com_link].ang_velocity_xyz[2],
                        # lstates[self._thigh_com_link].pose.position[0],
                        # lstates[self._thigh_com_link].pose.position[1],
                        # lstates[self._thigh_com_link].pose.position[2],
                        # thigh_ang_pos_x,
                        # thigh_ang_pos_y,
                        # thigh_ang_pos_z,
                        # lstates[self._shin_com_link].pose.position[0],
                        # lstates[self._shin_com_link].pose.position[1],
                        # lstates[self._shin_com_link].pose.position[2],
                        # shin_ang_pos_x,
                        # shin_ang_pos_y,
                        # shin_ang_pos_z,
                        self.BASE_STATE_IDXS.IMPULSES_SUM : abs_impulses_sum,
                        self.BASE_STATE_IDXS.FORCES_SUM : abs_forces_sum,
                        self.BASE_STATE_IDXS.FORCES_NUM : abs_forces_num,
                        self.BASE_STATE_IDXS.IMPULSES_SUM_AVG : abs_impulses_sum_avg,
                        self.BASE_STATE_IDXS.SAFETY_TRIGGERED : 1 if safety_triggered else 0,
                        self.BASE_STATE_IDXS.SMOOTHED_GOAL_DIST : smoothed_goal_dist}
        bstate_history[0][list(dictstate.keys())] = th.as_tensor(list(dictstate.values()),
                                                                 dtype=self._configuration.obs_dtype,
                                                                 device=self._configuration.th_device)
        
        # self._robot_model.set_joint_pose(np.array([ jstates[self._rail_joint].position[0],
        #                                             jstates[self._hip_joint].position[0],
        #                                             jstates[self._knee_joint].position[0]]))
        # ggLog.info(f"sim shin_pose = {lstates[self._shin_com_link].pose.array_xyz_xyzw()}")
        # ggLog.info(f"comp shin_pose = {self._robot_model.get_frame_poses()[self._shin_com_link[1]]}")

        # ggLog.info(f"current_vstate = {current_vstate}")
        bstate_history[0] = normalize(bstate_history[0],self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])                
        bstate_history[0][self._stacked_obs_part] += adarl.utils.utils.randn_like(bstate_history[0][self._stacked_obs_part],
                                                                                mu  = self._current_episode_config.obs_noise_mustd[0],
                                                                                std = self._current_episode_config.obs_noise_mustd[1],
                                                                                generator=self._rng)
        
        astate_history = self._current_state[self.STATE_ACT]
        for i in range(astate_history.size()[0]-2):
            astate_history[i+1] = astate_history[i]
        astate_history[0] = self._last_out_action
        
        if self._stepCounter == 0:
            bstate_history[1:] = bstate_history[0].expand(bstate_history.size()[0]-1,-1)
            astate_history[1:] = astate_history[0].expand(astate_history.size()[0]-1,-1)

        if self._configuration.obs_only_vec:
            istate = th.empty(size=(0,), dtype = th.uint8, device = self._configuration.th_device)
        else:
            import torchvision.transforms.functional # only import it if needed
            istate, _ = self._render_image(bstate_history)
            # ggLog.info(f"img size = {istate.size()}")

        self._current_state = {self.STATE_BASE : bstate_history.detach().clone(),
                            self.STATE_ACT  : astate_history.detach().clone(),
                            self.STATE_IMG  : istate}
        
    def getState(self) -> Dict[Any, th.Tensor]:
        """Update and return the current state
        """                
        return self._current_state

    def _update_dbg_info(self):
        rew_dbg_info = {}
        r = self.computeReward({},
                               self._current_state, 
                               th.tensor([]), 
                               env_conf=self.get_configuration(),
                               dbg_info=rew_dbg_info)
        if self._configuration.use_contacts:
            if not isinstance(self._environmentController, PyBulletAdapter):
                raise RuntimeError(f"Required to use contacts, but environment adapter does not support contacts")
            contacts = self._environmentController.get_contacts()
            abs_impulses = []
            abs_contacts = []
            for simsteps_contacts in contacts:
                abs_impulses += [abs(contact[3]*contact[4]) for contact in simsteps_contacts]
                abs_contacts += [abs(contact[3]) for contact in simsteps_contacts]
            if len(abs_impulses)>0:
                self._stats.ep_max_abs_impulse = max(self._stats.ep_max_abs_impulse, max(abs_impulses))
                self._stats.ep_max_abs_impulses_sum = max(self._stats.ep_max_abs_impulses_sum, sum(abs_impulses))
                self._stats.ep_max_abs_contact = max(self._stats.ep_max_abs_contact, max(abs_contacts))
                self._stats.ep_max_abs_contacts_sum = max(self._stats.ep_max_abs_contacts_sum, sum(abs_contacts))
        vstate_unnorm = unnormalize(self._current_state[self.STATE_BASE][0],self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])
        goal_dist = abs(vstate_unnorm[self.BASE_STATE_IDXS.HIP_GOAL_Z]-vstate_unnorm[self.BASE_STATE_IDXS.HIP_POS_Z])
        self._stats.cumulative_dist_to_goal += goal_dist
        self._stats.cumulative_knee_torque += abs(vstate_unnorm[self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT])
        self._stats.cumulative_hip_torque += abs(vstate_unnorm[self.BASE_STATE_IDXS.HIP_JOINT_EFFORT])
        self._stats.max_knee_torque = th.maximum(self._stats.max_knee_torque, abs(vstate_unnorm[self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]))
        self._stats.max_hip_torque = th.maximum(self._stats.max_hip_torque, th.abs(vstate_unnorm[self.BASE_STATE_IDXS.HIP_JOINT_EFFORT]))
        self._stats.last_abs_impulses_sum = vstate_unnorm[self.BASE_STATE_IDXS.IMPULSES_SUM]
        self._stats.dists_to_goal[self._stepCounter%len(self._stats.dists_to_goal)] = goal_dist
        self._stats.cumulated_abs_impulses += self._stats.last_abs_impulses_sum
        self._reward_dbg_info = rew_dbg_info
        return rew_dbg_info

    def performStep(self):
        super().performStep()
        self._update_state()
        self._update_dbg_info()
        self._last_step_simtime = self._environmentController.getEnvTimeFromReset()





    def getInfo(self,state) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        # ggLog.info(f"getInfo(): {self._stepCounter}")
        # i["step_count"] = self._stepCounter
        bstate = state[self.STATE_BASE][0]
        bstate_unnorm = unnormalize(bstate,self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])
        i["hip_goal_z"] = bstate_unnorm[self.BASE_STATE_IDXS.HIP_GOAL_Z]
        i["avg_dist"] = self._stats.cumulative_dist_to_goal/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg10_dist"] = th.mean(self._stats.dists_to_goal)
        i["avg_knee_torque"] = self._stats.cumulative_knee_torque/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg_hip_torque"] = self._stats.cumulative_hip_torque/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg_abs_impulse"] = self._stats.cumulated_abs_impulses/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["max_abs_impulse"] = self._stats.ep_max_abs_impulse
        i["max_abs_impulses_sum"] = self._stats.ep_max_abs_impulses_sum
        i["max_abs_normimps_sum"] = self._stats.ep_max_abs_impulses_sum/self._configuration.stepLength_sec
        i["max_abs_contact"] = self._stats.ep_max_abs_contact
        i["max_abs_contacts_sum"] = self._stats.ep_max_abs_contacts_sum
        i["max_abs_normconts_sum"] = self._stats.ep_max_abs_contacts_sum/self._configuration.stepLength_sec
        i["max_knee_torque"] = self._stats.max_knee_torque
        i["max_hip_torque"] = self._stats.max_hip_torque
        i["impulses_sum"] = self._stats.last_abs_impulses_sum
        i["step_count"] = self._stepCounter
        i["thigh_vel_x_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.THIGH_VEL_X,self.BASE_STATE_IDXS.THIGH_VEL_Z]]
        i["shin_vel_x_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.SHIN_VEL_X,self.BASE_STATE_IDXS.SHIN_VEL_Z]]
        i["hip_joint_vel"] = bstate_unnorm[self.BASE_STATE_IDXS.HIP_JOINT_VEL]
        i["thigh_ang_vel_y"] = bstate_unnorm[self.BASE_STATE_IDXS.THIGH_ANG_VEL_Y]
        # i["thigh_pos_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.THIGH_POS_Z]]
        # i["shin_pos_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.SHIN_POS_Z]]
        statenames = [e.name for e in self.BASE_STATE_IDXS]
        stateindxs = [e.value for e in self.BASE_STATE_IDXS]
        i["action"] = self._last_out_action
        i["cbstate_norm"] = bstate[stateindxs]
        i["cbstate"] = bstate_unnorm[stateindxs]
        i["cbstate_labels"] = th.as_tensor([list(n.encode("utf-8").ljust(16)[:16]) for n in statenames], dtype=th.uint8) # ugly, but simple
        i.update(self._reward_dbg_info)
        # i["config"] = dataclasses.asdict(self._configuration)
        i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        i["safety_triggered"] = bstate_unnorm[self.BASE_STATE_IDXS.SAFETY_TRIGGERED]
        i["success"] = i["avg10_dist"] < 0.05
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i

    def get_configuration(self):
        return dataclasses.asdict(self._configuration)
    
        
    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.as_tensor(False, device=self._configuration.th_device)
        r = state[self.STATE_BASE][0][self.BASE_STATE_IDXS.SAFETY_TRIGGERED] > 0
        if r:
            ggLog.info(f"Terminated at step {self._stepCounter}")
        return r
    
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
