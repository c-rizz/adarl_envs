#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


from __future__ import annotations
import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List, Literal
import adarl.utils.dbg.ggLog as ggLog
import random
import adarl.utils.spaces as spaces

from adarl.envs.ControlledEnv import ControlledEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState, LinkState, quat_swing_twist_decomposition, quat_angle, MoveFailError, exc_to_str
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import torch as th
import adarl.utils.utils
from enum import IntEnum
from adarl.adapters.BaseAdapter import BaseAdapter
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
import dataclasses
from dataclasses import dataclass
import time



class LegJumpEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""

    metadata = {'render.modes': ['rgb_array']}
    VECTOR_PART = "vec"
    IMAGE_PART = "img"
    STATE = IntEnum("STATE", [ "HIP_JOINT_POS",
                            "HIP_JOINT_VEL",
                            "HIP_JOINT_EFFORT",
                            "KNEE_JOINT_POS",
                            "KNEE_JOINT_VEL",
                            "KNEE_JOINT_EFFORT",
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
                            "THIGH_POS_X",
                            "THIGH_POS_Y",
                            "THIGH_POS_Z",
                            "THIGH_ANG_POS_X",
                            "THIGH_ANG_POS_Y",
                            "THIGH_ANG_POS_Z",
                            "SHIN_POS_X",
                            "SHIN_POS_Y",
                            "SHIN_POS_Z",
                            "SHIN_ANG_POS_X",
                            "SHIN_ANG_POS_Y",
                            "SHIN_ANG_POS_Z",
                            "IMPULSES_SUM",
                            "FORCES_SUM",
                            "FORCES_NUM",
                            "IMPULSES_SUM_AVG",
                            "SAFETY_TRIGGERED"], start=0)
    
    EPISODE_CONFIG = IntEnum("EPISODE_CONFIG", ["HIP_GOAL_Z",
                                                "SUPPORT1_POS_X",
                                                "SUPPORT1_POS_Z",
                                                "SUPPORT2_POS_X",
                                                "SUPPORT2_POS_Z",
                                                "REWARD_CONTACTS_WEIGHT"], start=0)
    
    CONTROL_MODES = IntEnum("EPISODE_CONFIG", [ "VELOCITY",
                                                "TORQUE",
                                                "POSITION",
                                                "IMPEDANCE",
                                                "IMPEDANCE_NO_GAINS",
                                                "POSITION_AND_TORQUES",
                                                "POSITION_AND_GAINS"], start=0)


    @dataclass
    class EpisodeConfiguration:
        hip_goal_z : th.Tensor
        support1_pos_x : th.Tensor
        support1_pos_z : th.Tensor
        support2_pos_x : th.Tensor
        support2_pos_z : th.Tensor
        reward_contacts_weights : th.Tensor
        obs_noise_mustd : th.Tensor

    @dataclass
    class EnvConfiguration:
        reward_contacts_weight : float
        reward_energy_weight : float
        reward_max_impulse : float
        reward_position_limit_weight : float
        reward_torque_limit_weight : float
        reward_torque_weight : float
        reward_tracking_weight : float
        reward_velocity_weight : float
        action_exp_smoothing_1s : float
        stepLength_sec : float
        position_phys_limits_hip : Tuple[float,float]
        position_phys_limits_knee : Tuple[float,float]
        torque_phys_limits_hip : Tuple[float,float]
        torque_phys_limits_knee : Tuple[float,float]
        velocity_phys_limits_hip : Tuple[float,float]
        velocity_phys_limits_knee : Tuple[float,float]
        position_safety_limits_hip : Tuple[float,float]
        position_safety_limits_knee : Tuple[float,float]
        torque_safety_limits_hip : Tuple[float,float]
        torque_safety_limits_knee : Tuple[float,float]
        velocity_safety_limits_hip : Tuple[float,float]
        velocity_safety_limits_knee : Tuple[float,float]
        torque_command_scale_hip : float
        torque_command_scale_knee : float
        velocity_command_scale_hip : float
        velocity_command_scale_knee : float
        vstate_minmax : th.Tensor
        reward_scale : float
        ep_obs_noise_mustd : th.Tensor
        step_obs_noise_std : th.Tensor
        action_noise_mustd : th.Tensor
        stop_on_safety : bool

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
                    stop_on_safety = True):
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



        

        self._knee_joint = ("leg","knee_joint_1")
        self._hip_joint = ("leg","hip_joint_1")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","foot_link")
        self._thigh_base_link = ("leg", "thigh_link_base")
        self._shin_base_link = ("leg", "shin_link_base")
        self._thigh_com_link = ("leg", "thigh_link_com")
        self._shin_com_link = ("leg", "shin_link_com")
        self._rendering_cam_name = "simple_camera"

        control_mode = control_mode.strip().lower()
        if control_mode == "velocity":
            self._control_mode = self.CONTROL_MODES.VELOCITY
        elif control_mode == "torque":
            self._control_mode = self.CONTROL_MODES.TORQUE
        elif control_mode == "position":
            self._control_mode = self.CONTROL_MODES.POSITION
        elif control_mode == "impedance":
            self._control_mode = self.CONTROL_MODES.IMPEDANCE
        elif control_mode == "impedance_no_gains":
            self._control_mode = self.CONTROL_MODES.IMPEDANCE_NO_GAINS
        elif control_mode == "position_and_torque":
            self._control_mode = self.CONTROL_MODES.POSITION_AND_TORQUES
        elif control_mode == "position_and_gains":
            self._control_mode = self.CONTROL_MODES.POSITION_AND_GAINS
        self._platform_randomization = platform_randomization


        self._obs_only_vec = obs_only_vec
        self._obs_only_img = obs_only_img
        self._obs_img_height = obs_img_height
        self._obs_img_width = obs_img_width
        self._th_device = th_device
        self._rendering_enabled = True
        self._start_height = 0.9
        self._show_goal = True
        self._rng = th.Generator(device=self._th_device)
        self._wall_sim_speed = wall_sim_speed
        self._original_max_epsteps = maxStepsPerEpisode
        self._use_contacts = use_contacts
        self._real = real
        self._obs_dtype = th.float32
        if self._use_contacts == False and reward_contacts_weight!=0:
            raise RuntimeError(f"use_contacts is False but reward_contacts_weight is not zero")

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
        halflife_s = 0.05
        action_exp_smoothing_1s = 0.5**(1/halflife_s)


        self._configuration = LegJumpEnv.EnvConfiguration(  reward_contacts_weight = reward_contacts_weight,
                                                            reward_energy_weight = reward_energy_weight,
                                                            reward_max_impulse = 10,
                                                            reward_position_limit_weight = reward_position_limit_weight,
                                                            reward_torque_limit_weight = reward_torque_limit_weight,
                                                            reward_torque_weight = reward_torque_weight,
                                                            reward_tracking_weight = reward_tracking_weight,
                                                            reward_velocity_weight = reward_velocity_weight,
                                                            action_exp_smoothing_1s = action_exp_smoothing_1s, 
                                                            stepLength_sec=stepLength_sec,
                                                            position_phys_limits_hip =  (-2.4, 2.4),
                                                            position_phys_limits_knee = (-2.4, 2.4),
                                                            torque_phys_limits_hip =  (-112, 112),
                                                            torque_phys_limits_knee = (-112, 112),
                                                            velocity_phys_limits_hip =  (-20, 20),
                                                            velocity_phys_limits_knee = (-20, 20),
                                                            position_safety_limits_hip =  (-2.3, 2.3),
                                                            position_safety_limits_knee = (-2.3, 2.3),
                                                            torque_safety_limits_hip =  (-100, 100),
                                                            torque_safety_limits_knee = (-100, 100),
                                                            velocity_safety_limits_hip =  (-18, 18),
                                                            velocity_safety_limits_knee = (-18, 18),
                                                            torque_command_scale_hip = 100,
                                                            torque_command_scale_knee = 100,
                                                            velocity_command_scale_hip = 20,
                                                            velocity_command_scale_knee = 20,
                                                            vstate_minmax = th.empty((0,)),
                                                            reward_scale = reward_scale,
                                                            ep_obs_noise_mustd=th.empty((0,)),
                                                            step_obs_noise_std=th.empty((0,)),
                                                            action_noise_mustd=th.empty((0,)),
                                                            stop_on_safety = stop_on_safety)
        if self._control_mode == self.CONTROL_MODES.IMPEDANCE:
            action_len = 10 
        elif self._control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            action_len = 6
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            action_len = 4
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_GAINS:
            action_len = 4
        elif self._control_mode == self.CONTROL_MODES.TORQUE:
            action_len = 2
        elif self._control_mode == self.CONTROL_MODES.VELOCITY:
            action_len = 2
        elif self._control_mode == self.CONTROL_MODES.POSITION:
            action_len = 2
        else:
            raise RuntimeError(f"invalid control mode {self._control_mode}")
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=th.tensor(0),
                                                                       support1_pos_x=th.tensor(0),
                                                                       support1_pos_z=th.tensor(0),
                                                                       support2_pos_x=th.tensor(0),
                                                                       support2_pos_z=th.tensor(0),
                                                                       reward_contacts_weights=th.tensor(0),
                                                                       obs_noise_mustd=th.tensor(0))
        # max_dact_dt = 100 #max change in action, i.e. da/dt
        # self._max_act_change = th.tensor(max_dact_dt*stepLength_sec,dtype=th.float32, device=self._th_device)
        # self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)
        self._action_size = action_len
        self._last_out_actions = th.empty((0,))
        self._history_length = 4
        self._frame_stack_length = 3
        self._vstate_history = th.zeros((self._history_length, len(self.STATE)), dtype=th.float32, device=self._th_device)
        self._new_history = th.zeros_like(self._vstate_history) # preallocate this

        self._max_hip_height_reached = th.tensor(0)
        self._spawned = False
        self._last_step_got_state = -1
        self._cumulative_dist_to_goal = 0
        self._dists_to_goal = th.zeros(size=(int(maxStepsPerEpisode/10),), dtype=th.float32, device=self._th_device)
        self._cumulative_knee_torque = 0
        self._cumulative_hip_torque = 0
        self._max_knee_torque = th.tensor(0)
        self._max_hip_torque = th.tensor(0)
        self._cumulated_abs_impulses = 0
        self._last_abs_impulses_sum = 0
        self._impulses_avg_alpha = 0.5
        self._last_abs_impulses_sum_avg = 0.0
        self._ep_max_abs_impulse = -1
        self._ep_max_abs_impulses_sum = -1
        self._ep_max_abs_contact = -1
        self._ep_max_abs_contacts_sum = -1
        self._last_external_work = 0
        self._last_state = {}
        self._dbg_info = {}
        self._dbg_info["external_work"] = 0
        self._dbg_info["thigh_work"] = 0
        self._dbg_info["shin_work"] = 0
        self._dbg_info["thigh_joint_work"] = 0
        self._dbg_info["shin_joint_work"] = 0
        self._dbg_info["new_thigh_energy"] = 0
        self._dbg_info["new_shin_energy"] = 0
        self._dbg_info["new_slider_energy"] = 0
        self._dbg_info["slider_work"] = 0
        
        vstate_min_max = {  self.STATE.HIP_JOINT_POS : self._configuration.position_phys_limits_hip,
                            self.STATE.HIP_JOINT_VEL : self._configuration.velocity_phys_limits_hip,
                            self.STATE.HIP_JOINT_EFFORT : self._configuration.torque_phys_limits_hip,
                            self.STATE.KNEE_JOINT_POS : self._configuration.position_phys_limits_knee,
                            self.STATE.KNEE_JOINT_VEL : self._configuration.velocity_phys_limits_knee,
                            self.STATE.KNEE_JOINT_EFFORT : self._configuration.torque_phys_limits_knee,
                            self.STATE.HIP_POS_Z : [0,3],
                            self.STATE.HIP_VEL_Z : [-100,100],
                            self.STATE.SUPPORT1_X : [-2,2],
                            self.STATE.SUPPORT1_Z : [0,2],
                            self.STATE.SUPPORT2_X : [-2,2],
                            self.STATE.SUPPORT2_Z : [0,2],
                            self.STATE.HIP_GOAL_Z : [0,2],
                            self.STATE.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                            self.STATE.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                            self.STATE.REWARD_VELOCITY_WEIGHT : [0,10],
                            self.STATE.REWARD_ENERGY_WEIGHT : [0,10],
                            self.STATE.REWARD_TRACKING_WEIGHT : [0,10],
                            self.STATE.REWARD_TORQUE_WEIGHT : [0,10],
                            self.STATE.REWARD_CONTACTS_WEIGHT : [0,10],
                            self.STATE.REWARD_IMPULSE_THRESHOLD : [0,10],
                            self.STATE.KNEE_TORQUE_CMD_SCALE : [0,150],
                            self.STATE.HIP_TORQUE_CMD_SCALE : [0,150],

                            self.STATE.THIGH_VEL_X : [-100,100],
                            self.STATE.THIGH_VEL_Y : [-100,100],
                            self.STATE.THIGH_VEL_Z : [-100,100],
                            self.STATE.THIGH_ANG_VEL_X : [-100,100],
                            self.STATE.THIGH_ANG_VEL_Y : [-100,100],
                            self.STATE.THIGH_ANG_VEL_Z : [-100,100],
                            self.STATE.SHIN_VEL_X : [-100,100],
                            self.STATE.SHIN_VEL_Y : [-100,100],
                            self.STATE.SHIN_VEL_Z : [-100,100],
                            self.STATE.SHIN_ANG_VEL_X : [-100,100],
                            self.STATE.SHIN_ANG_VEL_Y : [-100,100],
                            self.STATE.SHIN_ANG_VEL_Z : [-100,100],
                            self.STATE.THIGH_POS_X : [-2,2],
                            self.STATE.THIGH_POS_Y : [-2,2],
                            self.STATE.THIGH_POS_Z : [-2,2],
                            self.STATE.THIGH_ANG_POS_X : [-100,100],
                            self.STATE.THIGH_ANG_POS_Y : [-100,100],
                            self.STATE.THIGH_ANG_POS_Z : [-100,100],
                            self.STATE.SHIN_POS_X : [-2,2],
                            self.STATE.SHIN_POS_Y : [-2,2],
                            self.STATE.SHIN_POS_Z : [-2,2],
                            self.STATE.SHIN_ANG_POS_X : [-100,100],
                            self.STATE.SHIN_ANG_POS_Y : [-100,100],
                            self.STATE.SHIN_ANG_POS_Z : [-100,100],
                            self.STATE.IMPULSES_SUM : [0,100],
                            self.STATE.FORCES_SUM : [0,1000],
                            self.STATE.FORCES_NUM : [0,1000],
                            self.STATE.IMPULSES_SUM_AVG : [0,100],
                            self.STATE.SAFETY_TRIGGERED : [0,1]}        
        self._configuration.vstate_minmax = th.tensor([vstate_min_max[k] for k in self.STATE], device = self._th_device)

        self._stacked_obs_part = th.as_tensor([ self.STATE.HIP_JOINT_POS,
                                                self.STATE.HIP_JOINT_VEL,
                                                self.STATE.HIP_JOINT_EFFORT,
                                                self.STATE.KNEE_JOINT_POS,
                                                self.STATE.KNEE_JOINT_VEL,
                                                self.STATE.KNEE_JOINT_EFFORT,
                                                self.STATE.HIP_POS_Z,
                                                self.STATE.HIP_VEL_Z,
                                                self.STATE.SUPPORT1_X,
                                                self.STATE.SUPPORT1_Z,
                                                self.STATE.SUPPORT2_X,
                                                self.STATE.SUPPORT2_Z], device=self._th_device)

        self._constant_obs_part = th.as_tensor([self.STATE.HIP_GOAL_Z,
                                                self.STATE.REWARD_TORQUE_LIMIT_WEIGHT,
                                                self.STATE.REWARD_POSITION_LIMIT_WEIGHT,
                                                self.STATE.REWARD_VELOCITY_WEIGHT,
                                                self.STATE.REWARD_ENERGY_WEIGHT,
                                                self.STATE.REWARD_TRACKING_WEIGHT,
                                                self.STATE.REWARD_TORQUE_WEIGHT,
                                                self.STATE.REWARD_CONTACTS_WEIGHT,
                                                self.STATE.REWARD_IMPULSE_THRESHOLD], device=self._th_device)
        self._vec_obs_size = self._stacked_obs_part.size()[0]*self._frame_stack_length + self._constant_obs_part.size()[0]

        vec_obs_space_high = np.array( [1.0]*self._vec_obs_size)
        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        
        self._img_channels = 3 if rgb else 1
        img_shape_chw = (self._img_channels,self._obs_img_height,self._obs_img_width)
        img_observation_space = spaces.gym_spaces.Box(low=0, high=255, shape=img_shape_chw, dtype=np.uint8)

        state_space = spaces.gym_spaces.Dict({self.VECTOR_PART: spaces.gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._history_length,len(LegJumpEnv.STATE),)),
                                                   self.IMAGE_PART: img_observation_space})
        
        if self._obs_only_vec:
            observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space})     
        elif self._obs_only_img:
            observation_space = spaces.gym_spaces.Dict({ self.IMAGE_PART  : img_observation_space})
        else:
            observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space,
                                                              self.IMAGE_PART  : img_observation_space})
            
        action_space_high = np.array([1]*action_len)
        action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high, seed=seed)

        step_obs_noise_std = th.tensor(step_obs_noise_std)
        step_obs_noise_std = step_obs_noise_std.expand((len(self._stacked_obs_part),))
        ep_obs_noise_mustd = th.tensor(ep_obs_noise_mustd)
        if ep_obs_noise_mustd.dim() == 1: ep_obs_noise_mustd = ep_obs_noise_mustd.unsqueeze(1)
        ep_obs_noise_mustd = ep_obs_noise_mustd.expand((2, len(self._stacked_obs_part)))

        self._configuration.ep_obs_noise_mustd = th.tensor(ep_obs_noise_mustd, dtype=self._obs_dtype, device=self._th_device)
        self._configuration.step_obs_noise_std = th.tensor(step_obs_noise_std, dtype=self._obs_dtype, device=self._th_device)
        self._configuration.action_noise_mustd = 0.0 * th.ones(size=(action_len,), dtype=th.float32, device=self._th_device)
        
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
        if self._use_contacts:
            self._environmentController.monitor_contacts([("leg",None,None,None)]) # Monitor the contacts between the leg and all the environment

        self._environmentController.startup()









    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    @staticmethod
    def _unnormalize(v, min, max):
        return min+(v+1)/2*(max-min)
    
    @staticmethod        
    def _normalize(value, min, max):
        return (value + (-min))/(max-min)*2-1


    def submitAction(self, action : th.Tensor) -> None:
        # ggLog.info(f"Submitting action {action}")
        action = th.as_tensor(action).detach().cpu()
        super().submitAction(action)
        dt = self._configuration.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        if self._actionsCounter != 0:
            action = action*(1-alpha) + self._last_out_actions[0]*alpha
        # action = th.clamp(action, min=self._last_out_action-self._max_act_change, max=self._last_out_action+self._max_act_change)
        action = th.clamp(action, min=-1, max=1)
        # action = th.tensor([0.,0.])
        self._last_out_actions[1:] = self._last_out_actions[:-1]
        self._last_out_actions[0] = action
        if self._control_mode == self.CONTROL_MODES.VELOCITY:
            # hvel = self._unnormalize(action[0],self._configuration.velocity_command_scale_hip,self._configuration.velocity_limits_hip[0],self._configuration.velocity_limits_hip[1])
            # kvel = self._unnormalize(action[1],self._configuration.velocity_command_scale_knee,self._configuration.velocity_limits_knee[0],self._configuration.velocity_limits_knee[1])
            hvel = action[0]*self._configuration.velocity_command_scale_hip
            kvel = action[1]*self._configuration.velocity_command_scale_knee
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (0,hvel.item(),0,0,30),
                                                                self._knee_joint:  (0,kvel.item(),0,0,30)})
        elif self._control_mode == self.CONTROL_MODES.POSITION:
            hpos = self._unnormalize(action[0],self._configuration.position_phys_limits_hip[0],self._configuration.position_phys_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_phys_limits_knee[0],self._configuration.position_phys_limits_knee[1])
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (hpos,0,0,300,30),
                                                                self._knee_joint:  (kpos,0,0,300,30)})
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            hpos = self._unnormalize(action[0],self._configuration.position_phys_limits_hip[0],self._configuration.position_phys_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_phys_limits_knee[0],self._configuration.position_phys_limits_knee[1])
            htorque = action[2]*self._configuration.torque_command_scale_hip
            ktorque = action[3]*self._configuration.torque_command_scale_knee
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (hpos,0,htorque.item(),300,30),
                                                                self._knee_joint:  (kpos,0,ktorque.item(),300,30)})
        elif self._control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            hpos = self._unnormalize(action[0],self._configuration.position_phys_limits_hip[0],self._configuration.position_phys_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_phys_limits_knee[0],self._configuration.position_phys_limits_knee[1])
            hvel = action[2]*self._configuration.velocity_command_scale_hip
            kvel = action[3]*self._configuration.velocity_command_scale_knee
            htorque = action[4]*self._configuration.torque_command_scale_hip
            ktorque = action[5]*self._configuration.torque_command_scale_knee
            
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (hpos,hvel.item(),htorque.item(),300,30),
                                                                self._knee_joint:  (kpos,kvel.item(),ktorque.item(),300,30)})
        elif self._control_mode == self.CONTROL_MODES.IMPEDANCE:
            hpos = self._unnormalize(action[0],self._configuration.position_phys_limits_hip[0],self._configuration.position_phys_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_phys_limits_knee[0],self._configuration.position_phys_limits_knee[1])
            hvel = action[2]*self._configuration.velocity_command_scale_hip
            kvel = action[3]*self._configuration.velocity_command_scale_knee
            htorque = action[4]*self._configuration.torque_command_scale_hip
            ktorque = action[5]*self._configuration.torque_command_scale_knee
            hpgain = action[6]*500
            kpgain = action[7]*500
            hvgain = action[8]*100
            kvgain = action[9]*100
            
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (hpos,hvel.item(),htorque.item(),hpgain.item(),hvgain.item()),
                                                                self._knee_joint:  (kpos,kvel.item(),ktorque.item(),kpgain.item(),kvgain.item())})
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_GAINS:
            hpos = self._unnormalize(action[0],self._configuration.position_phys_limits_hip[0],self._configuration.position_phys_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_phys_limits_knee[0],self._configuration.position_phys_limits_knee[1])
            hpgain = (action[2]+1)/2*400
            kpgain = (action[3]+1)/2*400
            
            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :   (hpos,0,0,hpgain.item(),30),
                                                                self._knee_joint:   (kpos,0,0,kpgain.item(),30)})
        elif self._control_mode == self.CONTROL_MODES.TORQUE:
            htorque = action[0]*self._configuration.torque_command_scale_hip
            ktorque = action[1]*self._configuration.torque_command_scale_knee

            self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = 
                                                            {   self._hip_joint :  (0,0,htorque.item(),0,0),
                                                                self._knee_joint:  (0,0,ktorque.item(),0,0)})
            



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
                                                        vel_x=vstate_unnorm[LegJumpEnv.STATE.THIGH_VEL_X],
                                                        vel_z=vstate_unnorm[LegJumpEnv.STATE.THIGH_VEL_Z],
                                                        ang_vel_y=vstate_unnorm[LegJumpEnv.STATE.THIGH_ANG_VEL_Y])
        shin_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = shin_mass,
                                                        inertia_moment=1/12*shin_mass*shin_length**2,
                                                        vel_x=vstate_unnorm[LegJumpEnv.STATE.SHIN_VEL_X],
                                                        vel_z=vstate_unnorm[LegJumpEnv.STATE.SHIN_VEL_Z],
                                                        ang_vel_y=vstate_unnorm[LegJumpEnv.STATE.SHIN_ANG_VEL_Y])
        slider_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = slider_mass,
                                                        inertia_moment=0,
                                                        vel_x=0,
                                                        vel_z=vstate_unnorm[LegJumpEnv.STATE.HIP_VEL_Z],
                                                        ang_vel_y=0)
        thigh_pot_energy = thigh_mass*g*vstate_unnorm[LegJumpEnv.STATE.THIGH_POS_Z]
        shin_pot_energy = shin_mass*g*vstate_unnorm[LegJumpEnv.STATE.SHIN_POS_Z]
        slider_pot_energy = slider_mass*g*vstate_unnorm[LegJumpEnv.STATE.HIP_POS_Z]
        return thigh_kin_energy+thigh_pot_energy, shin_kin_energy+shin_pot_energy, slider_kin_energy+slider_pot_energy

    @staticmethod
    def computeReward(previousState : Dict[str,th.Tensor],
                      state : Dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : Optional[Dict[str,th.Tensor]] = None, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        vstate_norm = state[LegJumpEnv.VECTOR_PART][0]
        pvstate_norm = state[LegJumpEnv.VECTOR_PART][-1]

        normtorques = vstate_norm[[LegJumpEnv.STATE.HIP_JOINT_EFFORT,LegJumpEnv.STATE.KNEE_JOINT_EFFORT]]
        normvelocities = vstate_norm[[LegJumpEnv.STATE.HIP_JOINT_VEL,LegJumpEnv.STATE.KNEE_JOINT_VEL]]
        normpositions = vstate_norm[[LegJumpEnv.STATE.HIP_JOINT_POS,LegJumpEnv.STATE.KNEE_JOINT_POS]]
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

        vstate_un = LegJumpEnv._unnormalize(vstate_norm,env_conf["vstate_minmax"][:,0],env_conf["vstate_minmax"][:,1])
        pvstate_un = LegJumpEnv._unnormalize(pvstate_norm,env_conf["vstate_minmax"][:,0],env_conf["vstate_minmax"][:,1])

        goal_dist = th.abs(vstate_un[LegJumpEnv.STATE.HIP_GOAL_Z] - vstate_un[LegJumpEnv.STATE.HIP_POS_Z])
        # tracking_reward = 1 - goal_dist
        tracking_reward = 1/(1+goal_dist/0.05) # halves at 0.05m
        impulse_threshold = pvstate_un[LegJumpEnv.STATE.REWARD_IMPULSE_THRESHOLD]
        contacts_reward = th.clamp(-(vstate_un[LegJumpEnv.STATE.IMPULSES_SUM_AVG]/impulse_threshold)**4, min = -1)


        ktorque = vstate_un[LegJumpEnv.STATE.KNEE_JOINT_EFFORT]
        htorque = vstate_un[LegJumpEnv.STATE.HIP_JOINT_EFFORT]
        shin_rotation = vstate_un[LegJumpEnv.STATE.SHIN_ANG_POS_Y] - pvstate_un[LegJumpEnv.STATE.SHIN_ANG_POS_Y]
        thigh_rotation = vstate_un[LegJumpEnv.STATE.THIGH_ANG_POS_Y] - pvstate_un[LegJumpEnv.STATE.THIGH_ANG_POS_Y]

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

        if dbg_info is not None:
            dbg_info["external_work"] = external_work
            dbg_info["thigh_work"] = thigh_work
            dbg_info["shin_work"] = shin_work
            dbg_info["slider_work"] = slider_work
            dbg_info["thigh_joint_work"] = thigh_joint_work
            dbg_info["shin_joint_work"] = shin_joint_work
            dbg_info["new_thigh_energy"] = new_thigh_energy
            dbg_info["new_shin_energy"] = new_shin_energy
            dbg_info["new_slider_energy"] = new_slider_energy
        # ggLog.info(f"new_thigh_energy={new_thigh_energy}, old_thigh_energy={old_thigh_energy}, new_shin_energy={new_shin_energy}, old_shin_energy={old_shin_energy}")
        # ggLog.info(f"knee_work={knee_work}\t hip_work={hip_work}\t thigh_work={thigh_work}\t shin_work={shin_work}")

        reward_scale = env_conf["reward_scale"]
        tracking_reward         = reward_scale * tracking_reward
        torque_reward           = reward_scale * torque_reward
        torque_limit_reward     = reward_scale * torque_limit_reward
        velocity_reward         = reward_scale * velocity_reward
        position_limit_reward   = reward_scale * position_limit_reward
        global_energy_reward    = reward_scale * global_energy_reward
        contacts_reward         = reward_scale * contacts_reward

        if sub_rewards is not None:
            sub_rewards["tracking_reward"] = tracking_reward
            sub_rewards["torque_reward"] = torque_reward
            sub_rewards["torque_limit_reward"] = torque_limit_reward
            sub_rewards["velocity_reward"] = velocity_reward
            sub_rewards["position_limit_reward"] = position_limit_reward
            sub_rewards["energy_reward"] = global_energy_reward
            sub_rewards["contacts_reward"] = contacts_reward

        torque_lim_weight = pvstate_un[LegJumpEnv.STATE.REWARD_TORQUE_LIMIT_WEIGHT]
        position_lim_weight = pvstate_un[LegJumpEnv.STATE.REWARD_POSITION_LIMIT_WEIGHT]
        velocity_weight = pvstate_un[LegJumpEnv.STATE.REWARD_VELOCITY_WEIGHT]
        energy_weight = pvstate_un[LegJumpEnv.STATE.REWARD_ENERGY_WEIGHT]
        tracking_weight = pvstate_un[LegJumpEnv.STATE.REWARD_TRACKING_WEIGHT]
        torque_weight = pvstate_un[LegJumpEnv.STATE.REWARD_TORQUE_WEIGHT]
        contacts_weight = pvstate_un[LegJumpEnv.STATE.REWARD_CONTACTS_WEIGHT]

        return (tracking_weight*tracking_reward + 
                torque_lim_weight*torque_limit_reward + 
                velocity_weight*velocity_reward+
                position_lim_weight*position_limit_reward+
                energy_weight*global_energy_reward+
                torque_weight*torque_reward+
                contacts_weight * contacts_reward )


    def initializeEpisode(self, options = {}) -> None:

        self._last_state = {}
        if not self._spawned and isinstance(self._environmentController, BaseSimulationAdapter):
            
            # supp1_pos = [-0.1,0.2]
            # supp2_pos = [-0.15,0.4]

            leg_model_name = "leg"
            cam_model_name = "camera"
            leg_pose = build_pose(0,0,0,0,0,0,1)
            self._spawned = True
            if isinstance(self._environmentController, PyBulletAdapter):
                leg_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_simple.urdf.xacro")
                # import rospkg
                # leg_file = rospkg.RosPack().get_path("protoleg")+"/description/urdf/protoleg_test_rig.urdf.xacro"
                name = self._environmentController.spawn_model(model_file=leg_file,
                                                                model_name=leg_model_name,
                                                                pose=leg_pose,
                                                                model_format="urdf.xacro")
            self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro"),
                                                    model_name=cam_model_name,
                                                    pose=build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707),
                                                    model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._intendedStepLength_sec},
                                                    model_format="sdf.xacro")
            # ggLog.info(f"Model spawned with name {name}")

            if self._show_goal:
                self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/red_intangible_ball.urdf.xacro"),
                                                        model_name="red_ball",
                                                        pose=leg_pose,
                                                        model_format="urdf.xacro",
                                                        model_kwargs={"add_world_link":str(isinstance(self._environmentController, PyBulletAdapter))})
            self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support1",
                                                    pose=build_pose(-0.5, 0.3, 0.2, 0,0,0,1),
                                                    model_format="urdf.xacro",
                                                    model_kwargs={"add_world_link":str(isinstance(self._environmentController, PyBulletAdapter))})
            
            self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support2",
                                                    pose=build_pose(-0.5, 0.3, 0.4, 0,0,0,1),
                                                    model_format="urdf.xacro",
                                                    model_kwargs={"add_world_link":str(isinstance(self._environmentController, PyBulletAdapter))})

        
        self._max_hip_height_reached = th.tensor(0)
        self._dists_to_goal = th.zeros(size=(int(self._maxStepsPerEpisode/10),), dtype=th.float32, device=self._th_device)
        self._cumulative_dist_to_goal = 0
        self._cumulative_knee_torque = 0
        self._cumulative_hip_torque = 0
        self._max_knee_torque = th.tensor(0)
        self._max_hip_torque = th.tensor(0)
        self._cumulated_abs_impulses = 0
        self._last_abs_impulses_sum = 0
        self._ep_max_abs_impulse = 0.0
        self._ep_max_abs_impulses_sum = 0.0
        self._ep_max_abs_contact = 0.0
        self._ep_max_abs_contacts_sum = 0.0
        self._last_external_work = 0
        self._last_step_got_state = -1
        self._last_abs_impulses_sum_avg = 0.0
        self._last_out_actions = th.zeros(size=(10,self._action_size),dtype=self._obs_dtype,device=self._th_device)

        # Having conservative values here will not make the policy learn to behave nice in unfeasible cases
        # Having too broad value will have unfeasible cases in training
        min_stretch_z = 0.4
        max_stretch_z = 0.65
        max_jump_z = 0.6

        min_goal_z = min_stretch_z
        max_goal_z = max_jump_z + max_stretch_z
        hip_goal_z = min_goal_z + th.rand(size=(1,), generator=self._rng, device=self._th_device)*(max_goal_z-min_goal_z) # uniform(0.4,1.2)
        min_plat_z = hip_goal_z-max_stretch_z
        max_plat_z = th.min(th.as_tensor([max_jump_z, hip_goal_z-min_stretch_z]))

        if self._platform_randomization == "double":
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
        elif self._platform_randomization == "single":
            s1_xz = th.tensor([-0.1-0.125, -0.3]) # hide platform 
            s2_area = th.tensor([[0.20, 0.30],  # minx, maxx
                                 [min_plat_z, max_plat_z]], # miny, maxy
                                device=self._th_device)
            s2_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            s2_xz = s2_xz*(s2_area[:,1]-s2_area[:,0])+s2_area[:,0]
            s2_xz[0] = s2_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
        elif self._platform_randomization == "fixed":
            s1_xz = th.tensor([-0.1-0.125, 0.3])
            s2_xz = th.tensor([-0.15-0.125, 0.6])
        elif self._platform_randomization == "no_platforms":
            s1_xz = th.tensor([10, 10])
            s2_xz = th.tensor([10, 11])        
        else:
            raise RuntimeError(f"Invalid platform_randomization mode '{self._platform_randomization}'")


        reward_contacts_weights = self._sample(self._configuration.reward_contacts_weight,
                                                                                                self._rng,
                                                                                                self._th_device)
        maxStepsPerEpisode = self._original_max_epsteps
        # These override previous configs
        if options is not None:
            if "support1_pos_x" in options: s1_xz[0] = options["support1_pos_x"]
            if "support1_pos_z" in options: s1_xz[1] = options["support2_pos_z"]
            if "support2_pos_x" in options: s2_xz[0] = options["support2_pos_x"]
            if "support2_pos_z" in options: s2_xz[1] = options["support2_pos_z"]
            if "hip_goal_z" in options: hip_goal_z = options["hip_goal_z"]
            if "reward_contacts_weights" in options: reward_contacts_weights = options["reward_contacts_weights"]
            if "max_ep_steps" in options: maxStepsPerEpisode = options["max_ep_steps"]

        self._maxStepsPerEpisode = maxStepsPerEpisode
            
        obs_noise_mu = self._configuration.ep_obs_noise_mustd[0] + self._configuration.ep_obs_noise_mustd[1]*th.randn(size=(len(self._stacked_obs_part),),
                                                                                                                      generator=self._rng,
                                                                                                                      dtype=self._obs_dtype,
                                                                                                                      device=self._th_device)
        #min 0.4, max support2_z+0.6
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=hip_goal_z,
                                                                       support1_pos_x=s1_xz[0],
                                                                       support1_pos_z=s1_xz[1],
                                                                       support2_pos_x=s2_xz[0],
                                                                       support2_pos_z=s2_xz[1],
                                                                       reward_contacts_weights=reward_contacts_weights,
                                                                       obs_noise_mustd=th.stack([obs_noise_mu, self._configuration.step_obs_noise_std]))

        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            moved = False
            while not moved:
                ggLog.info(f"Cannot automatically initialize episode with non-simulated adapter. Lift up the robot and press ENTER.")
                input()
                if isinstance(self._environmentController, BaseJointPositionAdapter):
                    if self._current_episode_config.support2_pos_x > 0:
                        rpos, hpos, kpos = self._start_height, 3.4159/4,  3.14159/2
                    else:
                        rpos, hpos, kpos = self._start_height, -3.4159/4, -3.14159/2
                    try:
                        self._environmentController.moveToJointPoseSync({self._hip_joint:  hpos,
                                                                        self._knee_joint: kpos})
                    except MoveFailError as e:
                        ggLog.warn(f"Failed to move to joint position. Error = {exc_to_str(e)}")
            # raise RuntimeError("")

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
                                                                                             support1_xz[1])),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.)))})
        self._environmentController.setLinksStateDirect({self._support2_base :
                                                        LinkState(position_xyz = th.tensor((support2_xz[0],
                                                                                            0.3,
                                                                                            support2_xz[1])),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.)))})
        if self._show_goal:
            self._environmentController.setLinksStateDirect({self._red_ball_base :
                                                            LinkState( position_xyz = th.tensor((0.,
                                                                                                0.2,
                                                                                                goal_z)),
                                                                        orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.)))})

    def _simulation_initialization(self):
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._place_objects(support1_xz=(10,10),
                                support2_xz=(10,10),
                                goal_z=10)
            if self._current_episode_config.support2_pos_x > 0:
                rpos, hpos, kpos = self._start_height, 3.4159/4,  3.14159/2
            else:
                rpos, hpos, kpos = self._start_height, -3.4159/4, -3.14159/2
            # ggLog.info(f"Directly setting jpos = {rpos, hpos, kpos}")
            self._environmentController.setJointsStateDirect({self._rail_joint: JointState(position = self._start_height, rate=0, effort=0),
                                                            self._hip_joint:  JointState(position = hpos, rate=0, effort=0),
                                                            self._knee_joint: JointState(position = kpos, rate=0, effort=0)})
            start_jimp = {  self._hip_joint: (hpos,0,0,200,50),
                            self._knee_joint:(kpos,0,0,200,50)}         
            self._environmentController.setJointsImpedanceCommand(start_jimp)
            self._environmentController.apply_joint_impedances(start_jimp)
            # if self._environmentController.__class__.__name__== "RosXbotGazeboAdapter":
            self._environmentController.run(3.0) # let the leg fall
            # ggLog.info(f"jpos set")
            self._place_objects(support1_xz=(self._current_episode_config.support1_pos_x,self._current_episode_config.support1_pos_z),
                                support2_xz=(self._current_episode_config.support2_pos_x, self._current_episode_config.support2_pos_z),
                                goal_z=self._current_episode_config.hip_goal_z)
            # jpos = {k:v.position for k,v in self._environmentController.getJointsState(requestedJoints=[self._rail_joint, self._hip_joint, self._knee_joint]).items()}
            # ggLog.info(f"Init: current jpos = {jpos}")
        else:
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        
    def getUiRendering(self) -> Tuple[Union[np.ndarray, th.Tensor, None], float]:
        try:
            img, time = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            if img is None:
                time = -1
            return img, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return None, -1


    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        if self._obs_only_vec:
            stacked_part =  state[self.VECTOR_PART][:self._frame_stack_length,self._stacked_obs_part].detach().clone()
            stacked_part = stacked_part.flatten()
            constant_part = state[self.VECTOR_PART][0,self._constant_obs_part]
            return {self.VECTOR_PART : th.cat([stacked_part,constant_part])}
        else:
            vec_obs = state[self.VECTOR_PART][:-2]
            img_obs = state[self.IMAGE_PART]
            if self._obs_only_img:
                return {self.IMAGE_PART : img_obs}
            else:
                return {self.IMAGE_PART : img_obs,
                        self.VECTOR_PART : vec_obs}
            
    
    def getState(self) -> Dict[Any, th.Tensor]:
        """Get an observation of the environment.
        """
        with th.no_grad():
            if self._stepCounter>self._last_step_got_state:
                # ggLog.info(f"_stepCounter = {self._stepCounter}")
                self._last_step_got_state = self._stepCounter
                
                jstates = self._environmentController.getJointsState(requestedJoints=[self._knee_joint, self._hip_joint])
                lstates : Dict[Tuple[str,str],LinkState] = self._environmentController.getLinksState(requestedLinks = [self._thigh_com_link,
                                                                                    self._shin_com_link,
                                                                                    self._thigh_base_link])
                hip_height = lstates[self._thigh_base_link].pose.position[2]
                hip_vel_z = lstates[self._thigh_base_link].pos_velocity_xyz[2]
                self._max_hip_height_reached = th.maximum(self._max_hip_height_reached,hip_height)

                # n = '\n'
                # ggLog.info(f"\nlstates = \n{n.join([str(i) for i in lstates.items()])}")

                # ggLog.info(f"contacts == {n.join([str(c) for c in contacts])}")
                thigh_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
                thigh_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
                thigh_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([0.0,0.0,1.0], device=self._th_device))[1])
                shin_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
                shin_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
                shin_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_xyzw[[3,0,1,2]],
                                                                                    th.tensor([0.0,0.0,1.0], device=self._th_device))[1])

                if self._use_contacts:
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
                    abs_impulses_sum_avg = self._impulses_avg_alpha*self._last_abs_impulses_sum_avg + self._impulses_avg_alpha*abs_impulses_sum
                else:
                    abs_impulses_sum = -1
                    abs_forces_sum = -1
                    abs_forces_num = -1
                    abs_impulses_sum_avg = -1
                # ggLog.info(f"jstates = {jstates}")

                if len(self._last_state)!=0 and self._last_state[self.VECTOR_PART][0][self.STATE.SAFETY_TRIGGERED] > 0:
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
                    safety_triggered = th.any(jstate_th < j_safety_lims[:,0]) or th.any(jstate_th > j_safety_lims[:,1])
                    if safety_triggered:                        
                        ggLog.info(f"SAFETY TRIGGERED:\n"
                                   f"    jstate_th      = {jstate_th}\n"
                                   f"    j_safety_lims  = {j_safety_lims} ")
                # ggLog.info(f"safety_triggered = {safety_triggered}")

                current_vstate = th.tensor((jstates[self._hip_joint].position[0],
                                    jstates[self._hip_joint].rate[0],
                                    jstates[self._hip_joint].effort[0],
                                    jstates[self._knee_joint].position[0],
                                    jstates[self._knee_joint].rate[0],
                                    jstates[self._knee_joint].effort[0],
                                    hip_height,
                                    hip_vel_z,
                                    self._current_episode_config.support1_pos_x,
                                    self._current_episode_config.support1_pos_z,
                                    self._current_episode_config.support2_pos_x,
                                    self._current_episode_config.support2_pos_z,
                                    self._current_episode_config.hip_goal_z,
                                    self._configuration.reward_torque_limit_weight,
                                    self._configuration.reward_position_limit_weight,
                                    self._configuration.reward_velocity_weight,
                                    self._configuration.reward_energy_weight,
                                    self._configuration.reward_tracking_weight,
                                    self._configuration.reward_torque_weight,
                                    self._current_episode_config.reward_contacts_weights,
                                    self._configuration.reward_max_impulse,
                                    self._configuration.torque_command_scale_knee,
                                    self._configuration.torque_command_scale_hip,
                                    lstates[self._thigh_com_link].pos_velocity_xyz[0],
                                    lstates[self._thigh_com_link].pos_velocity_xyz[1],
                                    lstates[self._thigh_com_link].pos_velocity_xyz[2],
                                    lstates[self._thigh_com_link].ang_velocity_xyz[0],
                                    lstates[self._thigh_com_link].ang_velocity_xyz[1],
                                    lstates[self._thigh_com_link].ang_velocity_xyz[2],
                                    lstates[self._shin_com_link].pos_velocity_xyz[0],
                                    lstates[self._shin_com_link].pos_velocity_xyz[1],
                                    lstates[self._shin_com_link].pos_velocity_xyz[2],
                                    lstates[self._shin_com_link].ang_velocity_xyz[0],
                                    lstates[self._shin_com_link].ang_velocity_xyz[1],
                                    lstates[self._shin_com_link].ang_velocity_xyz[2],
                                    lstates[self._thigh_com_link].pose.position[0],
                                    lstates[self._thigh_com_link].pose.position[1],
                                    lstates[self._thigh_com_link].pose.position[2],
                                    thigh_ang_pos_x,
                                    thigh_ang_pos_y,
                                    thigh_ang_pos_z,
                                    lstates[self._shin_com_link].pose.position[0],
                                    lstates[self._shin_com_link].pose.position[1],
                                    lstates[self._shin_com_link].pose.position[2],
                                    shin_ang_pos_x,
                                    shin_ang_pos_y,
                                    shin_ang_pos_z,
                                    abs_impulses_sum,
                                    abs_forces_sum,
                                    abs_forces_num,
                                    abs_impulses_sum_avg,
                                    1 if safety_triggered else 0),
                                dtype = th.float32,
                                device = self._th_device)
                
                # ggLog.info(f"current_vstate = {current_vstate}")
                current_vstate = self._normalize(current_vstate,self._configuration.vstate_minmax[:,0],self._configuration.vstate_minmax[:,1])
                # ggLog.info(f"norm current_vstate = {current_vstate}")
                
                current_vstate[self._stacked_obs_part] += adarl.utils.utils.randn_like(current_vstate[self._stacked_obs_part],
                                                                                       mu  = self._current_episode_config.obs_noise_mustd[0],
                                                                                       std = self._current_episode_config.obs_noise_mustd[1],
                                                                                       generator=self._rng)
                # ggLog.info(f"noisy current_vstate = {current_vstate}")
                
                
                if self._stepCounter > 0:
                    self._new_history[1:] = self._vstate_history[0:-1]
                    self._new_history[0] = current_vstate
                else:
                    self._new_history[:] = current_vstate.repeat(self._history_length,1)
                tmp = self._vstate_history
                self._vstate_history = self._new_history
                self._new_history = tmp
                # self._vstate_history[self._stepCounter%self._history_length] = current_vstate
                # ggLog.info(f"vstate = {vstate}")
                # ggLog.info(f"thigh = {lstates}")
                if not self._obs_only_vec:
                    istate, tmp = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
                    istate = th.tensor(istate, dtype = th.uint8, device = self._th_device)
                else:
                    istate = th.empty(size=(0,), dtype = th.uint8, device = self._th_device)

                state = {self.VECTOR_PART : self._vstate_history.detach().clone(),
                        self.IMAGE_PART : istate}
                self._last_state = state
                # subrews = {}
                # self.computeReward(None,state,None,self.get_configuration(),subrews)
                # ggLog.info(f"subrewards = {subrews}")
            else:
                state = self._last_state
                
            # ggLog.info(f"state['vec'].size() = {state['vec'].size()}")
            return state

    def _compute_dbg_info(self):
        rew_dbg_info = {}
        sub_rewards = {}
        r = self.computeReward(None,
                               self._last_state, 
                               None, 
                               env_conf=self.get_configuration(),
                               dbg_info=rew_dbg_info,
                               sub_rewards=sub_rewards)
        if self._use_contacts:
            contacts = self._environmentController.get_contacts()
            abs_impulses = []
            abs_contacts = []
            for simsteps_contacts in contacts:
                abs_impulses += [abs(contact[3]*contact[4]) for contact in simsteps_contacts]
                abs_contacts += [abs(contact[3]) for contact in simsteps_contacts]
            if len(abs_impulses)>0:
                self._ep_max_abs_impulse = max(self._ep_max_abs_impulse, max(abs_impulses))
                self._ep_max_abs_impulses_sum = max(self._ep_max_abs_impulses_sum, sum(abs_impulses))
                self._ep_max_abs_contact = max(self._ep_max_abs_contact, max(abs_contacts))
                self._ep_max_abs_contacts_sum = max(self._ep_max_abs_contacts_sum, sum(abs_contacts))
        vstate_unnorm = self._unnormalize(self._last_state[self.VECTOR_PART][0],self._configuration.vstate_minmax[:,0],self._configuration.vstate_minmax[:,1])
        goal_dist = abs(vstate_unnorm[self.STATE.HIP_GOAL_Z]-vstate_unnorm[self.STATE.HIP_POS_Z])
        self._cumulative_dist_to_goal += goal_dist
        self._cumulative_knee_torque += abs(vstate_unnorm[self.STATE.KNEE_JOINT_EFFORT])
        self._cumulative_hip_torque += abs(vstate_unnorm[self.STATE.HIP_JOINT_EFFORT])
        self._max_knee_torque = th.maximum(self._max_knee_torque, abs(vstate_unnorm[self.STATE.KNEE_JOINT_EFFORT]))
        self._max_hip_torque = th.maximum(self._max_hip_torque, th.abs(vstate_unnorm[self.STATE.HIP_JOINT_EFFORT]))
        self._last_abs_impulses_sum = vstate_unnorm[self.STATE.IMPULSES_SUM]
        self._dists_to_goal[self._stepCounter%len(self._dists_to_goal)] = goal_dist
        self._cumulated_abs_impulses += self._last_abs_impulses_sum

        return rew_dbg_info

    def performStep(self):
        super().performStep()
        self._dbg_info = self._compute_dbg_info()



    def buildSimulation(self):
        # ggLog.info("Building env")
        envCtrlName = type(self._environmentController).__name__
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._environmentController.build_scenario(None)
            self._rendering_cam_name = "simple_camera"

            self._knee_joint = ("leg","knee_joint_1")
            self._hip_joint = ("leg","hip_joint_1")
            self._rail_joint = ("leg","rail_joint")

            self._foot_link = ("leg","tip1")
            self._thigh_base_link = ("leg", "thigh_link1")
            self._shin_base_link = ("leg", "shin_link1")
            self._thigh_com_link = ("leg", "thigh_link1_com")
            self._shin_com_link = ("leg", "shin_link1_com")
            self._rendering_cam_name = "simple_camera"
            self._support1_base = ("support1","world")
            self._support2_base = ("support2","world")
            self._red_ball_base = ("red_ball","world")
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._real:
                raise NotImplementedError()
            else:
                self._environmentController.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path("jumping_leg",
                                                                                                                          "gazebo/all_gazebo_xbot.launch"),
                                                           launch_file_args={"gui":"false"})
                self._knee_joint = ("leg","knee_joint_1")
                self._hip_joint = ("leg","hip_joint_1")
                self._rail_joint = ("leg","rail_joint")

                self._foot_link = ("leg","tip1")
                self._thigh_base_link = ("leg", "thigh_link1")
                self._shin_base_link = ("leg", "shin_link1")
                self._thigh_com_link = ("leg", "thigh_link1_com")
                self._shin_com_link = ("leg", "shin_link1_com")
                self._rendering_cam_name = "simple_camera"
                self._support1_base = ("support1","plate")
                self._support2_base = ("support2","plate")
                self._red_ball_base = ("red_ball","sphere_link")
        # elif envCtrlName in ["GazeboAdapter", "GazeboAdapterNoPlugin"]:
        #     # ggLog.info(f"sim_img_width  = {sim_img_width}")
        #     # ggLog.info(f"sim_img_height = {sim_img_height}")
        #     if not self._rendering_enabled:
        #         worldpath = "\"$(find adarl_ros)/worlds/ground_plane_world_plugin.world\""
        #     else:
        #         worldpath = "\"$(find adarl_ros)/worlds/fixed_camera_world_plugin.world\""
        #     self._environmentController.build_scenario( launch_file_pkg_and_path=("adarl_ros","/launch/gazebo_server.launch"),
        #                                                 launch_file_args={  "gui":"false",
        #                                                                     "paused":"true",
        #                                                                     "physics_engine":"bullet",
        #                                                                     "limit_sim_speed":"false",
        #                                                                     "world_name":worldpath,
        #                                                                     "gazebo_seed":f"{self._envSeed}",
        #                                                                     "wall_sim_speed":f"{self._wall_sim_speed}"})
        #     self._rendering_cam_name = "camera"
        # elif envCtrlName == "GzController":
        #     self._environmentController.build_scenario(sdf_file = ("adarl_ros2","/worlds/empty_cams.sdf"))
        #     # self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro"),
        #     #                                         model_name=None,
        #     #                                         pose=build_pose(0,2,0.5,0,0.0,-0.707,0.707),
        #     #                                         model_kwargs={"camera_width":"1920","camera_height":"1080","frame_rate":1/self._intendedStepLength_sec},
        #     #                                         model_format="sdf.xacro")
        #     self._rendering_cam_name = "simple_camera"
        else:
            raise NotImplementedError("environmentController "+envCtrlName+" not supported")





    def _destroySimulation(self):
        self._environmentController.destroy_scenario()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        # ggLog.info(f"getInfo(): {self._stepCounter}")
        # i["step_count"] = self._stepCounter
        current_vstate_unnorm = self._unnormalize(state[self.VECTOR_PART][0],self._configuration.vstate_minmax[:,0],self._configuration.vstate_minmax[:,1])
        i["hip_goal_z"] = current_vstate_unnorm[self.STATE.HIP_GOAL_Z]
        i["avg_dist"] = self._cumulative_dist_to_goal/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg10_dist"] = th.mean(self._dists_to_goal)
        i["avg_knee_torque"] = self._cumulative_knee_torque/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg_hip_torque"] = self._cumulative_hip_torque/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["avg_abs_impulse"] = self._cumulated_abs_impulses/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["max_abs_impulse"] = self._ep_max_abs_impulse
        i["max_abs_impulses_sum"] = self._ep_max_abs_impulses_sum
        i["max_abs_normimps_sum"] = self._ep_max_abs_impulses_sum/self._configuration.stepLength_sec
        i["max_abs_contact"] = self._ep_max_abs_contact
        i["max_abs_contacts_sum"] = self._ep_max_abs_contacts_sum
        i["max_abs_normconts_sum"] = self._ep_max_abs_contacts_sum/self._configuration.stepLength_sec
        i["max_knee_torque"] = self._max_knee_torque
        i["max_hip_torque"] = self._max_hip_torque
        i["impulses_sum"] = self._last_abs_impulses_sum
        i["step_count"] = self._stepCounter
        i["thigh_vel_x_z"] = current_vstate_unnorm[[self.STATE.THIGH_VEL_X,self.STATE.THIGH_VEL_Z]]
        i["shin_vel_x_z"] = current_vstate_unnorm[[self.STATE.SHIN_VEL_X,self.STATE.SHIN_VEL_Z]]
        i["hip_joint_vel"] = current_vstate_unnorm[self.STATE.HIP_JOINT_VEL]
        i["thigh_ang_vel_y"] = current_vstate_unnorm[self.STATE.THIGH_ANG_VEL_Y]
        i["thigh_pos_z"] = current_vstate_unnorm[[self.STATE.THIGH_POS_Z]]
        i["shin_pos_z"] = current_vstate_unnorm[[self.STATE.SHIN_POS_Z]]
        statenames = [e.name for e in self.STATE]
        i["vstate"] = {statenames[i]:current_vstate_unnorm[i] for i in range(len(statenames))}
        i.update(self._dbg_info)
        # i["config"] = dataclasses.asdict(self._configuration)
        i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i

    def get_configuration(self):
        return dataclasses.asdict(self._configuration)
    
        
    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.as_tensor(False, device=self._th_device)
        r = state[self.VECTOR_PART][0][self.STATE.SAFETY_TRIGGERED] > 0
        if r:
            ggLog.info(f"truncation at step {self._stepCounter}")
        return r