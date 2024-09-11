#!/usr/bin/env python3
from __future__ import annotations

from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.envs.ControlledEnv import ControlledEnv
from adarl.utils.robot_helpers import Robot
from adarl.utils.state_helper import StateNoiseGenerator, ThBoxStateHelper, DictStateHelper, RobotStateHelper
from adarl.utils.tensor_trees import TensorDict, map_tensor_tree
from adarl.utils.utils import build_pose, JointState, LinkState, quat_swing_twist_decomposition, quat_angle, MoveFailError, exc_to_str, to_string_tensor
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional, List, Literal, TypeVar, SupportsFloat, TypedDict, cast
import adarl
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces
import adarl.utils.spaces as spaces
import adarl.utils.utils
import dataclasses
import numpy as np
import os
import time
import torch as th
import copy

_T = TypeVar('_T', float, th.Tensor)




def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1


class LegJumpEnv(ControlledEnv):

    metadata = {'render.modes': ['rgb_array']}
    # STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT = "action" # component of the state that is the last performed action (it has a different size depending onthe configuration)
    STATE_IMG = "image" # componento f thestate that contains the rendered image
    STATE_ROBOT = "robot"
    STATE_EXTRINSIC = "extrinsic"
    STATE_INTERNAL = "internal"

    class State(TypedDict):
        action : th.Tensor
        image : th.Tensor
        robot : th.Tensor
        extrinsic : th.Tensor
        internal : th.Tensor


    EXTRINSIC_FIELDS = IntEnum("EXTRINSIC_FIELDS", ["HIP_POS_Z",
                                                    "HIP_VEL_Z",
                                                    "SUPPORT1_X",
                                                    "SUPPORT1_Z",
                                                    "SUPPORT2_X",
                                                    "SUPPORT2_Z"], start=0)
    # I don't like the GOAL_STATE name, but for now let's call it like that
    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", ["HIP_GOAL_Z",
                                                "REWARD_TORQUE_LIMIT_WEIGHT",
                                                "REWARD_POSITION_LIMIT_WEIGHT",
                                                "REWARD_VELOCITY_LIMIT_WEIGHT",
                                                "REWARD_VELOCITY_WEIGHT",
                                                "REWARD_ACCELERATION_WEIGHT",
                                                "REWARD_ENERGY_WEIGHT",
                                                "REWARD_TRACKING_WEIGHT",
                                                "REWARD_TORQUE_WEIGHT",
                                                "REWARD_TORQUEDIFF_WEIGHT",
                                                "REWARD_CONTACTS_WEIGHT",
                                                "REWARD_IMPULSE_THRESHOLD",
                                                "KNEE_TORQUE_CMD_SCALE",
                                                "HIP_TORQUE_CMD_SCALE",
                                                "IMPULSES_SUM",
                                                "FORCES_SUM",
                                                "FORCES_NUM",
                                                "IMPULSES_SUM_AVG",
                                                "SAFETY_TRIGGERED",
                                                "SMOOTHED_GOAL_DIST",
                                                "STEP_COUNT"], start=0)
    ACT_FIELDS = IntEnum("ACT_FIELDS", ["ACTION"], start=0)
    IMG_FIELDS = IntEnum("IMG_FIELDS", ["IMAGE"], start=0)

    

    # BASE_STATE_IDXS = IntEnum("BASE_STATE", [
                            
    #                         "THIGH_VEL_X",
    #                         "THIGH_VEL_Y",
    #                         "THIGH_VEL_Z",
    #                         "THIGH_ANG_VEL_X",
    #                         "THIGH_ANG_VEL_Y",
    #                         "THIGH_ANG_VEL_Z",
    #                         "SHIN_VEL_X",
    #                         "SHIN_VEL_Y",
    #                         "SHIN_VEL_Z",
    #                         "SHIN_ANG_VEL_X",
    #                         "SHIN_ANG_VEL_Y",
    #                         "SHIN_ANG_VEL_Z"], start=0)
    
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
                                                "POSITION_AND_STIFFNESS"], start=0)
    action_lengths = {
        CONTROL_MODES.IMPEDANCE: 10 ,
        CONTROL_MODES.IMPEDANCE_NO_GAINS: 6,
        CONTROL_MODES.POSITION_AND_TORQUES: 4,
        CONTROL_MODES.POSITION_AND_STIFFNESS: 4,
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
        damping_minmax : tuple[float,float]
        ep_obs_noise_mustd : th.Tensor
        frame_stack_length : int
        goal_dist_exp_smoothing_1s : float
        history_length : int
        img_shape_chw : tuple[int,int,int]
        impulses_avg_alpha : float
        joint_physical_limits_minmax_pve : dict[tuple[str,str],np.ndarray]
        joint_safety_limits_minmax_pve : dict[tuple[str,str],np.ndarray]
        leg_max_height : float
        leg_max_jump : float
        leg_min_height : float
        obs_dtype : th.dtype
        obs_only_img : bool
        obs_only_vec : bool
        original_max_epsteps : int
        platform_randomization : str
        position_cmd_limits_hip : Tuple[float,float]
        position_cmd_limits_knee : Tuple[float,float]
        rail_initial_position_limits : tuple[float,float]
        randomize_initial_pose : bool
        real : bool
        rendering_enabled :bool
        reward_acceleration_weight : float
        reward_contacts_weight : float
        reward_energy_weight : float
        reward_max_impulse : float
        reward_position_limit_weight : float
        reward_scale : float
        reward_torque_limit_weight : float
        reward_torque_weight : float
        reward_torquediff_weight : float
        reward_tracking_weight : float
        reward_velocity_limit_weight : float
        reward_velocity_weight : float
        safe_damping : float
        safe_stiffness : float
        show_goal : bool
        start_height : float
        stepLength_sec : float
        step_obs_noise_std : th.Tensor
        stiffness_minmax : tuple[float,float]
        stop_on_safety : bool
        th_device : th.device
        torque_command_scale_hip : float
        torque_command_scale_knee : float
        use_contacts : bool
        velocity_command_scale_hip : float
        velocity_command_scale_knee : float
        wall_sim_speed : bool
        

    @dataclass
    class Statistics:
        dists_to_goal : th.Tensor
        max_knee_torque : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        max_hip_torque : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        max_knee_vel : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        max_hip_vel : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        cumulated_abs_impulses : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        last_abs_impulses_sum : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        ep_max_abs_impulse : float = 0.0
        ep_max_abs_impulses_sum : float = 0.0
        ep_max_abs_contact : float = 0.0
        ep_max_abs_contacts_sum : float = 0.0
        last_external_work : float = 0.0
        last_step_got_state : int = -1
        last_abs_impulses_sum_avg : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(0.0))
        avg_dist_to_goal : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_knee_torque : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_hip_torque : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_knee_power : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_hip_power : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_knee_pos_err : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_hip_pos_err : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_knee_torqeref : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        avg_hip_torqueref : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        rewards : dict = dataclasses.field(default_factory=lambda: {})

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
                    reward_velocity_limit_weight = 1.0,
                    reward_velocity_weight = 0.0,
                    reward_acceleration_weight = 0.0,
                    reward_energy_weight = 0.01,
                    reward_tracking_weight = 1.0,
                    reward_torque_weight = 0.0,
                    reward_torquediff_weight = 0.0,
                    reward_contacts_weight = 0.0,
                    control_mode = Literal["impedance","impedance_no_gains","position_and_torques",
                                           "position_and_gains","torque","velocity","position"],
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
                    randomize_initial_pose : bool = False,
                    safety_limits_factor = 0.95):
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
        self._foot_link = ("leg","tip1")
        self._thigh_base_link = ("leg", "thigh_link1")
        self._shin_base_link = ("leg", "shin_link1")
        self._thigh_com_link = ("leg", "thigh_link1_com")
        self._shin_com_link = ("leg", "shin_link1_com")
        self._rendering_cam_name = "simple_camera"
        # max_dact_dt = 100 #max change in action, i.e. da/dt
        # self._max_act_change = th.tensor(max_dact_dt*stepLength_sec,dtype=th.float32, device=self._th_device)
        # self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)
        self._last_out_action = th.empty((0,))
        self._spawned = False

        self._rng = th.Generator(device=th_device)
        self._stats = copy.deepcopy(self.Statistics(dists_to_goal=th.empty((0,))))
        
        self._leg_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")
        self._robot_model = Robot(adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(self._leg_file).read_text()))
        phys_limits_minmax_pve = {("leg",k):l for k,l in self._robot_model.get_joint_limits([self._hip_joint[1],self._knee_joint[1]]).items()}
        safe_limits_minmax_pve = {k:(lims_minmax-0.5*(lims_minmax[1]+lims_minmax[0]))*safety_limits_factor+0.5*(lims_minmax[1]+lims_minmax[0])
                                   for k,lims_minmax in phys_limits_minmax_pve.items()}
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
        if obs_only_vec:
            img_shape_chw = (0,0,0)
        else:
            img_shape_chw = ((3 if rgb else 1),obs_img_height,obs_img_width)
        self._configuration = LegJumpEnv.EnvConfiguration(  action_delay_mustd = th.as_tensor(action_delay_mustd,
                                                                                              dtype=obs_dtype,
                                                                                              device=th_device),
                                                            action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                            action_len = LegJumpEnv.action_lengths[control_mode],
                                                            action_noise_mustd=th.empty((0,)),
                                                            bstate_minmax = th.empty((0,)),
                                                            control_mode=control_mode,
                                                            damping_minmax=(10,200),
                                                            ep_obs_noise_mustd=ep_obs_noise_mustd,
                                                            frame_stack_length = 3,
                                                            goal_dist_exp_smoothing_1s = goal_dist_exp_smoothing_1s,
                                                            history_length = 4,
                                                            img_shape_chw=img_shape_chw,
                                                            impulses_avg_alpha = 0.5,
                                                            joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                            joint_safety_limits_minmax_pve = safe_limits_minmax_pve,
                                                            leg_max_height = leg_max_height,
                                                            leg_max_jump = leg_max_jump,
                                                            leg_min_height = leg_min_height,
                                                            obs_dtype = obs_dtype,
                                                            obs_only_img=obs_only_img,
                                                            obs_only_vec=obs_only_vec,
                                                            original_max_epsteps = maxStepsPerEpisode,
                                                            platform_randomization = platform_randomization,
                                                            position_cmd_limits_hip =  (-2.2, 2.2),
                                                            position_cmd_limits_knee = (-2.2, 2.2),
                                                            rail_initial_position_limits = (0.1,1.5),
                                                            randomize_initial_pose = randomize_initial_pose,
                                                            real = real,
                                                            rendering_enabled = True,
                                                            reward_acceleration_weight = reward_acceleration_weight,
                                                            reward_contacts_weight = reward_contacts_weight,
                                                            reward_energy_weight = reward_energy_weight,
                                                            reward_max_impulse = 10,
                                                            reward_position_limit_weight = reward_position_limit_weight,
                                                            reward_scale = reward_scale,
                                                            reward_torque_limit_weight = reward_torque_limit_weight,
                                                            reward_torque_weight = reward_torque_weight,
                                                            reward_torquediff_weight = reward_torquediff_weight,
                                                            reward_tracking_weight = reward_tracking_weight,
                                                            reward_velocity_limit_weight = reward_velocity_limit_weight,
                                                            reward_velocity_weight = reward_velocity_weight,
                                                            safe_damping = 50,
                                                            safe_stiffness = 200,
                                                            show_goal = True,
                                                            start_height = 0.9,
                                                            stepLength_sec=stepLength_sec,
                                                            step_obs_noise_std=step_obs_noise_std,
                                                            stiffness_minmax=(10,1000),
                                                            stop_on_safety = stop_on_safety,
                                                            th_device = th_device,
                                                            torque_command_scale_hip = 100,
                                                            torque_command_scale_knee = 100,
                                                            use_contacts = use_contacts,
                                                            velocity_command_scale_hip = 20,
                                                            velocity_command_scale_knee = 20,
                                                            wall_sim_speed = wall_sim_speed
                                                            )
        
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=th.tensor(0, device=self._configuration.th_device),
                                                                       support1_pos_x=th.tensor(0, device=self._configuration.th_device),
                                                                       support1_pos_z=th.tensor(0, device=self._configuration.th_device),
                                                                       support2_pos_x=th.tensor(0, device=self._configuration.th_device),
                                                                       support2_pos_z=th.tensor(0, device=self._configuration.th_device),
                                                                       reward_contacts_weights=th.tensor(0, device=self._configuration.th_device),
                                                                       initial_joint_pose_rhk=th.tensor([1.0,1.0,1.0], device=self._configuration.th_device),
                                                                       max_ep_steps=th.tensor(self._configuration.original_max_epsteps, device=self._configuration.th_device))
        
        robot_state_helper = RobotStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                              stiffness_minmax=self._configuration.stiffness_minmax,
                                              damping_minmax=self._configuration.damping_minmax,
                                              obs_dtype=self._configuration.obs_dtype,
                                              th_device=self._configuration.th_device,
                                              history_length=self._configuration.frame_stack_length)
        self._safety_limits = robot_state_helper.build_robot_limits(  joint_limit_minmax_pve=self._configuration.joint_safety_limits_minmax_pve,
                                                                stiffness_minmax=self._configuration.stiffness_minmax,
                                                                damping_minmax=self._configuration.damping_minmax)
        internal_state_helper =   ThBoxStateHelper(field_names=[e.value for e in self.INTERNAL_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.INTERNAL_FIELDS.HIP_GOAL_Z : [0,2],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_ENERGY_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_CONTACTS_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_IMPULSE_THRESHOLD : [0,10],
                                                                self.INTERNAL_FIELDS.KNEE_TORQUE_CMD_SCALE : [0,150],
                                                                self.INTERNAL_FIELDS.HIP_TORQUE_CMD_SCALE : [0,150],
                                                                self.INTERNAL_FIELDS.IMPULSES_SUM : [-1,10000],
                                                                self.INTERNAL_FIELDS.FORCES_SUM : [-1,100000],
                                                                self.INTERNAL_FIELDS.FORCES_NUM : [-1,100000],
                                                                self.INTERNAL_FIELDS.IMPULSES_SUM_AVG : [-1,10000],
                                                                self.INTERNAL_FIELDS.SAFETY_TRIGGERED : [0,1],
                                                                self.INTERNAL_FIELDS.SMOOTHED_GOAL_DIST : [0,10],
                                                                self.INTERNAL_FIELDS.STEP_COUNT : [-1,1000_000_000]},
                                                observable_fields=[self.INTERNAL_FIELDS.HIP_GOAL_Z])
        extrinsic_state_helper =  ThBoxStateHelper(field_names=[e.value for e in self.EXTRINSIC_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.EXTRINSIC_FIELDS.HIP_POS_Z : [-0.5,3],
                                                                self.EXTRINSIC_FIELDS.HIP_VEL_Z : [-100,100],
                                                                self.EXTRINSIC_FIELDS.SUPPORT1_X : [-2,2],
                                                                self.EXTRINSIC_FIELDS.SUPPORT1_Z : [-1,3],
                                                                self.EXTRINSIC_FIELDS.SUPPORT2_X : [-2,2],
                                                                self.EXTRINSIC_FIELDS.SUPPORT2_Z : [-1,3]},
                                               history_length=self._configuration.frame_stack_length)
        act_history_state_helper = ThBoxStateHelper(field_names=[self.ACT_FIELDS.ACTION],
                                               obs_dtype=th.float32,
                                               th_device=th_device,
                                               field_size=(self._configuration.action_len,),
                                               fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]})
        img_state_helper = ThBoxStateHelper(field_names=[self.IMG_FIELDS.IMAGE],
                                       obs_dtype=th.uint8,
                                       th_device=th_device,
                                       field_size=self._configuration.img_shape_chw,
                                       fields_minmax={self.IMG_FIELDS.IMAGE:th.stack([    th.zeros(self._configuration.img_shape_chw,device=th_device,dtype=th.uint8),
                                                                        255*th.ones(self._configuration.img_shape_chw,device=th_device,dtype=th.uint8)])})
        if self._configuration.obs_only_vec:
            observable_fields = [   self.STATE_ROBOT,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_INTERNAL]
            vec_fields = observable_fields
        elif self._configuration.obs_only_img:
            observable_fields = [self.STATE_IMG]
        else:
            observable_fields = [   self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_IMG]
            vec_fields = [self.STATE_ROBOT, 
                          self.STATE_INTERNAL]
            
        robot_state_noise =  StateNoiseGenerator(robot_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = self._configuration.ep_obs_noise_mustd,
                                            step_std = self._configuration.step_obs_noise_std)
        extrinsic_state_noise =  StateNoiseGenerator(extrinsic_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = self._configuration.ep_obs_noise_mustd,
                                            step_std = self._configuration.step_obs_noise_std)        
        self._state_helper = DictStateHelper({self.STATE_ROBOT : robot_state_helper,
                                              self.STATE_EXTRINSIC : extrinsic_state_helper,
                                              self.STATE_INTERNAL : internal_state_helper,
                                              self.STATE_IMG : img_state_helper,
                                              self.STATE_ACT: act_history_state_helper},
                                              observable_fields=observable_fields,
                                              noise = {
                                                    self.STATE_ROBOT : robot_state_noise,
                                                    self.STATE_EXTRINSIC : extrinsic_state_noise},
                                             flatten_in_obs=vec_fields,
                                             flattened_part_name="vec")
        
        state_space = self._state_helper.get_space()
        observation_space = self._state_helper.get_obs_space()

        # ggLog.info(f"State space = {state_space}")
        # ggLog.info(f"Observation space = {observation_space}")
            
        action_space_high = np.array([1]*self._configuration.action_len)
        action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high, seed=seed)

        self._configuration.action_noise_mustd = 0.0 * th.ones(size=(self._configuration.action_len,), dtype=th.float32, device=self._configuration.th_device)
        
        # # delay noises will actually be discretized by the step_length
        # action_delay_noise_mustd = th.tensor([0.01,0.01], dtype=th.float32, device=self._th_device)
        # obs_delay_noise_mustd = th.tensor([0.01,0.01], dtype=th.float32, device=self._th_device)
        self._build_action_indexes()

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

    def _minmax_pvesd(self):
        hp_lim = self._configuration.position_cmd_limits_hip
        hv_lim = self._configuration.velocity_command_scale_hip
        he_lim = self._configuration.torque_command_scale_hip
        kp_lim = self._configuration.position_cmd_limits_knee
        kv_lim = self._configuration.velocity_command_scale_knee
        ke_lim = self._configuration.torque_command_scale_knee
        min_stiffness, max_stiffness = self._configuration.stiffness_minmax
        min_damping, max_damping = self._configuration.damping_minmax
        minmax_hipknee_pvesd = th.tensor([[[hp_lim[0], -hv_lim, -he_lim, min_stiffness, min_damping],
                                           [kp_lim[0], -kv_lim, -ke_lim, min_stiffness, min_damping]],

                                          [[hp_lim[1], hv_lim, he_lim, max_stiffness, max_damping],
                                           [kp_lim[1], kv_lim, ke_lim, max_stiffness, max_damping]]], device=self._configuration.th_device)
        return minmax_hipknee_pvesd

    
    def _build_action_indexes(self):
        min_stiffness, max_stiffness = self._configuration.stiffness_minmax
        min_damping, max_damping = self._configuration.damping_minmax
        jnums = 2
        s = normalize(self._configuration.safe_stiffness,min= min_stiffness,max=max_stiffness)
        d = normalize(self._configuration.safe_damping,min= min_damping,max=max_damping)
        if self._configuration.control_mode == self.CONTROL_MODES.VELOCITY:
            act_to_pvesd =  [1]
            base_pvesd = [0.0, 0.0, 0.0, -1.0, d]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION:
            act_to_pvesd =  [0]
            base_pvesd =  [0.0, 0.0, 0.0, s, d]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            act_to_pvesd =  [0,2]
            base_pvesd =  [0.0, 0.0, 0.0, s, d]
        elif self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            act_to_pvesd =  [0,1,2]
            base_pvesd =  [0.0, 0.0, 0.0, s, d]
        elif self._configuration.control_mode == self.CONTROL_MODES.IMPEDANCE:
            act_to_pvesd =  [0,1,2,3,4]
            base_pvesd =  [0.0, 0.0, 0.0, 0.0, 0.0]
        elif self._configuration.control_mode == self.CONTROL_MODES.POSITION_AND_STIFFNESS:
            act_to_pvesd =  [0,3]
            base_pvesd =  [0.0, 0.0, 0.0, 0.0, d]
        elif self._configuration.control_mode == self.CONTROL_MODES.TORQUE:
            act_to_pvesd =  [2]
            base_pvesd =  [0.0, 0.0, 0.0, -1.0, -1.0]
        else:
            raise RuntimeError(f"Invalid control mode {self._configuration.control_mode}")
        self._base_pvesd = th.as_tensor(base_pvesd).repeat(jnums,1)
        self._act_to_pvesd_idx = th.as_tensor(act_to_pvesd,
                                              dtype=th.int32,
                                              device=self._configuration.th_device)
        # self._act_to_pvesd_idx = th.as_tensor([[[j,i] for i in act_to_pvesd] for j in range(jnums)],
        #                                       dtype=th.int32,
        #                                       device=self._configuration.th_device)
        

    def _pvesd_to_action(self, cmds_pvesd : dict[tuple[str,str], tuple[float,float,float,float,float]]):
        minmax_hipknee_pvesd = self._minmax_pvesd()
        cmd_joint_pvesd = th.as_tensor([cmds_pvesd[self._hip_joint], cmds_pvesd[self._knee_joint]])
        cmd_joint_pvesd = normalize(th.as_tensor(cmds_pvesd[self._hip_joint]), min=minmax_hipknee_pvesd[0], max=minmax_hipknee_pvesd[1])
        action = cmd_joint_pvesd[:,self._act_to_pvesd_idx].flatten()
        return action

    def _action_to_pvesd(self, action: th.Tensor) -> dict[tuple[str,str],tuple[float,float,float,float,float]]:
        minmax_hipknee_pvesd = self._minmax_pvesd()

        jnums = 2
        cmd_joint_pvesd = self._base_pvesd.detach().clone()
        cmd_joint_pvesd[:,self._act_to_pvesd_idx] = action.view((jnums, -1))
        cmd_joint_pvesd = unnormalize(cmd_joint_pvesd, min=minmax_hipknee_pvesd[0], max=minmax_hipknee_pvesd[1])
        if th.any(cmd_joint_pvesd[:,[3,4]] <0 ):
            ggLog.warn(f"Negative stiffness or damping!! {cmd_joint_pvesd}")
        
        return {self._hip_joint :  tuple(cmd_joint_pvesd[0].tolist()),
                self._knee_joint:  tuple(cmd_joint_pvesd[1].tolist())}

    def submitAction(self, action : th.Tensor) -> None:
        with th.no_grad():
            action = th.as_tensor(action).detach().cpu().squeeze()
            super().submitAction(action)
            dt = self._configuration.stepLength_sec
            alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
            prev_action = self._current_state[self.STATE_ACT][0,self.ACT_FIELDS.ACTION].detach().cpu()
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


    # @staticmethod
    # def _compute_mechanical_energies(vstate_unnorm):
    #     thigh_mass = 3.37
    #     thigh_length = 0.3
    #     shin_mass = 1.3
    #     shin_length = 0.45
    #     slider_mass = 8
    #     g = 9.8
        
    #     thigh_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = thigh_mass,
    #                                                     inertia_moment=1/12*thigh_mass*thigh_length**2,
    #                                                     vel_x=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_VEL_X],
    #                                                     vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_VEL_Z],
    #                                                     ang_vel_y=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_VEL_Y])
    #     shin_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = shin_mass,
    #                                                     inertia_moment=1/12*shin_mass*shin_length**2,
    #                                                     vel_x=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_VEL_X],
    #                                                     vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_VEL_Z],
    #                                                     ang_vel_y=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_VEL_Y])
    #     slider_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = slider_mass,
    #                                                     inertia_moment=0,
    #                                                     vel_x=0,
    #                                                     vel_z=vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.HIP_VEL_Z],
    #                                                     ang_vel_y=0)
    #     thigh_pot_energy = 0 #thigh_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.THIGH_POS_Z]
    #     shin_pot_energy = 0 #shin_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.SHIN_POS_Z]
    #     slider_pot_energy = slider_mass*g*vstate_unnorm[LegJumpEnv.BASE_STATE_IDXS.HIP_POS_Z]
    #     return thigh_kin_energy+thigh_pot_energy, shin_kin_energy+shin_pot_energy, slider_kin_energy+slider_pot_energy

    def computeReward(self, previousState : Dict[str,th.Tensor],
                      state : Dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : Dict[str,th.Tensor] = {}, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = 100

        robot_state_norm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT])
        # normpositions = robot_state_norm[:,0]
        normvelocities = robot_state_norm[0][:,1]
        normtorques = robot_state_norm[0][:,2]
        normaccelerations = robot_state_norm[0][:,1] - robot_state_norm[1][:,1]
        normtorquediff = robot_state_norm[0][:,2] - robot_state_norm[1][:,2]

        torque_reward = - th.clamp(th.mean(th.pow(normtorques,4)),-max_rew,max_rew)
        velocity_reward = - th.clamp(th.mean(th.pow(normvelocities,4)),-max_rew,max_rew)
        acceleration_reward = - th.clamp(th.mean(th.pow(normaccelerations,2)),-max_rew,max_rew)
        torquediff_reward = - th.clamp(th.mean(th.pow(normaccelerations,2)),-max_rew,max_rew)
        
        robot_state_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)[0]
        position_safenorm = robot_state_safenorm[:,0]
        velocities_safenorm = robot_state_safenorm[:,1]
        torque_safenorm = robot_state_safenorm[:,2]

        torque_limit_reward = -th.clamp(th.mean(th.pow(torque_safenorm,50)),-max_rew,max_rew)
        position_limit_reward = -th.clamp(th.mean(th.pow(position_safenorm,50)),-max_rew,max_rew)
        velocity_limit_reward = -th.clamp(th.mean(th.pow(velocities_safenorm,50)),-max_rew,max_rew)


        internal_state = state[LegJumpEnv.STATE_INTERNAL][0]
        goal_dist = internal_state[LegJumpEnv.INTERNAL_FIELDS.SMOOTHED_GOAL_DIST]
        impulse_threshold = internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_IMPULSE_THRESHOLD]
        # tracking_reward = 1 - goal_dist
        # tracking_reward = 1/(1+goal_dist/0.05)       # 0.50 at 0.05m, 0.35 at 0.10m, 0.2 at 0.2
        tracking_reward = 1/(1+(goal_dist/0.1)**2) # 0.75 at 0.05m, 0.50 at 0.10m, 0.2 at 0.2
        contacts_reward = th.clamp(-(internal_state[LegJumpEnv.INTERNAL_FIELDS.IMPULSES_SUM_AVG]/impulse_threshold)**4, min = -1)

        # j_effs = self._state_helper.get(state, [(self.STATE_ROBOT, ([self._knee_joint,self._hip_joint],["eff"]))])
        ktorque = float("nan") #j_effs[0]
        htorque = float("nan") #j_effs[1]

        # shin_rotation = vstate_un[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_POS_Y] - pvstate_un[LegJumpEnv.BASE_STATE_IDXS.SHIN_ANG_POS_Y]
        # thigh_rotation = vstate_un[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_POS_Y] - pvstate_un[LegJumpEnv.BASE_STATE_IDXS.THIGH_ANG_POS_Y]
        shin_rotation = 0
        thigh_rotation = 0


        # new_thigh_energy, new_shin_energy, new_slider_energy = LegJumpEnv._compute_mechanical_energies(vstate_un)
        # old_thigh_energy, old_shin_energy, old_slider_energy = LegJumpEnv._compute_mechanical_energies(pvstate_un)
        new_thigh_energy, new_shin_energy, new_slider_energy = 0,0,0
        old_thigh_energy, old_shin_energy, old_slider_energy = 0,0,0
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
        sub_rewards["reward_acceleration"] = acceleration_reward
        sub_rewards["reward_position_limit"] = position_limit_reward
        sub_rewards["reward_velocity_limit"] = velocity_limit_reward
        sub_rewards["reward_torquediff"] = torquediff_reward
        # sub_rewards["reward_energy"] = global_energy_reward
        sub_rewards["reward_contacts"] = contacts_reward
        sub_rewards["reward_health"] = th.tensor(1, device=internal_state.device)
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards.items()}

        weights = { "reward_tracking" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT],
                    "reward_torque" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT],
                    "reward_torquediff" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT],
                    "reward_torque_limit" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "reward_velocity" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT],
                    "reward_acceleration" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT],
                    "reward_position_limit" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT],
                    "reward_velocity_limit" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT],
                    "reward_energy" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_ENERGY_WEIGHT],
                    "reward_contacts" : internal_state[LegJumpEnv.INTERNAL_FIELDS.REWARD_CONTACTS_WEIGHT],
                    "reward_health" : 1}
        for k in sub_rewards:
            sub_rewards[k] = sub_rewards[k]*env_conf["reward_scale"]*weights[k]
        sub_rewards = {k:v.squeeze() for k,v in sub_rewards.items()}
        sub_rewards_unscaled = {k:v.squeeze() for k,v in sub_rewards_unscaled.items()}
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
        if sub_rewards["reward_contacts"] != 0:
            raise RuntimeError(f"reward_contacts is {sub_rewards['reward_contacts']}, weights = {weights}, state = {state}")
        return reward
























    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------

    def initializeEpisode(self, options = {}) -> None:

        self._current_state : LegJumpEnv.State = self._state_helper.reset_state(initial_values={
            self.STATE_EXTRINSIC : th.tensor(0.0),
            self.STATE_ROBOT : th.tensor(0.0),
            self.STATE_ACT : th.tensor(0.0),
            self.STATE_INTERNAL : th.tensor(0.0),
            self.STATE_IMG : th.zeros(size=self._configuration.img_shape_chw, device=self._configuration.th_device, dtype=th.uint8)
        })
        self._current_state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.STEP_COUNT] = th.tensor(-1.)
        self._last_obs = None

        
        if not self._spawned and isinstance(self._environmentController, BaseSimulationAdapter):
            leg_pose = build_pose(0,0,0,0,0,0,1)
            camera_pose = build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707)
            support1_pose = build_pose(-0.5, 0.2, 0.2, 0,0,0,1)
            support2_pose = build_pose(-0.5, 0.2, 0.4, 0,0,0,1)
            red_ball_pose = leg_pose
            self._spawned = True
            support_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro")
            camera_file = adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro")
            supports_xacro_args = { "add_world_link":str(isinstance(self._environmentController, PyBulletAdapter)),
                                    "size_x":0.2,
                                    "size_y":0.2,
                                    "size_z":0.005}
            if isinstance(self._environmentController, PyBulletAdapter):
                self._environmentController.spawn_model(model_file=self._leg_file,
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
        self._stats = copy.deepcopy(self.Statistics(dists_to_goal=th.full(size=(int(self._maxStepsPerEpisode/10),),
                                                    fill_value=initial_dist,
                                                    dtype=th.float32, device=self._configuration.th_device)))
        
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            self._realworld_initialization()
        self._last_out_action = th.clamp(self._pvesd_to_action(self._last_sent_pvesd), min=-1, max=1)
        # ggLog.info(f"initial action {self._last_out_action}, pvesd = {self._last_sent_pvesd}")

        self._update_state()
        self._update_stats()

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
        if not isinstance(self._environmentController, BaseSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        rail_pos, hip_pos, knee_pos = self._current_episode_config.initial_joint_pose_rhk
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
        envCtrlName = type(self._environmentController).__name__
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

    def _render_image(self, state):
        with th.no_grad():
            import torchvision.transforms.functional # only import it if needed
            img, time = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            img = th.tensor(img, dtype = th.uint8, device = self._configuration.th_device)
            img = th.permute(img,(2,0,1)) # hwc to chw
            if img.size()[0] == self._configuration.img_shape_chw[0]:
                pass
            elif img.size()[0] == 3 and self._configuration.img_shape_chw[0] == 1:
                img = torchvision.transforms.functional.rgb_to_grayscale(img)
            else:
                raise RuntimeError(f"Cannot adapt image shape {img.size()} to required {self._configuration.img_shape_chw}")
            img = torchvision.transforms.functional.resized_crop(img,
                                                                top=0,
                                                                left=int(0.29*img.size()[2]),
                                                                height=img.size()[1],
                                                                width=int(0.4*img.size()[2]),
                                                                size = list(self._configuration.img_shape_chw[1:]))
            img = img.permute(1,2,0)
            img = img.cpu().numpy()
            # ggLog.info(f"img = {img.shape}")
            if img is None:
                time = -1
            return img, time


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
        self._last_obs = self._state_helper.observe(state)
        if th.any(th.abs(self._last_obs["vec"]) > 100):
            raise RuntimeError(f"Values over 100 in obs {self._last_obs}")
        return self._last_obs
            
    def getState(self) -> Dict[Any, th.Tensor]:
        return self._current_state
    
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


        internal_state = self._current_state[self.STATE_INTERNAL][0]
        step_count = internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        if step_count!=-1 and internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0:
            safety_triggered = True
        else:
            jstate_th = th.as_tensor([jstates[self._hip_joint].position[0],
                                    jstates[self._hip_joint].rate[0],
                                    jstates[self._hip_joint].effort[0],
                                    jstates[self._knee_joint].position[0],
                                    jstates[self._knee_joint].rate[0],
                                    jstates[self._knee_joint].effort[0]])
            j_safety_lims = th.as_tensor([  self._configuration.joint_safety_limits_minmax_pve[self._hip_joint][:,0],
                                            self._configuration.joint_safety_limits_minmax_pve[self._hip_joint][:,1],
                                            self._configuration.joint_safety_limits_minmax_pve[self._hip_joint][:,2],
                                            self._configuration.joint_safety_limits_minmax_pve[self._knee_joint][:,0],
                                            self._configuration.joint_safety_limits_minmax_pve[self._knee_joint][:,1],
                                            self._configuration.joint_safety_limits_minmax_pve[self._knee_joint][:,2]])
            triggered_limits = th.logical_or(jstate_th < j_safety_lims[:,0], jstate_th > j_safety_lims[:,1])
            safety_triggered = th.any(triggered_limits)
            if safety_triggered:       
                elements = ["hip_pos","hip_vel","hip_eff","knee_pos","knee_vel","knee_eff"]
                triggered = []
                for i in range(len(elements)):
                    if triggered_limits[i]:
                        triggered.append(elements[i])
                ggLog.info( f"SAFETY TRIGGERED:\n"
                            f"    triggered      = {triggered}\n"
                            f"    jstate_th      = {jstate_th}\n"
                            f"    j_safety_lims  = {j_safety_lims} ")




        goal_dist = abs(hip_height - self._current_episode_config.hip_goal_z)
        prev_goal_dist = internal_state[self.INTERNAL_FIELDS.SMOOTHED_GOAL_DIST]
        alpha = self._configuration.goal_dist_exp_smoothing_1s**(self._configuration.stepLength_sec)
        # ggLog.info(f"prev_goal_dist = {prev_goal_dist}")
        # ggLog.info(f"alpha = {alpha}")
        if prev_goal_dist > 0:
            smoothed_goal_dist = goal_dist*(1-alpha) + prev_goal_dist*alpha
        else:
            smoothed_goal_dist = 1
        # ggLog.info(f"smoothed_goal_dist = {smoothed_goal_dist}")

        new_internal_state = {
                        self.INTERNAL_FIELDS.HIP_GOAL_Z : self._current_episode_config.hip_goal_z,
                        self.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : self._configuration.reward_torque_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : self._configuration.reward_position_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : self._configuration.reward_velocity_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT : self._configuration.reward_velocity_weight,
                        self.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT : self._configuration.reward_acceleration_weight,
                        self.INTERNAL_FIELDS.REWARD_ENERGY_WEIGHT : self._configuration.reward_energy_weight,
                        self.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT : self._configuration.reward_tracking_weight,
                        self.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT : self._configuration.reward_torque_weight,
                        self.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT : self._configuration.reward_torquediff_weight,
                        self.INTERNAL_FIELDS.REWARD_CONTACTS_WEIGHT : self._current_episode_config.reward_contacts_weights,
                        self.INTERNAL_FIELDS.REWARD_IMPULSE_THRESHOLD : self._configuration.reward_max_impulse,
                        self.INTERNAL_FIELDS.KNEE_TORQUE_CMD_SCALE : self._configuration.torque_command_scale_knee,
                        self.INTERNAL_FIELDS.HIP_TORQUE_CMD_SCALE : self._configuration.torque_command_scale_hip,
                        self.INTERNAL_FIELDS.IMPULSES_SUM : abs_impulses_sum,
                        self.INTERNAL_FIELDS.FORCES_SUM : abs_forces_sum,
                        self.INTERNAL_FIELDS.FORCES_NUM : abs_forces_num,
                        self.INTERNAL_FIELDS.IMPULSES_SUM_AVG : abs_impulses_sum_avg,
                        self.INTERNAL_FIELDS.SAFETY_TRIGGERED : 1 if safety_triggered else 0,
                        self.INTERNAL_FIELDS.SMOOTHED_GOAL_DIST : smoothed_goal_dist,
                        self.INTERNAL_FIELDS.STEP_COUNT : step_count+1}
        new_robot_state = {self._hip_joint :    th.concat([ jstates[self._hip_joint].position[[0]],
                                                            jstates[self._hip_joint].rate[[0]],
                                                            jstates[self._hip_joint].effort[[0]],
                                                            th.as_tensor(self._last_sent_pvesd[self._hip_joint])]),
                            self._knee_joint :  th.concat([ jstates[self._knee_joint].position[[0]],
                                                            jstates[self._knee_joint].rate[[0]],
                                                            jstates[self._knee_joint].effort[[0]],
                                                            th.as_tensor(self._last_sent_pvesd[self._knee_joint])])}
        if th.any(th.concat([new_robot_state[self._hip_joint][6:],new_robot_state[self._knee_joint][6:]])<0):
            ggLog.warn(f"negative gains in new_robot_state = {new_robot_state}")
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.HIP_POS_Z : hip_height,
                                self.EXTRINSIC_FIELDS.HIP_VEL_Z : hip_vel_z,
                                self.EXTRINSIC_FIELDS.SUPPORT1_X : self._current_episode_config.support1_pos_x,
                                self.EXTRINSIC_FIELDS.SUPPORT1_Z : self._current_episode_config.support1_pos_z,
                                self.EXTRINSIC_FIELDS.SUPPORT2_X : self._current_episode_config.support2_pos_x,
                                self.EXTRINSIC_FIELDS.SUPPORT2_Z : self._current_episode_config.support2_pos_z}
        new_act_state = {self.ACT_FIELDS.ACTION : self._last_out_action}
        instantaneous_state = {self.STATE_EXTRINSIC : new_extrinsic_state,
                                self.STATE_ACT : new_act_state,
                                self.STATE_INTERNAL : new_internal_state,
                                self.STATE_ROBOT : new_robot_state}
        
        if self._configuration.obs_only_vec:
            img = th.zeros(size=self._configuration.img_shape_chw, dtype = th.uint8, device = self._configuration.th_device)
        else:
            import torchvision.transforms.functional # only import it if needed
            img, _ = self._render_image(instantaneous_state)
        new_img_state = {self.IMG_FIELDS.IMAGE : img}        
        instantaneous_state[self.STATE_IMG] = new_img_state

              
        
        
        if step_count <= 0:
            self._current_state = self._state_helper.reset_state(instantaneous_state)
        else:
            self._state_helper.update(instantaneous_state, state=self._current_state)
        
        map_tensor_tree(self._current_state, lambda t: t.detach().clone())
        

    def _update_stats(self):
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
        robot_state_j_pvepvesd = self._current_state[self.STATE_ROBOT][0]
        internal_state = self._current_state[self.STATE_INTERNAL][0]
        extrinsic_state = self._current_state[self.STATE_EXTRINSIC][0]
        goal_dist = abs(internal_state[self.INTERNAL_FIELDS.HIP_GOAL_Z]-extrinsic_state[self.EXTRINSIC_FIELDS.HIP_POS_Z])
        if self._stepCounter>0:
            self._stats.avg_dist_to_goal = (self._stats.avg_dist_to_goal*(self._stepCounter-1) + goal_dist.squeeze())/self._stepCounter
            self._stats.avg_knee_torque = (self._stats.avg_knee_torque*(self._stepCounter-1) + abs(robot_state_j_pvepvesd[0,2]).squeeze())/self._stepCounter
            self._stats.avg_hip_torque = (self._stats.avg_hip_torque*(self._stepCounter-1) +  abs(robot_state_j_pvepvesd[1,2]).squeeze())/self._stepCounter
            self._stats.avg_knee_power = (self._stats.avg_knee_power*(self._stepCounter-1) + abs(robot_state_j_pvepvesd[0,2]*robot_state_j_pvepvesd[0,1]).squeeze())/self._stepCounter
            self._stats.avg_hip_power = (self._stats.avg_hip_power*(self._stepCounter-1) +  abs(robot_state_j_pvepvesd[1,2]*robot_state_j_pvepvesd[1,1]).squeeze())/self._stepCounter
            self._stats.avg_knee_pos_err = (self._stats.avg_knee_pos_err*(self._stepCounter-1) + abs(robot_state_j_pvepvesd[0,0]-robot_state_j_pvepvesd[0,3]).squeeze())/self._stepCounter
            self._stats.avg_hip_pos_err = (self._stats.avg_hip_pos_err*(self._stepCounter-1) +  abs(robot_state_j_pvepvesd[1,0]-robot_state_j_pvepvesd[1,3]).squeeze())/self._stepCounter
            self._stats.avg_knee_torqeref = (self._stats.avg_knee_torqeref*(self._stepCounter-1) +  abs(robot_state_j_pvepvesd[0,5]).squeeze())/self._stepCounter
            self._stats.avg_hip_torqueref = (self._stats.avg_hip_torqueref*(self._stepCounter-1) + abs(robot_state_j_pvepvesd[1,5]).squeeze())/self._stepCounter
        self._stats.max_knee_torque = th.maximum(self._stats.max_knee_torque, abs(robot_state_j_pvepvesd[0,2])).squeeze()
        self._stats.max_hip_torque = th.maximum(self._stats.max_hip_torque, th.abs(robot_state_j_pvepvesd[1,2])).squeeze()
        self._stats.max_knee_vel = th.maximum(self._stats.max_knee_vel, abs(robot_state_j_pvepvesd[0,1])).squeeze()
        self._stats.max_hip_vel = th.maximum(self._stats.max_hip_vel, th.abs(robot_state_j_pvepvesd[1,1])).squeeze()
        self._stats.last_abs_impulses_sum = self._current_state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.IMPULSES_SUM].squeeze()
        self._stats.dists_to_goal[self._stepCounter%len(self._stats.dists_to_goal)] = goal_dist.squeeze()
        self._stats.cumulated_abs_impulses += self._stats.last_abs_impulses_sum.squeeze()
        self._stats.rewards = rew_dbg_info
        return rew_dbg_info

    def performStep(self):
        super().performStep()
        self._update_state()
        self._update_stats()
        self._last_step_simtime = self._environmentController.getEnvTimeFromReset()





    def getInfo(self,state) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        # ggLog.info(f"getInfo(): {self._stepCounter}")
        # i["step_count"] = self._stepCounter
        robot_state = state[self.STATE_ROBOT][0]
        internal_state = state[self.STATE_INTERNAL][0]
        i["hip_goal_z"] = internal_state[self.INTERNAL_FIELDS.HIP_GOAL_Z]
        i.update(dataclasses.asdict(self._stats))
        i["avg_dist"] = self._stats.avg_dist_to_goal
        i["avg10_dist"] = th.mean(self._stats.dists_to_goal)
        i["avg_abs_impulse"] = self._stats.cumulated_abs_impulses/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["max_abs_normimps_sum"] = self._stats.ep_max_abs_impulses_sum/self._configuration.stepLength_sec
        i["max_abs_normconts_sum"] = self._stats.ep_max_abs_contacts_sum/self._configuration.stepLength_sec
        i["step_count"] = self._stepCounter
        # i["thigh_vel_x_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.THIGH_VEL_X,self.BASE_STATE_IDXS.THIGH_VEL_Z]]
        # i["shin_vel_x_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.SHIN_VEL_X,self.BASE_STATE_IDXS.SHIN_VEL_Z]]
        i["hip_joint_vel"] = robot_state[1,1]
        # i["thigh_ang_vel_y"] = bstate_unnorm[self.BASE_STATE_IDXS.THIGH_ANG_VEL_Y]
        # i["thigh_pos_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.THIGH_POS_Z]]
        # i["shin_pos_z"] = bstate_unnorm[[self.BASE_STATE_IDXS.SHIN_POS_Z]]
        

        statenorm = self._state_helper.normalize(state)
        for substate in [self.STATE_ROBOT, self.STATE_EXTRINSIC, self.STATE_INTERNAL, self.STATE_ACT]:
            i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
            i["state_"+substate+"_labels"] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
            i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
            i["statenorm_"+substate+"_labels"] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
        # statenames = [e.name for e in self.BASE_STATE_IDXS]
        # stateindxs = [e.value for e in self.BASE_STATE_IDXS]
        # i["action"] = self._last_out_action
        # i["cbstate_norm"] = bstate[stateindxs]
        # i["cbstate"] = bstate_unnorm[stateindxs]
        # i["cbstate_labels"] = th.as_tensor([list(n.encode("utf-8").ljust(16)[:16]) for n in statenames], dtype=th.uint8) # ugly, but simple
        i.update(self._stats.rewards)
        # i["config"] = dataclasses.asdict(self._configuration)
        i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        i["safety_triggered"] = internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED]
        i["success"] = i["avg10_dist"] < 0.05
        i["vec_obs"] = self._last_obs["vec"]
        obslabels = [n.encode("utf-8").ljust(64)[:64] for n in self._state_helper.observation_names()["vec"]]
        # ggLog.info(f"obslabels = {obslabels}")
        i["vec_obs_labels"] = th.as_tensor(obslabels, dtype=th.uint8)
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i

    def get_configuration(self):
        return dataclasses.asdict(self._configuration)
    
        
    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.as_tensor(False, device=self._configuration.th_device)
        r = state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        if r:
            ggLog.info(f"Terminated at step {self._stepCounter}")
        return r
    
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
