from __future__ import annotations

from adarl_envs.env.env_utils import bell_reward, double_bell_reward, flattened_joint_penalty_reward, joint_penalty_reward, norm_penalty, penalty_reward, planar_tracking_error_vec, set_column, smooth_clip, smoothclip_flattener
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check_size, dbg_check
from adarl.utils.spaces import ThBox
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
from adarl.utils.utils import (LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, isinstance_noimport,
                                quat_xyzw_between_vecs_py, masked_assign, quat_mul_xyzw, quat_angle_xyzw, vectors_angle, ros_rpy_to_quaternion_xyzw_th)
from adarl.utils.vec_state_helper import ThBoxStateHelper, unnormalize, normalize
from adarl_envs.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS, DistributionDef, DistributionDefTh, DistributionTh, RobotVecEnvInitArgs
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from git import Tree
from gymnasium.vector.utils.spaces import batch_space
from requests import head
from typing import Sequence, Literal, TypedDict, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.session as session
import adarl.utils.utils
import numpy as np
import pprint
import torch as th
import os
import dataclasses
import adarl.utils.async_cuda2cpu_queue as async_cuda2cpu_queue
from adarl.utils.base_utils import record_time, record_region_start, record_region_end

disable_compile = os.environ.get("DISABLE_ENV_TH_COMPILE", False)

@dataclass
class LocomotionVecEnvInitArgs():
    robot_init_args : RobotVecEnvInitArgs
    disallowed_contact_links : list[tuple[str,str]]
    feet_links : list[tuple[str,str]]
    goal_height_minmax : tuple[float,float]
    goal_speed_minmax : tuple[float, float]
    goal_yaw_minmax : tuple[float, float]
    goal_yaw_vel_minmax : tuple[float, float]
    goal_yaw_vel_zero_ratio : float
    reward_superweight_joint_penalties : DistributionDef
    reward_joint_acceleration_weight : DistributionDef
    reward_joint_acc_on_vel_weight : DistributionDef
    reward_joint_actacc_weight : DistributionDef
    reward_joint_actdiff_weight : DistributionDef
    reward_contacts_weight : float
    reward_joint_energy_weight : float
    reward_joint_power_weight : float
    reward_failure_weight : float
    reward_feet_air_time_weight : float
    reward_feet_ground_time_weight : float
    reward_feet_on_ground_weight : float
    reward_feet_step_height_weight : float
    reward_heading_weight : float
    reward_heading_velocity_weight : float
    reward_health_weight : float
    reward_height_velocity_weight : float
    reward_height_position_weight : float
    reward_pitchnroll_weight : float
    reward_pitchnroll_velocity_weight : float
    reward_joint_posref_vel_weight : DistributionDef
    reward_joint_posref_acc_weight : DistributionDef
    reward_joint_position_limit_weight : float
    reward_joint_position_weight : float
    reward_joint_stand_position_weight : float
    reward_scale : float
    reward_joint_sensed_effort_weight : float
    reward_safety_triggered_weight : float
    reward_slip_weight : float
    reward_joint_torque_limit_weight : float
    reward_joint_cmdtorque_weight : DistributionDef
    reward_joint_torquediff_weight : float
    reward_joint_torqueref_weight : float
    reward_tracking_weight : float
    reward_joint_velocity_limit_weight : float
    reward_joint_velocity_weight : float
    reward_joint_velref_weight : float
    reward_yaw_vel_tracking_weight : float
    desired_foot_clearance : float
    terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]]
    use_contacts : bool
    max_good_step_duration : float
    min_good_step_duration : float
    heightmap_resolution : int = -1
    goal_resampling_probability_per_sec : float = 0.0
    max_goal_height_pos_change_speed : float = 0.25
    max_height_speed_goal : float = 1.0
    feet_air_time_avg_alpha = 0.8
    split_rewards : bool = False
    terminate_on_crash : bool = True
    playground_style_reward : bool = False
    
class LocomotionVecEnv(RobotVecEnv):
    STATE_LOCOMOTION = "loco"
    STATE_FEET = "feet"
    STATE_HEIGHTMAP = "heightmap"
    STATE_REWARDS = "reward_weights"

    @dataclass
    class LocomotionConfiguration:
        desired_foot_clearance : float
        disallowed_contact_links : list[tuple[str,str]]
        goal_speed_minmax : th.Tensor
        goal_abs_yaw_minmax : th.Tensor
        goal_yaw_vel_minmax : th.Tensor
        goal_yaw_vel_zero_ratio : th.Tensor
        reward_scale : th.Tensor
        reward_superweight_joint_penalties : DistributionDefTh
        reward_weight_joint_acceleration : DistributionDefTh
        reward_weight_joint_acc_on_vel : DistributionDefTh
        reward_weight_contacts : th.Tensor
        reward_weight_joint_energy : th.Tensor
        reward_weight_feet_step_height : th.Tensor
        reward_weight_health : th.Tensor
        reward_weight_height_velocity : th.Tensor
        reward_weight_height_position : th.Tensor
        reward_weight_pitchnroll : th.Tensor
        reward_weight_pitchnroll_velocity : th.Tensor
        reward_weight_joint_actdiff : DistributionDefTh
        reward_weight_joint_actacc : DistributionDefTh
        reward_weight_joint_power : th.Tensor
        reward_weight_joint_position_limit : th.Tensor
        reward_weight_joint_position : th.Tensor
        reward_weight_heading : th.Tensor
        reward_weight_heading_velocity : th.Tensor
        reward_weight_yaw_vel_tracking : th.Tensor
        reward_weight_joint_torque_limit : th.Tensor
        reward_weight_joint_torque : DistributionDefTh
        reward_weight_joint_torquediff : th.Tensor
        reward_weight_tracking : th.Tensor
        reward_weight_joint_velocity_limit : th.Tensor
        reward_weight_joint_velocity : th.Tensor
        reward_weight_feet_air_time : th.Tensor
        reward_weight_feet_ground_time : th.Tensor
        reward_weight_feet_on_ground : th.Tensor
        reward_weight_failure : th.Tensor
        reward_weight_joint_sensed_effort : th.Tensor
        reward_weight_joint_stand_position : th.Tensor
        reward_weight_safety_triggered : th.Tensor
        reward_weight_slip : th.Tensor
        reward_weight_joint_velref : th.Tensor
        reward_weight_joint_torqueref : th.Tensor
        reward_weight_joint_posref_vel : DistributionDefTh
        reward_weight_joint_posref_acc : DistributionDefTh
        terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]]
        use_contacts : bool
        height_reward_settle_point : th.Tensor
        height_reward_2_settle_point : th.Tensor
        height_reward_2_weight : th.Tensor
        pitchnroll_reward_settle_point : th.Tensor
        heading_reward_settle_point : th.Tensor
        vel_reward_goalrelative_weight : th.Tensor
        reward_vel_goal_relative_width : th.Tensor
        reward_vel_goal_relative_width_offset : th.Tensor
        reward_vel_goal_absolute_width : th.Tensor
        feet_links : list[tuple[str,str]]
        heightmap_resolution_xy : tuple[int,int]
        min_good_step_air_duration : float
        max_good_step_duration : float
        min_good_ground_duration : float
        max_good_ground_duration : float
        goal_height_minmax : tuple[float,float]
        goal_resampling_probability_per_sec : th.Tensor
        goal_resampling_enabled : bool
        terminate_on_crash : bool
        max_goal_height_pos_change_speed : float
        """ Max speed at which the goal height can change, in m/s. Used to prevent too sudden changes."""
        max_height_speed_goal : float
        """ Maximum goal speed for the speed-based height reward."""
        max_heading_speed_goal : th.Tensor
        """ Maximum goal speed for the speed-based heading reward."""
        heading_kp : th.Tensor
        """ Proportional gain used to compute the goal heading angular velocity from the heading error."""
        heading_kd : th.Tensor
        """ Derivative gain used to compute the goal heading angular velocity from the heading error."""
        feet_air_time_avg_alpha : float
        """ Exponential smoothing factor (for 1 second) used to compute the average step duration of each foot for the feet_air_time_uniformity reward."""
        max_pitchnroll_speed_goal : th.Tensor
        """ Maximum goal speed for the speed-based pitchnroll reward."""
        pitchnroll_kp : th.Tensor
        """ Proportional gain used to compute the goal pitch and roll angular velocity from the pitch and roll error."""
        pitchnroll_kd : th.Tensor
        """ Derivative gain used to compute the goal pitch and roll angular velocity from the pitch and roll error."""
        goal_heading_rel_yaw_minmax : th.Tensor
        """ Min and max yaw angle for the goal heading, relative to the linvel goal direction."""
        split_rewards : bool
        """ Split the rewards into separate components, making the reward n-dimensional."""
        enabled_rewards : list[str]
        """ List of enabled rewards. If None, all rewards are enabled."""
        playground_style_reward : bool
        """ Whether to use a reward structure more similar to the one used in the playground spot environment."""
        reward_yaw_vel_goal_absolute_width : th.Tensor
        reward_yaw_vel_goal_relative_width : th.Tensor
        reward_yaw_vel_goal_relative_width_offset : th.Tensor
        reward_yaw_vel_goal_relative_weight : th.Tensor
        is_jumping_alpha_1s : float
        """ Smoothing factor for the is_jumping signal, expressed as the alpha of an exponential moving average with a time dt of 1 second."""

    @dataclass
    class EpisodeLocomConfiguration:
        goal_abs_vel_vec_xys : th.Tensor | None
        """Absolute frame linvel goal, if goal_rel_vel_vec_xy_speed_xy is not None, it is ignored. Expressed as 3 
           scalars: xy direction of the velocity and speed."""
        goal_rel_vel_vec_xys : th.Tensor | None
        """Relative frame linvel goal, overrides goal_abs_vel_vec_xy. Expressed as 3 scalars: xy direction of the
           velocity and speed"""
        goal_abs_gravity_vec_xyz : th.Tensor
        goal_abs_height_vec_z : th.Tensor
        goal_heading_rel_vec_yaw : th.Tensor
        goal_yaw_vel_vec : th.Tensor
        reward_weight_actacc : th.Tensor
        reward_weight_actdiff : th.Tensor
        reward_weight_posref_vel : th.Tensor
        reward_weight_torque : th.Tensor
        reward_weight_acceleration : th.Tensor
        reward_weight_acc_on_vel : th.Tensor
        reward_weight_posref_acc : th.Tensor

    @dataclass
    class SubRewards:
        contacts : th.Tensor
        feet_air_time : th.Tensor
        feet_ground_time : th.Tensor
        feet_on_ground : th.Tensor
        feet_step_height : th.Tensor
        heading : th.Tensor
        heading_velocity : th.Tensor
        health : th.Tensor
        height_position : th.Tensor
        height_velocity : th.Tensor
        joint_acc_on_vel : th.Tensor
        joint_acceleration : th.Tensor
        joint_actacc : th.Tensor
        joint_actdiff : th.Tensor
        joint_position : th.Tensor
        joint_position_limit : th.Tensor
        joint_posref_acc : th.Tensor
        joint_posref_vel : th.Tensor
        joint_power : th.Tensor
        joint_sensed_effort : th.Tensor
        joint_stand_position : th.Tensor
        joint_torque : th.Tensor
        joint_torque_limit : th.Tensor
        joint_torque_refs : th.Tensor
        joint_torquediff : th.Tensor
        joint_velocity : th.Tensor
        joint_velocity_limit : th.Tensor
        joint_velocity_refs : th.Tensor
        pitchnroll : th.Tensor
        pitchnroll_velocity : th.Tensor
        safety_triggered : th.Tensor
        slip : th.Tensor
        tracking : th.Tensor
        yaw_vel_tracking : th.Tensor

    LOCOMOTION_FIELDS = IntEnum("LOCOMOTION_FIELDS", [  
                                                "COLLISON_COUNT",
                                                "GOAL_LINVEL_REL_DIRECTION_X",
                                                "GOAL_LINVEL_REL_DIRECTION_Y",
                                                "GOAL_LINVEL_REL_DIRECTION_Z",
                                                "GOAL_LINVEL_SPEED",
                                                "GOAL_VELOCITY_ABS_X",
                                                "GOAL_VELOCITY_ABS_Y",
                                                "GOAL_VELOCITY_ABS_Z",
                                                "GOAL_BODY_HEIGHT",
                                                "GOAL_REL_HEADING_YAW_X",
                                                "GOAL_REL_HEADING_YAW_Y",
                                                "GOAL_GRAVITY_ABS_X",
                                                "GOAL_GRAVITY_ABS_Y",
                                                "GOAL_GRAVITY_ABS_Z",
                                                "GOAL_YAW_VEL",
                                                "SMOOTHED_TRACKING_ERROR",
                                                "SMOOTHED_HEIGHT_ERROR",
                                                "SMOOTHED_PITCHNROLL_ERROR",
                                                "SMOOTHED_HEADING_ERROR",
                                                "SMOOTHED_YAW_VEL_ERROR",
                                                "SMOOTHED_PITCHNROLL_ERROR_VELOCITY",
                                                "SMOOTHED_GOAL_BODY_HEIGHT",
                                                "SUM_IMPULSES",
                                                "CRASHED",
                                                "SMOOTHED_IS_JUMPING",
                                                "SMOOTHED_NUM_FEET_ON_GROUND"
                                                ], start=0)
    
    REWARD_WEIGHTS_FIELDS = IntEnum("REWARD_FIELDS", [  # Important! keep sorted alphabetically
                                                "CONTACTS",
                                                "FAILURE",
                                                "FEET_AIR_TIME",
                                                "FEET_GROUND_TIME",
                                                "FEET_ON_GROUND",
                                                "FEET_STEP_HEIGHT",
                                                "HEADING",
                                                "HEADING_VELOCITY",
                                                "HEALTH",
                                                "HEIGHT_POSITION",
                                                "HEIGHT_VELOCITY",
                                                "JOINT_ACCELERATION",
                                                "JOINT_ACC_ON_VEL",
                                                "JOINT_ACTACC",
                                                "JOINT_ACTDIFF",
                                                "JOINT_POSITION",
                                                "JOINT_POSITION_LIMIT",
                                                "JOINT_POSREF_ACC",
                                                "JOINT_POSREF_VEL",
                                                "JOINT_POWER",
                                                "JOINT_SENSED_EFFORT",
                                                "JOINT_STAND_POSITION",
                                                "JOINT_TORQUE",
                                                "JOINT_TORQUEDIFF",
                                                "JOINT_TORQUE_LIMIT",
                                                "JOINT_TORQUE_REFS",
                                                "JOINT_VELOCITY",
                                                "JOINT_VELOCITY_LIMIT",
                                                "JOINT_VELOCITY_REFS",
                                                "PITCHNROLL",
                                                "PITCHNROLL_VELOCITY",
                                                "SAFETY_TRIGGERED",
                                                "SLIP",
                                                "TRACKING",
                                                "YAW_VEL_TRACKING"
                                                ], start=0)

    FEET_FIELDS = IntEnum("FEET_FIELDS" , [     "FEET_AIR_DURATIONS",
                                                "FEET_GROUND_DURATIONS",
                                                "FEET_VEL_X",
                                                "FEET_VEL_Y",
                                                "FEET_REL_POS_X",
                                                "FEET_REL_POS_Y",
                                                "FEET_REL_POS_Z",
                                                "FEET_ABS_POS_Z",
                                                "PEAK_POS_Z",
                                                "AVG_FEET_STEP_DURATIONS"
                                                ], start=0)

    def __init__(self,  init_args : LocomotionVecEnvInitArgs):
        robot_init_args = init_args.robot_init_args
        adapter = robot_init_args.adapter
        th_device = robot_init_args.th_device
        num_envs = adapter.vec_size()
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((num_envs,), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((num_envs,), device=th_device, dtype=th.bool)
        self._unit_3d_vector = self._thtens([1.0, 0.0, 0.0])
        self._unit_quaternion = self._thtens([0.0, 0.0, 0.0, 1.0])
        self._zero = self._thtens([0.0])
        self._loco_conf = LocomotionVecEnv.LocomotionConfiguration(
                        desired_foot_clearance = init_args.desired_foot_clearance,
                        reward_scale  = self._thtens(init_args.reward_scale) ,
                        reward_superweight_joint_penalties =    self._distr_to_tensor(init_args.reward_superweight_joint_penalties),
                        reward_weight_contacts  =               self._thtens(init_args.reward_contacts_weight) ,
                        reward_weight_failure =                 self._thtens(init_args.reward_failure_weight),
                        reward_weight_feet_air_time =           self._thtens(init_args.reward_feet_air_time_weight),
                        reward_weight_feet_ground_time =        self._thtens(init_args.reward_feet_ground_time_weight),
                        reward_weight_feet_on_ground =          self._thtens(init_args.reward_feet_on_ground_weight),
                        reward_weight_feet_step_height =        self._thtens(init_args.reward_feet_step_height_weight),
                        reward_weight_heading =                 self._thtens(init_args.reward_heading_weight),
                        reward_weight_heading_velocity =        self._thtens(init_args.reward_heading_velocity_weight),
                        reward_weight_health =                  self._thtens(init_args.reward_health_weight),
                        reward_weight_height_position =         self._thtens(init_args.reward_height_position_weight),
                        reward_weight_height_velocity =         self._thtens(init_args.reward_height_velocity_weight),
                        reward_weight_joint_acc_on_vel =        self._distr_to_tensor(init_args.reward_joint_acc_on_vel_weight),
                        reward_weight_joint_acceleration =      self._distr_to_tensor(init_args.reward_joint_acceleration_weight),
                        reward_weight_joint_actacc =            self._distr_to_tensor(init_args.reward_joint_actacc_weight),
                        reward_weight_joint_actdiff =           self._distr_to_tensor(init_args.reward_joint_actdiff_weight),
                        reward_weight_joint_energy =            self._thtens(init_args.reward_joint_energy_weight) ,
                        reward_weight_joint_position =          self._thtens(init_args.reward_joint_position_weight),
                        reward_weight_joint_position_limit =    self._thtens(init_args.reward_joint_position_limit_weight) ,
                        reward_weight_joint_posref_acc =        self._distr_to_tensor(init_args.reward_joint_posref_acc_weight),
                        reward_weight_joint_posref_vel =        self._distr_to_tensor(init_args.reward_joint_posref_vel_weight),
                        reward_weight_joint_power =             self._thtens(init_args.reward_joint_power_weight) ,
                        reward_weight_joint_sensed_effort =     self._thtens(init_args.reward_joint_sensed_effort_weight),
                        reward_weight_joint_stand_position =    self._thtens(init_args.reward_joint_stand_position_weight),
                        reward_weight_joint_torque =            self._distr_to_tensor(init_args.reward_joint_cmdtorque_weight),
                        reward_weight_joint_torque_limit =      self._thtens(init_args.reward_joint_torque_limit_weight) ,
                        reward_weight_joint_torquediff =        self._thtens(init_args.reward_joint_torquediff_weight),
                        reward_weight_joint_torqueref =         self._thtens(init_args.reward_joint_torqueref_weight),
                        reward_weight_joint_velocity =          self._thtens(init_args.reward_joint_velocity_weight),
                        reward_weight_joint_velocity_limit =    self._thtens(init_args.reward_joint_velocity_limit_weight),
                        reward_weight_joint_velref =            self._thtens(init_args.reward_joint_velref_weight),
                        reward_weight_pitchnroll =              self._thtens(init_args.reward_pitchnroll_weight),
                        reward_weight_pitchnroll_velocity =     self._thtens(init_args.reward_pitchnroll_velocity_weight),
                        reward_weight_safety_triggered =        self._thtens(init_args.reward_safety_triggered_weight),
                        reward_weight_slip =                    self._thtens(init_args.reward_slip_weight),
                        reward_weight_tracking =                self._thtens(init_args.reward_tracking_weight),
                        reward_weight_yaw_vel_tracking =        self._thtens(init_args.reward_yaw_vel_tracking_weight),
                        use_contacts = init_args.use_contacts,
                        disallowed_contact_links=init_args.disallowed_contact_links,
                        terminating_contact_pairs=init_args.terminating_contact_pairs,
                        goal_speed_minmax = th.as_tensor(init_args.goal_speed_minmax, device=th_device, dtype=th.float32),
                        goal_abs_yaw_minmax = th.as_tensor(init_args.goal_yaw_minmax, device=th_device, dtype=th.float32),
                        goal_yaw_vel_minmax = th.as_tensor(init_args.goal_yaw_vel_minmax, device=th_device, dtype=th.float32),
                        goal_yaw_vel_zero_ratio = self._thtens(init_args.goal_yaw_vel_zero_ratio),
                        height_reward_settle_point=self._thtens(0.05), # A narrow reward bell
                        height_reward_2_settle_point=self._thtens(0.3), # A wider reward bell
                        height_reward_2_weight=self._thtens(0.5), # weight of the wide reward bell over the narrow one
                        pitchnroll_reward_settle_point=self._thtens(th.pi/4), # ~zero reward after this angle
                        heading_reward_settle_point = self._thtens(3.14159/16), # ~zero reward after this distance (w component of the quat difference)
                        vel_reward_goalrelative_weight = self._thtens(0.25),
                        reward_vel_goal_relative_width = self._thtens(1.5),
                        reward_vel_goal_absolute_width = self._thtens(0.5),
                        reward_vel_goal_relative_width_offset = self._thtens(0.1),
                        feet_links = init_args.feet_links,
                        heightmap_resolution_xy = (init_args.heightmap_resolution,init_args.heightmap_resolution),
                        min_good_step_air_duration=init_args.min_good_step_duration,
                        max_good_step_duration=init_args.max_good_step_duration,
                        min_good_ground_duration=init_args.min_good_step_duration,
                        max_good_ground_duration=init_args.max_good_step_duration,
                        goal_height_minmax = init_args.goal_height_minmax,
                        goal_resampling_probability_per_sec = self._thtens(init_args.goal_resampling_probability_per_sec),
                        goal_resampling_enabled = init_args.goal_resampling_probability_per_sec > 0.0,
                        max_goal_height_pos_change_speed = init_args.max_goal_height_pos_change_speed,
                        max_height_speed_goal = init_args.max_height_speed_goal,
                        heading_kp=self._thtens(1.0),
                        heading_kd=self._thtens(0.01),
                        max_heading_speed_goal=self._thtens(th.pi/4),
                        feet_air_time_avg_alpha = init_args.feet_air_time_avg_alpha,
                        max_pitchnroll_speed_goal=self._thtens(th.pi/2),
                        pitchnroll_kp=self._thtens(1.0),
                        pitchnroll_kd=self._thtens(0.01),
                        goal_heading_rel_yaw_minmax=self._thtens([-th.pi, th.pi]),
                        split_rewards = init_args.split_rewards,
                        enabled_rewards = None, #type: ignore Will be set below,
                        terminate_on_crash = init_args.terminate_on_crash,
                        playground_style_reward = init_args.playground_style_reward,
                        reward_yaw_vel_goal_absolute_width = self._thtens(0.5),
                        reward_yaw_vel_goal_relative_width = self._thtens(1.5),
                        reward_yaw_vel_goal_relative_width_offset = self._thtens(0.1),
                        reward_yaw_vel_goal_relative_weight = self._thtens(0.25),
                        is_jumping_alpha_1s = 0.7
                        )
        self._loco_conf.enabled_rewards = []
        prefix = "reward_weight_"
        for k,v in asdict(self._loco_conf).items():
            if k.startswith(prefix):
                if v != 0:
                    self._loco_conf.enabled_rewards.append(k[len(prefix):])
        ggLog.info(f"Locomotion rewards enabled: {self._loco_conf.enabled_rewards}")
        self._sub_rewards_bounds = LocomotionVecEnv.SubRewards(
            joint_acceleration =    self._thtens([float("-inf"), 0            ]),
            joint_acc_on_vel =      self._thtens([float("-inf"), 0            ]),
            joint_actacc =          self._thtens([float("-inf"), 0            ]),
            joint_actdiff =         self._thtens([float("-inf"), 0            ]),
            contacts =              self._thtens([float("-inf"), float("+inf")]),
            feet_air_time =         self._thtens([float("-inf"), float("+inf")]),
            feet_ground_time =      self._thtens([float("-inf"), float("+inf")]),
            feet_on_ground =        self._thtens([float("-inf"), float("+inf")]),
            feet_step_height =      self._thtens([float("-inf"), float("+inf")]),
            heading =               self._thtens([0,             float("+inf")]),
            heading_velocity =      self._thtens([0,             float("+inf")]),
            health =                self._thtens([0,             float("+inf")]),
            height_position =       self._thtens([0,             float("+inf")]),
            height_velocity =       self._thtens([0,             float("+inf")]),
            pitchnroll =            self._thtens([0,             float("+inf")]),
            pitchnroll_velocity =   self._thtens([0,             float("+inf")]),
            joint_position =        self._thtens([float("-inf"), 0            ]),
            joint_position_limit =  self._thtens([float("-inf"), 0            ]),
            joint_power =           self._thtens([float("-inf"), 0            ]),
            joint_posref_acc =      self._thtens([float("-inf"), 0            ]),
            joint_posref_vel =      self._thtens([float("-inf"), 0            ]),
            joint_sensed_effort =   self._thtens([float("-inf"), 0            ]),
            safety_triggered =      self._thtens([float("-inf"), 0            ]),
            slip =                  self._thtens([float("-inf"), 0            ]),
            joint_stand_position =  self._thtens([float("-inf"), 0            ]),
            joint_torque =          self._thtens([float("-inf"), 0            ]),
            joint_torque_limit =    self._thtens([float("-inf"), 0            ]),
            joint_torque_refs =     self._thtens([float("-inf"), 0            ]),
            joint_torquediff =      self._thtens([float("-inf"), 0            ]),
            tracking =              self._thtens([0,             float("+inf")]),
            joint_velocity =        self._thtens([float("-inf"), 0            ]),
            joint_velocity_limit =  self._thtens([float("-inf"), 0            ]),
            joint_velocity_refs =   self._thtens([float("-inf"), 0            ]),
            yaw_vel_tracking =      self._thtens([0,             float("+inf")])
        )
        if self._loco_conf.split_rewards:
            bounds_tensor = th.stack([getattr(self._sub_rewards_bounds, r.lower()) for r in self._loco_conf.enabled_rewards], dim=0)
            single_reward_space=ThBox( low=bounds_tensor[:,0],
                                            high=bounds_tensor[:,1],
                                            torch_device=th_device,
                                            labels=np.array(self._loco_conf.enabled_rewards))
        else:
            single_reward_space=ThBox( low=th.tensor(float("-inf"), device=th_device),
                                            high=th.tensor(float("inf"), device=th_device),
                                            shape=tuple(), torch_device=th_device)
        
        self._locomotion_episode_config = LocomotionVecEnv.EpisodeLocomConfiguration(
                                            goal_abs_vel_vec_xys       = None,
                                            goal_rel_vel_vec_xys       = self._thtens([1.0,0.0,0.0]).expand(adapter.vec_size(), 3).clone(),
                                            goal_abs_gravity_vec_xyz   = self._thtens([0.0,0.0,-1.0]).expand(adapter.vec_size(), 3).clone(),
                                            goal_abs_height_vec_z      = self._thtens([sum(self._loco_conf.goal_height_minmax)/2]).expand(adapter.vec_size(), 1).detach().clone(),
                                            goal_heading_rel_vec_yaw   = self._thtens([0.0]).expand(adapter.vec_size(), 1).detach().clone(),
                                            goal_yaw_vel_vec           = self._thtens([0.0]).expand(adapter.vec_size(), 1).detach().clone(),
                                            reward_weight_actacc       = self._thzeros((num_envs,1)),
                                            reward_weight_actdiff      = self._thzeros((num_envs,1)),
                                            reward_weight_posref_vel   = self._thzeros((num_envs,1)),
                                            reward_weight_torque       = self._thzeros((num_envs,1)),
                                            reward_weight_acceleration = self._thzeros((num_envs,1)),
                                            reward_weight_acc_on_vel   = self._thzeros((num_envs,1)),
                                            reward_weight_posref_acc   = self._thzeros((num_envs,1)))
        robot_init_args.single_reward_space = single_reward_space
        super().__init__(robot_init_args)

        sub_rewards = {}
        reward = self.compute_rewards(self._current_state, sub_rewards)
        self.vec_reward_space=batch_space(self.single_reward_space, self._configuration.vec_size)
        obs_labels = self._state_helper.observation_names()
        ggLog.info(f"LocomotionVecEnv: reward shape = {reward.shape}")
        ggLog.info(f"LocomotionVecEnv: single_reward_space = {self.single_reward_space}")
        ggLog.info(f"LocomotionVecEnv: single_reward_space.labels = {self.single_reward_space.labels}")
        ggLog.info(f"observation_space = {self._state_helper.get_vec_obs_space()}")
        ggLog.info(f"Obs labels = \n{pprint.pformat(obs_labels)}")
        # ggLog.info(f"Env constructed")

    @override
    def _build_stats(self):
        self._stats = {}
        self._buff_sizes = int(self._configuration.original_max_epsteps/10)
        self._stats["vel_errs_vec"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["height_errs_vec"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["pitchnroll_errs_vec"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["body_speeds_vec"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["ep_avg_vel_err_vec"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_avg_height_err_vec"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_avg_pitchnroll_err_vec"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_avg_bodyspeed_vec"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_max_javg_sensed_effort"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_max_peak_sensed_effort"] = self._thzeros((self._configuration.vec_size,))

    @override
    def _build(self):
        # Set up monitored collision pairs for MjxAdapter
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            from adarl.adapters.MjxAdapter import MjxAdapter
            mjx_adapter : MjxAdapter = self._adapter #type: ignore
            # Create collision pairs: each foot vs ground
            feet_ground_collision_pairs = [(foot, self._configuration.ground_link) for foot in self._loco_conf.feet_links]
            mjx_adapter.set_monitored_collision_pairs(feet_ground_collision_pairs)
        super()._build()
        # self._feet_link_ids = self._adapter.get_links_ids(self._loco_conf.feet_links)
        # self._ground_link_id = self._adapter.get_links_ids([self._configuration.ground_link])
        self._adapter.set_monitored_links(self._adapter.get_monitored_links() + self._loco_conf.feet_links)
        self._feet_and_body_link_ids = self._adapter.get_monitored_links_ids(self._loco_conf.feet_links + [self._configuration.main_body_link])
        



    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        base_loco_fields = [self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED,
                            self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT,
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X,
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y,
                            self.LOCOMOTION_FIELDS.GOAL_YAW_VEL,
                            self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR]
        privileged_loco_fields = [  self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY]
        self._locomotion_state_helper = ThBoxStateHelper( field_names=[e for e in self.LOCOMOTION_FIELDS],
                                                    fields_minmax={ self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED : [0,10],
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_YAW_VEL : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SUM_IMPULSES : [0,10000],
                                                                    self.LOCOMOTION_FIELDS.COLLISON_COUNT : [0,1000],
                                                                    self.LOCOMOTION_FIELDS.CRASHED : [0,1],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_IS_JUMPING : [0,1],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_NUM_FEET_ON_GROUND : [0,4]},
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    history_length=3,
                                                    observation_definitions={  
                                                        "base" : ThBoxStateHelper.SimpleObsDef(observable_fields=base_loco_fields),
                                                        "privileged" : ThBoxStateHelper.SimpleObsDef(observable_fields=base_loco_fields+privileged_loco_fields)},
                                                    vec_size=adapter.vec_size())
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_LOCOMOTION,
                                                            self._locomotion_state_helper,
                                                            obs_defs={"base":{"observable":True,"concatenate":True,"noise":None},
                                                                      "privileged":{"observable":True,"concatenate":True,"noise":None}})
        observe_rew_weights = True
        self._reward_weights_state_helper = ThBoxStateHelper( field_names=self._loco_conf.enabled_rewards,
                                                    fields_minmax={n : [-1.0,1.0] for n in self._loco_conf.enabled_rewards}, # -1,1, makes it so that it does not get scaled
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    history_length=1,
                                                    observation_definitions={
                                                        "base" : ThBoxStateHelper.SimpleObsDef.fully_observable() if observe_rew_weights else ThBoxStateHelper.SimpleObsDef.not_observable(),
                                                        "privileged" : ThBoxStateHelper.SimpleObsDef.fully_observable() if observe_rew_weights else ThBoxStateHelper.SimpleObsDef.not_observable()},
                                                    flatten_observation=True,
                                                    vec_size=adapter.vec_size())
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_REWARDS,
                                                            self._reward_weights_state_helper,
                                                            obs_defs={"base":{"observable":True,"concatenate":False,"noise":None},
                                                                      "privileged":{"observable":True,"concatenate":False,"noise":None}})
        
        feet_num = len(self._loco_conf.feet_links)
        self._feet_state_helper = ThBoxStateHelper( field_names=[e for e in self.FEET_FIELDS],
                                                    fields_minmax={ 
                                                        self.FEET_FIELDS.FEET_AIR_DURATIONS     : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_GROUND_DURATIONS  : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_VEL_X : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_VEL_Y : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_REL_POS_X : th.as_tensor([[-1.0],[1.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_REL_POS_Y : th.as_tensor([[-1.0],[1.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_REL_POS_Z : th.as_tensor([[-1.0],[1.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_ABS_POS_Z : th.as_tensor([[-1.0],[1.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.PEAK_POS_Z : th.as_tensor([[-2.0],[2.0]]).expand(2,feet_num)},
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(len(self._loco_conf.feet_links),),
                                                    vec_size=adapter.vec_size(),
                                                    history_length=1,
                                                    observation_definitions=
                                                        {   "privileged" : ThBoxStateHelper.SimpleObsDef(observable_fields=[self.FEET_FIELDS.FEET_AIR_DURATIONS,
                                                                                                                            self.FEET_FIELDS.FEET_GROUND_DURATIONS,
                                                                                                                            self.FEET_FIELDS.FEET_VEL_X,
                                                                                                                            self.FEET_FIELDS.FEET_VEL_Y,
                                                                                                                            self.FEET_FIELDS.FEET_REL_POS_X,
                                                                                                                            self.FEET_FIELDS.FEET_REL_POS_Y,
                                                                                                                            self.FEET_FIELDS.FEET_REL_POS_Z,
                                                                                                                            self.FEET_FIELDS.FEET_ABS_POS_Z,
                                                                                                                            self.FEET_FIELDS.PEAK_POS_Z]),
                                                            "base" : ThBoxStateHelper.SimpleObsDef(observable_fields=[  self.FEET_FIELDS.FEET_REL_POS_X,
                                                                                                                        self.FEET_FIELDS.FEET_REL_POS_Y,
                                                                                                                        self.FEET_FIELDS.FEET_REL_POS_Z])})
        feet_noise = None #TODO add noise
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_FEET,
                                                            self._feet_state_helper,
                                                            obs_defs={"privileged":{"observable":True,"concatenate":True,"noise":None},
                                                                      "base":{"observable":True,"concatenate":True,"noise":feet_noise}})
        if self._loco_conf.heightmap_resolution_xy[0] > 0:
            heightmap_state_helper = ThBoxStateHelper( field_names=["map"],
                                                    fields_minmax={"map" : self._thtens([-10.0, 10.0])},
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=self._loco_conf.heightmap_resolution_xy,
                                                    vec_size=adapter.vec_size())
            self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_HEIGHTMAP,
                                                                 heightmap_state_helper,
                                                                 obs_defs={"base":{"observable":True,"concatenate":False,"noise":None}})
        ggLog.info(f"Built state/obs/action helpers")

    def _reset_state_full(self):
        super()._reset_state_full()
        # These need to be set to something valid to avoid issues at the start
        self._current_state[self.STATE_LOCOMOTION][:,:,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X] = 1.0
        self._current_state[self.STATE_LOCOMOTION][:,:,self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X] = 1.0
        self._current_state[self.STATE_LOCOMOTION][:,:,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z] = -1.0

        
    def _get_loco_adapter_data(self, super_adapter_data):
        if isinstance(self._adapter,BaseVecSimulationAdapter):
            lstates = self._adapter.getLinksState(requestedLinks = self._feet_and_body_link_ids, use_com_pose = False)
            nfeet = lstates.shape[1] - 1  # number of feet links
            feet_linvels_vec_foot_xyz = lstates[:,:4,7:10]
            bstate = lstates[:,4]
            borient_quat_vec_xyzw = bstate[:,3:7] # (nenvs,4)
            body_pos_vec_xyz = bstate[:,0:3].unsqueeze(1)  # (nenvs,1,3)
            feet_abs_pos_vec_foot_xyz = lstates[:,:nfeet,0:3] # (nenvs,nfeet,3)
            feet_rel_pos_vec_foot_xyz = th_quat_rotate(feet_abs_pos_vec_foot_xyz - body_pos_vec_xyz,
                                                   th_quat_conj(borient_quat_vec_xyzw).unsqueeze(1).expand(-1,nfeet,-1)) # (nenvs,nfeet,3)
        else:
            if self.num_envs == 1:
                jpos = super_adapter_data[1][0,:,0]
                self._robot_model.set_joint_pose_by_names({jn[1]:jpos[i] for i,jn in enumerate(self._configuration.joints_agent_controlled)} )
                feet_poses_dict = self._robot_model.get_frame_poses_xyzxyzw(self._configuration.main_body_link[1],[l[1] for l in self._loco_conf.feet_links])
                feet_positions_xyz = self._thtens([fp[:3] for fp in feet_poses_dict.values()])
                feet_rel_pos_vec_foot_xyz = feet_positions_xyz.unsqueeze(0)
            else:
                raise NotImplementedError("Feet positions are only implemented for single env when not using a simulation adapter")
            feet_linvels_vec_foot_xyz = self._thzeros((self.num_envs,4,3))
            feet_abs_pos_vec_foot_xyz = self._thzeros((self.num_envs,4,3))
            borient_quat_vec_xyzw = self._unit_quaternion.expand((self.num_envs,4))
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            from adarl.adapters.MjxAdapter import MjxAdapter
            mjx_adapter : MjxAdapter = self._adapter #type: ignore
            feet_are_touching_ground = mjx_adapter.check_colliding_links()  # Returns all monitored pairs (feet vs ground)
        else:
            feet_are_touching_ground = self._thzeros((self.num_envs,4))
        return feet_linvels_vec_foot_xyz, feet_rel_pos_vec_foot_xyz, feet_abs_pos_vec_foot_xyz, feet_are_touching_ground, borient_quat_vec_xyzw

    @override
    def _get_adapter_data_raw(self):
        record_region_start("LocomotionVecEnv._get_adapter_data_raw")
        super_adapter_data = super()._get_adapter_data_raw()
        loco_adapter_data = self._get_loco_adapter_data(super_adapter_data)
        record_region_end("LocomotionVecEnv._get_adapter_data_raw")
        return loco_adapter_data, super_adapter_data

    @override
    def _get_new_instantaneous_state(self, adapter_data):
        nenvs = self.num_envs
        loco_adapter_data, super_adapter_data = adapter_data
        feet_linvels_vec_foot_xyz, feet_rel_pos_vec_foot_xyz, feet_abs_pos_vec_foot_xyz, feet_are_touching_ground, borient_quat_vec_xyzw = loco_adapter_data

        # CAREFUL WITH THESE PREV STATES! Ensure they stay consistent at resets
        prev_locom_state =      self._current_state[self.STATE_LOCOMOTION][:, 0]
        prev_internal_state =   self._current_state[self.STATE_INTERNAL][:, 0]
        prev_extrinsic_state =  self._current_state[self.STATE_EXTRINSIC][:, 0]
        prev_feet_state =       self._current_state[self.STATE_FEET][:, 0]

        eps_resetting = (prev_internal_state[:, self.INTERNAL_FIELDS.STEP_COUNT] == -1).view((nenvs,))

        new_inst_state = super()._get_new_instantaneous_state(super_adapter_data)

        new_internal_state = new_inst_state[self.STATE_INTERNAL]
        new_extrinsic_state = new_inst_state[self.STATE_EXTRINSIC]
        
        # sadly in this point everything is a dict, so things must be addressed like this, maybe something could be done about this
        body_rel_linvel_vec_xyz = th.cat([new_extrinsic_state[k] for k in
                                        [self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                         self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                         self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z]], dim = 1)
        gravity_rel_vec_xyz     = th.cat([new_extrinsic_state[k] for k in 
                                        [self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                         self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                         self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z]], dim = 1)
        prev_gravity_rel_vec_xyz = prev_extrinsic_state[:,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1].view((nenvs,3))
        masked_assign(prev_gravity_rel_vec_xyz, eps_resetting, gravity_rel_vec_xyz) # at episode start prev values may be invalid

        curr_goal_abs_gravity_vec_xyz = self._locomotion_episode_config.goal_abs_gravity_vec_xyz
        prev_goal_abs_gravity_vec_xyz = prev_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X:self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z+1].view((nenvs,3))
        masked_assign(prev_goal_abs_gravity_vec_xyz, eps_resetting, curr_goal_abs_gravity_vec_xyz) # at episode start prev state values may be invalid
                
        max_goal_height_diff = self._loco_conf.max_goal_height_pos_change_speed*self._configuration.stepLength_sec
        goal_height = self._locomotion_episode_config.goal_abs_height_vec_z
        prev_smoothed_goal_height = prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        smoothed_goal_height = prev_smoothed_goal_height + th.clamp(goal_height - prev_smoothed_goal_height, min=-max_goal_height_diff, max=max_goal_height_diff)
        # smoothed_goal_height = goal_height*(1-goal_smoothing_alpha) + prev_smoothed_goal_height*goal_smoothing_alpha
        masked_assign(smoothed_goal_height,         eps_resetting,   goal_height)

        if self._locomotion_episode_config.goal_abs_vel_vec_xys is not None:
            if not isinstance_noimport(self._adapter, "BaseVecSimulationAdapter"):
                raise RuntimeError(f"Absolute velocity goals are supported only in simulation adapters, but adapter is of type {type(self._adapter)}")
            # Get the relative goal from the absolute one
            # Only possible with body pose (i.e. in simulation)
            goal_speed = self._locomotion_episode_config.goal_abs_vel_vec_xys[:,2].view((nenvs,1))
            abs_goal_linvel_direction_xy = self._locomotion_episode_config.goal_abs_vel_vec_xys[:,:2]/th.linalg.norm(self._locomotion_episode_config.goal_abs_vel_vec_xys[:,:2], dim=-1, keepdim=True)
            abs_goal_linvel_direction_xyz = th.cat([abs_goal_linvel_direction_xy, th.zeros_like(abs_goal_linvel_direction_xy[:,:1])], dim = 1) # should be always planar
            abs_goal_linvel_xyz = abs_goal_linvel_direction_xyz * goal_speed
            # abs_planar_linvelgoal_dir_quat = quat_xyzw_between_vecs_py(self._unit_3d_vector_vec_x, abs_planar_linvel_goal) # orientation of the linvel goal (quat that aligns (1,0,0) to it)
            rel_goal_linvel_dir_xyz = th_quat_rotate(abs_goal_linvel_direction_xyz, th_quat_conj(borient_quat_vec_xyzw))
        elif self._locomotion_episode_config.goal_rel_vel_vec_xys is not None:
            # The relative goal is expressed in the plane orthogonal to gravity
            # So the full realtive goal must be converted in the frame of the body.
            # In this formulation, we can see the planar relative goal direction as a twist around the gravity vector,
            # we have then to add a swing rotation, perpendicular to the gravity vector.
            # The swing can be obtained directly from the gravity vector, as the rotation that brings it to 0,0,-1
            goal_speed = self._locomotion_episode_config.goal_rel_vel_vec_xys[:,2].view((nenvs,1))
            rel_planar_goal_linvel_direction_xy = self._locomotion_episode_config.goal_rel_vel_vec_xys[:,:2]
            rel_planar_goal_linvel_direction_xyz = th.cat([rel_planar_goal_linvel_direction_xy[:,:2], th.zeros_like(rel_planar_goal_linvel_direction_xy[:,:1])], dim=1)
            swing = quat_xyzw_between_vecs_py(gravity_rel_vec_xyz, self._abs_gravity_dir.expand_as(gravity_rel_vec_xyz))
            twist = quat_xyzw_between_vecs_py(self._unit_3d_vector.expand_as(gravity_rel_vec_xyz), rel_planar_goal_linvel_direction_xyz)
            dir_quat = quat_mul_xyzw(twist,swing) #first swing then twist, i think
            rel_goal_linvel_dir_xyz = th_quat_rotate(self._unit_3d_vector.expand_as(gravity_rel_vec_xyz), dir_quat)
            abs_goal_linvel_xyz = th.zeros_like(rel_goal_linvel_dir_xyz) # not used in this branch
        else:
            raise RuntimeError(f"Neither absolute nor relative goal velocity is set, cannot compute goals")
        rel_goal_linvel_xyz = rel_goal_linvel_dir_xyz*goal_speed # relative to the body orientation
        rel_curr_heading_quat = quat_xyzw_between_vecs_py(rel_goal_linvel_dir_xyz, self._unit_3d_vector.expand(nenvs,3)) # orientation of the body with respect to linvel goal (quat that aligns linvel to the body)

        # compute linvel error
        tracking_err_vec = planar_tracking_error_vec(body_rel_linvel_vec_xyz, gravity_rel_vec_xyz, rel_goal_linvel_xyz).view(nenvs,1)
        
        # compute heading (yaw) error
        goal_rel_heading_yaw = self._locomotion_episode_config.goal_heading_rel_vec_yaw
        rel_goal_heading_quat = th.cat([self._thzeros((nenvs,2)),
                                        th.sin(goal_rel_heading_yaw/2).view((nenvs,1)),
                                        th.cos(goal_rel_heading_yaw/2).view((nenvs,1))], dim = 1)
        # the w component is by itself a measure of the size of the rotation, 2acos(w) would be the actual angle, but it is numerically unstable
        # in practice at w=1 the orientations are close, at -1 they are 180 degrees apart
        heading_error_vec = (1-quat_mul_xyzw(th_quat_conj(rel_goal_heading_quat), rel_curr_heading_quat)[:,3]).view(nenvs,1)
        
        # compute height error
        height_err_vec = new_extrinsic_state[self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] - smoothed_goal_height

        # compute pitch and roll error
        # pitchnroll_err_vec = th.linalg.norm(gravity_rel_vec_xyz-self._locomotion_episode_config.goal_abs_gravity_vec_xyz, dim = 1, keepdim=True) # Would be nice to use geodesic distance or somethinglike that
        pitchnroll_err_vec      = vectors_angle(gravity_rel_vec_xyz,      curr_goal_abs_gravity_vec_xyz).view((nenvs,1))
        pitchnroll_vel          = vectors_angle(gravity_rel_vec_xyz,      prev_gravity_rel_vec_xyz).view((nenvs,1))/new_internal_state[self.INTERNAL_FIELDS.LAST_STEP_DT].view((nenvs,1))
        pitchnroll_err_vel_vec = pitchnroll_vel

        # Compute absolute yaw velocity by projecting body angular velocity onto gravity (vertical) axis
        goal_yaw_vel = self._locomotion_episode_config.goal_yaw_vel_vec
        body_rel_angvel_xyz = th.cat([new_extrinsic_state[k] for k in
                                      [self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X,
                                       self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y,
                                       self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z]], dim=1)
        g_normalized = gravity_rel_vec_xyz / th.linalg.norm(gravity_rel_vec_xyz, dim=-1, keepdim=True)
        abs_yaw_vel = -th.sum(body_rel_angvel_xyz * g_normalized, dim=-1, keepdim=True)  # negate because gravity points down
        yaw_vel_error_vec = (goal_yaw_vel - abs_yaw_vel).view((nenvs,1))

        num_feet_on_ground = feet_are_touching_ground.sum(dim=1, keepdim=True)
        is_jumping = (num_feet_on_ground == 0).float()

        a_g = self._configuration.goal_err_exp_smoothing_1s**(self._configuration.stepLength_sec)
        smoothed_tracking_err_vec =         tracking_err_vec*(1-a_g) +        prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]*a_g
        smoothed_height_error =             height_err_vec*(1-a_g) +          prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR]*a_g
        smoothed_pitchnroll_error =         pitchnroll_err_vec*(1-a_g) +      prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR]*a_g
        smoothed_pitchnroll_error_vel_vec = pitchnroll_err_vel_vec*(1-a_g) +  prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY]*a_g
        smoothed_heading_error_vec =        heading_error_vec*(1-a_g) +       prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR]*a_g
        smoothed_yaw_vel_error_vec =        yaw_vel_error_vec*(1-a_g) +       prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR]*a_g
        a_j = self._loco_conf.is_jumping_alpha_1s**(self._configuration.stepLength_sec)
        smoothed_is_jumping =               is_jumping*(1-a_j) +              prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_IS_JUMPING]*a_j
        smoothed_num_feet_on_ground =       num_feet_on_ground*(1-a_j) +      prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_NUM_FEET_ON_GROUND]*a_j

        masked_assign(smoothed_tracking_err_vec,            eps_resetting,   tracking_err_vec)
        masked_assign(smoothed_height_error,                eps_resetting,   height_err_vec)
        masked_assign(smoothed_pitchnroll_error,            eps_resetting,   pitchnroll_err_vec)
        masked_assign(smoothed_heading_error_vec,           eps_resetting,   heading_error_vec)
        masked_assign(smoothed_pitchnroll_error_vel_vec,    eps_resetting,   pitchnroll_err_vel_vec)
        masked_assign(smoothed_yaw_vel_error_vec,           eps_resetting,   yaw_vel_error_vec)
        masked_assign(smoothed_is_jumping,                  eps_resetting,   is_jumping)
        masked_assign(smoothed_num_feet_on_ground,          eps_resetting,   num_feet_on_ground)

        if self._loco_conf.use_contacts:
            if not isinstance_noimport(self._adapter, "PyBulletAdapter"):
                raise RuntimeError(f"Contacts are supported only in pybullet for now")
            raise NotImplementedError()
            contacts = self._adapter.get_contacts()
            substep_count = len(contacts)
            contacts = sum(contacts,[]) # merge the contacts from all the substeps
            bad_contacts = [c for c in contacts if c[0] in self._loco_conf.disallowed_contact_links or c[1] in self._loco_conf.disallowed_contact_links]
            collision_count = len(contacts)/substep_count if substep_count != 0 else 0
            bad_forces = np.array([c[3] for c in bad_contacts])
            bad_durations = np.array([c[4] for c in bad_contacts])
            sum_bad_impulses = np.sum(np.abs(bad_forces*bad_durations))

            crashed = prev_locom_state[self.LOCOMOTION_FIELDS.CRASHED]
            if not crashed:
                # pairs = {(c[0],c[1]) for c in contacts}
                # print(f"contact pairs = {pairs}")
                for c in contacts:
                    if (c[0],c[1]) in self._loco_conf.terminating_contact_pairs or (c[1],c[0]) in self._loco_conf.terminating_contact_pairs:
                        crashed = 1
                        break
        else:
            collision_count_vec = th.zeros(size=(nenvs, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            sum_bad_impulses_vec = th.zeros(size=(nenvs, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            crashed_vec = th.zeros(size=(nenvs, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)



        new_reward_state = { 
            self.REWARD_WEIGHTS_FIELDS.CONTACTS               : self._loco_conf.reward_weight_contacts.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.FAILURE                : self._loco_conf.reward_weight_failure.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.FEET_AIR_TIME          : self._loco_conf.reward_weight_feet_air_time.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.FEET_GROUND_TIME       : self._loco_conf.reward_weight_feet_ground_time.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.FEET_ON_GROUND         : self._loco_conf.reward_weight_feet_on_ground.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.FEET_STEP_HEIGHT       : self._loco_conf.reward_weight_feet_step_height.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.HEADING                : self._loco_conf.reward_weight_heading.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.HEADING_VELOCITY       : self._loco_conf.reward_weight_heading_velocity.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.HEALTH                 : self._loco_conf.reward_weight_health.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.HEIGHT_POSITION        : self._loco_conf.reward_weight_height_position.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.HEIGHT_VELOCITY        : self._loco_conf.reward_weight_height_velocity.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_ACCELERATION     : self._locomotion_episode_config.reward_weight_acceleration.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_ACC_ON_VEL       : self._locomotion_episode_config.reward_weight_acc_on_vel.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_ACTACC           : self._locomotion_episode_config.reward_weight_actacc.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_ACTDIFF          : self._locomotion_episode_config.reward_weight_actdiff.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_POSITION         : self._loco_conf.reward_weight_joint_position.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_POSITION_LIMIT   : self._loco_conf.reward_weight_joint_position_limit.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_POSREF_ACC       : self._locomotion_episode_config.reward_weight_posref_acc.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_POSREF_VEL       : self._locomotion_episode_config.reward_weight_posref_vel.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_POWER            : self._loco_conf.reward_weight_joint_power.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_SENSED_EFFORT    : self._loco_conf.reward_weight_joint_sensed_effort.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_STAND_POSITION   : self._loco_conf.reward_weight_joint_stand_position.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_TORQUE           : self._locomotion_episode_config.reward_weight_torque.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_TORQUEDIFF       : self._loco_conf.reward_weight_joint_torquediff.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_TORQUE_LIMIT     : self._loco_conf.reward_weight_joint_torque_limit.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_TORQUE_REFS      : self._loco_conf.reward_weight_joint_torqueref.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_VELOCITY         : self._loco_conf.reward_weight_joint_velocity.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_VELOCITY_LIMIT   : self._loco_conf.reward_weight_joint_velocity_limit.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.JOINT_VELOCITY_REFS    : self._loco_conf.reward_weight_joint_velref.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.PITCHNROLL             : self._loco_conf.reward_weight_pitchnroll.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.PITCHNROLL_VELOCITY    : self._loco_conf.reward_weight_pitchnroll_velocity.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.SAFETY_TRIGGERED       : self._loco_conf.reward_weight_safety_triggered.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.SLIP                   : self._loco_conf.reward_weight_slip.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.TRACKING               : self._loco_conf.reward_weight_tracking.expand(nenvs,1),
            self.REWARD_WEIGHTS_FIELDS.YAW_VEL_TRACKING       : self._loco_conf.reward_weight_yaw_vel_tracking.expand(nenvs,1)
        }
        new_reward_state = {k:new_reward_state[self.REWARD_WEIGHTS_FIELDS[k.upper()]] for k in self._loco_conf.enabled_rewards}
        new_locom_state = { 
            self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : smoothed_tracking_err_vec.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR : smoothed_height_error.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR : smoothed_pitchnroll_error.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR : smoothed_heading_error_vec.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY : smoothed_pitchnroll_error_vel_vec.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_IS_JUMPING : smoothed_is_jumping.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_NUM_FEET_ON_GROUND : smoothed_num_feet_on_ground.view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X : rel_goal_linvel_dir_xyz[:,0].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y : rel_goal_linvel_dir_xyz[:,1].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z : rel_goal_linvel_dir_xyz[:,2].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED : goal_speed.view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X : abs_goal_linvel_xyz[:,0].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y : abs_goal_linvel_xyz[:,1].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z : abs_goal_linvel_xyz[:,2].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : self._locomotion_episode_config.goal_abs_height_vec_z,
            self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT : smoothed_goal_height.view(nenvs,1),
            self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR : smoothed_yaw_vel_error_vec.view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_YAW_VEL : goal_yaw_vel.view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X : th.cos(goal_rel_heading_yaw).view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y : th.sin(goal_rel_heading_yaw).view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,0].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,1].view(nenvs,1),
            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,2].view(nenvs,1),
            self.LOCOMOTION_FIELDS.SUM_IMPULSES : sum_bad_impulses_vec,
            self.LOCOMOTION_FIELDS.COLLISON_COUNT :collision_count_vec,
            self.LOCOMOTION_FIELDS.CRASHED : crashed_vec}
        
        nenv_nfeet = (nenvs,len(self._loco_conf.feet_links))
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            prev_feet_air_durations_vec_foot_t = prev_feet_state[:,self.FEET_FIELDS.FEET_AIR_DURATIONS]
            prev_feet_ground_durations_vec_foot_t = prev_feet_state[:,self.FEET_FIELDS.FEET_GROUND_DURATIONS]
            prev_avg_feet_step_durations = prev_feet_state[:,self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS]
            prev_peak_z = prev_feet_state[:,self.FEET_FIELDS.PEAK_POS_Z]

            eps_resetting_ext = eps_resetting.unsqueeze(1).expand(nenv_nfeet)
            prev_feet_air_durations_vec_foot_t    = th.where(condition=eps_resetting_ext, 
                                                          input=self._thzeros(nenv_nfeet), 
                                                          other=prev_feet_air_durations_vec_foot_t)
            prev_feet_ground_durations_vec_foot_t = th.where(condition=eps_resetting_ext,
                                                            input=self._thzeros(nenv_nfeet),
                                                            other=prev_feet_ground_durations_vec_foot_t)
            prev_avg_feet_step_durations          = th.where(condition=eps_resetting_ext,
                                                            input=self._thzeros(nenv_nfeet),
                                                            other=prev_avg_feet_step_durations)
            prev_peak_z                           = th.where(condition=eps_resetting_ext,
                                                            input=self._thzeros(nenv_nfeet),
                                                            other=prev_peak_z)

            dt = new_internal_state[self.INTERNAL_FIELDS.LAST_STEP_DT].view((nenvs,))
            feet_were_touching_ground = prev_feet_air_durations_vec_foot_t <= 0
            # if foot is just lifting off, mark the time in the state
            # if foot is already up, and stays up, leave the time there
            # if it is just now touching down, flip the time to negative and add the current time (so it becomes the negative step duration)
            # if it was already down, and stays down, write zero to it
            # lifting_off = th.logical_and(th.logical_not(feet_are_touching_ground), feet_were_touching_ground)
            just_touching_down = th.logical_and(th.logical_not(feet_were_touching_ground), feet_are_touching_ground)
            # new_feet_liftoffs_vec_foot_t = th.zeros_like(prev_feet_state)
            # ggLog.info(f"dt = {dt}")
            # ggLog.info(f"feet_were_touching_ground = \n{feet_were_touching_ground}")
            # ggLog.info(f"feet_are_touching_ground = \n{feet_are_touching_ground}")
            new_feet_air_durations_vec_foot_t  = prev_feet_air_durations_vec_foot_t.clone()
            th.where(condition=just_touching_down, #.expand(nenv_nfeet),
                     input = -prev_feet_air_durations_vec_foot_t, # if touching down, write negative step duration
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            th.where(condition=feet_were_touching_ground, #.expand(nenv_nfeet),
                     input = self._thtens(0.0), # if was touching ground already in the previous step, write zero
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            th.where(condition=th.logical_not(feet_are_touching_ground), #.expand(nenv_nfeet),
                     input = new_feet_air_durations_vec_foot_t+dt.unsqueeze(1).expand(nenv_nfeet), # if is up, increase time
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            
            feet_were_in_air = th.logical_not(feet_were_touching_ground)
            feet_are_in_air = th.logical_not(feet_are_touching_ground)
            just_lifting_up =    th.logical_and(feet_were_touching_ground, feet_are_in_air)
            new_feet_ground_durations_vec_foot_t  = prev_feet_ground_durations_vec_foot_t.clone()
            th.where(condition=just_lifting_up.expand(nenv_nfeet),
                     input = -prev_feet_ground_durations_vec_foot_t, # if touching down, write negative step duration
                     other = new_feet_ground_durations_vec_foot_t,
                     out   = new_feet_ground_durations_vec_foot_t)
            th.where(condition=feet_were_in_air.expand(nenv_nfeet),
                     input = self._thtens(0.0), # if was in the air already in the previous step, write zero
                     other = new_feet_ground_durations_vec_foot_t,
                     out   = new_feet_ground_durations_vec_foot_t)
            th.where(condition=feet_are_touching_ground.expand(nenv_nfeet),
                     input = new_feet_ground_durations_vec_foot_t+dt.unsqueeze(1).expand(nenv_nfeet), # if is on ground, increase time
                     other = new_feet_ground_durations_vec_foot_t,
                     out   = new_feet_ground_durations_vec_foot_t)
            max_ground_time_sec = 0.5
            new_feet_ground_durations_vec_foot_t = th.clamp(new_feet_ground_durations_vec_foot_t, max=max_ground_time_sec) # to avoid unbounded growth and limit state space

            # ggLog.info(f"prev_feet_step_durations_vec_foot_t = \n{prev_feet_step_durations_vec_foot_t}")
            # ggLog.info(f"new_feet_step_durations_vec_foot_t = \n{new_feet_step_durations_vec_foot_t}")
            new_avg_feet_step_durations = prev_avg_feet_step_durations.clone()
            a = self._loco_conf.feet_air_time_avg_alpha
            th.where(condition=just_touching_down.expand(nenv_nfeet),
                     input = prev_feet_air_durations_vec_foot_t*(1-a) + prev_avg_feet_step_durations*a,
                     other = new_avg_feet_step_durations,
                     out   = new_avg_feet_step_durations)
            feet_z = feet_abs_pos_vec_foot_xyz[:,:,2]
            peak_feet_z = th.where(condition=just_lifting_up.expand(nenv_nfeet),
                                    input = feet_z, # if just lifted up, record the foot height as the peak height of the step
                                    other = prev_peak_z)
            peak_feet_z = th.where(condition=feet_are_in_air.expand(nenv_nfeet),
                                    input = th.maximum(feet_z, peak_feet_z), # if just lifted up, record the foot height as the peak height of the step
                                    other = peak_feet_z) # otherwise keep the previous value (which is either the
        else:
            new_feet_air_durations_vec_foot_t = self._thtens([0.0]).expand(nenv_nfeet)
            new_feet_ground_durations_vec_foot_t = self._thtens([0.0]).expand(nenv_nfeet)
            new_avg_feet_step_durations = self._thtens([0.0]).expand(nenv_nfeet)
            peak_feet_z = self._thtens([0.0]).expand(nenv_nfeet)
        new_feet_state = {  self.FEET_FIELDS.FEET_AIR_DURATIONS : new_feet_air_durations_vec_foot_t,
                            self.FEET_FIELDS.FEET_GROUND_DURATIONS : new_feet_ground_durations_vec_foot_t,
                            self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS : new_avg_feet_step_durations,
                            self.FEET_FIELDS.FEET_VEL_X : feet_linvels_vec_foot_xyz[:,:,0],
                            self.FEET_FIELDS.FEET_VEL_Y : feet_linvels_vec_foot_xyz[:,:,1],
                            self.FEET_FIELDS.FEET_REL_POS_X : feet_rel_pos_vec_foot_xyz[:,:,0],
                            self.FEET_FIELDS.FEET_REL_POS_Y : feet_rel_pos_vec_foot_xyz[:,:,1],
                            self.FEET_FIELDS.FEET_REL_POS_Z : feet_rel_pos_vec_foot_xyz[:,:,2],
                            self.FEET_FIELDS.FEET_ABS_POS_Z : feet_abs_pos_vec_foot_xyz[:,:,2],
                            self.FEET_FIELDS.PEAK_POS_Z : peak_feet_z
                            }

        new_inst_state[self.STATE_LOCOMOTION] = new_locom_state
        new_inst_state[self.STATE_REWARDS] = new_reward_state
        new_inst_state[self.STATE_FEET] = new_feet_state
        return new_inst_state

    def _height_velocity_reward(self, curr_state_extr_vec, current_state_locom_vec, current_state_internal, prev_state_extr_vec):
        curr_pos = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        prev_pos = prev_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        goal_pos = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        dt = current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT]
        z_velocity = (curr_pos - prev_pos)/dt

        height_err = curr_pos-goal_pos
        # max_speed = self._loco_conf.max_height_speed_goal
        # kp = 2
        # goal_height_velocity = th.clamp(-height_err*kp, min=-max_speed, max=max_speed) 
        # reward_height = double_bell_reward(z_velocity-goal_height_velocity,
        #                                    bell_width_a=self._thtens(0.05),
        #                                    bell_width_b=goal_height_velocity*2+0.025,
        #                                    bell_b_weight=self._thtens(0.5))
        goal_height_velocity = th.zeros_like(z_velocity) # we want to maintain a constant height, so the goal velocity is zero
        reward_height = -z_velocity**2
        return reward_height, z_velocity, goal_height_velocity, dt, height_err
    
    def _velocity_from_position_reward(self,    curr_pos : th.Tensor,
                                                prev_pos : th.Tensor,
                                                prev_prev_pos : th.Tensor,
                                                goal_pos : th.Tensor,
                                                dt : th.Tensor,
                                                max_speed : th.Tensor,
                                                kp : th.Tensor,
                                                kd : th.Tensor,
                                                relative_bell_weight : float = 0.5,
                                                relative_bell_width : float = 2.0,
                                                relative_bell_epsilon : float = 0.025,
                                                absolute_bell_width : float = 0.1,
                                                final_reward_proportional_term : float = 0.0) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        curr_pos = curr_pos.view((self.num_envs,))
        prev_pos = prev_pos.view((self.num_envs,))
        goal_pos = goal_pos.view((self.num_envs,))
        dt = dt.view((self.num_envs,))
        max_speed = max_speed.view((1,))
        kp = kp.view((1,))

        err = prev_pos-goal_pos # This is the error that we were supposed to fix at the previous step
        prev_err = prev_prev_pos-goal_pos
        err_dt = (err - prev_err)/dt
        goal_velocity = th.clamp(-err*kp - err_dt*kd, min=-max_speed, max=max_speed) 
        velocity = (curr_pos - prev_pos)/dt
        velocity_err = velocity-goal_velocity
        reward : th.Tensor = double_bell_reward(velocity_err,
                                           bell_width_a=self._thtens(absolute_bell_width),
                                           bell_width_b=goal_velocity*self._thtens(relative_bell_width)+self._thtens(relative_bell_epsilon),
                                           bell_b_weight=self._thtens(relative_bell_weight))
        reward = reward + final_reward_proportional_term * (1 - err.abs())
        return reward, velocity, goal_velocity, velocity_err

    def _pitchnroll_velocity_penalty_reward(self, state):
        # curr_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        # prev_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,1,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        # angle_diff = vectors_angle(curr_rel_gravity_vec_xyz, prev_rel_gravity_vec_xyz).view((self.num_envs,))
        # angle_vel = angle_diff/self._configuration.stepLength_sec
        # return penalty_reward(angle_vel, max_rew=1, exponent=2)
        angvel_xy = state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X:self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y+1,0].view((self.num_envs,2))
        return -(angvel_xy.norm(dim=1)**2)

    def _pitchnroll_velocity_reward(self, state, dt : th.Tensor):
        curr_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prev_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,1,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prevprev_rel_gravity_vec_xyz  = state[self.STATE_EXTRINSIC][:,2,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prev_goal_gravity_vec_xyz     = state[self.STATE_LOCOMOTION][:,1,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X:self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z+1,0].view((self.num_envs,3))
        curr_goal_gravity_vec_xyz     = state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X:self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z+1,0].view((self.num_envs,3))
        prevprev_goal_gravity_vec_xyz = state[self.STATE_LOCOMOTION][:,2,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X:self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z+1,0].view((self.num_envs,3))

        curr_err = vectors_angle(curr_rel_gravity_vec_xyz, curr_goal_gravity_vec_xyz).view((self.num_envs,))
        prev_err = vectors_angle(prev_rel_gravity_vec_xyz, prev_goal_gravity_vec_xyz).view((self.num_envs,))
        prevprev_err = vectors_angle(prevprev_rel_gravity_vec_xyz, prevprev_goal_gravity_vec_xyz).view((self.num_envs,))

        return *self._velocity_from_position_reward(curr_pos = curr_err,
                                                    prev_pos = prev_err,
                                                    prev_prev_pos = prevprev_err,
                                                    goal_pos = self._thtens(0.0).expand(self.num_envs),
                                                    dt=dt,
                                                    max_speed=self._loco_conf.max_pitchnroll_speed_goal,
                                                    kp=self._loco_conf.pitchnroll_kp,
                                                    kd=self._loco_conf.pitchnroll_kd,
                                                    absolute_bell_width=th.pi/8), curr_err

    def _heading_velocity_reward(self, state, dt : th.Tensor):
        state_loco = state[self.STATE_LOCOMOTION]
        state_extr = state[self.STATE_EXTRINSIC]
        rel_goal_linvel_dir_xyz          = state_loco[:, 0,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        prev_rel_goal_linvel_dir_xyz     = state_loco[:, 1,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        prevprev_rel_goal_linvel_dir_xyz = state_loco[:, 2,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        rel_gravity_vec_xyz              = state_extr[:, 0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prev_rel_gravity_vec_xyz         = state_extr[:, 1,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prevprev_rel_gravity_vec_xyz     = state_extr[:, 2,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        rel_body_direction_xyz = self._unit_3d_vector.expand_as(rel_gravity_vec_xyz) # The body orientation in the body frame
        
        flattening_rotation          = quat_xyzw_between_vecs_py(rel_gravity_vec_xyz,           self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        prev_flattening_rotation     = quat_xyzw_between_vecs_py(prev_rel_gravity_vec_xyz,      self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        prevprev_flattening_rotation = quat_xyzw_between_vecs_py(prevprev_rel_gravity_vec_xyz,  self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        
        # rotate body and goal direction into the flattened frame
        curr_bodydir_flatbodyframe      = th_quat_rotate(rel_body_direction_xyz, flattening_rotation)
        prev_bodydir_flatbodyframe      = th_quat_rotate(rel_body_direction_xyz, prev_flattening_rotation)
        prevprev_bodydir_flatbodyframe  = th_quat_rotate(rel_body_direction_xyz, prevprev_flattening_rotation)
        curr_linvelgoaldir_flatbodyframe     = th_quat_rotate(rel_goal_linvel_dir_xyz,          flattening_rotation)
        prev_linvelgoaldir_flatbodyframe     = th_quat_rotate(prev_rel_goal_linvel_dir_xyz,     prev_flattening_rotation)
        prevprev_linvelgoaldir_flatbodyframe = th_quat_rotate(prevprev_rel_goal_linvel_dir_xyz, prevprev_flattening_rotation)

        eps = 1e-3
        dbg_check(lambda : th.all(th.stack([th.all(curr_linvelgoaldir_flatbodyframe[:,2]<eps),
                                            th.all(prev_linvelgoaldir_flatbodyframe[:,2]<eps),
                                            th.all(prevprev_linvelgoaldir_flatbodyframe[:,2]<eps)])),
                  assert_msg="flattened goal vectors not parallel to ground",
                  async_assert=True,
                  build_msg=lambda : f"flattened goal vectors not parallel to ground:\n"
                                     f" envs: {th.arange(self.num_envs, device=rel_goal_linvel_dir_xyz.device)[(curr_linvelgoaldir_flatbodyframe[:,2]>=eps)|(prev_linvelgoaldir_flatbodyframe[:,2]>=eps)|(prevprev_linvelgoaldir_flatbodyframe[:,2]>=eps)]}\n"
                                     f" curr_linvelgoaldir_flatbodyframe[:,2]={curr_linvelgoaldir_flatbodyframe[:,2]}\n"
                                     f"   violations = {curr_linvelgoaldir_flatbodyframe[:,2][curr_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   rel_goal_linvel_dir_xyz = {rel_goal_linvel_dir_xyz[curr_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   flattening_rotation = {flattening_rotation[curr_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   (failing steps = {state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT][curr_linvelgoaldir_flatbodyframe[:,2]>=eps]})\n"
                                     f" prev_linvelgoaldir_flatbodyframe[:,2]={prev_linvelgoaldir_flatbodyframe[:,2]}\n"
                                     f"   violations = {prev_linvelgoaldir_flatbodyframe[:,2][prev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   prev_rel_goal_linvel_dir_xyz = {prev_rel_goal_linvel_dir_xyz[prev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   prev_flattening_rotation = {prev_flattening_rotation[prev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   (failing steps = {state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT][prev_linvelgoaldir_flatbodyframe[:,2]>=eps]})\n"
                                     f" prevprev_linvelgoaldir_flatbodyframe[:,2]={prevprev_linvelgoaldir_flatbodyframe[:,2]}\n"
                                     f"   violations = {prevprev_linvelgoaldir_flatbodyframe[:,2][prevprev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   prevprev_rel_goal_linvel_dir_xyz = {prevprev_rel_goal_linvel_dir_xyz[prevprev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   prevprev_flattening_rotation = {prevprev_flattening_rotation[prevprev_linvelgoaldir_flatbodyframe[:,2]>=eps]}\n"
                                     f"   (failing steps = {state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT][prev_linvelgoaldir_flatbodyframe[:,2]>=eps]})\n")
        
        curr_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        curr_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prev_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prev_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prevprev_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prevprev_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane

        rel_headinggoal_flatgoalframe = th.zeros_like(curr_bodydir_flatbodyframe)
        rel_headinggoal_flatgoalframe[:,:2] = state_loco[:,0,self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X:self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y+1,0]
        headinggoal_flatbodyframe = th_quat_rotate(rel_headinggoal_flatgoalframe, quat_xyzw_between_vecs_py(self._unit_3d_vector.expand_as(rel_headinggoal_flatgoalframe), curr_linvelgoaldir_flatbodyframe))


        curr_yaw_err = vectors_angle(curr_bodydir_flatbodyframe, headinggoal_flatbodyframe)
        prev_yaw_err = vectors_angle(prev_bodydir_flatbodyframe, headinggoal_flatbodyframe)
        prevprev_yaw_err = vectors_angle(prevprev_bodydir_flatbodyframe, headinggoal_flatbodyframe)
        
        return *self._velocity_from_position_reward(curr_pos = curr_yaw_err,
                                                    prev_pos = prev_yaw_err,
                                                    prev_prev_pos = prevprev_yaw_err,
                                                    goal_pos = th.zeros_like(prev_yaw_err),
                                                    dt=dt,
                                                    max_speed=self._loco_conf.max_heading_speed_goal,
                                                    kp=self._loco_conf.heading_kp,
                                                    kd=self._loco_conf.heading_kd,
                                                    relative_bell_width=1.1), curr_yaw_err


    @override
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        rewards, sub_rewards_dict = self._compute_rewards(state) # Avoid input mutation for compiled function
        sub_rewards_return.update(sub_rewards_dict)
        return rewards

    @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune",
                                    #   skip_eval_unsafe_warmup=100, skip_eval_unsafe_manual_arg_guard=0,
                                      disable=disable_compile)
    def _compute_rewards(self,   state : dict[str,th.Tensor]) -> tuple[th.Tensor, dict[str,th.Tensor]]:
        if self._loco_conf.playground_style_reward:
            raise NotImplementedError("playground style reward has been removed")

        sub_rewards_return = {}
        if self._configuration.fixed_reward:
            rews_num = len(self._loco_conf.enabled_rewards)
            sub_rewards_return = {k:self._thones((self.num_envs,))/rews_num for k in self._loco_conf.enabled_rewards}
            if self._loco_conf.split_rewards:
                reward = self._thones((self.num_envs, len(sub_rewards_return)))/rews_num
            else:
                reward = self._thones((self.num_envs, 1))
            return reward, sub_rewards_return
        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = self._configuration.reward_penalties_max
        current_state_locom_vec = state[self.STATE_LOCOMOTION][:, 0,:,0]
        current_state_rewards_vec = state[self.STATE_REWARDS][:, 0,:,0]
        curr_state_extr_vec = state[self.STATE_EXTRINSIC][:, 0,:,0]
        prev_state_extr_vec = state[self.STATE_EXTRINSIC][:, 1,:,0]
        current_state_internal = state[self.STATE_INTERNAL][:, 0,:,0]
        state_action_raw_vec = state[self.STATE_ACT_RAW_HIST]
        state_stats_v_h_j_minmaxavgstd_pvaeep = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 6)
        last_step_dt = current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT].view((self.num_envs,))
        goal_speed = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED].view((self.num_envs,))
        goal_yaw_speed = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.GOAL_YAW_VEL].view((self.num_envs,))
        should_be_moving = th.logical_or(goal_speed.view((self.num_envs,1)) > 0.05,
                                         goal_yaw_speed.view((self.num_envs,1)).abs() > 10/180*th.pi)

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_position, lims[0,:,0], lims[1,:,0])
        state_robot = state[self.STATE_ROBOT]
        state_robot_norm        = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state_robot, warn_limits_violation=False)
        longterm_stats_pos_norm = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)
        joints_num = state_robot_norm.size()[2]
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state_robot, self._safety_limits, warn_limits_violation=False)
        norm_posstathomingdiff    = longterm_stats_pos_norm[:,0,0] - normhoming
        norm_positions       = state_robot_norm[:,0,:,0]
        norm_velocities      = state_robot_norm[:,0,:,1]
        norm_velocity_refs   = state_robot_norm[:,0,:,6]
        norm_torque_refs     = state_robot_norm[:,0,:,7]
        norm_poshomingdiff   = norm_positions - normhoming
        posref_vel           = (state_robot[:,0,:,5] - state_robot[:,1,:,5])/last_step_dt.unsqueeze(1).expand((self.num_envs,joints_num))
        prev_posref_vel      = (state_robot[:,1,:,5] - state_robot[:,2,:,5])/last_step_dt.unsqueeze(1).expand((self.num_envs,joints_num))
        posref_acc    = (posref_vel-prev_posref_vel)/last_step_dt.unsqueeze(1).expand((self.num_envs,joints_num))
        max_jacc = 1_000 # max expected joint acceleration (not really a strict max)
        max_sensed_effort = state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,0:2,4].abs().amax(dim=2)
        norm_accelerations   = state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,2,2]/max_jacc # normalized average accelearation
        actdiff             = th.flatten((state_action_raw_vec[:,0] - state_action_raw_vec[:,1])/2, start_dim=1) # divide by 2 to keep it in [-1,1]
        prev_actdiff        = th.flatten((state_action_raw_vec[:,1] - state_action_raw_vec[:,2])/2, start_dim=1)
        act_acc             = (actdiff - prev_actdiff)/2

        position_safenorm   = state_robot_safenorm[:,0,:,0]
        velocities_safenorm = state_robot_safenorm[:,0,:,1]
        torque_safenorm     = state_robot_safenorm[:,0,:,2]



        # ---------------- JOINT-LEVEL PENALTIES ----------------

        bad_effort_threshold = 200.0
        flattened_max_sensed_effort = max_sensed_effort*smoothclip_flattener(max_sensed_effort, bad_effort_threshold, bad_effort_threshold/10)
        reward_sensed_effort    = norm_penalty(flattened_max_sensed_effort, norm=4, power=2, squash_max=1.0, squash_smoothness=4.0)
        reward_velocity         = joint_penalty_reward(norm_velocities,max_rew=max_rew,exponent=2)
        reward_acceleration     = norm_penalty(norm_accelerations, norm=4.0, power=2.0, squash_max=1.0, squash_smoothness=4.0)
        reward_position         = flattened_joint_penalty_reward(norm_posstathomingdiff,max_rew=max_rew, exponent=2.0, flattening_scale=0.02)
        reward_actdiff          = joint_penalty_reward(actdiff,max_rew=1, exponent=2, presquash_factor=10)
        reward_actacc           = joint_penalty_reward(act_acc,max_rew=1, exponent=2, presquash_factor=100)
        reward_torque_limit     = joint_penalty_reward(torque_safenorm,max_rew=1,exponent=50)
        reward_position_limit   = joint_penalty_reward(position_safenorm,max_rew=1,exponent=50)
        reward_velocity_limit   = joint_penalty_reward(velocities_safenorm,max_rew=1,exponent=50)

        acc_on_vel = norm_accelerations/(norm_velocities+0.1)
        reward_acc_on_vel = norm_penalty(acc_on_vel, norm=4.0, power=2.0, squash_max=1.0, squash_smoothness=4.0)
        avg_cmd_torque = state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,2,3] # average torque of each joint over the simulation substeps
        avg_mechanical_power = state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,2,5] # average power of each joint over the simulation substeps
        reward_power = norm_penalty(avg_mechanical_power, norm=1.0, power=1.0, squash_max=100000.0, 
                                    squash_smoothness=4.0)/joints_num
        # We try to make it so that the cmdtorque reward expresses roughly the 
        # motor copper power losses.
        # Following this logic 
        # - K_t represents the "motor torque constant", in Nm/A, the ratio between torque and current, tau = K_t * I
        # - R   represents the "motor resistance", in Ohms, the ratio between voltage and current, so Power = I^2*R = (tau/K_t)^2 * R
        # - K_m represents the "motor constant", in Nm/sqrt(W), K_m = K_t/sqrt(R), so Power = (tau/K_m)^2
        # With this the L2 norm becomes the total copper loss of the motors, in W, same scale as the power reward
        K_m = 2.5 # reasonable froma B1-kyon sized quadruped, maybe, can be compensated by the reward weight
        reward_cmdtorque = norm_penalty(avg_cmd_torque/K_m, norm=2.0, power=2.0, squash_max=100000.0, 
                                        squash_smoothness=4.0)/joints_num

        torque_diff = state_robot[:,0,:,2] - state_robot[:,1,:,2]
        reward_torquediff = norm_penalty(torque_diff, norm=2.0, power=2.0, squash_max=100000.0,
                                        squash_smoothness=4.0)/joints_num


        reward_velocity_refs    = joint_penalty_reward(norm_velocity_refs,   max_rew=max_rew,exponent=2)
        reward_torque_refs      = joint_penalty_reward(norm_torque_refs,     max_rew=max_rew,exponent=2)
        posref_vel_threshold = 6.5
        posref_vel_max = 15.0
        abs_posref_vel = posref_vel.abs()
        flattened_posref_vels = abs_posref_vel*smoothclip_flattener(abs_posref_vel, posref_vel_threshold, 1.0)
        reward_posref_vel       = norm_penalty(flattened_posref_vels/posref_vel_max,     norm=2, power=2, squash_max=1.0, squash_smoothness=4.0)
        reward_posref_acc       = norm_penalty(posref_acc/1_000,  norm=4, power=1.0, squash_max=2.0, squash_smoothness=2.0)
        

        # ---------------- STAND JOINT_POSITION REWARD ----------------
        
        reward_stand_position   = joint_penalty_reward(norm_poshomingdiff,    max_rew=max_rew, exponent=1.0)*(th.logical_not(should_be_moving.view((self.num_envs,))))
        
        # ---------------- TRACKING REWARDS ----------------
        
        # ---- Height ----
        # Height of the pelvis
        reward_height_velocity, _, _, _, _ = self._height_velocity_reward(curr_state_extr_vec, current_state_locom_vec, current_state_internal, prev_state_extr_vec)
        height_err = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]-current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        reward_height_position = double_bell_reward( error=height_err,
                                            bell_width_a=self._loco_conf.height_reward_settle_point,
                                            bell_width_b=self._loco_conf.height_reward_2_settle_point,
                                            bell_b_weight=self._loco_conf.height_reward_2_weight)
        
        # ---- Pitch and Roll ----
        # Pitch and Roll of the pelvis
        reward_pitchnroll = 1+penalty_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR]*4, max_rew=1, exponent=2.0)
        reward_pitchnroll_velocity = self._pitchnroll_velocity_penalty_reward(state)

        # ---- Heading ----
        # This is the direction the robot should be facing, relative to the direction it should be moving.
        reward_heading_position = bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR],
                                            zero_rew_dist=self._loco_conf.heading_reward_settle_point)
        reward_heading_velocity, _, _, _, _ = self._heading_velocity_reward(state, last_step_dt)

        # ---- Linear Velocity Tracking ----
        # Linear velocity of the pelvis
        goalrelative_weight = self._loco_conf.vel_reward_goalrelative_weight
        rel_goal_bell_width = self._loco_conf.reward_vel_goal_relative_width
        rel_goal_offset = self._loco_conf.reward_vel_goal_relative_width_offset
        abs_goal_bell_width = self._loco_conf.reward_vel_goal_absolute_width
        velocity_tracking_err_vec = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        reward_velocity_tracking = double_bell_reward(velocity_tracking_err_vec,
                                                      abs_goal_bell_width,
                                                      rel_goal_bell_width*(goal_speed+rel_goal_offset),
                                                      goalrelative_weight)
        
        # ---- Yaw Velocity Tracking ----
        # Yaw velocity of the pelvis
        abs_yaw_vel_bell_width = self._loco_conf.reward_yaw_vel_goal_absolute_width
        rel_yaw_vel_bell_width = self._loco_conf.reward_yaw_vel_goal_relative_width
        rel_goal_bell_width_offset = self._loco_conf.reward_yaw_vel_goal_relative_width_offset
        yaw_vel_relative_weight = self._loco_conf.reward_yaw_vel_goal_relative_weight
        reward_yaw_vel_track = double_bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR],
                                                  abs_yaw_vel_bell_width,
                                                  rel_yaw_vel_bell_width*(goal_speed+rel_goal_bell_width_offset),
                                                  yaw_vel_relative_weight)

        # ---------------- FEET AIR TIME REWARD ----------------
        # This reward tries to encourage good durations of feet being in the air
        feet_state = state[self.STATE_FEET][:,0] # vec_size*history*fields*nfeet -> vec_size*fields*nfeet
        feet_air_durations_secs = feet_state[:,self.FEET_FIELDS.FEET_AIR_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        steps_finishing = feet_air_durations_secs < 0
        feet_air_durations_secs = -feet_air_durations_secs # When the value is positive then it is the duration of a currently ongoing step, which we dont consider
        # subtracting 0.1 from the durations makes it so that very short steps are actually penalized with a negative reward
        # this makes doing small steps worse than doing nothing
        # squash the durations to max_good step, and offset it o that steps shorter than min_good_step are negative
        clipped_feet_air_durations_secs = smooth_clip(feet_air_durations_secs, 
                                              self._loco_conf.max_good_step_duration,
                                              softness=8.0)
        corrected_air_durations = clipped_feet_air_durations_secs - self._loco_conf.min_good_step_air_duration
        # corrected_air_durations = th.tanh((feet_air_durations_secs - self._loco_conf.min_good_step_air_duration)/self._loco_conf.max_good_step_duration)*self._loco_conf.max_good_step_duration
        # # only keep the reward for finishing steps, and use reward quadratic in the duration (so two small steps are worse than one long one).
        # # But keep the sign of the reward
        # # Also, add add a linear term to keep a good gradient at min_good_step_duration
        # feet_air_rewards = steps_finishing*(th.sign(corrected_air_durations)*(corrected_air_durations)**2+corrected_air_durations)
        feet_air_rewards = steps_finishing*corrected_air_durations
        # step_is_good = (feet_air_rewards>0)
        step_air_time_is_bad  = (feet_air_rewards<=0)
        # feet_rewards = step_is_good*squashed_feet_rewards + step_is_bad*feet_rewards # only squash the positive rewards
        feet_air_rewards = feet_air_rewards*(th.logical_or(should_be_moving, step_air_time_is_bad)) # Only enable if speed is > 0.05 or the reward is a small step penalty
        reward_feet_air_time = th.mean(feet_air_rewards, dim=1) # average across the feet

        # ---------------- FEET GROUND TIME REWARD ----------------
        # Similar to feet air time, but rewarding good durations of ground contact
        feet_ground_durations_secs = feet_state[:,self.FEET_FIELDS.FEET_GROUND_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        steps_starting = feet_ground_durations_secs < 0
        feet_ground_durations = -feet_ground_durations_secs # When the value is positive then it is the duration of a currently ongoing step, which we dont consider
        corrected_ground_durations = th.tanh((feet_ground_durations - self._loco_conf.min_good_ground_duration)/self._loco_conf.max_good_ground_duration)*self._loco_conf.max_good_ground_duration
        feet_ground_rewards = steps_starting*(th.sign(corrected_ground_durations)*(corrected_ground_durations)**2+corrected_ground_durations)
        step_ground_time_is_bad  = (feet_ground_rewards<=0)
        feet_ground_rewards = feet_ground_rewards*(th.logical_or(should_be_moving, step_ground_time_is_bad))
        reward_feet_ground_time = th.mean(feet_ground_rewards, dim=1) # average across the feet

        # ---------------- FEET AIR UNIFORMITY REWARD ----------------
        # This reward tries to encourage the robot to have similar air times for all feet
        avg_feet_step_durations = feet_state[:,self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        avg_all_feet_step_duration = th.mean(avg_feet_step_durations, dim=1) # vec_size*nfeet -> vec_size
        step_duration_difformity = steps_finishing*(feet_air_durations_secs-avg_all_feet_step_duration.unsqueeze(1).expand_as(feet_air_durations_secs)) # vec_size*nfeet
        reward_feet_air_time_uniformity = joint_penalty_reward(step_duration_difformity, max_rew=1, exponent=2)
        
        # ---------------- FEET SLIP REWARD ----------------
        # This reward tries to encourage the robot to not slide its feet when they are in contact with the ground
        feet_linvels_xy = feet_state[:,1:3] # vec_size*fields*nfeet -> vec_size*2*nfeet
        feet_linvels = th.linalg.norm(feet_linvels_xy, dim=1) # vec_size*fields*nfeet -> vec_size*nfeet
        feet_touching_ground = feet_state[:,self.FEET_FIELDS.FEET_AIR_DURATIONS] <= 0
        feet_sliding_linvel = feet_linvels*feet_touching_ground
        reward_slip = joint_penalty_reward(feet_sliding_linvel, max_rew=1, exponent=2)

        # ---------------- FEET ON GROUND REWARD ----------------
        # Reward for having feet on the ground, to encourage the robot to stand still when it should
        feet_touching_ground = feet_state[:,0] <= 0
        reward_feet_on_ground = th.mean(feet_touching_ground.to(th.float32), dim=1) * th.logical_not(should_be_moving.view((self.num_envs,)))

        # ---------------- FEET STEP HEIGHT REWARD ----------------
        # This reward tries to encourage the robot to lift its feet high enough from the ground,
        feet_swing_top_height = feet_state[:,self.FEET_FIELDS.PEAK_POS_Z]
        did_good_step = th.logical_not(step_air_time_is_bad) & should_be_moving & steps_finishing
        peak_error = feet_swing_top_height - self._loco_conf.desired_foot_clearance
        max_error = 0.05
        peak_error = smooth_clip(peak_error, max_error, softness=4.0)
        feet_rews = did_good_step*(1 - (peak_error/max_error)**2) # reward is 1 for perfect height, and goes down to 0 as the error approaches max_error, and is 0 for errors larger than max_error
        reward_feet_step_height = th.mean(feet_rews, dim=1)


        # ---------------- CONTACT REWARD ----------------
        # This reward tries to encourage smooth and gentle contacts, by penalizing high impulses
        reward_contacts = - th.clamp(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SUM_IMPULSES], -max_rew, max_rew)

        # ---------------- SAFETY TRIGGERED REWARD ----------------
        # This is a penalty for triggering safety mechanisms
        safety_triggered = th.logical_or(state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED,0],
                                         state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED,0])
        reward_safety_triggered = -1*safety_triggered

        # FAILURE SCALING
        failed = (curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] < 0)
        if self._configuration.fail_on_safety:
            failed = th.logical_or(failed, safety_triggered)

        raw_rewards = LocomotionVecEnv.SubRewards(
            contacts = reward_contacts,
            feet_air_time = reward_feet_air_time,
            feet_ground_time = reward_feet_ground_time,
            feet_on_ground = reward_feet_on_ground,
            feet_step_height = reward_feet_step_height,
            heading = reward_heading_position,
            heading_velocity = reward_heading_velocity,
            health = th.ones((current_state_locom_vec.size()[0],), device=current_state_locom_vec.device),
            height_position = reward_height_position,
            height_velocity = reward_height_velocity,
            joint_acc_on_vel = reward_acc_on_vel,
            joint_acceleration = reward_acceleration,
            joint_actacc = reward_actacc,
            joint_actdiff = reward_actdiff,
            joint_position = reward_position,
            joint_position_limit = reward_position_limit,
            joint_posref_acc = reward_posref_acc,
            joint_posref_vel = reward_posref_vel,
            joint_power = reward_power,
            joint_sensed_effort = reward_sensed_effort,
            joint_stand_position = reward_stand_position,
            joint_torque = reward_cmdtorque,
            joint_torque_limit = reward_torque_limit,
            joint_torque_refs = reward_torque_refs,
            joint_torquediff = reward_torquediff,
            joint_velocity = reward_velocity,
            joint_velocity_limit = reward_velocity_limit,
            joint_velocity_refs = reward_velocity_refs,
            pitchnroll = reward_pitchnroll,
            pitchnroll_velocity = reward_pitchnroll_velocity,
            safety_triggered = reward_safety_triggered,
            slip = reward_slip,
            tracking = reward_velocity_tracking,
            yaw_vel_tracking = reward_yaw_vel_track
        )


        enabled_rewards = self._reward_weights_state_helper.field_names
        for n in enabled_rewards:
            n = str(n).lower()
            sub_rewards_return[n] = getattr(raw_rewards, n)
        # sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards_return.items()}
        
        weights = {n:current_state_rewards_vec[:,i] for i,n in enumerate(enabled_rewards)}
        for k,r in sub_rewards_return.items():
            r = self._loco_conf.reward_scale*r*weights.get(k, 0.0)
            sub_rewards_return[k] = r*th.where(th.logical_and(failed, r>0), 0.001, 1.0).view((self.num_envs,))
        
        if len(sub_rewards_return) != len(enabled_rewards):
            missing = set(enabled_rewards)-set(sub_rewards_return.keys())
            raise ValueError(f"Some enabled rewards are missing in the computed rewards: {missing}")
        # sub_rewards_unscaled = {k:v.view(self._adapter.vec_size(),) for k,v in sub_rewards_unscaled.items()}
        stacked_rewards = th.stack(list(sub_rewards_return.values()), dim = 1)
        if self._loco_conf.split_rewards:
            reward = stacked_rewards
            dbg_check_size(reward, (self._adapter.vec_size(),len(sub_rewards_return)), f"Unexpected reward size")
        else:
            reward = th.sum(stacked_rewards, dim =1, keepdim=True)
            dbg_check_size(reward, (self._adapter.vec_size(),1), f"Unexpected reward size")
        reward = th.clamp(reward, -self._configuration.reward_clamp, self._configuration.reward_clamp)
        
        # ggLog.info(f"sub_rewards_return = {sub_rewards_return}")
        dbg_check_finite(sub_rewards_return, async_assert=True, assert_msg="Nonfinite sub rewards detected")
        
        return reward, sub_rewards_return
        

    def _update_stats(self):
        record_region_start("LocomotionVecEnv._update_stats")
        super()._update_stats()
        record_time("LocomotionVecEnv._update_stats: super done")
        if not self._configuration.minimal_infos:
            body_rel_linvel_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z)) #type:ignore
            body_rel_gravity_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z)) #type:ignore
            goal_vel_rel_dir_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z)) #type:ignore
            goal_grav_abs_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X,
                                                                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y,
                                                                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z)) #type:ignore
            body_rel_linvel_xyz = self._current_state[self.STATE_EXTRINSIC][:,0,body_rel_linvel_xyz_idx,0]
            gravity_rel_vec_xyz = self._current_state[self.STATE_EXTRINSIC][:,0,body_rel_gravity_xyz_idx,0]
            goal_rel_linvel_dir_vec_xyz = self._current_state[self.STATE_LOCOMOTION][:,0,goal_vel_rel_dir_xyz_idx,0].view(self.num_envs,3)
            goal_speed = self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED,0].view(self.num_envs,1)
            goal_rel_linvel_vec_xyz = goal_rel_linvel_dir_vec_xyz*goal_speed
            body_speed_vec = th.linalg.norm(body_rel_linvel_xyz[:,:2], dim=-1)
            body_height_vec = self._current_state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z,0]
            goal_height_vec = self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT,0]
            smoothed_goal_height_vec = self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT,0]
            goal_gravity_vec = self._current_state[self.STATE_LOCOMOTION][:,0, goal_grav_abs_xyz_idx,0]
            vel_error_vec = planar_tracking_error_vec(
                                            body_rel_linvel_vec_xyz = body_rel_linvel_xyz,
                                            gravity_rel_vec_xyz = gravity_rel_vec_xyz,
                                            goal_rel_linvel_vec_xyz = goal_rel_linvel_vec_xyz)
            dbg_check_size(vel_error_vec, (self._adapter.vec_size(),))
            height_error_vec = th.abs(body_height_vec-smoothed_goal_height_vec)
            pitchnroll_err_vec = th.linalg.norm(gravity_rel_vec_xyz-goal_gravity_vec, dim = -1)
            step_counts = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT,0].to(th.long)
            dbg_check_size(pitchnroll_err_vec, (self._adapter.vec_size(),))
            dbg_check_size(step_counts, (self._adapter.vec_size(),))
            
            # Update episode averages
            self._stats["ep_avg_vel_err_vec"]          = (self._stats["ep_avg_vel_err_vec"]*(step_counts-1) + vel_error_vec)/step_counts # Elements with step_count == 0 will be inf
            self._stats["ep_avg_height_err_vec"]       = (self._stats["ep_avg_height_err_vec"]*(step_counts-1) + height_error_vec)/step_counts # Elements with step_count == 0 will be inf
            self._stats["ep_avg_pitchnroll_err_vec"]   = (self._stats["ep_avg_pitchnroll_err_vec"]*(step_counts-1) + pitchnroll_err_vec)/step_counts # Elements with step_count == 0 will be inf
            self._stats["ep_avg_bodyspeed_vec"]        = (self._stats["ep_avg_bodyspeed_vec"]*(step_counts-1) + body_speed_vec)/step_counts # Elements with step_count == 0 will be inf
            # Correct the episode averages for episodes that have just started
            starting_eps = step_counts==0
            masked_assign(self._stats["ep_avg_vel_err_vec"],starting_eps,vel_error_vec)
            masked_assign(self._stats["ep_avg_height_err_vec"],starting_eps,height_error_vec)
            masked_assign(self._stats["ep_avg_pitchnroll_err_vec"],starting_eps,pitchnroll_err_vec)
            masked_assign(self._stats["ep_avg_bodyspeed_vec"],starting_eps,body_speed_vec)

            # Fill the buffers for episodes that have just staretd
            masked_assign(self._stats["vel_errs_vec"],        starting_eps, vel_error_vec.unsqueeze(1).expand(-1, self._buff_sizes))
            masked_assign(self._stats["height_errs_vec"],     starting_eps, height_error_vec.unsqueeze(1).expand(-1, self._buff_sizes))
            masked_assign(self._stats["pitchnroll_errs_vec"], starting_eps, pitchnroll_err_vec.unsqueeze(1).expand(-1, self._buff_sizes))
            masked_assign(self._stats["body_speeds_vec"],     starting_eps, body_speed_vec.unsqueeze(1).expand(-1, self._buff_sizes))
            # Update the buffers
            # idxs = step_counts%self._buff_sizes
            idx : th.Tensor = self._th_tot_step_counter.view(tuple())%self._stats["vel_errs_vec"].size()[1]
            set_column(self._stats["vel_errs_vec"],     idx, vel_error_vec.view(self.num_envs,))
            set_column(self._stats["height_errs_vec"],  idx, height_error_vec.view(self.num_envs,))
            set_column(self._stats["pitchnroll_errs_vec"], idx, pitchnroll_err_vec.view(self.num_envs,))
            set_column(self._stats["body_speeds_vec"],  idx, body_speed_vec.view(self.num_envs,))

            state_stats_v_h_j_minmaxavgstd_pvaeep : th.Tensor = self._current_state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 6)
            masked_assign(self._stats["ep_max_javg_sensed_effort"], starting_eps, 0)
            masked_assign(self._stats["ep_max_peak_sensed_effort"], starting_eps, 0)
            self._stats["ep_max_javg_sensed_effort"] = th.maximum(self._stats["ep_max_javg_sensed_effort"], state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,2,4].mean(dim=1)).view((self.num_envs,)) 
            self._stats["ep_max_peak_sensed_effort"] = th.maximum(self._stats["ep_max_peak_sensed_effort"], state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,0:2,4].abs().amax(dim=[1,2])).view((self.num_envs,))
        record_region_end("LocomotionVecEnv._update_stats")
   
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        record_region_start("LocomotionVecEnv.get_infos")
        
        goal_vel_abs_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z)) #type:ignore 
        body_linvel_abs_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z)) #type: ignore
        curr_locom_state = state[self.STATE_LOCOMOTION][:,0]
        smoothed_linvel_error =       curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR].view(self.num_envs)
        smoothed_yawvel_error =       curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_YAW_VEL_ERROR].view(self.num_envs)
        smoothed_is_jumping =         curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_IS_JUMPING].view(self.num_envs)
        smoothed_num_feet_on_ground = curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_NUM_FEET_ON_GROUND].view(self.num_envs)
        smoothed_height_error =       curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR].view(self.num_envs)
        goal_abs =                    curr_locom_state[:,goal_vel_abs_xyz_idx]
        curr_extri_state = state[self.STATE_EXTRINSIC][:,0]
        abs_linvel = curr_extri_state[:,body_linvel_abs_xyz_idx]
        i = super().get_infos(state=state, labels=labels)
        avg_air_duration_idx = self._state_helper.sub_helpers[self.STATE_FEET].field_idx(self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS) #type: ignore
        avg_air_durations = state[self.STATE_FEET][:,0,avg_air_duration_idx,:] # vec_size*fields*nfeet -> vec_size*nfeet
        avg_air_duration = th.mean(avg_air_durations, dim=1) # vec_size*nfeet -> vec_size
        i["smoothed_linvel_error"] = smoothed_linvel_error
        i["smoothed_yawvel_error"] = th.abs(smoothed_yawvel_error)
        i["goal_abs_xyz_vec"] = goal_abs
        i["body_abs_linvel"] = abs_linvel
        i["body_abs_linspeed"] = abs_linvel.norm(dim=1)
        i["avg_air_duration"] = avg_air_duration
        i["smoothed_is_jumping"] = smoothed_is_jumping
        i["smoothed_num_feet_on_ground"] = smoothed_num_feet_on_ground
        i["smoothed_height_error"] = smoothed_height_error

        if self._configuration.minimal_infos:
            record_region_end("LocomotionVecEnv.get_infos")
            return i
        curr_rewar_state = state[self.STATE_REWARDS][:,0]
        prev_extri_state = state[self.STATE_EXTRINSIC][:,1]
        curr_inter_state = state[self.STATE_INTERNAL][:,0]
        
        goal_vel_rel_dir_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z)) #type:ignore
        body_linvel_rel_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z)) #type: ignore
        
        goal_dir = curr_locom_state[:,goal_vel_rel_dir_xyz_idx].view(self.num_envs,3)
        goal_speed = curr_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED].view(self.num_envs,1)
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)
        state_stats_v_h_j_minmaxavgstd_pvaeep : th.Tensor = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 6)
        avg10_vel_errs_vec = th.mean(self._stats["vel_errs_vec"], dim = 1).view(self.num_envs)
        goal_height = curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        height = curr_extri_state[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        loco_info = {
            "goal_rel_xyz_vec" : goal_dir*goal_speed,
            "goal_height" : goal_height,
            "goal_yaw_vel" : curr_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_YAW_VEL],
            "height" : height,
            "height_err" : th.abs(goal_height - height),
            "goal_abs_speed_vec" : th.linalg.norm(goal_abs,dim=1),
            "goal_abs_yaw_vec" : th.atan2(goal_abs[:,1],goal_abs[:,0]),
            "body_rel_linvel" : curr_extri_state[:,body_linvel_rel_xyz_idx],
            "linvel_error" : goal_abs - abs_linvel,
            "ep_avg_vel_err_vec" : self._stats["ep_avg_vel_err_vec"],
            "ep_avg_height_err_vec" : self._stats["ep_avg_height_err_vec"],
            "ep_avg_pitchnroll_err_vec" : self._stats["ep_avg_pitchnroll_err_vec"],
            "ep_avg_bodyspeed_vec" : self._stats["ep_avg_bodyspeed_vec"],
            "avg10_vel_errs_vec" : avg10_vel_errs_vec,
            "avg10_height_errs_vec" : th.mean(self._stats["height_errs_vec"], dim = 1).view(self.num_envs),
            "avg10_pitchnroll_errs_vec" : th.mean(self._stats["pitchnroll_errs_vec"], dim = 1).view(self.num_envs),
            "avg10_body_speeds_vec" : th.mean(self._stats["body_speeds_vec"], dim = 1).view(self.num_envs),
            "success_vec" : avg10_vel_errs_vec < 0.05,
            "joint_pos_safenorm" : state_robot_safenorm[:,0,:,0],
            "avg_sensed_effort" : state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,2,4].mean(dim=1),
            "avg_peak_sensed_effort" : state_stats_v_h_j_minmaxavgstd_pvaeep[:,0,:,0:2,4].abs().amax(dim=2).amax(dim=1)
        }
        rew_weights_needed = (   "actdiff", "actacc", "posref_vel", "posref_acc")
        enabled_needed_weights  = tuple([w for w in rew_weights_needed if w in self._loco_conf.enabled_rewards])
        disabled_needed_weights = tuple([w for w in rew_weights_needed if w not in self._loco_conf.enabled_rewards])
        enabled_needed_weights_idxs = self._state_helper.sub_helpers[self.STATE_REWARDS].field_idx(enabled_needed_weights) #type: ignore
        enabled_rew_weights = curr_rewar_state[:,enabled_needed_weights_idxs]
        for ri,rn in enumerate(enabled_needed_weights):
            i[rn+"_weight"] = enabled_rew_weights[:,ri]
        for rn in disabled_needed_weights:
            i[rn+"_weight"] = th.zeros((self.num_envs,), device=curr_rewar_state.device)
        
        _, i["height_velocity"], i["goal_height_velocity"], last_dt, i["height_err_raw"] = self._height_velocity_reward(curr_state_extr_vec = curr_extri_state,
                                                                                                       current_state_locom_vec = curr_locom_state,
                                                                                                       current_state_internal = curr_inter_state,
                                                                                                       prev_state_extr_vec = prev_extri_state)
        _, i["heading_vel"], i["goal_heading_vel"], i["heading_vel_err"], i["heading_err"] = self._heading_velocity_reward(state=state, dt=last_dt)
        _, i["pitchnroll_velocity"], i["goal_pitchnroll_velocity"], i["pitchnroll_vel_err"], i["pitchnroll_err"] = self._pitchnroll_velocity_reward(state=state,
                                                                                                                                                    dt=last_dt)
        i.update(loco_info)

        if labels is not None: # Generate at least some labels
            for k in i:
                if k not in labels:
                    if isinstance(i[k], th.Tensor):
                        nelements = i[k].shape[1] if len(i[k].shape)>1 else 1
                        labels[k] = to_string_tensor([k+f"[{i}]" for i in range(nelements)])
                    else:
                        labels[k] = to_string_tensor([k])

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_LOCOMOTION, self.STATE_FEET]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())

        record_region_end("LocomotionVecEnv.get_infos")
        return i
    
    def _sample_abs_goals(self):
        goal_speeds = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_speed_minmax[0],
                                    max=self._loco_conf.goal_speed_minmax[1])
        goal_yaws = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_abs_yaw_minmax[0],
                                    max=self._loco_conf.goal_abs_yaw_minmax[1])
        goal_abs_height = self._thrand(size=(self.num_envs,))*(self._loco_conf.goal_height_minmax[1]-self._loco_conf.goal_height_minmax[0])+self._loco_conf.goal_height_minmax[0]
        goal_abs_linvel_vec_xys = th.stack([  th.cos(goal_yaws),
                                                th.sin(goal_yaws),
                                                goal_speeds],
                                             dim=1)
        goal_heading_yaws = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_heading_rel_yaw_minmax[0],
                                    max=self._loco_conf.goal_heading_rel_yaw_minmax[1])
        goal_yaw_vels = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_yaw_vel_minmax[0],
                                    max=self._loco_conf.goal_yaw_vel_minmax[1])
        goal_yaw_vels = th.where(self._thrand(size=(self.num_envs,)) < self._loco_conf.goal_yaw_vel_zero_ratio,
                                 0.0,
                                 goal_yaw_vels)
        return goal_abs_linvel_vec_xys, goal_abs_height, goal_heading_yaws, goal_yaw_vels

    @override
    @th.compile(mode="max-autotune-no-cudagraphs", disable=disable_compile, dynamic=False)
    def pre_step(self):
        with th.no_grad():
            super().pre_step()
            if self._loco_conf.goal_resampling_enabled>0:
                resample_prob_per_env_dt = 1-th.pow(1-self._loco_conf.goal_resampling_probability_per_sec, self._intendedStepLength_sec)
                vec_mask = self._thrand((self.num_envs,)) < resample_prob_per_env_dt
                goal_abs_linvel_vec_xys, goal_height, goal_heading_yaws, goal_yaw_vels = self._sample_abs_goals()
                self.set_goal(goal_abs_linvel_vec_xys, 
                        goal_abs_height=goal_height,
                        vec_mask=vec_mask,
                        goal_heading_yaw=goal_heading_yaws,
                        goal_yaw_vel=goal_yaw_vels)

    @staticmethod
    def _set_linvel_global_stats(tensors):
        linvel_err_cpu = tensors["linvel_err"]
        session.default_session.run_info["extras"]["linvel_q95"] = linvel_err_cpu.quantile(0.95).item()
        session.default_session.run_info["extras"]["linvel_avg"] = linvel_err_cpu.mean().item()

    def post_step(self):
        record_region_start("LocomotionVecEnv.post_step")
        super().post_step()
        record_time("LocomotionVecEnv.post_step super done")
        linvel_err = self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR,0]
        async_cuda2cpu_queue.run_async_job({"linvel_err": linvel_err},
                                           self._set_linvel_global_stats)
        record_region_end("LocomotionVecEnv.post_step")

    @override
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        sampled_goal_linvel_vec_xys, sampled_goal_height, goal_heading_yaws, sampled_goal_yaw_vels = self._sample_abs_goals()
        if "goal_abs_linvel_vec_xys" in reset_options:
            if "goal_rel_linvel_xys" in reset_options:
                raise ValueError("Cannot specify both goal_abs_linvel_vec_xys and goal_rel_linvel_xys in reset options")
            goal_abs_linvel_vec_xys = th.as_tensor(reset_options["goal_abs_linvel_vec_xys"],device=self._configuration.th_device).view(self.num_envs,3)
            goal_rel_linvel_xys = None
        elif "goal_rel_linvel_xys" in reset_options:
            goal_rel_linvel_xys = th.as_tensor(reset_options["goal_rel_linvel_xys"],device=self._configuration.th_device).view(self.num_envs,3)
            goal_abs_linvel_vec_xys = None
        else:
            if isinstance(self._adapter, BaseVecSimulationAdapter):
                goal_abs_linvel_vec_xys = sampled_goal_linvel_vec_xys
                goal_rel_linvel_xys = None
            else:
                goal_abs_linvel_vec_xys = None
                goal_rel_linvel_xys = sampled_goal_linvel_vec_xys

        if "goal_abs_height" in reset_options:
            goal_abs_height = th.as_tensor(reset_options["goal_abs_height"],device=self._configuration.th_device).view(self.num_envs)
        else:
            goal_abs_height = sampled_goal_height

        if "goal_yaw_vels" in reset_options:
            goal_yaw_vels = th.as_tensor(reset_options["goal_yaw_vels"],device=self._configuration.th_device).view(self.num_envs)
        else:
            goal_yaw_vels = sampled_goal_yaw_vels
        self.set_goal(  goal_abs_linvel_vec_xys=goal_abs_linvel_vec_xys, 
                        goal_rel_linvel_xys=goal_rel_linvel_xys,
                        goal_abs_height=goal_abs_height,
                        vec_mask=vec_mask,
                        goal_heading_yaw=goal_heading_yaws,
                        goal_yaw_vel=goal_yaw_vels)
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
        self.set_max_episode_steps(reset_options.get("max_ep_steps",self._current_episode_config.vec_max_ep_steps))
        sw = self._sample_distr((self.num_envs,1), self._loco_conf.reward_superweight_joint_penalties)
        new_reward_weight_acceleration          = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_acceleration)
        new_reward_weight_acc_on_vel            = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_acc_on_vel)
        new_reward_weight_actacc                = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_actacc)
        new_reward_weight_actdiff               = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_actdiff)
        new_reward_weight_posref_vel            = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_posref_vel)
        new_reward_weight_torque                = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_torque)
        new_reward_weight_posref_acc            = sw*self._sample_distr((self.num_envs,1), self._loco_conf.reward_weight_joint_posref_acc)
        masked_assign(self._locomotion_episode_config.reward_weight_acceleration,   vec_mask, new_reward_weight_acceleration)
        masked_assign(self._locomotion_episode_config.reward_weight_acc_on_vel,     vec_mask, new_reward_weight_acc_on_vel)
        masked_assign(self._locomotion_episode_config.reward_weight_actacc,         vec_mask, new_reward_weight_actacc)
        masked_assign(self._locomotion_episode_config.reward_weight_actdiff,        vec_mask, new_reward_weight_actdiff)
        masked_assign(self._locomotion_episode_config.reward_weight_posref_vel,     vec_mask, new_reward_weight_posref_vel)
        masked_assign(self._locomotion_episode_config.reward_weight_posref_acc,     vec_mask, new_reward_weight_posref_acc)
        masked_assign(self._locomotion_episode_config.reward_weight_torque,         vec_mask, new_reward_weight_torque)
        

    def set_goal(self,  goal_abs_linvel_vec_xys : Sequence[tuple[float,float,float]] | tuple[float,float,float] | th.Tensor | None = None,
                        # goal_diff_linvel_speed_yaw : tuple[float,float] | th.Tensor | None = None,
                        goal_rel_linvel_xys : tuple[float,float,float] | th.Tensor | None = None,
                        goal_abs_height : float | th.Tensor | None = None,
                        goal_heading_yaw : float | th.Tensor | None = None,
                        goal_yaw_vel : float | th.Tensor | None = None,
                        vec_mask : th.Tensor | None = None):
        if vec_mask is None:
            vec_mask = self._all_vecs

        if goal_abs_linvel_vec_xys is not None:
            if self._locomotion_episode_config.goal_abs_vel_vec_xys is None:
                self._locomotion_episode_config.goal_abs_vel_vec_xys = self._thtens([1.0,0.0,0.0]).expand(self.num_envs,3).clone()
            self._locomotion_episode_config.goal_rel_vel_vec_xys = None
            goal_abs_linvel_vec_xys = self._thtens(goal_abs_linvel_vec_xys).expand(self.num_envs,3)
            masked_assign(self._locomotion_episode_config.goal_abs_vel_vec_xys,
                          vec_mask,
                          goal_abs_linvel_vec_xys)
        # elif goal_diff_linvel_speed_yaw is not None:
        #     raise NotImplementedError("goal_diff_linvel_speed_yaw has been disabled for now.")
        #     if self._locomotion_episode_config.goal_abs_vel_vec_xys is None:
        #         self._locomotion_episode_config.goal_abs_vel_vec_xys = self._thtens([1.0,0.0,0.0]).expand(self.num_envs,3)
        #     self._locomotion_episode_config.goal_rel_vel_vec_xys = None
        #     if isinstance(goal_diff_linvel_speed_yaw, Sequence):
        #         goal_diff_linvel_speed_yaw = self._thtens(goal_diff_linvel_speed_yaw)
        #     elif not isinstance(goal_diff_linvel_speed_yaw, th.Tensor):
        #         raise RuntimeError(f"Unexpected type {type(goal_diff_linvel_speed_yaw)} for goal_velocity_diff_speed_yaw")
        #     prev_goal_abs_vel_vec_xys = self._locomotion_episode_config.goal_abs_vel_vec_xys
        #     curr_goal_speeds : th.Tensor = prev_goal_abs_vel_vec_xys[:,2]
        #     curr_goal_yaws = th.atan2(prev_goal_abs_vel_vec_xys[:,1], prev_goal_abs_vel_vec_xys[:,0])
        #     new_goal_speeds = curr_goal_speeds + goal_diff_linvel_speed_yaw[0]
        #     new_goal_yaws = curr_goal_yaws + goal_diff_linvel_speed_yaw[1]
        #     new_goal_dirs = th.stack([th.cos(new_goal_yaws), th.sin(new_goal_yaws)], dim = 1)    
        #     new_goals_xys = th.cat([new_goal_dirs, new_goal_speeds.unsqueeze(1)], dim = 1)
        #     masked_assign(self._locomotion_episode_config.goal_abs_vel_vec_xys,
        #                   vec_mask,
        #                   new_goals_xys)
        elif goal_rel_linvel_xys is not None:
            self._locomotion_episode_config.goal_rel_vel_vec_xys = self._thtens(goal_rel_linvel_xys).view(self.num_envs,3)
            self._locomotion_episode_config.goal_abs_vel_vec_xys = None
        # else:
        #     raise RuntimeError("One of goal_abs_linvel_vec_xys, goal_diff_linvel_speed_yaw or goal_rel_linvel_xys must be provided")

        if goal_abs_height is not None:
            goal_abs_height = self._thtens(goal_abs_height).expand(1,self.num_envs).permute(1,0)
            masked_assign(self._locomotion_episode_config.goal_abs_height_vec_z,
                          vec_mask,
                          goal_abs_height)
        if goal_heading_yaw is not None:
            goal_heading_yaw = self._thtens(goal_heading_yaw).expand(self.num_envs,).unsqueeze(1)
            masked_assign(self._locomotion_episode_config.goal_heading_rel_vec_yaw,
                          vec_mask,
                          goal_heading_yaw)
        if goal_yaw_vel is not None:
            goal_yaw_vel = self._thtens(goal_yaw_vel).expand(self.num_envs,).unsqueeze(1)
            masked_assign(self._locomotion_episode_config.goal_yaw_vel_vec,
                          vec_mask,
                          goal_yaw_vel)


    def get_goals(self):
        return {"abs_linvel_xys" : self._locomotion_episode_config.goal_abs_vel_vec_xys,
                "rel_linvel_xys" : self._locomotion_episode_config.goal_rel_vel_vec_xys,
                "abs_gravity" : self._locomotion_episode_config.goal_abs_gravity_vec_xyz,
                "abs_height" : self._locomotion_episode_config.goal_abs_height_vec_z,
                "heading_rel" : self._locomotion_episode_config.goal_heading_rel_vec_yaw,
                "yaw_vel" : self._locomotion_episode_config.goal_yaw_vel_vec}

    @override
    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune")
    def are_states_terminal(self, states) -> th.Tensor:
        r = super().are_states_terminal(states)
        if self._loco_conf.terminate_on_crash:
            r = th.logical_or(r, states[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.CRASHED,0]).view((self.num_envs,))
            below_ground = states[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z,0] < 0.0
            r = th.logical_or(r, below_ground)
            non_vertical = states[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z,0] > -0.85
            r = th.logical_or(r, non_vertical)
        return r

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        super()._initialize_episodes(vec_mask=vec_mask, options=options)
        if self._loco_conf.use_contacts:
            raise NotImplementedError("Contacts not implemented yet")
            self._adapter.monitor_contacts([(self._configuration.robot_name, None)])

    def _set_arrow_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            goal_abs_vel_vec_xys = self._locomotion_episode_config.goal_abs_vel_vec_xys
            if goal_abs_vel_vec_xys is None:
                goal_abs_vel_vec_xys = self._thtens([1.0,0.0,0.0]).expand(self.num_envs,3)
            goals_corrected = goal_abs_vel_vec_xys.detach().clone()
            zero_goals = th.linalg.norm(goals_corrected, dim = 1) < 0.0001
            if len(zero_goals)>0:
                masked_assign(goals_corrected,zero_goals,self._thtens([0,0,-1.0]))
            speed = goals_corrected[:,2]
            direction = goals_corrected
            direction[:,2] = 0.0 # Arrow is always on the ground
            linvel_dir_quat = quat_xyzw_between_vecs_py(self._thtens([1.0,0,0]).expand((self._adapter.vec_size(),3)), direction)
            bstates_vec_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_pose = False)[:,0,:]
            goalvel_arrow_pose = bstates_vec_13[:,:7].clone()
            goalvel_arrow_pose[:,2] = speed
            goalvel_arrow_pose[:,3:7] = linvel_dir_quat
            # goalvel_arrow_pose[1:] = goalvel_arrow_pose[0] # is on a fixed link, so it must be set to the same pose across all links
            goalvel_arrow_state = th.cat([goalvel_arrow_pose, self._thzeros((goalvel_arrow_pose.size()[0],6,))], dim = 1)

            heading_arrow_pose = bstates_vec_13[:,:7].clone()
            heading_rpy = self._thzeros((self._adapter.vec_size(),3))
            heading_rpy[:,2] = self._locomotion_episode_config.goal_heading_rel_vec_yaw.view(self.num_envs,)
            rel_heading_quat = ros_rpy_to_quaternion_xyzw_th(heading_rpy)
            heading_arrow_pose[:,3:7] = quat_mul_xyzw(rel_heading_quat, linvel_dir_quat)
            heading_arrow_pose[:,2] = 0.0 # Arrow is always on the ground
            # heading_arrow_pose[1:] = heading_arrow_pose[0]
            heading_arrow_state = th.cat([heading_arrow_pose, self._thzeros((heading_arrow_pose.size()[0],6,))], dim = 1)

            self._adapter.setLinksStateDirect(link_names=[self._arrow_yellow, self._arrow_base],
                                                link_states_pose_vel=th.stack([heading_arrow_state, goalvel_arrow_state], dim = 1),
                                                vec_mask=vec_mask)

    @override
    def _simulation_initialization(self, vec_mask : th.Tensor):
        super()._simulation_initialization(vec_mask = vec_mask)
        # if self._configuration.show_goal:
        #     self._set_arrow_pose(vec_mask=self._all_vecs)
            
    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._set_arrow_pose(vec_mask=self._all_vecs)
        return super().get_ui_renderings(vec_mask=vec_mask)
    
class TransitionAugmentor:

    def __init__(self, reward_space : ThBox,
                       reward_weights_distr : DistributionTh.DistributionDef,
                       augmented_samples : int,
                       generator : th.Generator | None = None):
        self._reward_weights_key = "base.reward_weights"
        self._randomized_rewards = ["posref_acc", "posref_vel"]
        # for rr in self._randomized_rewards:
        #     if rr not in reward_space.labels:
        #         raise ValueError(f"Reward '{rr}' not found in reward space labels")
        self._th_device = reward_space.th_device
        self._reward_mask = th.as_tensor([rn in self._randomized_rewards for rn in reward_space.labels], device=self._th_device)
        self._rewards_distribution = DistributionTh(distribution_def=reward_weights_distr,
                                                    device=self._th_device,
                                                    dtype=reward_space.torch_dtype,
                                                    generator=generator)
        self._augmented_samples = augmented_samples

    def augment_transition(self, observations : dict[str, th.Tensor],
                                actions : th.Tensor,
                                next_observations : dict[str, th.Tensor],
                                rewards : th.Tensor,
                                terminateds : th.Tensor
                                ) -> tuple[dict[str, th.Tensor], th.Tensor, dict[str, th.Tensor], th.Tensor, th.Tensor]:
        with th.no_grad():
            original_reward_weights = observations[self._reward_weights_key]
            
            # We will augment just the first augmented_samples elements of the batch

            # Sample augmented_samples new reward weights
            new_rews_size = (self._augmented_samples,) + original_reward_weights.size()[1:]
            sampled_weights = self._rewards_distribution.sample(size=new_rews_size,)
            # Apply the new weight just to the rewards specified by the mask
            sampled_weights = th.where(self._reward_mask.unsqueeze(0),
                                    sampled_weights,
                                    original_reward_weights[:self._augmented_samples])
            # Compute the new rewards accoridng to these weights
            rescaled_rewards = rewards[:self._augmented_samples]/original_reward_weights[:self._augmented_samples]*sampled_weights

            # Substitute these reward in the original batch
            new_rewards = rewards.clone()
            new_rewards[:self._augmented_samples] = rescaled_rewards
            # Substitute the weights in the observation and next observation
            new_reward_weights = original_reward_weights.clone()
            new_reward_weights[:self._augmented_samples] = sampled_weights
            new_obss = {k:v for k,v in observations.items()}
            new_obss[self._reward_weights_key] = new_reward_weights
            new_next_obss = {k:v for k,v in next_observations.items()}
            new_next_obss[self._reward_weights_key] = new_reward_weights # What if the next_obs weights were different, they shoudlnt be for how the env is formulated now

        return new_obss, actions, new_next_obss, new_rewards, terminateds