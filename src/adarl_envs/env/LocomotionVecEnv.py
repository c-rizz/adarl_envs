from __future__ import annotations
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.utils.utils import (LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, vector_projection, isinstance_noimport, 
                               quat_xyzw_between_vecs_py, masked_assign, quat_mul_xyzw, quat_angle_xyzw, vectors_angle, ros_rpy_to_quaternion_xyzw_th)
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check_size, dbg_check
import adarl.utils.utils
from adarl.utils.vec_state_helper import ThBoxStateHelper, unnormalize, normalize
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any
from git import Tree
from requests import head
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
import torch as th
from adarl_envs.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS, DistributionDef
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
import pprint

disable_compile = False

@th.jit.script
def bell_reward(error : th.Tensor, zero_rew_dist : th.Tensor):
    """A bell-shaped reward function. It's 1 at error = 0, it reaches about zero (~0.0183) at error = zero_rew_dist

    Parameters
    ----------
    error : th.Tensor
        Error value
    zero_rew_dist : th.Tensor | float
        Error value at which the reward should start to settle around zero

    Returns
    -------
    th.Tensor
        Reward value
    """
    return th.exp(-(2*error/zero_rew_dist)**2)

@th.jit.script
def double_bell_reward(error : th.Tensor, bell_width_a : th.Tensor, bell_width_b : th.Tensor, bell_b_weight : th.Tensor):
    return (   bell_b_weight  * bell_reward(error, zero_rew_dist=bell_width_b)+
            (1-bell_b_weight) * bell_reward(error, zero_rew_dist=bell_width_a))

@th.jit.script
def ramp_reward(error : th.Tensor, zero_rew_dist : th.Tensor):
    return 1-error/zero_rew_dist

@th.jit.script
def flattened_joint_penalty_reward(x, max_rew, exponent : float, flattening_scale : float):
    """A penalty produced by raising abs(x) at the power of exponent, and flattening it with
        a flipped exponential, scaled with flattening_scale. With exponent=1.5 and 
        flattening_scale=0.1 results in an x^1.5 that is quite flat below 0.1.
        This then is squashed with a tanh to be under max_rew.
        In formulas (not squashed): x^exponent * (-e^(-x^2/flattening_scale)+1)
    """
    return -th.tanh((th.mean(th.pow(th.abs(x),exponent)*(1-th.exp(-(x/flattening_scale)**2)), dim=1))/max_rew)*max_rew

@th.jit.script
def joint_penalty_reward(x, max_rew : float, exponent : float, reduction : str = "mean", presquash_factor : float = 1.0):
    """A penalty produced by raising abs(x) at the power of exponent, and squashing
        it with a tanh to be under max_rew.
    """
    if reduction == "mean":
        return -th.tanh(th.mean(th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    elif reduction == "sum":
        return -th.tanh(th.sum( th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    elif reduction == "max":
        return -th.tanh(th.amax(th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    

@th.jit.script
def penalty_reward(x, max_rew : float, exponent : float):
    """A penalty produced by raising abs(x) at the power of exponent, and squashing
        it with a tanh to be under max_rew.
    """
    return -th.tanh(th.pow(th.abs(x),exponent)/max_rew)*max_rew

def planar_tracking_error_vec(body_rel_linvel_vec_xyz : th.Tensor, gravity_rel_vec_xyz : th.Tensor, goal_rel_linvel_vec_xyz : th.Tensor) -> th.Tensor:
        """_summary_

        Parameters
        ----------
        body_rel_linvel_vec_xyz : th.Tensor
            current linvel, relative to the body frame
        gravity_rel_vec_xyz : th.Tensor
            gravity vector, relative to the body frame
        goal_rel_linvel_vec_xyz : th.Tensor
            linvel goal, relative to the body frame


        Returns
        -------
        th.Tensor
            _description_
        """
        body_planar_rel_linvel_xyz = body_rel_linvel_vec_xyz - vector_projection(body_rel_linvel_vec_xyz,gravity_rel_vec_xyz)
        # goal_rel_linvel_xyz should already be "planar", it's projection along gravity_rel should be zero
        norms = th.norm(vector_projection(goal_rel_linvel_vec_xyz,gravity_rel_vec_xyz), dim = 1)
        dbg_check(lambda: th.all(norms < 0.1) == True,
                  lambda:   f"goal_rel_linvel_xyz is not horizontal (th.all(norms < 0.1) = {th.all(norms < 0.1)}), projection is "
                            f"{vector_projection(goal_rel_linvel_vec_xyz, gravity_rel_vec_xyz)[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f"goal={goal_rel_linvel_vec_xyz[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f"gravity={gravity_rel_vec_xyz[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f" big={th.nonzero(norms >= 0.1)}"
                            f" isnan={th.nonzero(th.isnan(norms))}"
                            f" isinf={th.nonzero(th.isinf(norms))}",
                    async_assert=True,
                    assert_msg="goal_rel_linvel_vec_xyz is not horizontal")
        return th.linalg.norm(body_planar_rel_linvel_xyz-goal_rel_linvel_vec_xyz, dim = 1)


def set_column(t : th.Tensor, idx :th.Tensor, value : th.Tensor) -> th.Tensor:
    """ Set a column of a 2D tensor. Equivalent to t[:, idx] = value, but more suited to th.compile.
    """
    t.T.index_put_([idx], value)
    return t







class LocomotionVecEnv(RobotVecEnv):
    STATE_LOCOMOTION = "loco"
    STATE_FEET = "feet"
    STATE_HEIGHTMAP = "heightmap"

    @dataclass
    class LocomotionConfiguration:
        disallowed_contact_links : list[tuple[str,str]]
        goal_speed_minmax : th.Tensor
        goal_abs_yaw_minmax : th.Tensor
        reward_scale : th.Tensor
        reward_weight_acceleration : th.Tensor
        reward_weight_contacts : th.Tensor
        reward_weight_energy : th.Tensor
        reward_weight_health : th.Tensor
        reward_weight_height_velocity : th.Tensor
        reward_weight_height_position : th.Tensor
        reward_weight_pitchnroll : th.Tensor
        reward_weight_pitchnroll_velocity : th.Tensor
        reward_weight_actdiff : th.Tensor
        reward_weight_actacc : th.Tensor
        reward_weight_position_limit : th.Tensor
        reward_weight_position : th.Tensor
        reward_weight_heading : th.Tensor
        reward_weight_heading_velocity : th.Tensor
        reward_weight_torque_limit : th.Tensor
        reward_weight_torque : th.Tensor
        reward_weight_torquediff : th.Tensor
        reward_weight_tracking : th.Tensor
        reward_weight_velocity_limit : th.Tensor
        reward_weight_velocity : th.Tensor
        reward_weight_feet_air_time : th.Tensor
        reward_weight_feet_ground_time : th.Tensor
        reward_weight_feet_on_ground : th.Tensor
        reward_weight_failure : th.Tensor
        reward_weight_sensed_effort : th.Tensor
        reward_weight_slip : th.Tensor
        reward_weight_velref : th.Tensor
        reward_weight_torqueref : th.Tensor
        reward_weight_pos2posref : th.Tensor
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

    @dataclass
    class EpisodeLocomConfiguration:
        goal_abs_vel_vec_xys : th.Tensor
        """Absolute frame linvel goal, if goal_rel_vel_vec_xy_speed_xy is not None, it is ignored. Expressed as 3 
           scalars: xy direction of the velocity and speed."""
        goal_rel_vel_vec_xy_speed : th.Tensor | None
        """Relative frame linvel goal, overrides goal_abs_vel_vec_xy. Expressed as 3 scalars: xy direction of the
           velocity and speed"""
        goal_abs_gravity_vec_xyz : th.Tensor
        goal_abs_height_vec_z : th.Tensor
        goal_heading_rel_vec_yaw : th.Tensor

    LOCOMOTION_FIELDS = IntEnum("LOCOMOTION_FIELDS", ["COLLISON_COUNT",
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
                                                    "REWARD_ACCELERATION_WEIGHT",
                                                    "REWARD_CONTACTS_WEIGHT",
                                                    "REWARD_HEALTH_WEIGHT",
                                                    "REWARD_POSITION_LIMIT_WEIGHT",
                                                    "REWARD_TORQUEDIFF_WEIGHT",
                                                    "REWARD_TORQUE_LIMIT_WEIGHT",
                                                    "REWARD_TORQUE_WEIGHT",
                                                    "REWARD_TRACKING_WEIGHT",
                                                    "REWARD_VELOCITY_LIMIT_WEIGHT",
                                                    "REWARD_VELOCITY_WEIGHT",
                                                    "REWARD_HEIGHT_VELOCITY_WEIGHT",
                                                    "REWARD_HEIGHT_POSITION_WEIGHT",
                                                    "REWARD_PITCHNROLL_WEIGHT",
                                                    "REWARD_PITCHNROLL_VELOCITY_WEIGHT",
                                                    "REWARD_ACTDIFF_WEIGHT",
                                                    "REWARD_ACTACC_WEIGHT",
                                                    "REWARD_FEET_AIR_TIME_WEIGHT",
                                                    "REWARD_FEET_GROUND_TIME_WEIGHT",
                                                    "REWARD_FEET_ON_GROUND_WEIGHT",
                                                    "REWARD_POSITION_WEIGHT",
                                                    "REWARD_HEADING_WEIGHT",
                                                    "REWARD_HEADING_VELOCITY_WEIGHT",
                                                    "REWARD_FAILURE_WEIGHT",
                                                    "REWARD_SENSED_EFFORT_WEIGHT",
                                                    "REWARD_SLIP_WEIGHT",
                                                    "REWARD_VELREF_WEIGHT",
                                                    "REWARD_TORQUEREF_WEIGHT",
                                                    "REWARD_POS2POSREF_WEIGHT",
                                                    "SMOOTHED_TRACKING_ERROR",
                                                    "SMOOTHED_HEIGHT_ERROR",
                                                    "SMOOTHED_PITCHNROLL_ERROR",
                                                    "SMOOTHED_HEADING_ERROR",
                                                    "SMOOTHED_PITCHNROLL_ERROR_VELOCITY",
                                                    "SMOOTHED_GOAL_BODY_HEIGHT",
                                                    "SUM_IMPULSES",
                                                    "CRASHED",
                                                    "SUPPORT_POLYGON_LINVEL_X",
                                                    "SUPPORT_POLYGON_LINVEL_Y",
                                                    "SUPPORT_POLYGON_LINVEL_Z"], start=0)
    
    FEET_FIELDS = IntEnum("FEET_FIELDS" , ["FEET_AIR_DURATIONS",
                                           "FEET_GROUND_DURATIONS",
                                           "FEET_VEL_X",
                                           "FEET_VEL_Y",
                                           "AVG_FEET_STEP_DURATIONS"], start=0)

    def __init__(self,  action_delay_mustd_std : tuple[float,float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseVecJointImpedanceAdapter,
                        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor],
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | JOINT_FILTERS],
                        disallowed_contact_links : list[tuple[str,str]],
                        enable_dbg_checks : bool,
                        fail_on_safety : bool,
                        feet_links : list[tuple[str,str]],
                        frame_stack_length : int,
                        free_joints : Sequence[str],
                        goal_err_smoothing_halflife_sec : float,
                        goal_height_minmax : tuple[float,float],
                        goal_speed_minmax : tuple[float, float],
                        goal_yaw_minmax : tuple[float, float],
                        ground_link : tuple[str,str],
                        held_joints_damping : float,
                        held_joints_stiffness : float,
                        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                        homing_joint_pose : dict[tuple[str,str], float],
                        init_on_reset_ratio : float,
                        initial_height_randomization_range_meters : float,
                        initial_joint_pose_randomization_range : float,
                        maxStepsPerEpisode : int,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        obs_noise_angvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_gravity_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_joints_pve_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_linacc_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_linvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_posz_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        quiet : bool,
                        reward_acceleration_weight : float,
                        reward_actacc_weight : float,
                        reward_actdiff_weight : float,
                        reward_contacts_weight : float,
                        reward_energy_weight : float,
                        reward_failure_weight : float,
                        reward_feet_air_time_weight : float,
                        reward_feet_ground_time_weight : float,
                        reward_feet_on_ground_weight : float,
                        reward_heading_weight : float,
                        reward_heading_velocity_weight : float,
                        reward_health_weight : float,
                        reward_height_velocity_weight : float,
                        reward_height_position_weight : float,
                        reward_pitchnroll_weight : float,
                        reward_pitchnroll_velocity_weight : float,
                        reward_pos2posref_weight : float,
                        reward_position_limit_weight : float,
                        reward_position_weight : float,
                        reward_scale : float,
                        reward_sensed_effort_weight : float,
                        reward_slip_weight : float,
                        reward_torque_limit_weight : float,
                        reward_cmdtorque_weight : float,
                        reward_torquediff_weight : float,
                        reward_torqueref_weight : float,
                        reward_tracking_weight : float,
                        reward_velocity_limit_weight : float,
                        reward_velocity_weight : float,
                        reward_velref_weight : float,
                        robot_main_body_link : str,
                        robot_name : str,
                        robot_root_link : str,
                        robot_urdf_string : str,
                        safe_damping : float,
                        safe_limits_position_offset : dict[tuple[str,str], float],
                        safe_stiffness : float,
                        safety_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float], 
                        seed : int,
                        stepLength_sec : float,
                        step_precision_tolerance : float,
                        stop_on_failure : bool,
                        terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]],
                        th_device : th.device,
                        use_contacts : bool,
                        verbose_infos : bool,
                        max_good_step_duration : float,
                        min_good_step_duration : float,
                        enable_limits_safety : bool = True,
                        enable_link_collisions : list[tuple[tuple[str,str],list[tuple[str,str]]]] | None = [],
                        enable_posref_safety : bool = True,
                        heightmap_resolution : int = -1,
                        impulse_duration_minmax : tuple[float,float ]= (0.01, 5.0),
                        impulse_mean_std : tuple[float,float ]= (50.0, 50.0),
                        impulse_probability_per_sec : float = 0.0,
                        just_health_reward : bool = False,
                        longterm_states_decimation_time : float = 0.0001,
                        merge_privileged : bool = False,
                        observe_full_robot_state : bool = False,
                        posref_safety_period = 0.001,
                        randomized_armature_joints : Sequence[tuple[str,str]] = [],
                        randomized_armature_ratios : float = 0.1,
                        randomized_com_links : list[tuple[str,str]] = [],
                        randomized_com_xyz_diff_distribution : DistributionDef = ("uniform", ([-0.01,-0.01,-0.01],[0.01, 0.01,0.01])),
                        randomized_friction_links : list[tuple[str,str]] = [],
                        randomized_friction_slide_spin_roll_ratios : tuple[float, float, float] = (0.1,0.1,0.1),
                        randomized_frictionloss_joints : Sequence[tuple[str,str]] = [],
                        randomized_frictionloss_ratios : float = 0.0,
                        randomized_gains_damping_ratio_epstd : float = 1.0,
                        randomized_gains_stiffness_ratio_epstd : float = 1.0,
                        randomized_mass_links : list[tuple[str,str]] = [],
                        randomized_mass_ratios_distr : DistributionDef = ("normal", (0.0, 0.05)),
                        randomized_reference_filter_distribution : DistributionDef | None = None,
                        recycle_pose_randomization : bool = False,
                        saturate_jimp_ref_limits : bool = True,
                        ui_camera_resolution_hw : tuple[int,int] = (256,144),
                        goal_resampling_probability_per_sec : float = 0.0,
                        max_goal_height_pos_change_speed : float = 0.25,
                        max_height_speed_goal : float = 1.0,
                        feet_air_time_avg_alpha = 1.0
                        ):
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._unit_3d_vector = self._thtens([1.0, 0.0, 0.0])
        self._unit_quaternion = self._thtens([0.0, 0.0, 0.0, 1.0])
        self._zero = self._thtens([0.0])
        self._loco_conf = LocomotionVecEnv.LocomotionConfiguration(
                        reward_weight_acceleration = self._thtens(reward_acceleration_weight),
                        reward_weight_contacts  = self._thtens(reward_contacts_weight) ,
                        reward_weight_health = self._thtens(reward_health_weight),
                        reward_weight_energy  = self._thtens(reward_energy_weight) ,
                        reward_weight_position_limit  = self._thtens(reward_position_limit_weight) ,
                        reward_scale  = self._thtens(reward_scale) ,
                        reward_weight_torque_limit  = self._thtens(reward_torque_limit_weight) ,
                        reward_weight_torque = self._thtens(reward_cmdtorque_weight),
                        reward_weight_torquediff = self._thtens(reward_torquediff_weight),
                        reward_weight_tracking = self._thtens(reward_tracking_weight),
                        reward_weight_velocity_limit = self._thtens(reward_velocity_limit_weight),
                        reward_weight_velocity = self._thtens(reward_velocity_weight),
                        reward_weight_position = self._thtens(reward_position_weight),
                        reward_weight_heading = self._thtens(reward_heading_weight),
                        reward_weight_heading_velocity = self._thtens(reward_heading_velocity_weight),
                        reward_weight_height_velocity = self._thtens(reward_height_velocity_weight),
                        reward_weight_height_position = self._thtens(reward_height_position_weight),
                        reward_weight_pitchnroll = self._thtens(reward_pitchnroll_weight),
                        reward_weight_pitchnroll_velocity = self._thtens(reward_pitchnroll_velocity_weight),
                        reward_weight_actdiff = self._thtens(reward_actdiff_weight),
                        reward_weight_actacc = self._thtens(reward_actacc_weight),
                        reward_weight_feet_air_time = self._thtens(reward_feet_air_time_weight),
                        reward_weight_feet_ground_time = self._thtens(reward_feet_ground_time_weight),
                        reward_weight_feet_on_ground = self._thtens(reward_feet_on_ground_weight),
                        reward_weight_failure = self._thtens(reward_failure_weight),
                        reward_weight_sensed_effort = self._thtens(reward_sensed_effort_weight),
                        reward_weight_slip=self._thtens(reward_slip_weight),
                        reward_weight_velref=self._thtens(reward_velref_weight),
                        reward_weight_torqueref=self._thtens(reward_torqueref_weight),
                        reward_weight_pos2posref=self._thtens(reward_pos2posref_weight),
                        use_contacts = use_contacts,
                        disallowed_contact_links=disallowed_contact_links,
                        terminating_contact_pairs=terminating_contact_pairs,
                        goal_speed_minmax = th.as_tensor(goal_speed_minmax, device=th_device, dtype=th.float32),
                        goal_abs_yaw_minmax = th.as_tensor(goal_yaw_minmax, device=th_device, dtype=th.float32),
                        height_reward_settle_point=self._thtens(0.025), # A narrow reward bell
                        height_reward_2_settle_point=self._thtens(0.3), # A wider reward bell
                        height_reward_2_weight=self._thtens(0.5), # weight of the wide reward bell over the narrow one
                        pitchnroll_reward_settle_point=self._thtens(th.pi/4), # ~zero reward after this angle
                        heading_reward_settle_point = self._thtens(3.14159/16), # ~zero reward after this distance (w component of the quat difference)
                        vel_reward_goalrelative_weight = self._thtens(0.25),
                        reward_vel_goal_relative_width = self._thtens(1.5),
                        reward_vel_goal_absolute_width = self._thtens(0.25),
                        reward_vel_goal_relative_width_offset = self._thtens(0.1),
                        feet_links = feet_links,
                        heightmap_resolution_xy = (heightmap_resolution,heightmap_resolution),
                        min_good_step_air_duration=min_good_step_duration,
                        max_good_step_duration=max_good_step_duration,
                        min_good_ground_duration=min_good_step_duration,
                        max_good_ground_duration=max_good_step_duration,
                        goal_height_minmax = goal_height_minmax,
                        goal_resampling_probability_per_sec = self._thtens(goal_resampling_probability_per_sec),
                        goal_resampling_enabled = goal_resampling_probability_per_sec > 0.0,
                        max_goal_height_pos_change_speed = max_goal_height_pos_change_speed,
                        max_height_speed_goal = max_height_speed_goal,
                        heading_kp=self._thtens(1.0),
                        heading_kd=self._thtens(0.0),
                        max_heading_speed_goal=self._thtens(th.pi/4),
                        feet_air_time_avg_alpha = feet_air_time_avg_alpha,
                        max_pitchnroll_speed_goal=self._thtens(th.pi/2),
                        pitchnroll_kp=self._thtens(1.0),
                        pitchnroll_kd=self._thtens(0.0),
                        goal_heading_rel_yaw_minmax=self._thtens([-th.pi, th.pi])
                        )
        
        self._locomotion_episode_config = LocomotionVecEnv.EpisodeLocomConfiguration(goal_abs_vel_vec_xys       = self._thtens([1.0,0.0,0.0]).expand(adapter.vec_size(), 3).detach().clone(),
                                                                                     goal_rel_vel_vec_xy_speed  = None,
                                                                                     goal_abs_gravity_vec_xyz   = self._thtens([0.0,0.0,-1.0]).expand(adapter.vec_size(), 3).detach().clone(),
                                                                                     goal_abs_height_vec_z      = self._thtens([sum(self._loco_conf.goal_height_minmax)/2]).expand(adapter.vec_size(), 1).detach().clone(),
                                                                                     goal_heading_rel_vec_yaw = self._thtens([0.0]).expand(adapter.vec_size(), 1).detach().clone())
        
        super().__init__(   action_delay_mustd_std = action_delay_mustd_std,
                            action_noise_mustd = action_noise_mustd, 
                            action_smoothing_halflife_sec = action_smoothing_halflife_sec,
                            adapter = adapter,
                            control_limits_minmax_pve = control_limits_minmax_pve,
                            control_mode = control_mode,
                            controlled_joints = controlled_joints,
                            enable_dbg_checks = enable_dbg_checks,
                            enable_limits_safety = enable_limits_safety,
                            enable_link_collisions = enable_link_collisions,
                            enable_posref_safety = enable_posref_safety,
                            fail_on_safety = fail_on_safety,
                            frame_stack_length=frame_stack_length,
                            free_joints=free_joints,
                            goal_err_smoothing_halflife_sec = goal_err_smoothing_halflife_sec,
                            ground_link=ground_link,
                            held_joints_damping = held_joints_damping,
                            held_joints_stiffness = held_joints_stiffness,
                            homing_body_pose_xyz_xyzw = homing_body_pose_xyz_xyzw,
                            homing_joint_pose = homing_joint_pose,
                            impulse_duration_minmax = impulse_duration_minmax,
                            impulse_mean_std = impulse_mean_std,
                            impulse_probability_per_sec = impulse_probability_per_sec,
                            init_on_reset_ratio = init_on_reset_ratio,
                            initial_joint_pose_randomization_range = initial_joint_pose_randomization_range,
                            initial_height_randomization_range_meters = initial_height_randomization_range_meters,
                            just_health_reward=just_health_reward,
                            longterm_states_decimation_time = longterm_states_decimation_time,
                            maxStepsPerEpisode = maxStepsPerEpisode,
                            merge_privileged=merge_privileged,
                            minmax_damping = minmax_damping,
                            minmax_stiffness = minmax_stiffness,
                            obs_noise_angvel_ep_mustd_step_std = obs_noise_angvel_ep_mustd_step_std,
                            obs_noise_gravity_ep_mustd_step_std = obs_noise_gravity_ep_mustd_step_std,
                            obs_noise_joints_pve_ep_mustd_step_std = obs_noise_joints_pve_ep_mustd_step_std,
                            obs_noise_linacc_ep_mustd_step_std = obs_noise_linacc_ep_mustd_step_std,
                            obs_noise_linvel_ep_mustd_step_std = obs_noise_linvel_ep_mustd_step_std,
                            obs_noise_posz_ep_mustd_step_std = obs_noise_posz_ep_mustd_step_std,
                            observe_full_robot_state = observe_full_robot_state,
                            posref_safety_period = posref_safety_period,
                            quiet = quiet,
                            randomized_armature_joints = randomized_armature_joints,
                            randomized_armature_ratios = randomized_armature_ratios,
                            randomized_com_links=randomized_com_links,
                            randomized_com_xyz_diff_distribution=randomized_com_xyz_diff_distribution,
                            randomized_friction_links=randomized_friction_links,
                            randomized_friction_slide_spin_roll_ratios=randomized_friction_slide_spin_roll_ratios,
                            randomized_frictionloss_joints=randomized_frictionloss_joints,
                            randomized_frictionloss_ratios=randomized_frictionloss_ratios,
                            randomized_gains_damping_ratio_epstd=randomized_gains_damping_ratio_epstd,
                            randomized_gains_stiffness_ratio_epstd=randomized_gains_stiffness_ratio_epstd,
                            randomized_mass_links=randomized_mass_links,
                            randomized_mass_ratios_distr=randomized_mass_ratios_distr,
                            randomized_reference_filter_distribution=randomized_reference_filter_distribution,
                            recycle_pose_randomization=recycle_pose_randomization,
                            robot_main_body_link = robot_main_body_link,
                            robot_name = robot_name,
                            robot_root_link = robot_root_link,
                            robot_urdf_string = robot_urdf_string,
                            safe_damping = safe_damping,
                            safe_limits_position_offset = safe_limits_position_offset,
                            safe_stiffness = safe_stiffness,
                            safety_limits_ratios_minmax_pve = safety_limits_ratios_minmax_pve,
                            saturate_jimp_ref_limits = saturate_jimp_ref_limits,
                            seed = seed,
                            stepLength_sec = stepLength_sec,
                            step_precision_tolerance = step_precision_tolerance,
                            stop_on_failure = stop_on_failure,
                            th_device = th_device,
                            ui_camera_resolution_hw = ui_camera_resolution_hw,
                            verbose_infos = verbose_infos
                        )

        
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__
        obs_labels = self._state_helper.observation_names()
        # ggLog.info(f"Obs labels = \n{pprint.pformat(obs_labels)}")
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
        super()._build()
        self._feet_link_ids = self._adapter.get_links_ids(self._loco_conf.feet_links)
        self._ground_link_id = self._adapter.get_links_ids([self._configuration.ground_link])



    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        base_loco_fields = [self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z,
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED,
                            self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT,
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X,
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y]
        privileged_loco_fields = [  self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR,
                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY]
        if self._configuration.merge_privileged:
            obs_defs = {"base" : ThBoxStateHelper.SimpleObsDef(observable_fields=base_loco_fields+privileged_loco_fields)}
        else:
            obs_defs = {"base" : ThBoxStateHelper.SimpleObsDef(observable_fields=base_loco_fields),
                        "privileged" : ThBoxStateHelper.SimpleObsDef(observable_fields=privileged_loco_fields)}
        self._locomotion_state_helper = ThBoxStateHelper( field_names=[e for e in self.LOCOMOTION_FIELDS],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
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
                                                                    self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEIGHT_VELOCITY_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEIGHT_POSITION_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_VELOCITY_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_ACTACC_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_FEET_AIR_TIME_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_FEET_GROUND_TIME_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_FEET_ON_GROUND_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEADING_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEADING_VELOCITY_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_FAILURE_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELREF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_SENSED_EFFORT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUEREF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POS2POSREF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_SLIP_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SUM_IMPULSES : [0,10000],
                                                                    self.LOCOMOTION_FIELDS.COLLISON_COUNT : [0,1000],
                                                                    self.LOCOMOTION_FIELDS.CRASHED : [0,1],
                                                                    self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_X : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_Y : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_Z : [-10,10]},
                                                    history_length=3,
                                                    observation_definitions=obs_defs,
                                                    vec_size=adapter.vec_size())
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_LOCOMOTION,
                                                            self._locomotion_state_helper,
                                                            obs_defs={"base":{"observable":True,"flatten":True,"noise":None}})
        feet_num = len(self._loco_conf.feet_links)

        feet_obs_type = "privileged" if not self._configuration.merge_privileged else "base"
        self._feet_state_helper = ThBoxStateHelper( field_names=[e for e in self.FEET_FIELDS],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(len(self._loco_conf.feet_links),),
                                                    fields_minmax={ 
                                                        self.FEET_FIELDS.FEET_AIR_DURATIONS     : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_GROUND_DURATIONS  : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS : th.as_tensor([[-10.0],[10.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_VEL_X : th.as_tensor([[-100.0],[100.0]]).expand(2,feet_num),
                                                        self.FEET_FIELDS.FEET_VEL_Y : th.as_tensor([[-100.0],[100.0]]).expand(2,feet_num)},
                                                    vec_size=adapter.vec_size(),
                                                    history_length=1,
                                                    observation_definitions=
                                                        {   feet_obs_type : ThBoxStateHelper.SimpleObsDef(observable_fields=[self.FEET_FIELDS.FEET_AIR_DURATIONS,
                                                                                                                            self.FEET_FIELDS.FEET_GROUND_DURATIONS,
                                                                                                                            self.FEET_FIELDS.FEET_VEL_X,
                                                                                                                            self.FEET_FIELDS.FEET_VEL_Y])})
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_FEET,
                                                            self._feet_state_helper,
                                                            obs_defs={feet_obs_type:{"observable":True,"flatten":True,"noise":None}})
        if self._loco_conf.heightmap_resolution_xy[0] > 0:
            heightmap_state_helper = ThBoxStateHelper( field_names=["map"],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=self._loco_conf.heightmap_resolution_xy,
                                                    fields_minmax={"map" : self._thtens([-10.0, 10.0])},
                                                    vec_size=adapter.vec_size())
            self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_HEIGHTMAP,
                                                                 heightmap_state_helper,
                                                                 obs_defs={"base":{"observable":True,"flatten":False,"noise":None}})
        ggLog.info(f"Built state/obs/action helpers")
        
    @th.compiler.disable
    def _get_loco_adapter_data(self):
        if isinstance(self._adapter,BaseVecSimulationAdapter):
            feet_linvels_vec_foot_xyz = self._adapter.getLinksState(self._feet_link_ids)[:,:,7:10]
        else:
            feet_linvels_vec_foot_xyz = self._thzeros((self.num_envs,4,3))
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            from adarl.adapters.MjxAdapter import MjxAdapter
            mjx_adapter : MjxAdapter = self._adapter #type: ignore
            feet_are_touching_ground = mjx_adapter.check_colliding_links(self._feet_link_ids, self._ground_link_id)
        else:
            feet_are_touching_ground = self._thzeros((self.num_envs,4))
        borient_quat_vec_xyzw = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_pose = False)[:,0,3:7]
        return feet_linvels_vec_foot_xyz, feet_are_touching_ground, borient_quat_vec_xyzw

    @override
    def _get_adapter_data(self):
        super_adapter_data = super()._get_adapter_data()
        loco_adapter_data = self._get_loco_adapter_data()
        return loco_adapter_data, super_adapter_data

    @override
    def _get_new_instantaneous_state(self, adapter_data):

        loco_adapter_data, super_adapter_data = adapter_data
        feet_linvels_vec_foot_xyz, feet_are_touching_ground, borient_quat_vec_xyzw = loco_adapter_data

        track_support_linvel = False
        prev_locom_state =    self._current_state[self.STATE_LOCOMOTION][:, 0]
        prev_internal_state = self._current_state[self.STATE_INTERNAL][:, 0]
        prev_extrinsic_state = self._current_state[self.STATE_EXTRINSIC][:, 0]
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
        
        step_counts = new_internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        starting_eps = (step_counts<=0).view((self.num_envs,))
        
        
        max_goal_height_diff = self._loco_conf.max_goal_height_pos_change_speed*self._configuration.stepLength_sec
        goal_height = self._locomotion_episode_config.goal_abs_height_vec_z
        prev_smoothed_goal_height = prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        smoothed_goal_height = prev_smoothed_goal_height + th.clamp(goal_height - prev_smoothed_goal_height, min=-max_goal_height_diff, max=max_goal_height_diff)
        # smoothed_goal_height = goal_height*(1-goal_smoothing_alpha) + prev_smoothed_goal_height*goal_smoothing_alpha
        masked_assign(smoothed_goal_height,         starting_eps,   goal_height)

        vsize = step_counts.size()[0]
        if self._locomotion_episode_config.goal_rel_vel_vec_xy_speed is None:
            # Get the relative goal from the absolute one
            # Only possible with body pose (i.e. in simulation)
            goal_speed = self._locomotion_episode_config.goal_abs_vel_vec_xys[:,2].view((vsize,1))
            abs_goal_linvel_direction_xy = self._locomotion_episode_config.goal_abs_vel_vec_xys[:,:2]/th.linalg.norm(self._locomotion_episode_config.goal_abs_vel_vec_xys[:,:2], dim=-1, keepdim=True)
            abs_goal_linvel_direction_xyz = th.cat([abs_goal_linvel_direction_xy, th.zeros_like(abs_goal_linvel_direction_xy[:,:1])], dim = 1) # should be always planar
            abs_goal_linvel_xyz = abs_goal_linvel_direction_xyz * goal_speed
            # abs_planar_linvelgoal_dir_quat = quat_xyzw_between_vecs_py(self._unit_3d_vector_vec_x, abs_planar_linvel_goal) # orientation of the linvel goal (quat that aligns (1,0,0) to it)
            rel_goal_linvel_dir_xyz = th_quat_rotate(abs_goal_linvel_direction_xyz, th_quat_conj(borient_quat_vec_xyzw))
        else:
            # The relative goal is expressed in the plane orthogonal to gravity
            # So the full realtive goal must be converted in the frame of the body.
            # In this formulation, we can see the planar relative goal direction as a twist around the gravity vector,
            # we have then to add a swing rotation, perpendicular to the gravity vector.
            # The swing can be obtained directly from the gravity vector, as the rotation that brings it to 0,0,-1
            goal_speed = self._locomotion_episode_config.goal_rel_vel_vec_xy_speed[:,2]
            rel_planar_goal_linvel_direction_xy = self._locomotion_episode_config.goal_rel_vel_vec_xy_speed[:,:2]
            rel_planar_goal_linvel_direction_xyz = th.cat([rel_planar_goal_linvel_direction_xy[:,:2], th.zeros_like(rel_planar_goal_linvel_direction_xy[:,:1])], dim=1)
            swing = quat_xyzw_between_vecs_py(gravity_rel_vec_xyz, self._abs_gravity_dir.expand_as(gravity_rel_vec_xyz))
            twist = quat_xyzw_between_vecs_py(self._unit_3d_vector.expand_as(gravity_rel_vec_xyz), rel_planar_goal_linvel_direction_xyz)
            dir_quat = quat_mul_xyzw(twist,swing) #first swing then twist, i think
            rel_goal_linvel_dir_xyz = th_quat_rotate(self._unit_3d_vector.expand_as(gravity_rel_vec_xyz), dir_quat)
            abs_goal_linvel_xyz = th.zeros_like(rel_goal_linvel_dir_xyz) # not used in this branch
        rel_goal_linvel_xyz = rel_goal_linvel_dir_xyz*goal_speed # relative to the body orientation
        rel_curr_heading_quat = quat_xyzw_between_vecs_py(rel_goal_linvel_dir_xyz, self._unit_3d_vector.expand(self.num_envs,3)) # orientation of the body with respect to linvel goal (quat that aligns linvel to the body)

        # compute linvel error
        support_polygon_linvel = th.mean(feet_linvels_vec_foot_xyz, dim=1) # average linvel across the feet
        if track_support_linvel:
            tracked_body_linvel = support_polygon_linvel
        else:
            tracked_body_linvel = body_rel_linvel_vec_xyz
        tracking_err_vec = planar_tracking_error_vec(tracked_body_linvel, gravity_rel_vec_xyz, rel_goal_linvel_xyz).view(vsize,1)
        
        # compute heading (yaw) error
        goal_rel_heading_yaw = self._locomotion_episode_config.goal_heading_rel_vec_yaw
        rel_goal_heading_quat = th.cat([self._thzeros((self.num_envs,2)),
                                        th.sin(goal_rel_heading_yaw/2).view((self.num_envs,1)),
                                        th.cos(goal_rel_heading_yaw/2).view((self.num_envs,1))], dim = 1)
        # abs_curr_heading_quat = th_quat_conj(borient_quat_vec_xyzw) # orientation of the body (quat that aligns (1,0,0) to it)
        # rel_curr_heading_quat = quat_mul_xyzw(abs_curr_heading_quat, th_quat_conj(abs_planar_linvelgoal_dir_quat)) # orientation of the body with respect to linvel goal (quat that aligns linvel to the body)
        # heading_error_vec = quat_angle_xyzw(quat_mul_xyzw(th_quat_conj(rel_goal_heading_quat), rel_curr_heading_quat)).view(vsize,1)
        # the w component is by itself a measure of the size of the rotation, 2acos(w) would be the actual angle, but it is numerically unstable
        # in practice at w=1 the orientations are close, at -1 they are 180 degrees apart
        heading_error_vec = (1-quat_mul_xyzw(th_quat_conj(rel_goal_heading_quat), rel_curr_heading_quat)[:,3]).view(vsize,1)
        
        # compute height error
        height_err_vec = new_extrinsic_state[self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] - smoothed_goal_height

        # compute pitch and roll error
        # pitchnroll_err_vec = th.linalg.norm(gravity_rel_vec_xyz-self._locomotion_episode_config.goal_abs_gravity_vec_xyz, dim = 1, keepdim=True) # Would be nice to use geodesic distance or somethinglike that
        pitchnroll_err_vec = vectors_angle(gravity_rel_vec_xyz, self._locomotion_episode_config.goal_abs_gravity_vec_xyz).view((self.num_envs,1))
        prev_gravity_rel_vec_xyz = prev_extrinsic_state[:,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1].view((self.num_envs,3))
        prev_goal_abs_gravity_vec_xyz = prev_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X:self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z+1].view((self.num_envs,3))
        pitchnroll_err_vel_vec = pitchnroll_err_vec - vectors_angle(prev_gravity_rel_vec_xyz, prev_goal_abs_gravity_vec_xyz).view((self.num_envs,1))

        alpha = self._configuration.goal_err_exp_smoothing_1s**(self._configuration.stepLength_sec)
        smoothed_tracking_err_vec = tracking_err_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]*alpha
        smoothed_height_error = height_err_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR]*alpha
        smoothed_pitchnroll_error = pitchnroll_err_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR]*alpha
        smoothed_heading_error_vec = heading_error_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR]*alpha
        smoothed_pitchnroll_error_vel_vec = pitchnroll_err_vel_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY]*alpha
        masked_assign(smoothed_tracking_err_vec,    starting_eps,   tracking_err_vec)
        masked_assign(smoothed_height_error,        starting_eps,   height_err_vec)
        masked_assign(smoothed_pitchnroll_error,    starting_eps,   pitchnroll_err_vec)
        masked_assign(smoothed_heading_error_vec,   starting_eps,   heading_error_vec)
        masked_assign(smoothed_pitchnroll_error_vel_vec,   starting_eps,   pitchnroll_err_vel_vec)

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
            collision_count_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            sum_bad_impulses_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            crashed_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)



        new_locom_state = { self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : self._loco_conf.reward_weight_torque_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : self._loco_conf.reward_weight_position_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : self._loco_conf.reward_weight_velocity_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT : self._loco_conf.reward_weight_velocity.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT : self._loco_conf.reward_weight_acceleration.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT : self._loco_conf.reward_weight_contacts.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT : self._loco_conf.reward_weight_health.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEIGHT_VELOCITY_WEIGHT : self._loco_conf.reward_weight_height_velocity.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEIGHT_POSITION_WEIGHT : self._loco_conf.reward_weight_height_position.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : self._loco_conf.reward_weight_pitchnroll.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_VELOCITY_WEIGHT : self._loco_conf.reward_weight_pitchnroll_velocity.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT : self._loco_conf.reward_weight_actdiff.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_ACTACC_WEIGHT : self._loco_conf.reward_weight_actacc.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_FEET_AIR_TIME_WEIGHT : self._loco_conf.reward_weight_feet_air_time.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_FEET_GROUND_TIME_WEIGHT : self._loco_conf.reward_weight_feet_ground_time.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_FEET_ON_GROUND_WEIGHT : self._loco_conf.reward_weight_feet_on_ground.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT : self._loco_conf.reward_weight_tracking.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT : self._loco_conf.reward_weight_torque.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : self._loco_conf.reward_weight_torquediff.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT : self._loco_conf.reward_weight_position.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEADING_WEIGHT : self._loco_conf.reward_weight_heading.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEADING_VELOCITY_WEIGHT : self._loco_conf.reward_weight_heading_velocity.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_FAILURE_WEIGHT : self._loco_conf.reward_weight_failure.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_SENSED_EFFORT_WEIGHT : self._loco_conf.reward_weight_sensed_effort.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_SLIP_WEIGHT : self._loco_conf.reward_weight_slip.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_VELREF_WEIGHT : self._loco_conf.reward_weight_velref.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TORQUEREF_WEIGHT : self._loco_conf.reward_weight_torqueref.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_POS2POSREF_WEIGHT : self._loco_conf.reward_weight_pos2posref.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : smoothed_tracking_err_vec.view(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_HEIGHT_ERROR : smoothed_height_error.view(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR : smoothed_pitchnroll_error.view(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR : smoothed_heading_error_vec.view(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR_VELOCITY : smoothed_pitchnroll_error_vel_vec.view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X : rel_goal_linvel_dir_xyz[:,0].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y : rel_goal_linvel_dir_xyz[:,1].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z : rel_goal_linvel_dir_xyz[:,2].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED : goal_speed.view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X : abs_goal_linvel_xyz[:,0].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y : abs_goal_linvel_xyz[:,1].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z : abs_goal_linvel_xyz[:,2].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : self._locomotion_episode_config.goal_abs_height_vec_z,
                            self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT : smoothed_goal_height.view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X : th.cos(goal_rel_heading_yaw).view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y : th.sin(goal_rel_heading_yaw).view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,0].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,1].view(vsize,1),
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,2].view(vsize,1),
                            self.LOCOMOTION_FIELDS.SUM_IMPULSES : sum_bad_impulses_vec,
                            self.LOCOMOTION_FIELDS.COLLISON_COUNT :collision_count_vec,
                            self.LOCOMOTION_FIELDS.CRASHED : crashed_vec,
                            self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_X : support_polygon_linvel[:,0].view(vsize,1),
                            self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_Y : support_polygon_linvel[:,1].view(vsize,1),
                            self.LOCOMOTION_FIELDS.SUPPORT_POLYGON_LINVEL_Z : support_polygon_linvel[:,2].view(vsize,1)}
        
        # fstates_vec_13 = self._adapter.getLinksState(requestedLinks = self._feet_link_ids, use_com_pose = False)
        # feet_lifted = fstates_vec_13[:,:,2] > self._feet_radius + 0.001
        nenv_nfeet = (self.num_envs,len(self._loco_conf.feet_links))
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            prev_time = prev_internal_state[:,self.INTERNAL_FIELDS.SIM_TIME].view((self.num_envs,))
            curr_time = new_internal_state[self.INTERNAL_FIELDS.SIM_TIME].view((self.num_envs,))
            dt : th.Tensor = curr_time - prev_time
            # ggLog.info(f"feet_state = {self._current_state[self.STATE_FEET]}")
            prev_feet_air_durations_vec_foot_t = self._current_state[self.STATE_FEET][:, 0, self.FEET_FIELDS.FEET_AIR_DURATIONS]
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
            new_feet_air_durations_vec_foot_t  = self._current_state[self.STATE_FEET][:, 0, self.FEET_FIELDS.FEET_AIR_DURATIONS].clone()
            th.where(condition=just_touching_down.expand(nenv_nfeet),
                     input = -prev_feet_air_durations_vec_foot_t, # if touching down, write negative step duration
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            th.where(condition=feet_were_touching_ground.expand(nenv_nfeet),
                     input = self._thtens(0.0), # if was touching ground already in the previous step, write zero
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            th.where(condition=th.logical_not(feet_are_touching_ground).expand(nenv_nfeet),
                     input = new_feet_air_durations_vec_foot_t+dt.unsqueeze(1).expand(nenv_nfeet), # if is up, increase time
                     other = new_feet_air_durations_vec_foot_t,
                     out   = new_feet_air_durations_vec_foot_t)
            
            prev_feet_ground_durations_vec_foot_t = self._current_state[self.STATE_FEET][:, 0, self.FEET_FIELDS.FEET_GROUND_DURATIONS]
            feet_were_in_air = th.logical_not(feet_were_touching_ground)
            just_lifting_up =    th.logical_and(feet_were_touching_ground, th.logical_not(feet_are_touching_ground))
            new_feet_ground_durations_vec_foot_t  = self._current_state[self.STATE_FEET][:, 0, self.FEET_FIELDS.FEET_GROUND_DURATIONS].clone()
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
            

            # ggLog.info(f"prev_feet_step_durations_vec_foot_t = \n{prev_feet_step_durations_vec_foot_t}")
            # ggLog.info(f"new_feet_step_durations_vec_foot_t = \n{new_feet_step_durations_vec_foot_t}")
            prev_avg_feet_step_durations = self._current_state[self.STATE_FEET][:, 0, self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS]
            new_avg_feet_step_durations = prev_avg_feet_step_durations.clone()
            alpha = self._loco_conf.feet_air_time_avg_alpha
            th.where(condition=just_touching_down.expand(nenv_nfeet),
                     input = -prev_feet_air_durations_vec_foot_t*(1-alpha) + prev_avg_feet_step_durations*alpha,
                     other = new_avg_feet_step_durations,
                     out   = new_avg_feet_step_durations)
        else:
            new_feet_air_durations_vec_foot_t = self._thtens([0.0]).expand(nenv_nfeet)
            new_avg_feet_step_durations = self._thtens([0.0]).expand(nenv_nfeet)
        new_feet_state = {  self.FEET_FIELDS.FEET_AIR_DURATIONS : new_feet_air_durations_vec_foot_t,
                            self.FEET_FIELDS.FEET_GROUND_DURATIONS : new_feet_ground_durations_vec_foot_t,
                            self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS : new_avg_feet_step_durations,
                            self.FEET_FIELDS.FEET_VEL_X : feet_linvels_vec_foot_xyz[:,:,0],
                            self.FEET_FIELDS.FEET_VEL_Y : feet_linvels_vec_foot_xyz[:,:,1]}

        new_inst_state[self.STATE_LOCOMOTION] = new_locom_state
        new_inst_state[self.STATE_FEET] = new_feet_state
        return new_inst_state

    def _height_velocity_reward(self, curr_state_extr_vec, current_state_locom_vec, current_state_internal, prev_state_extr_vec):
        curr_pos = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        prev_pos = prev_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        goal_pos = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        dt = current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT]
        max_speed = self._loco_conf.max_height_speed_goal
        kp = 2
        height_err = curr_pos-goal_pos
        goal_height_velocity = th.clamp(-height_err*kp, min=-max_speed, max=max_speed) 
        z_velocity = (curr_pos - prev_pos)/dt
        reward_height = double_bell_reward(z_velocity-goal_height_velocity,
                                           bell_width_a=self._thtens(0.05),
                                           bell_width_b=goal_height_velocity*2+0.025,
                                           bell_b_weight=self._thtens(0.5))
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
        curr_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prev_rel_gravity_vec_xyz      = state[self.STATE_EXTRINSIC][:,1,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        
        angle_vel = vectors_angle(curr_rel_gravity_vec_xyz, prev_rel_gravity_vec_xyz).view((self.num_envs,))
        return -penalty_reward(angle_vel, max_rew=1, exponent=2)

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
        rel_gravity_vec_xyz              = state[self.STATE_EXTRINSIC][:, 0,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        rel_goal_linvel_dir_xyz          = state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        prev_rel_gravity_vec_xyz         = state[self.STATE_EXTRINSIC][:, 1,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prev_rel_goal_linvel_dir_xyz     = state[self.STATE_LOCOMOTION][:,1,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        prevprev_rel_gravity_vec_xyz     = state[self.STATE_EXTRINSIC][:, 2,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X:self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z+1,0].view((self.num_envs,3))
        prevprev_rel_goal_linvel_dir_xyz = state[self.STATE_LOCOMOTION][:,2,self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X:self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z+1,0].view((self.num_envs,3))
        rel_body_direction_xyz = self._unit_3d_vector.expand_as(rel_gravity_vec_xyz) # The body orientation in the body frame
        
        flattening_rotation = quat_xyzw_between_vecs_py(rel_gravity_vec_xyz, self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        prev_flattening_rotation = quat_xyzw_between_vecs_py(prev_rel_gravity_vec_xyz, self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        prevprev_flattening_rotation = quat_xyzw_between_vecs_py(prevprev_rel_gravity_vec_xyz, self._abs_gravity_dir.expand_as(rel_gravity_vec_xyz)) # rotation that brings gravity to 0,0,-1
        
        # rotate body and goal direction into the flattened frame
        curr_bodydir_flatbodyframe = th_quat_rotate(rel_body_direction_xyz,  flattening_rotation)
        curr_linvelgoaldir_flatbodyframe = th_quat_rotate(rel_goal_linvel_dir_xyz, flattening_rotation)
        prev_bodydir_flatbodyframe = th_quat_rotate(rel_body_direction_xyz,       prev_flattening_rotation)
        prev_linvelgoaldir_flatbodyframe = th_quat_rotate(prev_rel_goal_linvel_dir_xyz, prev_flattening_rotation)
        prevprev_bodydir_flatbodyframe = th_quat_rotate(rel_body_direction_xyz,           prevprev_flattening_rotation)
        prevprev_linvelgoaldir_flatbodyframe = th_quat_rotate(prevprev_rel_goal_linvel_dir_xyz, prevprev_flattening_rotation)

        eps = 1e-3
        dbg_check(lambda : th.all(th.stack([th.all(curr_linvelgoaldir_flatbodyframe[:,2]<eps),
                                            th.all(prev_linvelgoaldir_flatbodyframe[:,2]<eps),
                                            th.all(prevprev_linvelgoaldir_flatbodyframe[:,2]<eps)])),
                  assert_msg="flattened goal vectors not parallel to ground", async_assert=True)
        
        curr_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        curr_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prev_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prev_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prevprev_bodydir_flatbodyframe[:,2] = 0 # project to the gravity plane
        prevprev_linvelgoaldir_flatbodyframe[:,2] = 0 # project to the gravity plane

        rel_headinggoal_flatgoalframe = th.zeros_like(curr_bodydir_flatbodyframe)
        rel_headinggoal_flatgoalframe[:,:2] = state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_X:self.LOCOMOTION_FIELDS.GOAL_REL_HEADING_YAW_Y+1,0]
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
                                                    kd=self._loco_conf.heading_kd), curr_yaw_err

    @override
    # @th.compile(mode="max-autotune")
    @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune", disable=disable_compile)
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        if self._configuration.just_health_reward:
            sub_rewards_return["health"] = th.ones((self.num_envs,), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            return self._thtens([1.0]).expand(self.num_envs)
        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = self._configuration.reward_penalties_max
        current_state_locom_vec = state[self.STATE_LOCOMOTION][:, 0,:,0]
        curr_state_extr_vec = state[self.STATE_EXTRINSIC][:, 0,:,0]
        prev_state_extr_vec = state[self.STATE_EXTRINSIC][:, 1,:,0]
        current_state_internal = state[self.STATE_INTERNAL][:, 0,:,0]
        state_action_raw_vec = state[self.STATE_ACT_RAW_HIST]
        state_stats_v_h_j_minmaxavgstd_pvaee = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 5)

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_pvesd[:,0], lims[0,:,0], lims[1,:,0])
        state_robot_norm        = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)
        longterm_stats_pos_norm = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)
        # self._warn_out_of_bounds(state_robot_norm)
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)
        # state_stats_norm = self._state_helper.sub_helpers[self.STATE_ROBOT_STATS].normalize(state_stats)
        normposhomingdiff    = longterm_stats_pos_norm[:,0,0] - normhoming
        normvelocities       = state_robot_norm[:,0,:,1]
        normcmdtorques       = state_robot_norm[:,0,:,2]
        norm_velocity_refs   = state_robot_norm[:,0,:,6]
        norm_torque_refs     = state_robot_norm[:,0,:,7]
        norm_pos2posref_diff = state_robot_norm[:,0,:,0] - state_robot_norm[:,0,:,5]
        # normaccelerations   = (state_robot_norm[:,0,:,1] - state_robot_norm[:,1,:,1])/2 # like this it should be between [-1,1] #self._configuration.stepLength_sec
        max_senseff = 1_000 # max expected sensed effort (not really a strict max)
        max_jacc = 1_000 # max expected joint acceleration (not really a strict max)
        norm_senseff        = th.clamp(state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,0:2,4].abs().amax(dim=2)/max_senseff, -1, 1) # normalized max abs sensed effort
        normaccelerations   = state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,2,2]/max_jacc # normalized average accelearation
        normtorquediff      = state_robot_norm[:,0,:,2] - state_robot_norm[:,1,:,2]
        actdiff             = th.flatten((state_action_raw_vec[:,0] - state_action_raw_vec[:,1])/2, start_dim=1) # divide by 2 to keep it in [-1,1]
        prev_actdiff        = th.flatten((state_action_raw_vec[:,1] - state_action_raw_vec[:,2])/2, start_dim=1)
        act_acc             = (actdiff - prev_actdiff)/2

        position_safenorm   = state_robot_safenorm[:,0,:,0]
        velocities_safenorm = state_robot_safenorm[:,0,:,1]
        torque_safenorm     = state_robot_safenorm[:,0,:,2]

        reward_sensed_effort    = joint_penalty_reward(norm_senseff,     max_rew=1.0, exponent=8.0, reduction="max")
        reward_cmdtorque        = joint_penalty_reward(normcmdtorques,   max_rew=max_rew, exponent=4.0)
        reward_velocity         = joint_penalty_reward(normvelocities,max_rew=max_rew,exponent=2)
        # reward_acceleration     = flattened_penalty_reward(normaccelerations,max_rew=max_rew, exponent=8.0, flattening_scale=0.1)
        reward_acceleration     = flattened_joint_penalty_reward(normaccelerations,max_rew=max_rew, exponent=2.0, flattening_scale=0.1)
        reward_position         = flattened_joint_penalty_reward(normposhomingdiff,max_rew=max_rew, exponent=0.5, flattening_scale=0.02)
        reward_torquediff       = joint_penalty_reward(normtorquediff,max_rew=max_rew,exponent=2)
        reward_actdiff          = joint_penalty_reward(actdiff,max_rew=max_rew,exponent=2)
        reward_actacc           = joint_penalty_reward(act_acc,max_rew=1, exponent=2, presquash_factor=100)
        reward_torque_limit     = joint_penalty_reward(torque_safenorm,max_rew=1,exponent=50)
        reward_position_limit   = joint_penalty_reward(position_safenorm,max_rew=1,exponent=50)
        reward_velocity_limit   = joint_penalty_reward(velocities_safenorm,max_rew=1,exponent=50)
        
        reward_velocity_refs    = joint_penalty_reward(norm_velocity_refs,   max_rew=max_rew,exponent=2)
        reward_torque_refs      = joint_penalty_reward(norm_torque_refs,     max_rew=max_rew,exponent=2)
        reward_pos2posref_diff  = joint_penalty_reward(norm_pos2posref_diff, max_rew=max_rew,exponent=2)
        # reward_position     = bell_reward(th.mean(th.abs(normposhomingdiff), dim=1),
        #                                     zero_rew_dist=self._thtens(0.02))
        reward_height_velocity, _, _, _, _ = self._height_velocity_reward(curr_state_extr_vec, current_state_locom_vec, current_state_internal, prev_state_extr_vec)
        height_err = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]-current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        reward_height_position = double_bell_reward( error=height_err,
                                            bell_width_a=self._loco_conf.height_reward_settle_point,
                                            bell_width_b=self._loco_conf.height_reward_2_settle_point,
                                            bell_b_weight=self._loco_conf.height_reward_2_weight)
        # reward_pitchnroll   = bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR],
        #                                     zero_rew_dist=self._loco_conf.pitchnroll_reward_settle_point)
        reward_pitchnroll = penalty_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_PITCHNROLL_ERROR], max_rew=1, exponent=2)
        # reward_pitchnroll_velocity, _, _, _, _ = self._pitchnroll_velocity_reward(state, current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT])
        reward_pitchnroll_velocity = self._pitchnroll_velocity_penalty_reward(state)
        reward_heading_position = bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_HEADING_ERROR],
                                            zero_rew_dist=self._loco_conf.heading_reward_settle_point)
        reward_heading_velocity, _, _, _, _ = self._heading_velocity_reward(state, current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT])

        goalrelative_weight = self._loco_conf.vel_reward_goalrelative_weight
        rel_goal_bell_width = self._loco_conf.reward_vel_goal_relative_width
        rel_goal_offset = self._loco_conf.reward_vel_goal_relative_width_offset
        abs_goal_bell_width = self._loco_conf.reward_vel_goal_absolute_width
        goal_speed = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED].view((self.num_envs,))
        velocity_tracking_err_vec = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        reward_velocity_tracking = double_bell_reward(velocity_tracking_err_vec,
                                                      abs_goal_bell_width,
                                                      rel_goal_bell_width*(goal_speed+rel_goal_offset),
                                                      goalrelative_weight)
        
        reward_contacts = - th.clamp(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SUM_IMPULSES], -max_rew, max_rew)

        # FEET AIR TIME REWARD
        should_be_moving = goal_speed.view((self.num_envs,1)) > 0.01
        feet_state = state[self.STATE_FEET][:,0] # vec_size*history*fields*nfeet -> vec_size*fields*nfeet
        feet_air_durations_secs = feet_state[:,self.FEET_FIELDS.FEET_AIR_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        steps_finishing = feet_air_durations_secs < 0
        feet_air_durations_secs = -feet_air_durations_secs # When the value is positive then it is the duration of a currently ongoing step, which we dont consider
        # subtracting 0.1 from the durations makes it so that very short steps are actually penalized with a negative reward
        # this makes doing small steps worse than doing nothing
        # squash the durations to max_good step, and offset it o that steps shorter than min_good_step are negative
        corrected_air_durations = th.tanh((feet_air_durations_secs - self._loco_conf.min_good_step_air_duration)/self._loco_conf.max_good_step_duration)*self._loco_conf.max_good_step_duration
        # only keep the reward for finishing steps, and use reward quadratic in the duration (so two small steps are worse than one long one).
        # But keep the sign of the reward
        # Also, add add a linear term to keep a good gradient at min_good_step_duration
        feet_air_rewards = steps_finishing*(th.sign(corrected_air_durations)*(corrected_air_durations)**2+corrected_air_durations)
        # step_is_good = (feet_air_rewards>0)
        step_air_time_is_bad  = (feet_air_rewards<=0)
        # feet_rewards = step_is_good*squashed_feet_rewards + step_is_bad*feet_rewards # only squash the positive rewards
        feet_air_rewards = feet_air_rewards*(th.logical_or(should_be_moving, step_air_time_is_bad)) # Only enable if speed is > 0.05 or the reward is a small step penalty
        reward_feet_air_time = th.mean(feet_air_rewards, dim=1) # average across the feet

        # FEET GROUND TIME REWARD
        feet_ground_durations_secs = feet_state[:,self.FEET_FIELDS.FEET_GROUND_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        steps_starting = feet_ground_durations_secs < 0
        feet_ground_durations = -feet_ground_durations_secs # When the value is positive then it is the duration of a currently ongoing step, which we dont consider
        corrected_ground_durations = th.tanh((feet_ground_durations - self._loco_conf.min_good_ground_duration)/self._loco_conf.max_good_ground_duration)*self._loco_conf.max_good_ground_duration
        feet_ground_rewards = steps_starting*(th.sign(corrected_ground_durations)*(corrected_ground_durations)**2+corrected_ground_durations)
        step_ground_time_is_bad  = (feet_ground_rewards<=0)
        feet_ground_rewards = feet_ground_rewards*(th.logical_or(should_be_moving, step_ground_time_is_bad))
        reward_feet_ground_time = th.mean(feet_ground_rewards, dim=1) # average across the feet

        # FEET AIR UNIFORMITY REWARD
        avg_feet_step_durations = feet_state[:,self.FEET_FIELDS.AVG_FEET_STEP_DURATIONS] # vec_size*fields*nfeet -> vec_size*nfeet
        avg_all_feet_step_duration = th.mean(avg_feet_step_durations, dim=1) # vec_size*nfeet -> vec_size
        step_duration_difformity = steps_finishing*(feet_air_durations_secs-avg_all_feet_step_duration.unsqueeze(1).expand_as(feet_air_durations_secs)) # vec_size*nfeet
        reward_feet_air_time_uniformity = joint_penalty_reward(step_duration_difformity, max_rew=1, exponent=2)
        
        # FEET SLIP REWARD
        feet_linvels_xy = feet_state[:,1:3] # vec_size*fields*nfeet -> vec_size*2*nfeet
        feet_linvels = th.linalg.norm(feet_linvels_xy, dim=1) # vec_size*fields*nfeet -> vec_size*nfeet
        feet_touching_ground = feet_state[:,self.FEET_FIELDS.FEET_AIR_DURATIONS] <= 0
        feet_sliding_linvel = feet_linvels*feet_touching_ground
        reward_slip = joint_penalty_reward(feet_sliding_linvel, max_rew=1, exponent=2)

        # FEET ON GROUND REWARD
        feet_touching_ground = feet_state[:,0] <= 0
        reward_feet_on_ground = th.mean(feet_touching_ground.to(th.float32), dim=1)

        # FAILURE REWARD
        failed = (curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] < 0)

        if self._configuration.fail_on_safety:
            failed = th.logical_or(failed, state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED,0])

        sub_rewards_return["acceleration"] = reward_acceleration
        sub_rewards_return["actacc"] = reward_actacc
        sub_rewards_return["actdiff"] = reward_actdiff
        sub_rewards_return["contacts"] = reward_contacts
        sub_rewards_return["feet_air_time"] = reward_feet_air_time
        sub_rewards_return["feet_ground_time"] = reward_feet_ground_time
        # sub_rewards_return["feet_air_time_uniformity"] = reward_feet_air_time_uniformity
        sub_rewards_return["feet_on_ground"] = reward_feet_on_ground
        sub_rewards_return["heading"] = reward_heading_position
        sub_rewards_return["heading_velocity"] = reward_heading_velocity
        sub_rewards_return["health"] = th.ones((current_state_locom_vec.size()[0],), device=current_state_locom_vec.device)
        sub_rewards_return["height_position"] = reward_height_position
        sub_rewards_return["height_velocity"] = reward_height_velocity
        sub_rewards_return["pitchnroll"] = reward_pitchnroll
        sub_rewards_return["pitchnroll_velocity"] = reward_pitchnroll_velocity
        sub_rewards_return["pos2posref_diff"] = reward_pos2posref_diff
        sub_rewards_return["position"] = reward_position
        sub_rewards_return["position_limit"] = reward_position_limit
        sub_rewards_return["sensed_effort"] = reward_sensed_effort
        sub_rewards_return["slip"] = reward_slip
        sub_rewards_return["torque"] = reward_cmdtorque
        sub_rewards_return["torque_limit"] = reward_torque_limit
        sub_rewards_return["torque_refs"] = reward_torque_refs
        sub_rewards_return["torquediff"] = reward_torquediff
        sub_rewards_return["tracking"] = reward_velocity_tracking
        sub_rewards_return["velocity"] = reward_velocity
        sub_rewards_return["velocity_limit"] = reward_velocity_limit
        sub_rewards_return["velocity_refs"] = reward_velocity_refs
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards_return.items()}

        # ggLog.info(f"heading_velocity = {reward_heading_velocity}")
        # ggLog.info(f"height_velocity = {reward_height_velocity}")
        # ggLog.info(f"dt = {current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT]}")
        # for k,v in sub_rewards_return.items():
        #     dbg_check_size(v, (self._adapter.vec_size(),), f"Unexpected size for sub_reward {k}")
        # dbg_check_finite(sub_rewards_return)
        
        weights = { "tracking" :           current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT],
                    "torque" :             current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT],
                    "torque_limit" :       current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "torquediff" :         current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT],
                    "velocity" :           current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT],
                    "velocity_limit" :     current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT],
                    "acceleration" :       current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT],
                    "position_limit" :     current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT],
                    "health" :             current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT],
                    "contacts" :           current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT],
                    "height_velocity" :    current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEIGHT_VELOCITY_WEIGHT],
                    "height_position" :    current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEIGHT_POSITION_WEIGHT],
                    "pitchnroll" :         current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT],
                    "pitchnroll_velocity": current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_VELOCITY_WEIGHT],
                    "actdiff" :            current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT],
                    "position" :           current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT],
                    "heading" :            current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEADING_WEIGHT],
                    "failure" :            current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_FAILURE_WEIGHT],
                    "actacc" :             current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_ACTACC_WEIGHT],
                    "feet_air_time" :      current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_FEET_AIR_TIME_WEIGHT],
                    "feet_ground_time" :   current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_FEET_GROUND_TIME_WEIGHT],
                    # "feet_air_time_uniformity" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_FEET_AIR_TIME_UNIFORMITY_WEIGHT],
                    "feet_on_ground" :     current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_FEET_ON_GROUND_WEIGHT],
                    "velocity_refs" :      current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_VELREF_WEIGHT],
                    "sensed_effort" :      current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_SENSED_EFFORT_WEIGHT],
                    "torque_refs" :        current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUEREF_WEIGHT],
                    "pos2posref_diff" :    current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_POS2POSREF_WEIGHT],
                    "slip" :               current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_SLIP_WEIGHT],
                    "heading_velocity":    current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEADING_VELOCITY_WEIGHT]}
        for k in sub_rewards_return:
            sub_rewards_return[k] = self._loco_conf.reward_scale*sub_rewards_return[k]*weights[k]
        scaled_rewards_vec = th.stack(list(sub_rewards_return.values()), dim = 1)
        sub_rewards_return["failure"] = -th.sum(scaled_rewards_vec*(scaled_rewards_vec>0), dim =1)*failed # negate all the positive rewards
        sub_rewards_return = {k:v.view(self._adapter.vec_size(),) for k,v in sub_rewards_return.items()}
        sub_rewards_unscaled = {k:v.view(self._adapter.vec_size(),) for k,v in sub_rewards_unscaled.items()}
        reward = th.sum(th.stack(list(sub_rewards_return.values()), dim = 1), dim =1)
        reward = th.clamp(reward, -self._configuration.reward_clamp, self._configuration.reward_clamp)
        # if dbg_info is not None:
        #     sub_rewards_scaled = {f"{k}_scaled":v for k,v in sub_rewards_return.items()}
        #     sub_rewards_scaled_agg = th.stack([sub_rewards_scaled[k] for k in sub_rewards_scaled.keys()])
        #     sub_rewards_scaled_agg_names = to_string_tensor([k for k in sub_rewards_scaled.keys()])
        #     sub_rewards_unscaled_agg = th.stack([sub_rewards_unscaled[k] for k in sub_rewards_unscaled.keys()])
        #     sub_rewards_unscaled_agg_names = sub_rewards_scaled_agg_names
        #     dbg_info["sub_rewards_unscaled"] = sub_rewards_unscaled_agg
        #     dbg_info["sub_rewards_unscaled_labels"] = sub_rewards_unscaled_agg_names
        #     dbg_info["sub_rewards_scaled"] = sub_rewards_scaled_agg
        #     dbg_info["sub_rewards_scaled_labels"] = sub_rewards_scaled_agg_names
        #     dbg_info.update({k:r.cpu().item() if isinstance(r,th.Tensor) else r for k,r in sub_rewards_return.items()})
        #     dbg_info["reward"] = reward
        # ggLog.info(f"sub_rewards_return = {sub_rewards_return}")
        # ggLog.info(f"sub_rewards_unscaled = {sub_rewards_unscaled}")
        # ggLog.info(f"reward_torque = {reward_torque}, normtorques = {normtorques}")
        # ggLog.info(f"torques = {state[self.STATE_ROBOT][:,0,:,2]}")
        dbg_check_size(reward, (self._adapter.vec_size(),), f"Unexpected reward size")
        dbg_check_finite(sub_rewards_return, async_assert=True, assert_msg="Nonfinite sub rewards detected")
        # dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(sub_rewards_return, -100, 100),
        #           lambda: f"{adarl.utils.tensor_trees.flatten_tensor_tree(map_tensor_tree(sub_rewards_return, lambda t: adarl.utils.tensor_trees.is_leaf_bounded(t,min=-100,max=100)))}",
        #           just_warn=True)
        # dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(reward, -100, 100),
        #           lambda: f"Reward over 100. sub_rewards = {map_tensor_tree(sub_rewards_return,lambda t: 'minmax='+str((th.min(t).cpu().item(), th.max(t).cpu().item())))}",
        #           just_warn=True)
        return reward
    










    def _update_stats(self):
        super()._update_stats()

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
        masked_assign(self._stats["body_speeds_vec"],     starting_eps, vel_error_vec.unsqueeze(1).expand(-1, self._buff_sizes))
        # Update the buffers
        # idxs = step_counts%self._buff_sizes
        idx : th.Tensor = self._th_tot_step_counter.view(tuple())%self._stats["vel_errs_vec"].size()[1]
        set_column(self._stats["vel_errs_vec"], idx, vel_error_vec.view(self.num_envs,))
        set_column(self._stats["height_errs_vec"], idx, height_error_vec.view(self.num_envs,))
        set_column(self._stats["pitchnroll_errs_vec"], idx, pitchnroll_err_vec.view(self.num_envs,))
        set_column(self._stats["body_speeds_vec"], idx, body_speed_vec.view(self.num_envs,))

        state_stats_v_h_j_minmaxavgstd_pvaee : th.Tensor = self._current_state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 5)
        masked_assign(self._stats["ep_max_javg_sensed_effort"], starting_eps, 0)
        masked_assign(self._stats["ep_max_peak_sensed_effort"], starting_eps, 0)
        self._stats["ep_max_javg_sensed_effort"] = th.maximum(self._stats["ep_max_javg_sensed_effort"], state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,2,4].mean(dim=1)).view((self.num_envs,)) 
        self._stats["ep_max_peak_sensed_effort"] = th.maximum(self._stats["ep_max_peak_sensed_effort"], state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,0:2,4].abs().amax(dim=[1,2])).view((self.num_envs,))

   
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        i = super().get_infos(state=state, labels=labels)
        curr_locom_state = state[self.STATE_LOCOMOTION][:,0]
        curr_extri_state = state[self.STATE_EXTRINSIC][:,0]
        prev_extri_state = state[self.STATE_EXTRINSIC][:,1]
        curr_inter_state = state[self.STATE_INTERNAL][:,0]
        
        goal_vel_rel_dir_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_X,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Y,
                                                                            self.LOCOMOTION_FIELDS.GOAL_LINVEL_REL_DIRECTION_Z)) #type:ignore
        goal_vel_abs_xyz_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z)) #type:ignore 
        smooth_track_err_idx = self._locomotion_state_helper.field_idx((self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR,)) #type: ignore
        body_linvel_abs_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z)) #type: ignore
        body_linvel_rel_xyz_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((  self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z)) #type: ignore
        goal_dir = curr_locom_state[:,goal_vel_rel_dir_xyz_idx].view(self.num_envs,3)
        goal_speed = curr_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED].view(self.num_envs,1)
        i["goal_rel_xyz_vec"] = goal_dir*goal_speed
        i["goal_height"] = curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_GOAL_BODY_HEIGHT]
        i["height"] = curr_extri_state[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]
        i["height_err"] = th.abs(i["goal_height"] - i["height"])
        goal_abs = curr_locom_state[:,goal_vel_abs_xyz_idx]
        i["goal_abs_speed_vec"] = th.linalg.norm(goal_abs,dim=1)
        i["goal_abs_yaw_vec"] = th.atan2(goal_abs[:,1],goal_abs[:,0])
        i["goal_abs_xyz_vec"] = goal_abs
        i["smoothed_linvel_error"] = curr_locom_state[:,smooth_track_err_idx].view(self.num_envs)
        i["body_abs_linvel"] = curr_extri_state[:,body_linvel_abs_xyz_idx]
        i["body_rel_linvel"] = curr_extri_state[:,body_linvel_rel_xyz_idx]
        i["linvel_error"] = i["goal_abs_xyz_vec"] - i["body_abs_linvel"]
        i["ep_avg_vel_err_vec"] = self._stats["ep_avg_vel_err_vec"]
        i["ep_avg_height_err_vec"] = self._stats["ep_avg_height_err_vec"]
        i["ep_avg_pitchnroll_err_vec"] = self._stats["ep_avg_pitchnroll_err_vec"]
        i["ep_avg_bodyspeed_vec"] = self._stats["ep_avg_bodyspeed_vec"]
        i["avg10_vel_errs_vec"] = th.mean(self._stats["vel_errs_vec"], dim = 1).view(self.num_envs)
        i["avg10_height_errs_vec"] = th.mean(self._stats["height_errs_vec"], dim = 1).view(self.num_envs)
        i["avg10_pitchnroll_errs_vec"] = th.mean(self._stats["pitchnroll_errs_vec"], dim = 1).view(self.num_envs)
        i["avg10_body_speeds_vec"] = th.mean(self._stats["body_speeds_vec"], dim = 1).view(self.num_envs)
        i["success_vec"] = i["avg10_vel_errs_vec"] < 0.05
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)
        i["joint_pos_safenorm"] = state_robot_safenorm[:,0,:,0]
        state_stats_v_h_j_minmaxavgstd_pvaee : th.Tensor = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 5)
        i["avg_sensed_effort"] = state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,2,4].mean(dim=1)
        i["avg_peak_sensed_effort"] = state_stats_v_h_j_minmaxavgstd_pvaee[:,0,:,0:2,4].abs().amax(dim=2).amax(dim=1)

        
        _, i["height_velocity"], i["goal_height_velocity"], last_dt, i["height_err_raw"] = self._height_velocity_reward(curr_state_extr_vec = curr_extri_state,
                                                                                                       current_state_locom_vec = curr_locom_state,
                                                                                                       current_state_internal = curr_inter_state,
                                                                                                       prev_state_extr_vec = prev_extri_state)
        _, i["heading_vel"], i["goal_heading_vel"], i["heading_vel_err"], i["heading_err"] = self._heading_velocity_reward(state=state, dt=last_dt)
        _, i["pitchnroll_velocity"], i["goal_pitchnroll_velocity"], i["pitchnroll_vel_err"], i["pitchnroll_err"] = self._pitchnroll_velocity_reward(state=state,
                                                                                                                                                    dt=last_dt)

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


        return i
    
    def _sample_goals(self):
        goal_speeds = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_speed_minmax[0],
                                    max=self._loco_conf.goal_speed_minmax[1])
        goal_yaws = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_abs_yaw_minmax[0],
                                    max=self._loco_conf.goal_abs_yaw_minmax[1])
        goal_height = self._thrand(size=(self.num_envs,))*(self._loco_conf.goal_height_minmax[1]-self._loco_conf.goal_height_minmax[0])+self._loco_conf.goal_height_minmax[0]
        goal_abs_linvel_vec_xys = th.stack([  th.cos(goal_yaws),
                                                th.sin(goal_yaws),
                                                goal_speeds],
                                             dim=1)
        goal_heading_yaws = unnormalize(self._thrand(size=(self.num_envs,))*2-1,
                                    min=self._loco_conf.goal_heading_rel_yaw_minmax[0],
                                    max=self._loco_conf.goal_heading_rel_yaw_minmax[1])
        return goal_abs_linvel_vec_xys, goal_height, goal_heading_yaws

    @override
    @th.compile(mode="max-autotune-no-cudagraphs", disable=disable_compile)
    def pre_step(self):
        super().pre_step()
        if self._loco_conf.goal_resampling_enabled>0:
            resample_prob_per_env_dt = 1-th.pow(1-self._loco_conf.goal_resampling_probability_per_sec, self._intendedStepLength_sec)
            vec_mask = self._thrand((self.num_envs,)) < resample_prob_per_env_dt
            goal_abs_linvel_vec_xys, goal_height, goal_heading_yaws = self._sample_goals()
            self.set_goal(goal_abs_linvel_vec_xys, 
                      goal_abs_height=goal_height,
                      vec_mask=vec_mask,
                      goal_heading_yaw=goal_heading_yaws)


    @override
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        goal_abs_linvel_vec_xys, goal_height, goal_heading_yaws = self._sample_goals()
        if "goal_velocity_xy" in reset_options:
            goal_velocity_vec_xy = th.as_tensor(reset_options["goal_velocity_xy"],device=self._configuration.th_device).view(self.num_envs,2)
            goal_speeds = th.linalg.norm(goal_velocity_vec_xy, dim=-1)
            goal_yaws = th.atan2(goal_velocity_vec_xy[:,1], goal_velocity_vec_xy[:,0])
            goal_abs_linvel_vec_xys = th.stack([  th.cos(goal_yaws),
                                                    th.sin(goal_yaws),
                                                    goal_speeds],
                                                dim=1)
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
        self.set_max_episode_steps(reset_options.get("reset_options",self._current_episode_config.vec_max_ep_steps))
        self.set_goal(goal_abs_linvel_vec_xys, 
                      goal_abs_height=goal_height,
                      vec_mask=vec_mask,
                      goal_heading_yaw=goal_heading_yaws)

    def set_goal(self,  goal_abs_linvel_vec_xys : Sequence[tuple[float,float,float]] | tuple[float,float,float] | th.Tensor | None = None,
                        goal_diff_linvel_speed_yaw : tuple[float,float] | th.Tensor | None = None,
                        goal_rel_linvel_xys : tuple[float,float,float] | th.Tensor | None = None,
                        goal_abs_height : float | th.Tensor | None = None,
                        goal_heading_yaw : float | th.Tensor | None = None, 
                        vec_mask : th.Tensor | None = None):
        if vec_mask is None:
            vec_mask = self._all_vecs
        if goal_abs_linvel_vec_xys is not None:
            goal_abs_linvel_vec_xys = self._thtens(goal_abs_linvel_vec_xys).expand(self.num_envs,3)
            masked_assign(self._locomotion_episode_config.goal_abs_vel_vec_xys,
                          vec_mask,
                          goal_abs_linvel_vec_xys)
        elif goal_diff_linvel_speed_yaw is not None:
            if isinstance(goal_diff_linvel_speed_yaw, Sequence):
                goal_diff_linvel_speed_yaw = self._thtens(goal_diff_linvel_speed_yaw)
            elif not isinstance(goal_diff_linvel_speed_yaw, th.Tensor):
                raise RuntimeError(f"Unexpected type {type(goal_diff_linvel_speed_yaw)} for goal_velocity_diff_speed_yaw")
            curr_goal_speeds : th.Tensor = self._locomotion_episode_config.goal_abs_vel_vec_xys[:,2]
            curr_goal_yaws = th.atan2(self._locomotion_episode_config.goal_abs_vel_vec_xys[:,1],
                           self._locomotion_episode_config.goal_abs_vel_vec_xys[:,0])
            new_goal_speeds = curr_goal_speeds + goal_diff_linvel_speed_yaw[0]
            new_goal_yaws = curr_goal_yaws + goal_diff_linvel_speed_yaw[1]
            new_goal_dirs = th.stack([th.cos(new_goal_yaws), th.sin(new_goal_yaws)], dim = 1)    
            new_goals_xys = th.cat([new_goal_dirs, new_goal_speeds.unsqueeze(1)], dim = 1)
            masked_assign(self._locomotion_episode_config.goal_abs_vel_vec_xys,
                          vec_mask,
                          new_goals_xys)
        else:
            self._locomotion_episode_config.goal_rel_vel_vec_xy_speed = self._thtens(goal_rel_linvel_xys).view(self.num_envs,3)
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


    def get_goals(self):
        return {"abs_linvel_xys" : self._locomotion_episode_config.goal_abs_vel_vec_xys,
                "rel_linvel_xys" : self._locomotion_episode_config.goal_rel_vel_vec_xy_speed,
                "abs_gravity" : self._locomotion_episode_config.goal_abs_gravity_vec_xyz,
                "abs_height" : self._locomotion_episode_config.goal_abs_height_vec_z,
                "heading_rel" : self._locomotion_episode_config.goal_heading_rel_vec_yaw}

    @override
    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune")
    def are_states_terminal(self, states) -> th.Tensor:
        r = super().are_states_terminal(states)
        return th.logical_and(r, states[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.CRASHED,0]).view((self.num_envs,))

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        super()._initialize_episodes(vec_mask=vec_mask, options=options)
        if self._loco_conf.use_contacts:
            raise NotImplementedError("Contacts not implemented yet")
            self._adapter.monitor_contacts([(self._configuration.robot_name, None)])

    def _set_arrow_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            goals_corrected = self._locomotion_episode_config.goal_abs_vel_vec_xys.detach().clone()
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
            self._adapter.setLinksStateDirect(link_names=[self._arrow_base],
                                                link_states_pose_vel=th.cat([goalvel_arrow_pose, self._thzeros((goalvel_arrow_pose.size()[0],6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)
            heading_arrow_pose = bstates_vec_13[:,:7].clone()
            heading_rpy = self._thzeros((self._adapter.vec_size(),3))
            heading_rpy[:,2] = self._locomotion_episode_config.goal_heading_rel_vec_yaw.view(self.num_envs,)
            rel_heading_quat = ros_rpy_to_quaternion_xyzw_th(heading_rpy)
            heading_arrow_pose[:,3:7] = quat_mul_xyzw(rel_heading_quat, linvel_dir_quat)
            heading_arrow_pose[:,2] = 0.0 # Arrow is always on the ground
            # heading_arrow_pose[1:] = heading_arrow_pose[0]
            self._adapter.setLinksStateDirect(link_names=[self._arrow_yellow],
                                                link_states_pose_vel=th.cat([heading_arrow_pose, self._thzeros((heading_arrow_pose.size()[0],6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)

    @override
    def _simulation_initialization(self, vec_mask : th.Tensor):
        super()._simulation_initialization(vec_mask = vec_mask)
        if self._configuration.show_goal:
            self._set_arrow_pose(vec_mask=self._all_vecs)
            
    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._set_arrow_pose(vec_mask=self._all_vecs)
        return super().get_ui_renderings(vec_mask=vec_mask)