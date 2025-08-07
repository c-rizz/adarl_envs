from __future__ import annotations
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.utils.utils import (LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, vector_projection, isinstance_noimport, 
                               quat_xyzw_between_vecs_py, masked_assign, quat_mul_xyzw, quat_angle_xyzw, ros_rpy_to_quaternion_xyzw_th)
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check, dbg_run
import adarl.utils.utils
from adarl.utils.vec_state_helper import ThBoxStateHelper, unnormalize, normalize
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
import torch as th
import math
import quaternion
from adarl_envs.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
import adarl.utils.tensor_trees
import traceback
from adarl.utils.spaces import get_space_labels
import pprint
import dataclasses
from pathlib import Path

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
def ramp_reward(error : th.Tensor, zero_rew_dist : th.Tensor):
    return 1-error/zero_rew_dist

class GraspVecEnv(RobotVecEnv):
    STATE_GRASPING = "grasp"
    STATE_CAMERA = "camera"

    @dataclass
    class GraspingConfiguration:
        reward_scale : th.Tensor
        target_object_link : tuple[str,str]
        gripper_link : tuple[str,str]
        observe_object_pose : bool
        camera_resolution_xy : tuple[int,int]
        init_obj_area_minmax_xyz : th.Tensor
        goal_obj_area_minmax_xyz : th.Tensor
        table_link : tuple[str,str]
        manipulator_links : list[tuple[str,str]]

    @dataclass
    class RewardConfiguration:
        acceleration : th.Tensor
        health : th.Tensor
        actdiff : th.Tensor
        actacc : th.Tensor
        position_limit : th.Tensor
        position : th.Tensor
        torque_limit : th.Tensor
        torque : th.Tensor
        torquediff : th.Tensor
        velocity_limit : th.Tensor
        velocity : th.Tensor
        failure : th.Tensor



    @dataclass
    class EpisodeGraspingConfiguration:
        initial_object_pose : th.Tensor
        goal_object_pose : th.Tensor

    GRASPING_FIELDS = IntEnum("GRASPING_FIELDS",   ["GOAL_POSE",
                                                    "OBJECT_POSE",
                                                    "GRIPPER_POSE"], start=0)

    CAMERA_FIELDS = IntEnum("CAMERA_FIELDS",   ["IMAGE"], start=0)
    

    def __init__(self,  action_delay_mustd : tuple[float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseVecJointImpedanceAdapter,
                        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor],
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | JOINT_FILTERS],
                        free_joints : Sequence[str],
                        held_joints_stiffness : float,
                        held_joints_damping : float,
                        frame_stack_length : int,
                        goal_err_smoothing_halflife_sec : float,
                        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                        homing_joint_pose : dict[tuple[str,str], float],
                        maxStepsPerEpisode : int,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        observe_body_velocity : bool,
                        reward_acceleration_weight : float,
                        reward_actdiff_weight : float,
                        reward_actacc_weight : float,
                        reward_health_weight : float,
                        reward_position_limit_weight : float,
                        reward_position_weight : float,
                        reward_scale : float,
                        reward_torque_limit_weight : float,
                        reward_torque_weight : float,
                        reward_torquediff_weight : float,
                        reward_velocity_limit_weight : float,
                        reward_velocity_weight : float,
                        robot_main_body_link : str,
                        robot_name : str,
                        robot_root_link : str,
                        robot_urdf_string : str,
                        safe_damping : float,
                        safe_stiffness : float,
                        safety_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float], 
                        safe_limits_position_offset : dict[tuple[str,str], float],
                        seed : int,
                        stepLength_sec : float,
                        step_precision_tolerance : float,
                        stop_on_failure : bool,
                        fail_on_safety : bool,
                        th_device : th.device,
                        verbose_infos : bool,
                        quiet : bool,
                        enable_dbg_checks : bool,
                        initial_pose_randomization_range : float,
                        init_on_reset_ratio : float,
                        obs_noise_joints_pve_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_linvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_angvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_posz_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_gravity_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        ground_link : tuple[str,str],
                        target_object_link : tuple[str,str],
                        gripper_link : tuple[str,str],
                        manipulator_links : list[tuple[str,str]],
                        ui_camera_resolution_hw : tuple[int,int] = (256,144),
                        enable_link_collisions : list[tuple[tuple[str,str],list[tuple[str,str]]]] | None = [],
                        mass_randomized_links : list[tuple[str,str]] = [],
                        mass_randomization_ratio : float = 0.1,
                        friction_randomized_links : list[tuple[str,str]] = [],
                        friction_slide_spin_roll_randomization_ratios : tuple[float, float, float] = (0.1,0.1,0.1),
                        impulse_probability_per_sec : float = 0.0,
                        impulse_duration_minmax : tuple[float,float ]= (0.01, 5.0),
                        impulse_mean_std : tuple[float,float ]= (50.0, 50.0),
                        longterm_states_decimation_time : float = 0.0001,
                        observe_object_pose : bool = False
                        ):
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._unit_3d_vector = self._thtens([1.0, 0.0, 0.0])
        self._unit_quaternion = self._thtens([0.0, 0.0, 0.0, 1.0])
        self._zero = self._thtens([0.0])
        self._grasping_conf = GraspVecEnv.GraspingConfiguration(
                        reward_scale = self._thtens(reward_scale),
                        target_object_link=target_object_link,
                        gripper_link=gripper_link,
                        observe_object_pose=observe_object_pose,
                        camera_resolution_xy = (64,64),
                        init_obj_area_minmax_xyz = th.as_tensor([[-0.5, -0.5, 0.5], [0.5, 0.5, 0.5]], device=th_device),
                        goal_obj_area_minmax_xyz = th.as_tensor([[-0.5, -0.5, 0.5], [0.5, 0.5, 1.0]], device=th_device),
                        table_link = ground_link,
                        manipulator_links = manipulator_links
                        )
        self._sub_reward_weights = GraspVecEnv.RewardConfiguration(
                        acceleration = self._thtens(reward_acceleration_weight),
                        actacc = self._thtens(reward_actacc_weight),
                        actdiff = self._thtens(reward_actdiff_weight),
                        failure = self._thtens(1.0),
                        health = self._thtens(reward_health_weight),
                        position = self._thtens(reward_position_weight),
                        position_limit  = self._thtens(reward_position_limit_weight) ,
                        torque = self._thtens(reward_torque_weight),
                        torque_limit  = self._thtens(reward_torque_limit_weight) ,
                        torquediff = self._thtens(reward_torquediff_weight),
                        velocity = self._thtens(reward_velocity_weight),
                        velocity_limit = self._thtens(reward_velocity_limit_weight)
                        )
        self._reward_weights = self._thtens([v for v in dataclasses.asdict(self._sub_reward_weights).values()])
        
        self._grasping_episode_config = GraspVecEnv.EpisodeGraspingConfiguration(initial_object_pose = self._thzeros((adapter.vec_size(), 7)),
                                                                                   goal_object_pose = self._thzeros((adapter.vec_size(), 7)))
        if enable_link_collisions is None:
            enable_link_collisions = []
        enable_link_collisions.append((self._grasping_conf.target_object_link, [self._grasping_conf.table_link]+self._grasping_conf.manipulator_links))
        super().__init__(   action_delay_mustd = action_delay_mustd,
                            action_noise_mustd = action_noise_mustd, 
                            action_smoothing_halflife_sec = action_smoothing_halflife_sec,
                            adapter = adapter,
                            control_mode = control_mode,
                            controlled_joints = controlled_joints,
                            goal_err_smoothing_halflife_sec = goal_err_smoothing_halflife_sec,
                            maxStepsPerEpisode = maxStepsPerEpisode,
                            minmax_damping = minmax_damping,
                            minmax_stiffness = minmax_stiffness,
                            robot_main_body_link = robot_main_body_link,
                            robot_name = robot_name,
                            robot_root_link = robot_root_link,
                            robot_urdf_string = robot_urdf_string,
                            safe_damping = safe_damping,
                            safe_stiffness = safe_stiffness,
                            safety_limits_ratios_minmax_pve = safety_limits_ratios_minmax_pve,
                            safe_limits_position_offset = safe_limits_position_offset,
                            seed = seed,
                            stepLength_sec = stepLength_sec,
                            step_precision_tolerance = step_precision_tolerance,
                            stop_on_failure = stop_on_failure,
                            th_device = th_device,
                            homing_body_pose_xyz_xyzw = homing_body_pose_xyz_xyzw,
                            homing_joint_pose = homing_joint_pose,
                            control_limits_minmax_pve = control_limits_minmax_pve,
                            observe_body_velocity = observe_body_velocity,
                            frame_stack_length=frame_stack_length,
                            verbose_infos = verbose_infos,
                            quiet = quiet,
                            enable_dbg_checks = enable_dbg_checks,
                            initial_joint_pose_randomization_range = initial_pose_randomization_range,
                            init_on_reset_ratio = init_on_reset_ratio,
                            obs_noise_joints_pve_ep_mustd_step_std = obs_noise_joints_pve_ep_mustd_step_std,
                            obs_noise_linvel_ep_mustd_step_std = obs_noise_linvel_ep_mustd_step_std,
                            obs_noise_angvel_ep_mustd_step_std = obs_noise_angvel_ep_mustd_step_std,
                            obs_noise_posz_ep_mustd_step_std = obs_noise_posz_ep_mustd_step_std,
                            obs_noise_gravity_ep_mustd_step_std = obs_noise_gravity_ep_mustd_step_std,
                            ui_camera_resolution_hw = ui_camera_resolution_hw,
                            enable_link_collisions = enable_link_collisions,
                            randomized_mass_links=mass_randomized_links,
                            randomized_mass_ratios=mass_randomization_ratio,
                            randomized_friction_links=friction_randomized_links,
                            randomized_friction_slide_spin_roll_ratios=friction_slide_spin_roll_randomization_ratios,
                            ground_link=ground_link,
                            impulse_probability_per_sec = impulse_probability_per_sec,
                            impulse_duration_minmax = impulse_duration_minmax,
                            impulse_mean_std = impulse_mean_std,
                            fail_on_safety = fail_on_safety,
                            longterm_states_decimation_time = longterm_states_decimation_time,
                            free_joints=free_joints,
                            held_joints_stiffness = held_joints_stiffness,
                            held_joints_damping = held_joints_damping
                        )

        
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__
        obs_labels = self._state_helper.observation_names()
        ggLog.info(f"Obs labels = \n{pprint.pformat(obs_labels)}")
        ggLog.info(f"Env constructed")

    @override
    def _build_stats(self):
        self._stats = {}
        self._buff_sizes = int(self._configuration.original_max_epsteps/10)
        self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))

    @override
    def _build(self):
        super()._build()
        self._object_link_id = self._adapter.get_links_ids([self._grasping_conf.target_object_link])
        self._gripper_link_id = self._adapter.get_links_ids([self._grasping_conf.gripper_link])



    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        if self._grasping_conf.observe_object_pose:
            observable_fields=[ self.GRASPING_FIELDS.GOAL_POSE,
                                self.GRASPING_FIELDS.OBJECT_POSE
                                ]
        else:
            observable_fields=[ self.GRASPING_FIELDS.GOAL_POSE]
        grasping_state_helper = ThBoxStateHelper( field_names=[e for e in self.GRASPING_FIELDS],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(7,),
                                                    fields_minmax={ self.GRASPING_FIELDS.GOAL_POSE :    [-10, 10],
                                                                    self.GRASPING_FIELDS.OBJECT_POSE :  [-10, 10],
                                                                    self.GRASPING_FIELDS.GRIPPER_POSE : [-10, 10]},
                                                    observable_fields=observable_fields, #type: ignore
                                                    vec_size=adapter.vec_size(),
                                                    history_length=2,
                                                    obs_history_length=1)
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_GRASPING,
                                                            grasping_state_helper,
                                                            observable = True,
                                                            flatten_obs = True)
        camera_state_helper = ThBoxStateHelper( field_names=[e for e in self.CAMERA_FIELDS],
                                                dtype=self._obs_dtype,
                                                th_device=self._th_device,
                                                field_size=self._grasping_conf.camera_resolution_xy,
                                                fields_minmax={ self.CAMERA_FIELDS.IMAGE : [-1,1]},
                                                observable_fields=observable_fields, #type: ignore
                                                vec_size=adapter.vec_size())
        # if not self._grasping_conf.observe_object_pose:
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_CAMERA,
                                                            camera_state_helper,
                                                            observable = not self._grasping_conf.observe_object_pose,
                                                            flatten_obs = False)
        ggLog.info(f"Built state/obs/action helpers")
        



    @override
    def _get_new_instantaneous_state(self):

        new_inst_state = super()._get_new_instantaneous_state()

        current_object_pose = self._adapter.getLinksState(self._object_link_id, use_com_pose=False)[:,0,:7]
        current_gripper_pose = self._adapter.getLinksState(self._gripper_link_id, use_com_pose=False)[:,0,:7]
        # ggLog.info(f"current_object_pose = {current_object_pose}")
        # ggLog.info(f"current_gripper_pose = {current_gripper_pose}")
        new_grasping_state = {self.GRASPING_FIELDS.GOAL_POSE   : self._grasping_episode_config.goal_object_pose.expand(self.num_envs,7),
                              self.GRASPING_FIELDS.OBJECT_POSE : current_object_pose.expand(self.num_envs,7),
                              self.GRASPING_FIELDS.GRIPPER_POSE : current_gripper_pose.expand(self.num_envs,7)}
        new_inst_state[self.STATE_GRASPING] = new_grasping_state
        
        # if not self._grasping_conf.observe_object_pose:
        new_camera_state = self._thzeros((self.num_envs,
                                        1,
                                        self._grasping_conf.camera_resolution_xy[0],
                                        self._grasping_conf.camera_resolution_xy[1]))
        new_inst_state[self.STATE_CAMERA] = new_camera_state
        return new_inst_state
    









    @staticmethod
    @th.jit.script
    def _flattened_penalty_reward(x, max_rew, exponent : float, flattening_scale : float):
        """A penalty produced by raising abs(x) at the power of exponent, and flattening it with
            a flipped exponential, scaled with flattening_scale. With exponent=15 and 
            flattening_scale=0.1 results in an x^1.5 that is quite flat below 0.1.
            This then is squashed with a tanh to be under max_rew.
            In formulas (not squashed): x^exponent * (-e^(-x^2/flattening_scale)+1)
        """
        return th.tanh((th.mean(th.pow(th.abs(x),exponent)*(1-th.exp(-(x/flattening_scale)**2)), dim=1))/max_rew)*max_rew
    
    @staticmethod
    @th.jit.script
    def _penalty_reward(x, max_rew, exponent : float):
        """A penalty produced by raising abs(x) at the power of exponent, and squashing
            it with a tanh to be under max_rew.
        """
        return th.tanh(th.mean(th.pow(th.abs(x),exponent),dim=1)/max_rew)*max_rew

    @override
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:

        max_rew = self._configuration.reward_penalties_max
        state_action_vec = state[self.STATE_ACT_RAW_HIST]
        state_stats = state[self.STATE_JOINT_STEP_STATS]

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_pvesd[:,0], lims[0,:,0], lims[1,:,0])
        state_robot_norm     = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)
        longterm_stats_pos_norm     = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)
        normposhomingdiff   = longterm_stats_pos_norm[:,0,0] - normhoming
        normvelocities      = state_robot_norm[:,0,:,1]
        normtorques         = state_robot_norm[:,0,:,2]
        normaccelerations   = state_stats[:,0,:,10]/1000 # average accelearation, normalized assuming a max of 1000 m/s^2
        normtorquediff      = state_robot_norm[:,0,:,2] - state_robot_norm[:,1,:,2]
        actdiff             = th.flatten((state_action_vec[:,0] - state_action_vec[:,1])/2, start_dim=1)
        prev_actdiff        = th.flatten((state_action_vec[:,1] - state_action_vec[:,2])/2, start_dim=1)
        act_acc             = actdiff - prev_actdiff

        position_safenorm   = state_robot_safenorm[:,0,:,0]
        velocities_safenorm = state_robot_safenorm[:,0,:,1]
        torque_safenorm     = state_robot_safenorm[:,0,:,2]

        reward_torque           = -self._penalty_reward(normtorques,max_rew=max_rew,exponent=2)
        reward_velocity         = -self._penalty_reward(normvelocities,max_rew=max_rew,exponent=2)
        reward_acceleration     = -self._flattened_penalty_reward(normaccelerations,max_rew=max_rew, exponent=1.5, flattening_scale=0.02)
        reward_position         = -self._flattened_penalty_reward(normposhomingdiff,max_rew=max_rew, exponent=0.5, flattening_scale=0.02)
        reward_torquediff       = -self._penalty_reward(normtorquediff,max_rew=max_rew,exponent=2)
        reward_actdiff          = -self._penalty_reward(actdiff,max_rew=max_rew,exponent=2)
        reward_actacc           = -self._flattened_penalty_reward(act_acc,max_rew=max_rew, exponent=0.5,flattening_scale=0.1)
        reward_torque_limit     = -self._penalty_reward(torque_safenorm,max_rew=max_rew,exponent=50)
        reward_position_limit   = -self._penalty_reward(position_safenorm,max_rew=max_rew,exponent=50)
        reward_velocity_limit   = -self._penalty_reward(velocities_safenorm,max_rew=max_rew,exponent=50)

        obj_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.OBJECT_POSE,:3]
        goal_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GOAL_POSE,:3]
        gripper_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GRIPPER_POSE,:3]
        obj2goal_dist = th.linalg.norm(obj_position - goal_position, dim = -1)
        obj2hand_dist = th.linalg.norm(obj_position - gripper_position, dim = -1)

        reward_object_pose = 1-th.tanh(obj2goal_dist/0.5)
        reward_gripper_pose = 1-th.tanh(obj2hand_dist/0.5)

        failed = self._no_vecs
        if self._configuration.fail_on_safety:
            failed = th.logical_or(failed, state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED,0])

        sub_rewards_unscaled_dict : dict[str,th.Tensor] = {
            "acceleration" : reward_acceleration,
            "actacc" : reward_actacc,
            "actdiff" : reward_actdiff,
            "failure" : self._thtens(0.0).expand(self.num_envs),
            "health" : self._thtens(1.0).expand(self.num_envs),
            "position" : reward_position,
            "position_limit" : reward_position_limit,
            "torque" : reward_torque,
            "torque_limit" : reward_torque_limit,
            "torquediff" : reward_torquediff,
            "velocity" : reward_velocity,
            "velocity_limit" : reward_velocity_limit,
            "object_pose" : reward_object_pose,
            "gripper_pose" : reward_gripper_pose
        }
        sub_rewards_unscaled_dict["failure"] = -th.sum(th.stack([rew*(rew>0.0) for rew in sub_rewards_unscaled_dict.values()], dim=1), dim =1)*failed
        
        for k,v in sub_rewards_unscaled_dict.items():
            dbg_check_size(v, (self._adapter.vec_size(),), f"Unexpected size for sub_reward {k}")

        rewards_unscaled = th.stack([sub_rewards_unscaled_dict[rn] for rn in dataclasses.asdict(self._sub_reward_weights).keys()], dim=1)
        rewards_scaled = rewards_unscaled*self._reward_weights*self._grasping_conf.reward_scale
        sub_rewards_return.update({rn:rewards_unscaled[:,i] for i,rn in enumerate(dataclasses.asdict(self._sub_reward_weights).keys())})
        reward = th.sum(rewards_scaled, dim =1)

        dbg_check_size(reward, (self._adapter.vec_size(),), f"Unexpected reward size")
        dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(sub_rewards_return, -100, 100),
                  lambda: f"{adarl.utils.tensor_trees.flatten_tensor_tree(map_tensor_tree(sub_rewards_return, lambda t: adarl.utils.tensor_trees.is_leaf_bounded(t,min=-100,max=100)))}",
                  just_warn=True)
        dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(reward, -100, 100),
                  lambda: f"Reward over 100. sub_rewards = {map_tensor_tree(sub_rewards_return,lambda t: 'minmax='+str((th.min(t).cpu().item(), th.max(t).cpu().item())))}",
                  just_warn=True)
        return reward
    










    def _update_stats(self):
        super()._update_stats()

        self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))

        obj_pose  = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.OBJECT_POSE]
        goal_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GOAL_POSE]
        gripper_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GRIPPER_POSE]
        obj2goal_dist = th.linalg.norm(obj_pose[:,:3]-goal_pose[:,:3], dim = -1)
        obj2hand_dist = th.linalg.norm(obj_pose[:,:3]-gripper_pose[:,:3], dim = -1)
        prev_obj_pose  = self._current_state[self.STATE_GRASPING][:,1,self.GRASPING_FIELDS.OBJECT_POSE]
        obj_travel = th.linalg.norm(obj_pose[:,:3]-prev_obj_pose[:,:3], dim = -1)

        step_counts = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT,0].to(th.long)
        dbg_check_size(step_counts, (self._adapter.vec_size(),))
        
        # Update episode averages
        self._stats["ep_obj2hand_dist"]          = (self._stats["ep_obj2hand_dist"]*(step_counts-1) + obj2hand_dist)/step_counts # Elements with step_count == 0 will be inf
        self._stats["ep_obj2goal_dist"]          = (self._stats["ep_obj2goal_dist"]*(step_counts-1) + obj2goal_dist)/step_counts # Elements with step_count == 0 will be inf
        self._stats["ep_obj_travel"]             = (self._stats["ep_obj_travel"] + obj_travel) # Elements with step_count == 0 will be inf
        # Correct the episode averages for episodes that have just started
        starting_eps = step_counts==0
        masked_assign(self._stats["ep_obj2hand_dist"],      starting_eps, obj2hand_dist)
        masked_assign(self._stats["ep_obj2goal_dist"],      starting_eps, obj2goal_dist)
        masked_assign(self._stats["ep_obj_travel"],         starting_eps, obj_travel)
        # Fill the buffers for episodes that have just staretd
        masked_assign(self._stats["obj2hand_dist"],     step_counts==0, obj2hand_dist.unsqueeze(1).expand(-1, self._buff_sizes))
        masked_assign(self._stats["obj2goal_dist"],     step_counts==0, obj2goal_dist.unsqueeze(1).expand(-1, self._buff_sizes))
        
        # Update the buffers
        # idxs = step_counts%self._buff_sizes
        idxs = step_counts%self._stats["obj2hand_dist"].size()[1]
        # print(f"torch.is_grad_enabled()) = {th.is_grad_enabled()}")
        # print(f"idx.size() = {idxs.size()}, idx = {idxs}")
        # print(f"vel_error_vec.size() = {vel_error_vec.size()}, {vel_error_vec}")
        self._stats["obj2hand_dist"][:,idxs] = obj2hand_dist
        self._stats["obj2goal_dist"][:,idxs] = obj2goal_dist
   
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        i = super().get_infos(state=state, labels=labels)
        
        i["obj2hand_dist"] = self._stats["obj2hand_dist"]
        i["obj2goal_dist"] = self._stats["obj2goal_dist"]
        i["ep_obj2hand_dist"] = self._stats["ep_obj2hand_dist"]
        i["ep_obj2goal_dist"] = self._stats["ep_obj2goal_dist"]
        i["ep_obj_travel"] = self._stats["ep_obj_travel"]
        i["avg10_obj2hand_dist"] = th.mean(self._stats["obj2hand_dist"], dim = 1)
        i["avg10_obj2goal_dist"] = th.mean(self._stats["obj2goal_dist"], dim = 1)
        i["success_vec"] = i["avg10_obj2goal_dist"] < 0.05
        sub_rews = {}
        self.compute_rewards(state, sub_rews)
        i["rewards"] = th.stack(list(sub_rews.values()), dim = 1) 
        # ggLog.info(f"i['rewards'] = {i['rewards'].size()}")
        if labels is not None:
            labels["rewards"] = to_string_tensor(list(sub_rews.keys())) 

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_GRASPING]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())

        return i
    
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        if "goal_pose" in reset_options:
            goal_obj_pose_v_xyzxyzw = th.as_tensor(reset_options["goal_pose"],device=self._configuration.th_device)
        else:
            goal_obj_pose_v_xyzxyzw = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                                self._grasping_conf.goal_obj_area_minmax_xyz[0],
                                                self._grasping_conf.goal_obj_area_minmax_xyz[1])
            goal_obj_quat = ros_rpy_to_quaternion_xyzw_th(self._thrand((self.num_envs,3))*math.pi*2)
            goal_obj_pose_v_xyzxyzw = th.cat([goal_obj_pose_v_xyzxyzw, goal_obj_quat], dim=1).view((self.num_envs,7))
            
        initial_object_position = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                              self._grasping_conf.init_obj_area_minmax_xyz[0],
                                              self._grasping_conf.init_obj_area_minmax_xyz[1])
        initial_object_yaw = self._thrand((self.num_envs,))*math.pi*2
        initial_obj_quat = ros_rpy_to_quaternion_xyzw_th(th.cat([self._thzeros((self.num_envs,2)),initial_object_yaw.unsqueeze(1)], dim=1))
        initial_obj_pose = th.cat([initial_object_position, initial_obj_quat], dim=1).view((self.num_envs,7))
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
        self._grasping_episode_config = GraspVecEnv.EpisodeGraspingConfiguration(   initial_object_pose=initial_obj_pose,
                                                                                    goal_object_pose=goal_obj_pose_v_xyzxyzw)
        self.set_max_episode_steps(reset_options.get("reset_options",self._current_episode_config.vec_max_ep_steps))
        self.set_goals(goal_obj_pose_v_xyzxyzw)

    def set_goals(self, goal_object_pose_vec_xyzxyzw : th.Tensor, reference_framce : Literal["absolute","relative"] = "absolute"):
        if reference_framce == "relative":
            goal_object_pose_vec_xyzxyzw[:,:3] = goal_object_pose_vec_xyzxyzw[:,:3] + self._grasping_episode_config.goal_object_pose[:,:3]
            goal_object_pose_vec_xyzxyzw[:,3:7] = quat_mul_xyzw(goal_object_pose_vec_xyzxyzw[:,3:7], self._grasping_episode_config.goal_object_pose[:,3:7])
        self._grasping_episode_config.goal_object_pose = goal_object_pose_vec_xyzxyzw

    def get_goals(self):
        return self._grasping_episode_config.goal_object_pose

    @override
    def are_states_terminal(self, states) -> th.Tensor:
        return super().are_states_terminal(states)

    def _set_goal_marker_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._adapter.setLinksStateDirect(  link_names=[self._goal_marker_base_link],
                                                link_states_pose_vel=th.cat([self._grasping_episode_config.goal_object_pose,
                                                                             self._thzeros((self.num_envs,6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)

    @override
    def _simulation_initialization(self, vec_mask : th.Tensor):
        super()._simulation_initialization(vec_mask = vec_mask)
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            obj_state = self._thzeros((self.num_envs,1,13))
            obj_state[:,0,:7] = self._grasping_episode_config.initial_object_pose[:]
            self._adapter.setLinksStateDirect(link_names=[self._grasping_conf.target_object_link],
                                              link_states_pose_vel=obj_state)
        else:
            raise RuntimeError(f"Cannot run simulation initialization on non-simulated adapter")
            
    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._set_goal_marker_pose(vec_mask=self._all_vecs)        
        return super().get_ui_renderings(vec_mask=vec_mask)
    
    @override
    def _get_spawn_defs(self):
        spawn_defs = super()._get_spawn_defs()
        is_pybullet = isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper) and adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("PyBulletJointImpedanceAdapter"))
        if not hasattr(self,"_cube_spawn_def"):
            self._cube_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/cube.urdf.xacro")).read_text(),
                                            name="cube",
                                            pose=None,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
        spawn_defs.append(self._cube_spawn_def)
        if not hasattr(self,"_axes_spawn_def"):
            self._axes_spawn_def = ModelSpawnDef(   definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/axes.urdf.xacro")).read_text(),
                                                    name="goal_axes",
                                                    pose=None,
                                                    format="urdf.xacro",
                                                    kwargs={"add_world_link":str(is_pybullet),
                                                            "size" : 0.2})
            if not is_pybullet:
                self._goal_marker_base_link = ("goal_axes","root")
            else:
                self._goal_marker_base_link = ("goal_axes","world")
        spawn_defs.append(self._axes_spawn_def)
        return spawn_defs