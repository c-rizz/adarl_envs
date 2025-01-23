from __future__ import annotations
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.utils.utils import LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, vector_projection, isinstance_noimport, quat_xyzw_between_vecs_py, dbg_check_size, dbg_check, dbg_run
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
from jumping_leg.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
import adarl.utils.tensor_trees
import traceback

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

class LocomotionVecEnv(RobotVecEnv):
    STATE_LOCOMOTION = "loco"

    @dataclass
    class LocomotionConfiguration:
        disallowed_contact_links : list[tuple[str,str]]
        goal_speed_minmax : th.Tensor
        reward_scale : th.Tensor
        reward_weight_acceleration : th.Tensor
        reward_weight_contacts : th.Tensor
        reward_weight_energy : th.Tensor
        reward_weight_health : th.Tensor
        reward_weight_height : th.Tensor
        reward_weight_pitchnroll : th.Tensor
        reward_weight_actdiff : th.Tensor
        reward_weight_position_limit : th.Tensor
        reward_weight_position : th.Tensor
        reward_weight_torque_limit : th.Tensor
        reward_weight_torque : th.Tensor
        reward_weight_torquediff : th.Tensor
        reward_weight_tracking : th.Tensor
        reward_weight_velocity_limit : th.Tensor
        reward_weight_velocity : th.Tensor
        terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]]
        use_contacts : bool
        height_reward_settle_point : th.Tensor
        pitchnroll_reward_settle_point : th.Tensor
        vel_reward_goalrelative_weight : th.Tensor
        reward_vel_goal_relative_width : th.Tensor
        reward_vel_goal_relative_width_offset : th.Tensor
        reward_vel_goal_absolute_width : th.Tensor


    @dataclass
    class EpisodeLocomConfiguration:
        goal_abs_vel_vec_xyz : th.Tensor
        goal_abs_gravity_vec_xyz : th.Tensor
        goal_abs_height_vec_z : th.Tensor

    LOCOMOTION_FIELDS = IntEnum("INTERNAL_FIELDS", ["COLLISON_COUNT",
                                                    "GOAL_VELOCITY_REL_X",
                                                    "GOAL_VELOCITY_REL_Y",
                                                    "GOAL_VELOCITY_REL_Z",
                                                    "GOAL_VELOCITY_ABS_X",
                                                    "GOAL_VELOCITY_ABS_Y",
                                                    "GOAL_VELOCITY_ABS_Z",
                                                    "GOAL_BODY_HEIGHT",
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
                                                    "REWARD_HEIGHT_WEIGHT",
                                                    "REWARD_PITCHNROLL_WEIGHT",
                                                    "REWARD_ACTDIFF_WEIGHT",
                                                    "REWARD_POSITION_WEIGHT",
                                                    "SMOOTHED_TRACKING_ERROR",
                                                    "HEIGHT_ERR",
                                                    "ORIENT_ERR",
                                                    "SUM_IMPULSES",
                                                    "CRASHED"], start=0)

    def __init__(self,  action_delay_mustd : tuple[float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseVecJointImpedanceAdapter,
                        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor],
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | JOINT_FILTERS],
                        disallowed_contact_links : list[tuple[str,str]],
                        frame_stack_length : int,
                        goal_err_smoothing_halflife_sec : float,
                        goal_speed_minmax : tuple[float, float],
                        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                        homing_joint_pose : dict[tuple[str,str], float],
                        maxStepsPerEpisode : int,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        observe_body_velocity : bool,
                        reward_acceleration_weight : float,
                        reward_actdiff_weight : float,
                        reward_contacts_weight : float,
                        reward_energy_weight : float,
                        reward_health_weight : float,
                        reward_height_weight : float,
                        reward_pitchnroll_weight : float,
                        reward_position_limit_weight : float,
                        reward_position_weight : float,
                        reward_scale : float,
                        reward_torque_limit_weight : float,
                        reward_torque_weight : float,
                        reward_torquediff_weight : float,
                        reward_tracking_weight : float,
                        reward_velocity_limit_weight : float,
                        reward_velocity_weight : float,
                        robot_main_body_link : str,
                        robot_name : str,
                        robot_root_link : str,
                        robot_urdf_string : str,
                        safe_damping : float,
                        safe_stiffness : float,
                        safety_limits_factor : float,
                        seed : int,
                        stepLength_sec : float,
                        step_precision_tolerance : float,
                        stop_on_safety : bool,
                        terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]],
                        th_device : th.device,
                        use_contacts : bool,
                        verbose_infos : bool,
                        quiet : bool,
                        enable_dbg_checks : bool,
                        initial_pose_randomization : float,
                        init_on_reset_ratio : float,
                        obs_noise_joints_pve_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_linvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_angvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_posz_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        obs_noise_gravity_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor,
                        ui_camera_resolution_hw : tuple[int,int] = (256,144)
                        ):
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._locomotion_conf = LocomotionVecEnv.LocomotionConfiguration(
                        reward_weight_acceleration = self._thtens(reward_acceleration_weight),
                        reward_weight_contacts  = self._thtens(reward_contacts_weight) ,
                        reward_weight_health = self._thtens(reward_health_weight),
                        reward_weight_energy  = self._thtens(reward_energy_weight) ,
                        reward_weight_position_limit  = self._thtens(reward_position_limit_weight) ,
                        reward_scale  = self._thtens(reward_scale) ,
                        reward_weight_torque_limit  = self._thtens(reward_torque_limit_weight) ,
                        reward_weight_torque = self._thtens(reward_torque_weight),
                        reward_weight_torquediff = self._thtens(reward_torquediff_weight),
                        reward_weight_tracking = self._thtens(reward_tracking_weight),
                        reward_weight_velocity_limit = self._thtens(reward_velocity_limit_weight),
                        reward_weight_velocity = self._thtens(reward_velocity_weight),
                        reward_weight_position = self._thtens(reward_position_weight),
                        use_contacts = use_contacts,
                        disallowed_contact_links=disallowed_contact_links,
                        terminating_contact_pairs=terminating_contact_pairs,
                        goal_speed_minmax = th.as_tensor(goal_speed_minmax, device=th_device, dtype=th.float32),
                        reward_weight_height = self._thtens(reward_height_weight),
                        reward_weight_pitchnroll = self._thtens(reward_pitchnroll_weight),
                        reward_weight_actdiff = self._thtens(reward_actdiff_weight),
                        height_reward_settle_point=self._thtens(0.2),
                        pitchnroll_reward_settle_point=self._thtens(0.2),
                        vel_reward_goalrelative_weight = self._thtens(0.25),
                        reward_vel_goal_relative_width = self._thtens(1.5),
                        reward_vel_goal_absolute_width = self._thtens(0.25),
                        reward_vel_goal_relative_width_offset = self._thtens(0.1))
        
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
                            safety_limits_factor = safety_limits_factor,
                            seed = seed,
                            stepLength_sec = stepLength_sec,
                            step_precision_tolerance = step_precision_tolerance,
                            stop_on_safety = stop_on_safety,
                            th_device = th_device,
                            homing_body_pose_xyz_xyzw = homing_body_pose_xyz_xyzw,
                            homing_joint_pose = homing_joint_pose,
                            control_limits_minmax_pve = control_limits_minmax_pve,
                            observe_body_velocity = observe_body_velocity,
                            frame_stack_length=frame_stack_length,
                            verbose_infos = verbose_infos,
                            quiet = quiet,
                            enable_dbg_checks = enable_dbg_checks,
                            initial_pose_randomization = initial_pose_randomization,
                            init_on_reset_ratio = init_on_reset_ratio,
                            obs_noise_joints_pve_ep_mustd_step_std = obs_noise_joints_pve_ep_mustd_step_std,
                            obs_noise_linvel_ep_mustd_step_std = obs_noise_linvel_ep_mustd_step_std,
                            obs_noise_angvel_ep_mustd_step_std = obs_noise_angvel_ep_mustd_step_std,
                            obs_noise_posz_ep_mustd_step_std = obs_noise_posz_ep_mustd_step_std,
                            obs_noise_gravity_ep_mustd_step_std = obs_noise_gravity_ep_mustd_step_std,
                            ui_camera_resolution_hw = ui_camera_resolution_hw
                        )

        
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__
        ggLog.info(f"Env constructed")

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

    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        locomotion_state_helper = ThBoxStateHelper( field_names=[e for e in self.LOCOMOTION_FIELDS],
                                                    obs_dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    fields_minmax={ self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : [-1,1], 
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
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.HEIGHT_ERR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.ORIENT_ERR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SUM_IMPULSES : [0,10000],
                                                                    self.LOCOMOTION_FIELDS.COLLISON_COUNT : [0,1000],
                                                                    self.LOCOMOTION_FIELDS.CRASHED : [0,1]},
                                                    observable_fields=[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z,
                                                                        self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR],
                                                    vec_size=adapter.vec_size())
        self._state_helper = self._state_helper.add_substate(LocomotionVecEnv.STATE_LOCOMOTION,
                                                            locomotion_state_helper,
                                                            observable = True,
                                                            flatten = True)
        ggLog.info(f"Built state/obs/action helpers")
        



    @override
    def _get_new_instantaneous_state(self):

        prev_locom_state = self._current_state[self.STATE_LOCOMOTION][:, 0]
        new_inst_state = super()._get_new_instantaneous_state()
        new_internal_state = new_inst_state[self.STATE_INTERNAL]
        new_extrinsic_state = new_inst_state[self.STATE_EXTRINSIC]

        bstates_vec_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_frame = True)[:,0,:]
        vsize = bstates_vec_13.size()[0]
        prev_goal_abs_vec_xyz = self._locomotion_episode_config.goal_abs_vel_vec_xyz
        # prev_goal_abs_vec_xyz = prev_locom_state[:,[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X,
        #                                             self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y,
        #                                             self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z]]
        goal_rel_linvel_vec_xyz = th_quat_rotate(prev_goal_abs_vec_xyz, th_quat_conj(bstates_vec_13[:,3:7]))
        goal_height_z = prev_locom_state[:,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT]

        # sadly right in this point everything is a dict, so things must be addressed like this, maybe something could be done about this
        body_rel_linvel_vec_xyz = th.cat([new_extrinsic_state[k] for k in
                                        [self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                         self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                         self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z]], dim = 1)
        gravity_rel_vec_xyz     = th.cat([new_extrinsic_state[k] for k in 
                                        [self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                         self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                         self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z]], dim = 1)
        # print(f"body_rel_linvel_vec_xyz.size() = {body_rel_linvel_vec_xyz.size()}")
        # print(f"gravity_rel_vec_xyz.size() = {gravity_rel_vec_xyz.size()}")
        # print(f"goal_rel_linvel_vec_xyz.size() = {goal_rel_linvel_vec_xyz.size()}")
        tracking_err_vec = self._tracking_error_vec(body_rel_linvel_vec_xyz, gravity_rel_vec_xyz, goal_rel_linvel_vec_xyz).unsqueeze(-1)
        # print(f"prev_locom_state.size() = {prev_locom_state.size()}")
        # print(f"tracking_err_vec.size() = {tracking_err_vec.size()}")
        # print(f"self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT].size() = {self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT].size()}")
        # print(f"new_extrinsic_state[self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z].size() = {new_extrinsic_state[self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z].size()}")
        # print(f"prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR].size() = {prev_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR].size()}")
        # print(f"goal_height_z.size() = {goal_height_z.size()}")
        # ggLog.info( f"abs_goal = {self._locomotion_episode_config.goal_abs_linvel_xyz}, body_rel_linvel_xyz = {body_rel_linvel_xyz}, goal_rel_linvel_xyz = {goal_rel_linvel_xyz}, gravity_rel_xyz={gravity_rel_xyz}\n"
        #             f"tracking_err = {tracking_error} = norm({body_planar_rel_linvel_xyz}-{goal_rel_linvel_xyz}) = norm({body_planar_rel_linvel_xyz-goal_rel_linvel_xyz})")
        height_err_vec = th.abs(new_extrinsic_state[self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] - goal_height_z)
        orient_err_vec = th.linalg.norm(gravity_rel_vec_xyz-self._locomotion_episode_config.goal_abs_gravity_vec_xyz, dim = 1, keepdim=True) # Would be nice to use geodesic distance or somethinglike that

        alpha = self._configuration.goal_err_exp_smoothing_1s**(self._configuration.stepLength_sec)
        smoothed_tracking_err_vec = tracking_err_vec*(1-alpha) + prev_locom_state[:, self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]*alpha
        smoothed_tracking_err_vec[new_internal_state[self.INTERNAL_FIELDS.STEP_COUNT]<=0] = tracking_err_vec

        if self._locomotion_conf.use_contacts:
            if not isinstance_noimport(self._adapter, "PyBulletAdapter"):
                raise RuntimeError(f"Contacts are supported only in pybullet for now")
            raise NotImplementedError()
            contacts = self._adapter.get_contacts()
            substep_count = len(contacts)
            contacts = sum(contacts,[]) # merge the contacts from all the substeps
            bad_contacts = [c for c in contacts if c[0] in self._locomotion_conf.disallowed_contact_links or c[1] in self._locomotion_conf.disallowed_contact_links]
            collision_count = len(contacts)/substep_count if substep_count != 0 else 0
            bad_forces = np.array([c[3] for c in bad_contacts])
            bad_durations = np.array([c[4] for c in bad_contacts])
            sum_bad_impulses = np.sum(np.abs(bad_forces*bad_durations))

            crashed = prev_locom_state[self.LOCOMOTION_FIELDS.CRASHED]
            if not crashed:
                # pairs = {(c[0],c[1]) for c in contacts}
                # print(f"contact pairs = {pairs}")
                for c in contacts:
                    if (c[0],c[1]) in self._locomotion_conf.terminating_contact_pairs or (c[1],c[0]) in self._locomotion_conf.terminating_contact_pairs:
                        crashed = 1
                        break
        else:
            collision_count_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            sum_bad_impulses_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
            crashed_vec = th.zeros(size=(vsize, 1), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)



        new_locom_state = { self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : self._locomotion_conf.reward_weight_torque_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : self._locomotion_conf.reward_weight_position_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : self._locomotion_conf.reward_weight_velocity_limit.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT : self._locomotion_conf.reward_weight_velocity.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT : self._locomotion_conf.reward_weight_acceleration.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT : self._locomotion_conf.reward_weight_contacts.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT : self._locomotion_conf.reward_weight_health.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT : self._locomotion_conf.reward_weight_height.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : self._locomotion_conf.reward_weight_pitchnroll.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT : self._locomotion_conf.reward_weight_actdiff.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT : self._locomotion_conf.reward_weight_tracking.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT : self._locomotion_conf.reward_weight_torque.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : self._locomotion_conf.reward_weight_torquediff.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT : self._locomotion_conf.reward_weight_position.expand(vsize,1),
                            self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : smoothed_tracking_err_vec,
                            self.LOCOMOTION_FIELDS.HEIGHT_ERR : height_err_vec,
                            self.LOCOMOTION_FIELDS.ORIENT_ERR : orient_err_vec,
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X : goal_rel_linvel_vec_xyz[:,[0]],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y : goal_rel_linvel_vec_xyz[:,[1]],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z : goal_rel_linvel_vec_xyz[:,[2]],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X : prev_goal_abs_vec_xyz[:,[0]],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y : prev_goal_abs_vec_xyz[:,[1]],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z : prev_goal_abs_vec_xyz[:,[2]],
                            self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : self._locomotion_episode_config.goal_abs_height_vec_z,
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,[0]],
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,[1]],
                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z : self._locomotion_episode_config.goal_abs_gravity_vec_xyz[:,[2]],
                            self.LOCOMOTION_FIELDS.SUM_IMPULSES : sum_bad_impulses_vec,
                            self.LOCOMOTION_FIELDS.COLLISON_COUNT :collision_count_vec,
                            self.LOCOMOTION_FIELDS.CRASHED : crashed_vec}
        new_inst_state[self.STATE_LOCOMOTION] = new_locom_state

        return new_inst_state
    










    def _tracking_error_vec(self, body_rel_linvel_vec_xyz : th.Tensor, gravity_rel_vec_xyz : th.Tensor, goal_rel_linvel_vec_xyz : th.Tensor):
        # ggLog.info(f"tracking_error_vec(body_rel_linvel_vec_xyz.size()={body_rel_linvel_vec_xyz.size()}, gravity_rel_vec_xyz.size()={gravity_rel_vec_xyz.size()}, goal_rel_linvel_vec_xyz.size()={goal_rel_linvel_vec_xyz.size()}")
        body_planar_rel_linvel_xyz = body_rel_linvel_vec_xyz - vector_projection(body_rel_linvel_vec_xyz,gravity_rel_vec_xyz)
        # ggLog.info(f" {body_rel_linvel_xyz.cpu().tolist()} + vector_projection({body_rel_linvel_xyz.cpu().tolist()},{gravity_rel_xyz.cpu().tolist()}) =\n"
        #            f" {body_rel_linvel_xyz.cpu().tolist()} + {vector_projection(body_rel_linvel_xyz,gravity_rel_xyz).cpu().tolist()} = \n"
        #            f"{body_planar_rel_linvel_xyz.cpu().tolist()}\n"
        #            f"norm({body_planar_rel_linvel_xyz.cpu().tolist()} - {goal_rel_linvel_xyz.cpu().tolist()})={th.linalg.norm(body_planar_rel_linvel_xyz-goal_rel_linvel_xyz).cpu().tolist()}")
        # time.sleep(0.1)
        # goal_rel_linvel_xyz should already be "planar", it's projection along gravity_rel should be zero
        norms = th.norm(vector_projection(goal_rel_linvel_vec_xyz,gravity_rel_vec_xyz), dim = 1)
        dbg_check(lambda: th.all(norms < 0.1),
                  lambda:   f"goal_rel_linvel_xyz is not horizontal (th.all(norms < 0.1) = {th.all(norms < 0.1)}), projection is "
                            f"{vector_projection(goal_rel_linvel_vec_xyz, gravity_rel_vec_xyz)[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f"goal={goal_rel_linvel_vec_xyz[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f"graity={gravity_rel_vec_xyz[th.logical_or(norms >= 0.1,th.logical_not(th.isfinite(norms)))]}"
                            f" big={th.nonzero(norms >= 0.1)}"
                            f" isnan={th.nonzero(th.isnan(norms))}"
                            f" isinf={th.nonzero(th.isinf(norms))}")
        return th.linalg.norm(body_planar_rel_linvel_xyz-goal_rel_linvel_vec_xyz, dim = 1)
    
    @override
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = 100
        current_state_locom_vec = state[self.STATE_LOCOMOTION][:, 0,:,0]
        state_action_vec = self._current_state[self.STATE_ACT]

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_pvesd[:,0], lims[0,:,0], lims[1,:,0])
        state_robot_norm     = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)
        dbg_run(lambda: self._warn_out_of_bounds(state_robot_norm))
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)
        # ggLog.info(f"current_state_locom_vec.size() = {current_state_locom_vec.size()}")
        # ggLog.info(f"state_robot_norm.size() = {state_robot_norm.size()}")
        # ggLog.info(f"state_robot_safenorm.size() = {state_robot_safenorm.size()}")
        normposhomingdiff   = state_robot_norm[:,0,:,0] - normhoming
        normvelocities      = state_robot_norm[:,0,:,1]
        normtorques         = state_robot_norm[:,0,:,2]
        normaccelerations   = (state_robot_norm[:,0,:,1] - state_robot_norm[:,1,:,1])/self._configuration.stepLength_sec
        normtorquediff      = state_robot_norm[:,0,:,2] - state_robot_norm[:,1,:,2]
        actdiff             = th.flatten(state_action_vec[:,0] - state_action_vec[:,1], start_dim=1)

        position_safenorm   = state_robot_safenorm[:,0,:,0]
        velocities_safenorm = state_robot_safenorm[:,0,:,1]
        torque_safenorm     = state_robot_safenorm[:,0,:,2]

        reward_torque           = -th.clamp(th.mean(th.pow(normtorques,2), dim=1),          -max_rew,max_rew)
        reward_velocity         = -th.clamp(th.mean(th.pow(normvelocities,2), dim=1),       -max_rew,max_rew)
        reward_acceleration     = -th.clamp(th.mean(th.pow(normaccelerations,2), dim=1),    -max_rew,max_rew)
        reward_position         = -th.clamp(th.mean(th.pow(normposhomingdiff,2), dim=1),    -max_rew,max_rew)
        reward_torquediff       = -th.clamp(th.mean(th.pow(normtorquediff,2), dim=1),       -max_rew,max_rew)
        reward_actdiff          = -th.clamp(th.mean(th.pow(actdiff,2), dim=1),              -max_rew,max_rew)
        reward_torque_limit     = -th.clamp(th.mean(th.pow(torque_safenorm,50), dim=1),     -1,1)
        reward_position_limit   = -th.clamp(th.mean(th.pow(position_safenorm,50), dim=1),   -1,1)
        reward_velocity_limit   = -th.clamp(th.mean(th.pow(velocities_safenorm,50), dim=1), -1,1)

        reward_height = bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.HEIGHT_ERR],
                                    zero_rew_dist=self._locomotion_conf.height_reward_settle_point)

        reward_pitchnroll = bell_reward(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.ORIENT_ERR],
                                        zero_rew_dist=self._locomotion_conf.pitchnroll_reward_settle_point)

        velocity_tracking_err_vec = current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        goalrelative_weight = self._locomotion_conf.vel_reward_goalrelative_weight
        rel_goal_bell_width = self._locomotion_conf.reward_vel_goal_relative_width
        rel_goal_offset = self._locomotion_conf.reward_vel_goal_relative_width_offset
        abs_goal_bell_width = self._locomotion_conf.reward_vel_goal_absolute_width
        goal_norm = th.linalg.norm(self._locomotion_episode_config.goal_abs_vel_vec_xyz, dim = -1)
        # ggLog.info(f"velocity_tracking_err_vec.size() = {velocity_tracking_err_vec.size()}")
        # ggLog.info(f"goal_norm.size() = {goal_norm.size()}")
        reward_velocity_tracking = (   goalrelative_weight  * bell_reward(velocity_tracking_err_vec,
                                                                          zero_rew_dist=rel_goal_bell_width*(goal_norm+rel_goal_offset))+
                                    (1-goalrelative_weight) * bell_reward(velocity_tracking_err_vec,
                                                                          zero_rew_dist=abs_goal_bell_width))
        
        reward_contacts = - th.clamp(current_state_locom_vec[:,self.LOCOMOTION_FIELDS.SUM_IMPULSES], -max_rew, max_rew)

        sub_rewards_return["reward_tracking"] = reward_velocity_tracking
        sub_rewards_return["reward_torque"] = reward_torque
        sub_rewards_return["reward_torque_limit"] = reward_torque_limit
        sub_rewards_return["reward_torquediff"] = reward_torquediff
        sub_rewards_return["reward_velocity"] = reward_velocity
        sub_rewards_return["reward_contacts"] = reward_contacts
        sub_rewards_return["reward_height"] = reward_height
        sub_rewards_return["reward_pitchnroll"] = reward_pitchnroll
        sub_rewards_return["reward_velocity_limit"] = reward_velocity_limit
        sub_rewards_return["reward_acceleration"] = reward_acceleration
        sub_rewards_return["reward_position_limit"] = reward_position_limit
        sub_rewards_return["reward_position"] = reward_position
        sub_rewards_return["reward_actdiff"] = reward_actdiff
        sub_rewards_return["reward_health"] = th.ones((current_state_locom_vec.size()[0],), device=current_state_locom_vec.device)
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards_return.items()}

        for k,v in sub_rewards_return.items():
            dbg_check_size(v, (self._adapter.vec_size(),), f"Unexpected size for sub_reward {k}")
        # dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(sub_rewards_return, -100, 100),
        #           lambda: f"{adarl.utils.tensor_trees.flatten_tensor_tree(map_tensor_tree(sub_rewards_return, lambda t: adarl.utils.tensor_trees.is_leaf_bounded(t,min=-100,max=100)))}")
        
        weights = { "reward_tracking" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT],
                    "reward_torque" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT],
                    "reward_torque_limit" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "reward_torquediff" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT],
                    "reward_velocity" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT],
                    "reward_velocity_limit" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT],
                    "reward_acceleration" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT],
                    "reward_position_limit" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT],
                    "reward_health" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT],
                    "reward_contacts" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT],
                    "reward_height" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT],
                    "reward_pitchnroll" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT],
                    "reward_actdiff" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_ACTDIFF_WEIGHT],
                    "reward_position" : current_state_locom_vec[:,self.LOCOMOTION_FIELDS.REWARD_POSITION_WEIGHT]
                    }
        for k in sub_rewards_return:
            sub_rewards_return[k] = self._locomotion_conf.reward_scale*sub_rewards_return[k]*weights[k]
        sub_rewards_return = {k:v.view(self._adapter.vec_size(),) for k,v in sub_rewards_return.items()}
        sub_rewards_unscaled = {k:v.view(self._adapter.vec_size(),) for k,v in sub_rewards_unscaled.items()}
        reward = th.sum(th.stack(list(sub_rewards_return.values()), dim = 1), dim =1)

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
        dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(sub_rewards_return, -100, 100),
                  lambda: f"{adarl.utils.tensor_trees.flatten_tensor_tree(map_tensor_tree(sub_rewards_return, lambda t: adarl.utils.tensor_trees.is_leaf_bounded(t,min=-100,max=100)))}",
                  just_warn=True)
        dbg_check(lambda: adarl.utils.tensor_trees.is_all_bounded(reward, -100, 100),
                  lambda: f"Reward over 100. sub_rewards = {map_tensor_tree(sub_rewards_return,lambda t: 'minmax='+str((th.min(t).cpu().item(), th.max(t).cpu().item())))}",
                  just_warn=True)
        return reward
    










    def _update_stats(self):
        super()._update_stats()

        vel_error_vec = self._tracking_error_vec(
                                        body_rel_linvel_vec_xyz = self._current_state[self.STATE_EXTRINSIC][:,0,
                                                                                                            [self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z],0],
                                        gravity_rel_vec_xyz = self._current_state[self.STATE_EXTRINSIC][:,0,
                                                                                                        [self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z],0],
                                        goal_rel_linvel_vec_xyz = self._current_state[self.STATE_LOCOMOTION][:,0,
                                                                                                            [self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X, 
                                                                                                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y, 
                                                                                                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z],0])
        dbg_check_size(vel_error_vec, (self._adapter.vec_size(),))
        body_linvel_xyz = self._current_state[self.STATE_EXTRINSIC][:, 0,  [self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z],0]
        body_speed_vec = th.linalg.norm(body_linvel_xyz[:,:2], dim=-1)
        body_height_vec = self._current_state[self.STATE_EXTRINSIC][:,0,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z,0]
        goal_height_vec = self._current_state[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT,0]
        height_error_vec = th.abs(body_height_vec-goal_height_vec)
        gravity_vec = self._current_state[self.STATE_EXTRINSIC][:,0,[self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                                                     self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                                                     self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z],0]
        goal_gravity_vec = self._current_state[self.STATE_LOCOMOTION][:,0, [self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_X,
                                                                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Y,
                                                                            self.LOCOMOTION_FIELDS.GOAL_GRAVITY_ABS_Z],0]
        pitchnroll_err_vec = th.linalg.norm(gravity_vec-goal_gravity_vec, dim = -1)
        step_counts = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT,0].to(th.long)
        dbg_check_size(pitchnroll_err_vec, (self._adapter.vec_size(),))
        dbg_check_size(step_counts, (self._adapter.vec_size(),))
        
        # Update episode averages
        self._stats["ep_avg_vel_err_vec"]          = (self._stats["ep_avg_vel_err_vec"]*(step_counts-1) + vel_error_vec)/step_counts # Elements with step_count == 0 will be inf
        self._stats["ep_avg_height_err_vec"]       = (self._stats["ep_avg_height_err_vec"]*(step_counts-1) + height_error_vec)/step_counts # Elements with step_count == 0 will be inf
        self._stats["ep_avg_pitchnroll_err_vec"]   = (self._stats["ep_avg_pitchnroll_err_vec"]*(step_counts-1) + pitchnroll_err_vec)/step_counts # Elements with step_count == 0 will be inf
        self._stats["ep_avg_bodyspeed_vec"]        = (self._stats["ep_avg_bodyspeed_vec"]*(step_counts-1) + body_speed_vec)/step_counts # Elements with step_count == 0 will be inf
        # Correct the episode averages for episodes that have just started
        self._stats["ep_avg_vel_err_vec"][step_counts==0]          = vel_error_vec[step_counts==0]
        self._stats["ep_avg_height_err_vec"][step_counts==0]       = height_error_vec[step_counts==0]
        self._stats["ep_avg_pitchnroll_err_vec"][step_counts==0]   = pitchnroll_err_vec[step_counts==0]
        self._stats["ep_avg_bodyspeed_vec"][step_counts==0]        = body_speed_vec[step_counts==0]

        # Fill the buffers for episodes that have just staretd
        self._stats["vel_errs_vec"   ][step_counts==0,:] = vel_error_vec[step_counts==0].unsqueeze(1).expand(-1, self._buff_sizes)
        self._stats["height_errs_vec"   ][step_counts==0,:] = height_error_vec[step_counts==0].unsqueeze(1).expand(-1, self._buff_sizes)
        self._stats["pitchnroll_errs_vec"   ][step_counts==0,:] = pitchnroll_err_vec[step_counts==0].unsqueeze(1).expand(-1, self._buff_sizes)
        self._stats["body_speeds_vec"   ][step_counts==0,:] = body_speed_vec[step_counts==0].unsqueeze(1).expand(-1, self._buff_sizes)
        # Update the buffers
        # idxs = step_counts%self._buff_sizes
        idxs = step_counts%self._stats["vel_errs_vec"].size()[1]
        # print(f"torch.is_grad_enabled()) = {th.is_grad_enabled()}")
        # print(f"idx.size() = {idxs.size()}, idx = {idxs}")
        # print(f"vel_error_vec.size() = {vel_error_vec.size()}, {vel_error_vec}")
        self._stats["vel_errs_vec"   ][:,idxs]    = vel_error_vec
        self._stats["height_errs_vec"][:,idxs] = height_error_vec
        self._stats["pitchnroll_errs_vec"  ][:,idxs]   = pitchnroll_err_vec
        self._stats["body_speeds_vec"      ][:,idxs]       = body_speed_vec
   
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        i = super().get_infos(state=state, labels=labels)
        curr_locom_state = state[self.STATE_LOCOMOTION][:,0]
        curr_extri_state = state[self.STATE_EXTRINSIC][:,0]
        i["goal_rel_xyz_vec"] = curr_locom_state[:,[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X,
                                                self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y,
                                                self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z]]
        i["goal_abs_xyz_vec"] = curr_locom_state[:,[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_X,
                                                self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Y,
                                                self.LOCOMOTION_FIELDS.GOAL_VELOCITY_ABS_Z]]
        i["smoothed_linvel_error"] = curr_locom_state[:,self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        i["body_abs_linvel"] = curr_extri_state[:,[self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X,
                                                   self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y,
                                                   self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z]]
        i["body_rel_linvel"] = curr_extri_state[:,[self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                   self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                   self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z]]
        i["linvel_error"] = i["goal_abs_xyz_vec"] - i["body_abs_linvel"]
        i["linvel_error"] = i["goal_abs_xyz_vec"] - i["body_abs_linvel"]
        i["ep_avg_vel_err_vec"] = self._stats["ep_avg_vel_err_vec"]
        i["ep_avg_height_err_vec"] = self._stats["ep_avg_height_err_vec"]
        i["ep_avg_pitchnroll_err_vec"] = self._stats["ep_avg_pitchnroll_err_vec"]
        i["ep_avg_bodyspeed_vec"] = self._stats["ep_avg_bodyspeed_vec"]
        i["avg10_vel_errs_vec"] = th.mean(self._stats["vel_errs_vec"], dim = 1)
        i["avg10_height_errs_vec"] = th.mean(self._stats["height_errs_vec"], dim = 1)
        i["avg10_pitchnroll_errs_vec"] = th.mean(self._stats["pitchnroll_errs_vec"], dim = 1)
        i["avg10_body_speeds_vec"] = th.mean(self._stats["body_speeds_vec"], dim = 1)
        i["success_vec"] = i["avg10_vel_errs_vec"] < 0.05
        sub_rews = {}
        self.compute_rewards(state, sub_rews)
        i["rewards"] = th.stack(list(sub_rews.values()), dim = 1) 
        # ggLog.info(f"i['rewards'] = {i['rewards'].size()}")
        if labels is not None:
            labels["rewards"] = to_string_tensor(list(sub_rews.keys())) 

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_LOCOMOTION]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())



        return i
    
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        if "goal_velocity_xy" in reset_options:
            goal_velocity_vec_xy = th.as_tensor(reset_options["goal_velocity_xy"],device=self._configuration.th_device)
        else:
            goal_speeds = unnormalize(self._thrand(size=(self._adapter.vec_size(),))*2-1,
                                        min=self._locomotion_conf.goal_speed_minmax[0],
                                        max=self._locomotion_conf.goal_speed_minmax[1])
            goal_yaws = self._thrand((self._adapter.vec_size(),))*math.pi*2
            goal_velocity_vec_xy = goal_speeds.unsqueeze(1)*th.stack([th.cos(goal_yaws), th.sin(goal_yaws)], dim = 1)
        
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
        self._locomotion_episode_config = LocomotionVecEnv.EpisodeLocomConfiguration(goal_abs_vel_vec_xyz     = self._thzeros((self._adapter.vec_size(), 3)),
                                                                                     goal_abs_gravity_vec_xyz = self._thtens([0.0,0.0,-1.0]).repeat(self._adapter.vec_size(), 1),
                                                                                     goal_abs_height_vec_z    = self._thtens([0.45]).repeat(self._adapter.vec_size(), 1))
        self.set_max_episode_steps(reset_options.get("reset_options",self._current_episode_config.vec_max_ep_steps))
        self.set_goal(goal_velocity_vec_xy)

    def set_goal(self, goal_velocity_vec_xy : Sequence[tuple[float,float]] | tuple[float,float] | th.Tensor | None = None,
                        goal_velocity_diff_speed_yaw : tuple[float,float] | th.Tensor | None = None):
        if goal_velocity_vec_xy is not None:
            goal_velocity_vec_xy = th.as_tensor(goal_velocity_vec_xy,device=self._configuration.th_device)
            goal_velocity_vec_xy = goal_velocity_vec_xy.expand(self._adapter.vec_size(),2)
        else:
            if isinstance(goal_velocity_diff_speed_yaw, Sequence):
                goal_velocity_diff_speed_yaw = th.as_tensor(goal_velocity_diff_speed_yaw,device=self._configuration.th_device)
            elif not isinstance(goal_velocity_diff_speed_yaw, th.Tensor):
                raise RuntimeError(f"Unexpected type {type(goal_velocity_diff_speed_yaw)} for goal_velocity_diff_speed_yaw")
            goal_speeds = th.norm(self._locomotion_episode_config.goal_abs_vel_vec_xyz[:,:2], dim=1)
            goal_yaws = th.atan2(self._locomotion_episode_config.goal_abs_vel_vec_xyz[:,1],
                           self._locomotion_episode_config.goal_abs_vel_vec_xyz[:,0])
            goal_speeds += goal_velocity_diff_speed_yaw[0]
            goal_yaws += goal_velocity_diff_speed_yaw[1]
            goal_velocity_vec_xy = goal_speeds.unsqueeze(1)*th.stack([th.cos(goal_yaws), th.sin(goal_yaws)], dim = 1)
            
        self._locomotion_episode_config.goal_abs_vel_vec_xyz[:,:2] = goal_velocity_vec_xy
        self._locomotion_episode_config.goal_abs_vel_vec_xyz[:,2] = 0

    def get_goals(self):
        return self._locomotion_episode_config.goal_abs_vel_vec_xyz

    @override
    def are_states_terminal(self, states) -> th.Tensor:
        r = super().are_states_terminal(states)
        return th.logical_and(r, states[self.STATE_LOCOMOTION][:,0,self.LOCOMOTION_FIELDS.CRASHED,0]).view((self.num_envs,))

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        super()._initialize_episodes(vec_mask=vec_mask, options=options)
        if self._locomotion_conf.use_contacts:
            raise NotImplementedError("Contects not implemented yet")
            self._adapter.monitor_contacts([(self._configuration.robot_name, None)])

    def _set_arrow_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            q = quat_xyzw_between_vecs_py(self._thtens([1.0,0,0]).expand((self._adapter.vec_size(),3)), self._locomotion_episode_config.goal_abs_vel_vec_xyz)
            bstates_vec_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_frame = True)[:,0,:]
            pose = bstates_vec_13[:,:7]
            pose[:,2] = th.linalg.norm(self._locomotion_episode_config.goal_abs_vel_vec_xyz, dim = 1)
            pose[:,3:7] = q
            pose[1:] = pose[0] #body_states13[:,:,:3] # Camera is on a fixed link, so it must be set to the same pose across all links
            # pose[:,:3] = self._thtens([0.,0.,1.0])
            self._adapter.setLinksStateDirect(link_names=[self._arrow_base],
                                                link_states_pose_vel=th.cat([pose, self._thzeros((pose.size()[0],6,))], dim = 1).unsqueeze(1),
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