from __future__ import annotations
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.utils.utils import LinkState, to_string_tensor, quat_rotate, quat_conjugate
from adarl.utils.state_helper import ThBoxStateHelper, unnormalize
import adarl.utils.utils
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
import torch as th
import math
import quaternion
from jumping_leg.env.RobotEnv import RobotEnv
from adarl.utils.tensor_trees import map_tensor_tree


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

class LocomotionEnv(RobotEnv):
    STATE_LOCOMOTION = "loco"

    @dataclass
    class LocomotionConfiguration:
        disallowed_contact_links : list[tuple[str,str]]
        goal_speed_minmax : th.Tensor
        reward_acceleration_weight : float
        reward_contacts_weight : float
        reward_energy_weight : float
        reward_health_weight : float
        reward_height_weight : float
        reward_pitchnroll_weight : float
        reward_position_limit_weight : float
        reward_scale : float
        reward_torque_limit_weight : float
        reward_torque_weight : float
        reward_torquediff_weight : float
        reward_tracking_weight : float
        reward_velocity_limit_weight : float
        reward_velocity_weight : float
        terminating_contact_pairs : list[tuple[tuple[str,str],tuple[str,str]]]
        use_contacts : bool
        height_reward_settle_point : th.Tensor
        pitchnroll_reward_settle_point : th.Tensor
        vel_tracking_reward_settle_point : th.Tensor


    @dataclass
    class EpisodeLocomConfiguration:
        goal_velocity_xy : th.Tensor

    LOCOMOTION_FIELDS = IntEnum("INTERNAL_FIELDS", ["COLLISON_COUNT",
                                                    "GOAL_VELOCITY_REL_X",
                                                    "GOAL_VELOCITY_REL_Y",
                                                    "GOAL_VELOCITY_REL_Z",
                                                    "GOAL_BODY_HEIGHT",
                                                    "GOAL_BODY_GRAVITY_X",
                                                    "GOAL_BODY_GRAVITY_Y",
                                                    "GOAL_BODY_GRAVITY_Z",
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
                                                    "SMOOTHED_TRACKING_ERROR",
                                                    "HEIGHT_ERR",
                                                    "ORIENT_ERR",
                                                    "SUM_IMPULSES",
                                                    "CRASHED"], start=0)

    def __init__(self,  action_delay_mustd : tuple[float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseJointImpedanceAdapter,
                        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor],
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | RobotEnv.JOINT_FILTERS],
                        disallowed_contact_links : list[tuple[str,str]],
                        frame_stack_length : int,
                        goal_err_smoothing_halflife_sec : float,
                        goal_speed_minmax : tuple[float, float],
                        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                        homing_joint_pose : dict[tuple[str,str], float],
                        maxStepsPerEpisode : int,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        obs_noise_ep_mustd : Sequence[float] | th.Tensor, 
                        obs_noise_step_std : Sequence[float] | float | th.Tensor,
                        observe_body_velocity : bool,
                        reward_acceleration_weight : float,
                        reward_contacts_weight : float,
                        reward_energy_weight : float,
                        reward_health_weight : float,
                        reward_height_weight : float,
                        reward_pitchnroll_weight : float,
                        reward_position_limit_weight : float,
                        reward_scale : float,
                        reward_torque_limit_weight : float,
                        reward_torque_weight : float,
                        reward_torquediff_weight : float,
                        reward_tracking_weight : float,
                        reward_velocity_limit_weight : float,
                        reward_velocity_weight : float,
                        robot_main_body_link : str,
                        robot_name : str,
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
                        verbose_infos : bool
                        ):
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
                            obs_noise_ep_mustd = obs_noise_ep_mustd, 
                            obs_noise_step_std = obs_noise_step_std,
                            robot_main_body_link = robot_main_body_link,
                            robot_name = robot_name,
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
                            verbose_infos = verbose_infos
                        )
        self._locomotion_conf = LocomotionEnv.LocomotionConfiguration(
                        reward_acceleration_weight = reward_acceleration_weight,
                        reward_contacts_weight  = reward_contacts_weight ,
                        reward_health_weight = reward_health_weight,
                        reward_energy_weight  = reward_energy_weight ,
                        reward_position_limit_weight  = reward_position_limit_weight ,
                        reward_scale  = reward_scale ,
                        reward_torque_limit_weight  = reward_torque_limit_weight ,
                        reward_torque_weight = reward_torque_weight,
                        reward_torquediff_weight = reward_torquediff_weight,
                        reward_tracking_weight = reward_tracking_weight,
                        reward_velocity_limit_weight = reward_velocity_limit_weight,
                        reward_velocity_weight = reward_velocity_weight,
                        use_contacts = use_contacts,
                        disallowed_contact_links=disallowed_contact_links,
                        terminating_contact_pairs=terminating_contact_pairs,
                        goal_speed_minmax = th.as_tensor(goal_speed_minmax, device=th_device, dtype=th.float32),
                        reward_height_weight=reward_height_weight,
                        reward_pitchnroll_weight=reward_pitchnroll_weight,
                        height_reward_settle_point=th.tensor(0.1, device=th_device),
                        pitchnroll_reward_settle_point=th.tensor(0.1, device=th_device),
                        vel_tracking_reward_settle_point=th.tensor(1.0, device=th_device))
        locomotion_state_helper = ThBoxStateHelper( field_names=[e for e in self.LOCOMOTION_FIELDS],
                                                    obs_dtype=th.float32,
                                                    th_device=th_device,
                                                    field_size=(1,),
                                                    fields_minmax={ self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X : [-10,10],
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z : [-10,10], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_X : [-1,1],
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Y : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Z : [-1,1], 
                                                                    self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.HEIGHT_ERR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.ORIENT_ERR : [0,10],
                                                                    self.LOCOMOTION_FIELDS.SUM_IMPULSES : [0,10000],
                                                                    self.LOCOMOTION_FIELDS.COLLISON_COUNT : [0,1000],
                                                                    self.LOCOMOTION_FIELDS.CRASHED : [0,1]},
                                                    observable_fields=[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y,
                                                                        self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z])
        self._state_helper = self._state_helper.add_substate(LocomotionEnv.STATE_LOCOMOTION,
                                                             locomotion_state_helper,
                                                             observable = True,
                                                             flatten = True)
        self.state_space = self._state_helper.get_space()
        self.observation_space = self._state_helper.get_obs_space()
        self.action_space = self._action_helper.action_space(seed=seed)

    @override
    def _get_new_instantaneous_state(self):
        locom_state = self._current_state[self.STATE_LOCOMOTION][0]
        prev_locom_state = self._current_state[self.STATE_LOCOMOTION][0]
        internal_state = self._current_state[self.STATE_INTERNAL][0]
        step_count = internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        
        body_state : LinkState = self._adapter.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
        body_linvel_xyz = body_state.pos_velocity_xyz
        tracking_error = th.linalg.norm(body_linvel_xyz[:2]-self._locomotion_episode_config.goal_velocity_xy).cpu().item()
        prev_tracking_error = locom_state[self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        alpha = self._configuration.goal_err_exp_smoothing_1s**(self._configuration.stepLength_sec)
        if step_count > 0:
            smoothed_goal_dist = tracking_error*(1-alpha) + prev_tracking_error*alpha
        else:
            smoothed_goal_dist = tracking_error

        if self._locomotion_conf.use_contacts:
            if not isinstance(self._adapter, PyBulletAdapter):
                raise RuntimeError(f"Contacts are supported only in pybullet for now")
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
                for c in contacts:
                    if (c[0],c[1]) in self._locomotion_conf.terminating_contact_pairs or (c[1],c[0]) in self._locomotion_conf.terminating_contact_pairs:
                        crashed = 1
                        break
        else:
            collision_count = 0
            sum_bad_impulses = 0

        goal_vel_xyz = np.array([self._locomotion_episode_config.goal_velocity_xy[0], self._locomotion_episode_config.goal_velocity_xy[1], 0.0])
        goal_vel_rel_xyz = quat_rotate(goal_vel_xyz, quat_conjugate(body_state.pose.orientation_xyzw))

        new_inst_state = super()._get_new_instantaneous_state()

        goal_body_height = 0.45
        goal_gravity_vec = th.tensor([0.0,0.0,-1.0], device = self._configuration.th_device)
        height_err = th.abs(new_inst_state[self.STATE_EXTRINSIC][self.EXTRINSIC_FIELDS.BODY_POS_Z] - goal_body_height)
        gravity_vec = th.as_tensor([new_inst_state[self.STATE_EXTRINSIC][k] for k in [self.EXTRINSIC_FIELDS.BODY_GRAVITY_X,self.EXTRINSIC_FIELDS.BODY_GRAVITY_Y,self.EXTRINSIC_FIELDS.BODY_GRAVITY_Z]], device = self._configuration.th_device)
        orient_err = th.norm(gravity_vec-goal_gravity_vec) # Would be nice to use geodesic distance or somethinglike that

        new_locom_state = { self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : self._locomotion_conf.reward_torque_limit_weight,
                            self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : self._locomotion_conf.reward_position_limit_weight,
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : self._locomotion_conf.reward_velocity_limit_weight,
                            self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT : self._locomotion_conf.reward_velocity_weight,
                            self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT : self._locomotion_conf.reward_acceleration_weight,
                            self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT : self._locomotion_conf.reward_contacts_weight,
                            self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT : self._locomotion_conf.reward_health_weight,
                            self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT : self._locomotion_conf.reward_height_weight,
                            self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT : self._locomotion_conf.reward_pitchnroll_weight,
                            self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT : self._locomotion_conf.reward_tracking_weight,
                            self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT : self._locomotion_conf.reward_torque_weight,
                            self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT : self._locomotion_conf.reward_torquediff_weight,
                            self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR : smoothed_goal_dist,
                            self.LOCOMOTION_FIELDS.HEIGHT_ERR : height_err,
                            self.LOCOMOTION_FIELDS.ORIENT_ERR : orient_err,
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X : goal_vel_rel_xyz[0],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y : goal_vel_rel_xyz[1],
                            self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z : goal_vel_rel_xyz[2],
                            self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT : goal_body_height,
                            self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_X : goal_gravity_vec[0],
                            self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Y : goal_gravity_vec[1],
                            self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Z : goal_gravity_vec[2],
                            self.LOCOMOTION_FIELDS.SUM_IMPULSES : sum_bad_impulses,
                            self.LOCOMOTION_FIELDS.COLLISON_COUNT :collision_count,
                            self.LOCOMOTION_FIELDS.CRASHED : crashed}
        new_inst_state[self.STATE_LOCOMOTION] = new_locom_state


        return new_inst_state
    
    @override
    def computeReward(self, previousState : dict[str,th.Tensor],
                      state : dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : dict[str,th.Tensor] = {}, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = 100
        locom_state = state[self.STATE_LOCOMOTION][0]

        robot_state_norm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT])
        # normpositions = robot_state_norm[:,0]
        normvelocities = robot_state_norm[0][:,1]
        normtorques = robot_state_norm[0][:,2]
        normaccelerations = robot_state_norm[0][:,1] - robot_state_norm[1][:,1]
        normtorquediff = robot_state_norm[0][:,2] - robot_state_norm[1][:,2]

        robot_state_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._safety_limits, warn_limits_violation=False)[0]
        position_safenorm = robot_state_safenorm[:,0]
        velocities_safenorm = robot_state_safenorm[:,1]
        torque_safenorm = robot_state_safenorm[:,2]

        torque_reward       = - th.clamp(th.mean(th.pow(normtorques,2)),-max_rew,max_rew)
        velocity_reward     = - th.clamp(th.mean(th.pow(normvelocities,2)),-max_rew,max_rew)
        acceleration_reward = - th.clamp(th.mean(th.pow(normaccelerations,2)),-max_rew,max_rew)
        torquediff_reward   = - th.clamp(th.mean(th.pow(normtorquediff,2)),-max_rew,max_rew)

        torque_limit_reward   = -th.clamp(th.mean(th.pow(torque_safenorm,50)),-max_rew,max_rew)
        position_limit_reward = -th.clamp(th.mean(th.pow(position_safenorm,50)),-max_rew,max_rew)
        velocity_limit_reward = -th.clamp(th.mean(th.pow(velocities_safenorm,50)),-max_rew,max_rew)

        reward_height = bell_reward(locom_state[self.LOCOMOTION_FIELDS.HEIGHT_ERR],
                                    zero_rew_dist=self._locomotion_conf.height_reward_settle_point)

        reward_pitchnroll = bell_reward(locom_state[self.LOCOMOTION_FIELDS.ORIENT_ERR],
                                        zero_rew_dist=self._locomotion_conf.pitchnroll_reward_settle_point)

        velocity_tracking_err = locom_state[self.LOCOMOTION_FIELDS.SMOOTHED_TRACKING_ERROR]
        velocity_tracking_reward = bell_reward(velocity_tracking_err, zero_rew_dist=self._locomotion_conf.vel_tracking_reward_settle_point)
        # tracking_reward = 1 - th.tanh(tracking_err/50)
        # tracking_reward = 1/(1+goal_dist/0.05)       # 0.50 at 0.05m, 0.35 at 0.10m, 0.2 at 0.2
        # tracking_reward = 1/(1+(goal_dist/0.1)**2) # 0.75 at 0.05m, 0.50 at 0.10m, 0.2 at 0.2

        contacts_reward = - th.clamp(locom_state[self.LOCOMOTION_FIELDS.SUM_IMPULSES], -max_rew, max_rew)

        sub_rewards["reward_tracking"] = velocity_tracking_reward
        sub_rewards["reward_torque"] = torque_reward
        sub_rewards["reward_torque_limit"] = torque_limit_reward
        sub_rewards["reward_torquediff"] = torquediff_reward
        sub_rewards["reward_velocity"] = velocity_reward
        sub_rewards["reward_contacts"] = contacts_reward
        sub_rewards["reward_height"] = reward_height
        sub_rewards["reward_pitchnroll"] = reward_pitchnroll
        sub_rewards["reward_velocity_limit"] = velocity_limit_reward
        sub_rewards["reward_acceleration"] = acceleration_reward
        sub_rewards["reward_position_limit"] = position_limit_reward
        sub_rewards["reward_health"] = th.tensor(1, device=locom_state.device)
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards.items()}

        weights = { "reward_tracking" : locom_state[self.LOCOMOTION_FIELDS.REWARD_TRACKING_WEIGHT],
                    "reward_torque" : locom_state[self.LOCOMOTION_FIELDS.REWARD_TORQUE_WEIGHT],
                    "reward_torque_limit" : locom_state[self.LOCOMOTION_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "reward_torquediff" : locom_state[self.LOCOMOTION_FIELDS.REWARD_TORQUEDIFF_WEIGHT],
                    "reward_velocity" : locom_state[self.LOCOMOTION_FIELDS.REWARD_VELOCITY_WEIGHT],
                    "reward_velocity_limit" : locom_state[self.LOCOMOTION_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT],
                    "reward_acceleration" : locom_state[self.LOCOMOTION_FIELDS.REWARD_ACCELERATION_WEIGHT],
                    "reward_position_limit" : locom_state[self.LOCOMOTION_FIELDS.REWARD_POSITION_LIMIT_WEIGHT],
                    "reward_health" : locom_state[self.LOCOMOTION_FIELDS.REWARD_HEALTH_WEIGHT],
                    "reward_contacts" : locom_state[self.LOCOMOTION_FIELDS.REWARD_CONTACTS_WEIGHT],
                    "reward_height" : locom_state[self.LOCOMOTION_FIELDS.REWARD_HEIGHT_WEIGHT],
                    "reward_pitchnroll" : locom_state[self.LOCOMOTION_FIELDS.REWARD_PITCHNROLL_WEIGHT]}
        for k in sub_rewards:
            sub_rewards[k] = sub_rewards[k]*self._locomotion_conf.reward_scale*weights[k]
        sub_rewards = {k:v.squeeze() for k,v in sub_rewards.items()}
        sub_rewards_unscaled = {k:v.squeeze() for k,v in sub_rewards_unscaled.items()}
        reward = th.as_tensor(sum(list(sub_rewards.values())))

        if dbg_info is not None:
            sub_rewards_scaled = {f"{k}_scaled":v for k,v in sub_rewards.items()}
            sub_rewards_scaled_agg = th.stack([sub_rewards_scaled[k] for k in sub_rewards_scaled.keys()])
            sub_rewards_scaled_agg_names = to_string_tensor([k for k in sub_rewards_scaled.keys()])
            sub_rewards_unscaled_agg = th.stack([sub_rewards_unscaled[k] for k in sub_rewards_unscaled.keys()])
            sub_rewards_unscaled_agg_names = sub_rewards_scaled_agg_names
            dbg_info["sub_rewards_unscaled"] = sub_rewards_unscaled_agg
            dbg_info["sub_rewards_unscaled_labels"] = sub_rewards_unscaled_agg_names
            dbg_info["sub_rewards_scaled"] = sub_rewards_scaled_agg
            dbg_info["sub_rewards_scaled_labels"] = sub_rewards_scaled_agg_names
            dbg_info.update({k:r.cpu().item() if isinstance(r,th.Tensor) else r for k,r in sub_rewards.items()})
            dbg_info["reward"] = reward
        return reward
    

    def _update_stats(self):
        super()._update_stats()

        step_count = self._current_state[self.STATE_INTERNAL][0][self.INTERNAL_FIELDS.STEP_COUNT]
        body_linvel_xyz = self._current_state[self.STATE_EXTRINSIC][0,[self.EXTRINSIC_FIELDS.BODY_LINVEL_X,self.EXTRINSIC_FIELDS.BODY_LINVEL_Y,self.EXTRINSIC_FIELDS.BODY_LINVEL_Z]]
        goal_velocity_xy = self._current_state[self.STATE_LOCOMOTION][0,[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X,self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y]]
        tracking_error = th.linalg.norm(body_linvel_xyz[:2] - goal_velocity_xy)
        body_height = self._current_state[self.STATE_EXTRINSIC][0,self.EXTRINSIC_FIELDS.BODY_POS_Z]
        goal_height = self._current_state[self.STATE_LOCOMOTION][0,self.LOCOMOTION_FIELDS.GOAL_BODY_HEIGHT]
        height_error = th.abs(body_height-goal_height)
        gravity_vec = self._current_state[self.STATE_EXTRINSIC][0,[self.EXTRINSIC_FIELDS.BODY_GRAVITY_X,self.EXTRINSIC_FIELDS.BODY_GRAVITY_Y,self.EXTRINSIC_FIELDS.BODY_GRAVITY_Z]]
        goal_gravity_vec = self._current_state[self.STATE_LOCOMOTION][0,[self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_X,
                                        self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Y,
                                        self.LOCOMOTION_FIELDS.GOAL_BODY_GRAVITY_Z]]
        pitchnroll_err = th.norm(gravity_vec-goal_gravity_vec)
        step_count = self._current_state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.STEP_COUNT].to(th.long).item()
        if step_count>0:
            self._stats["avg_vel_track_err"] = ((self._stats["avg_vel_track_err"]*(step_count-1) + tracking_error.squeeze())/step_count).item()
            self._stats["vel_track_errs"][step_count%len(self._stats["vel_track_errs"])] = tracking_error.cpu().item()
            self._stats["avg_height_track_err"] = ((self._stats["avg_height_track_err"]*(step_count-1) + height_error.squeeze())/step_count).item()
            self._stats["height_track_errs"][step_count%len(self._stats["height_track_errs"])] = height_error.cpu().item()
            self._stats["avg_pitchnroll_err"] = ((self._stats["avg_pitchnroll_err"]*(step_count-1) + pitchnroll_err.squeeze())/step_count).item()
            self._stats["pitchnroll_errs"][step_count%len(self._stats["pitchnroll_errs"])] = pitchnroll_err.cpu().item()
        else:
            self._stats["avg_vel_track_err"] = tracking_error.item()
            self._stats["vel_track_errs"] =  th.full(  size=(int(self._maxStepsPerEpisode/10),),
                                                        fill_value=tracking_error,
                                                        dtype=th.float32, device=self._configuration.th_device)
            self._stats["avg_height_track_err"] = height_error.item()
            self._stats["height_track_errs"] =  th.full(  size=(int(self._maxStepsPerEpisode/10),),
                                                        fill_value=height_error.item(),
                                                        dtype=th.float32, device=self._configuration.th_device)
            
            self._stats["avg_pitchnroll_err"] = height_error.item()
            self._stats["pitchnroll_errs"] =  th.full(  size=(int(self._maxStepsPerEpisode/10),),
                                                        fill_value=pitchnroll_err.item(),
                                                        dtype=th.float32, device=self._configuration.th_device)
            
    @override
    def getInfo(self,state) -> dict[Any,Any]:
        i = super().getInfo(state=state)
        internal_state = state[self.STATE_LOCOMOTION][0]
        i["goal_rel_x"] = internal_state[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_X]
        i["goal_rel_y"] = internal_state[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Y]
        i["goal_rel_z"] = internal_state[self.LOCOMOTION_FIELDS.GOAL_VELOCITY_REL_Z]
        i["goal_x"] = self._locomotion_episode_config.goal_velocity_xy[0]
        i["goal_y"] = self._locomotion_episode_config.goal_velocity_xy[1]
        i["avg_vel_track_err"] = self._stats["avg_vel_track_err"]
        i["avg10_vel_track_errs"] = th.mean(self._stats["vel_track_errs"])
        i["avg_height_track_err"] = self._stats["avg_height_track_err"]
        i["avg10_height_track_errs"] = th.mean(self._stats["height_track_errs"])
        i["avg_pitchnroll_err"] = self._stats["avg_pitchnroll_err"]
        i["avg10_pitchnroll_errs"] = th.mean(self._stats["pitchnroll_errs"])
        i["success"] = i["avg10_vel_track_errs"] < 0.05

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_LOCOMOTION]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["state_"+substate+"_labels"] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                i["statenorm_"+substate+"_labels"] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())

        return i
    
    def _set_current_ep_config(self, reset_options : dict = {}):
        super()._set_current_ep_config(reset_options=reset_options)
        
        goal_speed = unnormalize(th.rand(size=(1,),generator=self._rng, device=self._configuration.th_device)*2-1,
                                    min=self._locomotion_conf.goal_speed_minmax[0],
                                    max=self._locomotion_conf.goal_speed_minmax[1])
        goal_direction = th.rand((1,),generator=self._rng, device=self._configuration.th_device)*math.pi*2
        goal_velocity_xy = reset_options.get("goal_velocity_xy", goal_speed*th.cat([th.cos(goal_direction), th.sin(goal_direction)]))
                                             
        # goal_velocity_xy = th.as_tensor((10.,0.), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)

        self._locomotion_episode_config = LocomotionEnv.EpisodeLocomConfiguration(goal_velocity_xy=goal_velocity_xy)
        self.set_max_episode_steps(self._current_episode_config.max_ep_steps)

    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if super().reachedTerminalState(previousState, state):
            return th.as_tensor(True, device=self._configuration.th_device)
        else:
            return state[self.STATE_LOCOMOTION][0][self.LOCOMOTION_FIELDS.CRASHED].squeeze()

    def initializeEpisode(self, options=...) -> None:
        super().initializeEpisode(options)
        if self._locomotion_conf.use_contacts:
            if not isinstance(self._adapter,PyBulletAdapter):
                raise RuntimeError(f"Contacts are supported only in pybullet for now")
            self._adapter.monitor_contacts([(self._configuration.robot_name, None)])

    def _simulation_initialization(self):
        if not isinstance(self._adapter, BaseSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        super()._simulation_initialization()
        if self._configuration.show_goal:
            body_state : LinkState = self._adapter.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
            q = quaternion.from_euler_angles([0.0,0.0,np.arctan2(*self._locomotion_episode_config.goal_velocity_xy[[1,0]].cpu().numpy())])
            self._adapter.setLinksStateDirect({self._arrow_base :
                                                            LinkState( position_xyz = th.tensor((body_state.pose.position[0],
                                                                                                 body_state.pose.position[1],
                                                                                                 body_state.pose.position[2]+0.1), device=self._configuration.th_device),
                                                                        orientation_xyzw = th.tensor((q.x, q.y, q.z, q.w), device=self._configuration.th_device),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
            
    @override
    def getUiRendering(self) -> tuple[np.ndarray | th.Tensor | None, float]:
        if isinstance(self._adapter, BaseSimulationAdapter):
            body_state : LinkState = self._adapter.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
            q = quaternion.from_euler_angles([0.0,0.0,np.arctan2(*self._locomotion_episode_config.goal_velocity_xy[[1,0]].cpu().numpy())])
            self._adapter.setLinksStateDirect({self._arrow_base :
                                                            LinkState( position_xyz = th.tensor((body_state.pose.position[0],
                                                                                                    body_state.pose.position[1],
                                                                                                    0.0), device=self._configuration.th_device),
                                                                        orientation_xyzw = th.tensor((q.x, q.y, q.z, q.w), device=self._configuration.th_device),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
        return super().getUiRendering()