
from __future__ import annotations
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.envs.ControlledEnv import ControlledEnv
from adarl.utils.robot_helpers import Robot
from adarl.utils.utils import to_string_tensor
from adarl.utils.state_helper import    JointImpedanceActionHelper, ThBoxStateHelper,\
                                        RobotStateHelper, RobotStatsStateHelper,\
                                        StateNoiseGenerator, DictStateHelper, unnormalize
from adarl.utils.tensor_trees import map_tensor_tree
from adarl.utils.utils import build_pose, JointState, Pose, LinkState
from dataclasses import dataclass
from gymnasium import Space
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.tensor_trees
import adarl.utils.utils
import copy
import dataclasses
import numpy as np
import torch as th

class LocomotionEnv(ControlledEnv[BaseJointImpedanceAdapter]):

    @dataclass
    class Configuration:
        action_delay_mustd : th.Tensor
        action_exp_smoothing_1s : float
        action_noise_mustd : th.Tensor
        control_mode : JointImpedanceActionHelper.CONTROL_MODES
        controlled_joints : Sequence[tuple[str,str]]
        frame_stack_length : int
        goal_err_exp_smoothing_1s : float
        history_length : int
        homing_joint_pose : dict[tuple[str,str],th.Tensor]
        joint_physical_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_damping : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_stiffness : dict[tuple[str,str],th.Tensor]
        main_body_link : tuple[str,str]
        model_urdf_string : str
        obs_dtype : th.dtype
        obs_noise_ep_mustd : th.Tensor
        obs_noise_step_std : th.Tensor
        original_max_epsteps : int
        randomize_initial_pose : bool
        real : bool
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
        robot_name : str
        safe_damping : float
        safe_stiffness : float
        seed : int
        show_goal : bool
        stepLength_sec : float
        stop_on_safety : bool
        th_device : th.device
        ui_camera_name : str
        ui_camera_link : tuple[str,str]

    metadata = {'render.modes': ['rgb_array']}
    # STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT = "action"
    STATE_ROBOT = "robot"
    STATE_ROBOT_STATS = "robot_stats"
    STATE_EXTRINSIC = "extrinsic"
    STATE_INTERNAL = "internal"
    
    
    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", [  "GOAL_VELOCITY_X",
                                                    "GOAL_VELOCITY_Y",
                                                    "SAFETY_TRIGGERED",
                                                    "STEP_COUNT",
                                                    "REWARD_TRACKING_WEIGHT",
                                                    "REWARD_TORQUE_WEIGHT",
                                                    "REWARD_TORQUE_LIMIT_WEIGHT",
                                                    "REWARD_VELOCITY_WEIGHT",
                                                    "REWARD_ACCELERATION_WEIGHT",
                                                    "REWARD_POSITION_LIMIT_WEIGHT",
                                                    "REWARD_VELOCITY_LIMIT_WEIGHT",
                                                    "REWARD_TORQUEDIFF_WEIGHT",
                                                    "SMOOTHED_TRACKING_ERROR"], start=0)

    EXTRINSIC_FIELDS = IntEnum("EXTRINSIC_FIELS", ["BODY_VEL_X",
                                                   "BODY_VEL_Y",
                                                   "BODY_VEL_Z"], start=0)
    ACT_FIELDS = IntEnum("ACT_FIELDS", ["ACTION"], start=0)
    
    JOINT_FILTERS = Enum("JointFilters",["ALL_REVOLUTE",
                                         "ALL"])
    
    joint_filters = {JOINT_FILTERS.ALL : lambda joint_name, robot_model: True,
                     JOINT_FILTERS.ALL_REVOLUTE : lambda joint_name, robot_model: robot_model.get_joint_properties([joint_name])[joint_name]["type"] == Robot.JOINT_TYPES.REVOLUTE}

    class State(TypedDict):
        action : th.Tensor
        robot : th.Tensor
        robot_stats : th.Tensor
        extrinsic : th.Tensor
        internal : th.Tensor

    @dataclass
    class EpisodeConfiguration:
        goal_velocity_xy : th.Tensor
        initial_joint_pose : th.Tensor
        max_ep_steps : th.Tensor

    @dataclass
    class Statistics:
        tracking_errors : th.Tensor
        avg_tracking_error : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        rewards : dict = dataclasses.field(default_factory=lambda: {})

    def  __init__(self, action_delay_mustd : tuple[float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseJointImpedanceAdapter,
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | JOINT_FILTERS],
                        goal_err_smoothing_halflife_sec : float,
                        maxStepsPerEpisode,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        obs_noise_ep_mustd : Sequence[float] | th.Tensor, 
                        obs_noise_step_std : Sequence[float] | float | th.Tensor,
                        reward_acceleration_weight : float,
                        reward_contacts_weight : float,
                        reward_energy_weight : float,
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
                        seed,
                        stepLength_sec,
                        step_precision_tolerance : float,
                        stop_on_safety : bool,
                        th_device : th.device
                        ):
        
        self._rng = th.Generator(device=th_device)
        self._spawned = False
        self._robot_model = Robot(adarl.utils.utils.compile_xacro_string(  model_definition_string=robot_urdf_string))

        controlled_joints_str = []
        for j in controlled_joints:
            if isinstance(j, str):
                controlled_joints_str.append(j)
            elif isinstance(j, LocomotionEnv.JOINT_FILTERS):
                for jn in self._robot_model.get_joint_names():
                    if LocomotionEnv.joint_filters[j](jn,self._robot_model):
                        controlled_joints_str.append(jn)

        controlled_joints_rn : list[tuple[str,str]] = [(robot_name,jn) for jn in controlled_joints_str]
        phys_limits_minmax_pve = {(robot_name,k):th.as_tensor(l,device=th_device) 
                                    for k,l in self._robot_model.get_joint_limits(controlled_joints_str).items()}
        safe_limits_minmax_pve = {k:(lims_minmax-0.5*(lims_minmax[1]+lims_minmax[0]))*safety_limits_factor+0.5*(lims_minmax[1]+lims_minmax[0])
                                    for k,lims_minmax in phys_limits_minmax_pve.items()}
        ggLog.info(f"phys_limits_minmax_pve = \n"+"\n".join([str(jn_lim) for jn_lim in phys_limits_minmax_pve.items()]))
        ggLog.info(f"safe_limits_minmax_pve = \n"+"\n".join([str(jn_lim) for jn_lim in safe_limits_minmax_pve.items()]))
        if isinstance(minmax_stiffness, tuple):
            minmax_stiffness_thdict = {k:th.as_tensor(minmax_stiffness, device=th_device) for k in phys_limits_minmax_pve.keys()}
        else:
            minmax_stiffness_thdict = {(robot_name,k):th.as_tensor(minmax, device=th_device) for k,minmax in minmax_stiffness.items()}
        if isinstance(minmax_damping, tuple):
            minmax_damping_thdict = {k:th.as_tensor(minmax_damping, device=th_device) for k in phys_limits_minmax_pve.keys()}
        else:
            minmax_damping_thdict = {(robot_name,k):th.as_tensor(minmax, device=th_device) for k,minmax in minmax_damping.items()}
        action_exp_smoothing_1s = 0.5**(1/action_smoothing_halflife_sec) if action_smoothing_halflife_sec>0 else 0.0
        goal_err_exp_smoothing_1s = 0.5**(1/goal_err_smoothing_halflife_sec) if goal_err_smoothing_halflife_sec>0 else 0.0
        homing_joint_pose = {jn: unnormalize(0.75, safe_limits_minmax_pve[jn][0,0].item(), safe_limits_minmax_pve[jn][1,0].item()) for jn in controlled_joints_rn}
        ggLog.info(f"homing_joint_pose = "+"\n".join([f"{jn}:{p}" for jn,p in homing_joint_pose.items()]))

        self._configuration = LocomotionEnv.Configuration(  action_delay_mustd = th.as_tensor(action_delay_mustd, device=th_device),
                                                            action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                            action_noise_mustd = th.as_tensor(action_noise_mustd, device=th_device),
                                                            control_mode = JointImpedanceActionHelper.CONTROL_MODES[control_mode.upper()],
                                                            controlled_joints = controlled_joints_rn,
                                                            frame_stack_length = 3,
                                                            goal_err_exp_smoothing_1s = goal_err_exp_smoothing_1s,
                                                            history_length = 3,
                                                            homing_joint_pose = homing_joint_pose,
                                                            joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                            joint_safe_limits_minmax_damping = minmax_damping_thdict,
                                                            joint_safe_limits_minmax_pve = safe_limits_minmax_pve,
                                                            joint_safe_limits_minmax_stiffness = minmax_stiffness_thdict,
                                                            main_body_link=(robot_name,robot_main_body_link),
                                                            model_urdf_string=robot_urdf_string,
                                                            obs_dtype = th.float32,
                                                            obs_noise_ep_mustd = th.as_tensor(obs_noise_ep_mustd, device=th_device),
                                                            obs_noise_step_std = th.as_tensor(obs_noise_step_std, device=th_device),
                                                            original_max_epsteps = maxStepsPerEpisode,
                                                            randomize_initial_pose = False,
                                                            real = False,
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
                                                            robot_name = robot_name,
                                                            safe_damping = safe_damping,
                                                            safe_stiffness = safe_stiffness,
                                                            show_goal = True,
                                                            stepLength_sec = stepLength_sec,
                                                            stop_on_safety = stop_on_safety,
                                                            th_device = th_device,
                                                            seed = seed,
                                                            ui_camera_name="simple_camera",
                                                            ui_camera_link = ("simple_camera", "simple_camera_link")
                                                            )

        self._safe_limits_minmax_j_pve = th.stack([safe_limits_minmax_pve[jn] for jn in controlled_joints_rn], dim=1)
        self._action_helper= JointImpedanceActionHelper(control_mode=self._configuration.control_mode,
                                joints=controlled_joints_rn,
                                joints_minmax_pvesd={jn:th.cat([safe_limits_minmax_pve[jn],
                                                                minmax_stiffness_thdict[jn].unsqueeze(1),
                                                                minmax_damping_thdict[jn].unsqueeze(1)], dim=1) 
                                                        for jn in controlled_joints_rn},
                                safe_stiffness=th.as_tensor([self._configuration.safe_stiffness]).repeat(len(controlled_joints_rn)),
                                safe_damping=th.as_tensor([self._configuration.safe_stiffness]).repeat(len(controlled_joints_rn)),
                                th_device=self._configuration.th_device)

        robot_state_helper = RobotStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                              stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                              damping_minmax=self._configuration.joint_safe_limits_minmax_damping,
                                              obs_dtype=self._configuration.obs_dtype,
                                              th_device=self._configuration.th_device,
                                              history_length=self._configuration.frame_stack_length)
        robot_stats_state_helper = RobotStatsStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                                        obs_dtype=self._configuration.obs_dtype,
                                                        th_device=self._configuration.th_device)
        internal_state_helper =   ThBoxStateHelper(field_names=[e.value for e in self.INTERNAL_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.INTERNAL_FIELDS.GOAL_VELOCITY_X : [-10,10],
                                                                self.INTERNAL_FIELDS.GOAL_VELOCITY_Y : [-10,10],
                                                                self.INTERNAL_FIELDS.SAFETY_TRIGGERED : [0,1],
                                                                self.INTERNAL_FIELDS.STEP_COUNT : [-1,1000_000_000],                                                                
                                                                self.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT : [0,10],
                                                                self.INTERNAL_FIELDS.SMOOTHED_TRACKING_ERROR : [0,10]},
                                                observable_fields=[self.INTERNAL_FIELDS.GOAL_VELOCITY_X,self.INTERNAL_FIELDS.GOAL_VELOCITY_Y])
        extrinsic_state_helper =  ThBoxStateHelper(field_names=[e.value for e in self.EXTRINSIC_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.EXTRINSIC_FIELDS.BODY_VEL_X : [-100,100],
                                                                self.EXTRINSIC_FIELDS.BODY_VEL_Y : [-100,100],
                                                                self.EXTRINSIC_FIELDS.BODY_VEL_Z : [-100,100]},
                                               history_length=self._configuration.frame_stack_length)
        act_history_state_helper = ThBoxStateHelper(field_names=[self.ACT_FIELDS.ACTION],
                                               obs_dtype=th.float32,
                                               th_device=th_device,
                                               field_size=(self._action_helper.action_len(),),
                                               fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]})
        robot_state_noise =  StateNoiseGenerator(robot_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = self._configuration.obs_noise_ep_mustd,
                                            step_std = self._configuration.obs_noise_step_std)
        extrinsic_state_noise =  StateNoiseGenerator(extrinsic_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = self._configuration.obs_noise_ep_mustd,
                                            step_std = self._configuration.obs_noise_step_std)        
        self._state_helper = DictStateHelper({self.STATE_ROBOT : robot_state_helper,
                                              self.STATE_ROBOT_STATS : robot_stats_state_helper,
                                              self.STATE_EXTRINSIC : extrinsic_state_helper,
                                              self.STATE_INTERNAL : internal_state_helper,
                                              self.STATE_ACT: act_history_state_helper},
                                              observable_fields=[   self.STATE_ROBOT,
                                                                    self.STATE_EXTRINSIC,
                                                                    self.STATE_INTERNAL],
                                              noise = {
                                                    self.STATE_ROBOT : robot_state_noise,
                                                    self.STATE_EXTRINSIC : extrinsic_state_noise},
                                              flatten_in_obs=[   self.STATE_ROBOT,
                                                                self.STATE_EXTRINSIC,
                                                                self.STATE_INTERNAL],
                                              flattened_part_name="vec")

        self._safety_limits = robot_state_helper.build_robot_limits(joint_limit_minmax_pve=self._configuration.joint_safe_limits_minmax_pve,
                                                                    stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                                                    damping_minmax=self._configuration.joint_safe_limits_minmax_damping)
        
        state_space = self._state_helper.get_space()
        observation_space = self._state_helper.get_obs_space()
        action_space = self._action_helper.action_space(seed=seed)

        self._current_state : LocomotionEnv.State = LocomotionEnv.State(action=th.empty(0),
                                                              robot=th.empty(0),
                                                              robot_stats=th.empty(0),
                                                              extrinsic=th.empty(0),
                                                              internal=th.empty(0))

        super().__init__(maxStepsPerEpisode,
                         stepLength_sec,
                         adapter,
                         action_space,
                         observation_space,
                         state_space,
                         startSimulation = True,
                         step_precision_tolerance = step_precision_tolerance)
        
        self._environmentController.set_monitored_links([self._configuration.main_body_link])
        self._environmentController.startup()


    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    @override
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
            jimp_pvesd = self._action_helper._action_to_pvesd(action)
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
            





    @override
    def computeReward(self, previousState : dict[str,th.Tensor],
                      state : dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : dict[str,th.Tensor] = {}, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        max_rew = 100

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

        torque_reward = - th.clamp(th.mean(th.pow(normtorques,4)),-max_rew,max_rew)
        velocity_reward = - th.clamp(th.mean(th.pow(normvelocities,4)),-max_rew,max_rew)
        acceleration_reward = - th.clamp(th.mean(th.pow(normaccelerations,2)),-max_rew,max_rew)
        torquediff_reward = - th.clamp(th.mean(th.pow(normtorquediff,2)),-max_rew,max_rew)

        torque_limit_reward = -th.clamp(th.mean(th.pow(torque_safenorm,50)),-max_rew,max_rew)
        position_limit_reward = -th.clamp(th.mean(th.pow(position_safenorm,50)),-max_rew,max_rew)
        velocity_limit_reward = -th.clamp(th.mean(th.pow(velocities_safenorm,50)),-max_rew,max_rew)


        internal_state = state[self.STATE_INTERNAL][0]
        goal_dist = internal_state[self.INTERNAL_FIELDS.SMOOTHED_TRACKING_ERROR]
        # tracking_reward = 1 - goal_dist
        # tracking_reward = 1/(1+goal_dist/0.05)       # 0.50 at 0.05m, 0.35 at 0.10m, 0.2 at 0.2
        tracking_reward = 1/(1+(goal_dist/0.1)**2) # 0.75 at 0.05m, 0.50 at 0.10m, 0.2 at 0.2

        sub_rewards["reward_tracking"] = tracking_reward
        sub_rewards["reward_torque"] = torque_reward
        sub_rewards["reward_torque_limit"] = torque_limit_reward
        sub_rewards["reward_torquediff"] = torquediff_reward
        sub_rewards["reward_velocity"] = velocity_reward
        sub_rewards["reward_velocity_limit"] = velocity_limit_reward
        sub_rewards["reward_acceleration"] = acceleration_reward
        sub_rewards["reward_position_limit"] = position_limit_reward
        sub_rewards["reward_health"] = th.tensor(1, device=internal_state.device)
        sub_rewards_unscaled = {f"{k}_unscaled":v for k,v in sub_rewards.items()}

        weights = { "reward_tracking" : internal_state[self.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT],
                    "reward_torque" : internal_state[self.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT],
                    "reward_torque_limit" : internal_state[self.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT],
                    "reward_torquediff" : internal_state[self.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT],
                    "reward_velocity" : internal_state[self.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT],
                    "reward_velocity_limit" : internal_state[self.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT],
                    "reward_acceleration" : internal_state[self.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT],
                    "reward_position_limit" : internal_state[self.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT],
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
            dbg_info["sub_rewards_unscaled"] = sub_rewards_unscaled_agg
            dbg_info["sub_rewards_unscaled_labels"] = sub_rewards_unscaled_agg_names
            dbg_info["sub_rewards_scaled"] = sub_rewards_scaled_agg
            dbg_info["sub_rewards_scaled_labels"] = sub_rewards_scaled_agg_names
            dbg_info.update({k:r.cpu().item() if isinstance(r,th.Tensor) else r for k,r in sub_rewards.items()})
            dbg_info["reward"] = reward
        return reward
    





    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------
    @override
    def initializeEpisode(self, options = {}) -> None:

        self._current_state : LocomotionEnv.State = self._state_helper.reset_state(initial_values={
            self.STATE_EXTRINSIC : th.tensor(0.0),
            self.STATE_ROBOT : th.tensor(0.0),
            self.STATE_ROBOT_STATS : th.tensor(0.0),
            self.STATE_ACT : th.tensor(0.0),
            self.STATE_INTERNAL : th.tensor(0.0)
        })
        self._current_state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.STEP_COUNT] = th.tensor(-1.)
        self._last_obs = None

        if not self._spawned and isinstance(self._environmentController, BaseSimulationAdapter):
            robot_pose = build_pose(0,0,1,0,0,0,1)
            camera_pose = build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707)
            red_ball_pose = robot_pose
            self._spawned = True
            camera_file = adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro")
            if isinstance(self._environmentController, PyBulletAdapter):
                self._environmentController.spawn_model(model_definition_string=self._configuration.model_urdf_string,
                                                        model_name=self._configuration.robot_name,
                                                        pose=robot_pose,
                                                        model_format="urdf")
            self._environmentController.spawn_model(model_file=camera_file,
                                                    model_name="simple_camera",
                                                    pose=camera_pose,
                                                    model_format="sdf.xacro",
                                                    model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._intendedStepLength_sec})
            if self._configuration.show_goal:
                self._environmentController.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/red_intangible_ball.urdf.xacro"),
                                                        model_name="red_ball",
                                                        pose=red_ball_pose,
                                                        model_format="urdf.xacro",
                                                        model_kwargs={"add_world_link":str(isinstance(self._environmentController, PyBulletAdapter))})
            # self._robot_model.disable_tree_self_collisions("rail_joint")
            # self._robot_model.remove_collision_pairs([("rail_link_0","slider_link_0")])            
            # self._ground_co_id = self._robot_model.add_collision_box( pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
            #                                                         collision_box_size_xyz=(10,10,1),
            #                                                         collision_obj_id="ground_collision")
            self._environmentController.set_monitored_joints(self._configuration.controlled_joints)

        
        self._set_current_ep_config(reset_options = options)

        body_state = self._environmentController.getLinksState([self._configuration.main_body_link], use_com_frame=True)[self._configuration.main_body_link]
        initial_tracking_error = th.linalg.norm(body_state.pos_velocity_xyz[:2]-self._current_episode_config.goal_velocity_xy).cpu().item()
        self._stats = copy.deepcopy(self.Statistics(tracking_errors=th.full(  size=(int(self._maxStepsPerEpisode/10),),
                                                                            fill_value=initial_tracking_error,
                                                                            dtype=th.float32, device=self._configuration.th_device)))
        
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            self._realworld_initialization()
        self._last_out_action = th.clamp(self._action_helper._pvesd_to_action(self._last_sent_pvesd), min=-1, max=1)
        # ggLog.info(f"initial action {self._last_out_action}, pvesd = {self._last_sent_pvesd}")

        self._update_state()
        self._update_stats()


    def _set_current_ep_config(self, reset_options : dict):
        if reset_options is not None:
            reset_options = {}
        # Having conservative values here will not make the policy learn to behave nice in unfeasible cases
        # Having too broad value will have unfeasible cases in training
        goal_velocity_xy = th.tanh(th.randn(size=(2,),generator=self._rng)/5)*5 * 2        
        if "goal_velocity_xy" in reset_options: goal_velocity_xy = reset_options["goal_velocity_xy"]
        goal_velocity_xy = th.as_tensor((1.,0.), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)

        maxStepsPerEpisode = reset_options.get("max_ep_steps", self._configuration.original_max_epsteps)
           
        found_good_configuration = False
        if self._configuration.randomize_initial_pose:
            raise NotImplementedError()
        if not found_good_configuration:
            initial_joint_pose = th.as_tensor([self._configuration.homing_joint_pose[jn] for jn in self._configuration.controlled_joints], device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
        
        # ggLog.info(f"{os.getpid()} init: chosen jpos= {rail_hip_knee_pos.cpu().tolist()}")        
        self._current_episode_config = LocomotionEnv.EpisodeConfiguration(   goal_velocity_xy=goal_velocity_xy,
                                                                        initial_joint_pose = initial_joint_pose,
                                                                        max_ep_steps = maxStepsPerEpisode)
        self.set_max_episode_steps(self._current_episode_config.max_ep_steps)

    def _realworld_initialization(self):
        raise NotImplementedError()
    
    def _simulation_initialization(self):
        if not isinstance(self._environmentController, BaseSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        
        self._environmentController.setLinksStateDirect({self._configuration.main_body_link :
                                                        LinkState( position_xyz = th.tensor((0.0, 0.0, 0.75), device=self._configuration.th_device),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0), device=self._configuration.th_device),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
        self._environmentController.setJointsStateDirect({jn:JointState(position=self._configuration.homing_joint_pose[jn],
                                                                        rate = 0,
                                                                        effort = 0) for jn in self._configuration.controlled_joints})
        start_jimp : dict[tuple[str,str], tuple] = {jn:(self._configuration.homing_joint_pose[jn],
                                                        0,
                                                        0,
                                                        self._configuration.safe_stiffness,
                                                        self._configuration.safe_damping) 
                                                    for jn in self._configuration.controlled_joints}         
        self._environmentController.setJointsImpedanceCommand(start_jimp)
        self._environmentController.apply_joint_impedances(start_jimp)
        self._last_sent_pvesd = start_jimp

        if self._configuration.show_goal:
            self._environmentController.setLinksStateDirect({self._red_ball_base :
                                                            LinkState( position_xyz = th.tensor((self._current_episode_config.goal_velocity_xy[0],
                                                                                                 self._current_episode_config.goal_velocity_xy[1],
                                                                                                 0.5), device=self._configuration.th_device),
                                                                        orientation_xyzw = th.tensor((0.,0.,0.,1.0), device=self._configuration.th_device),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
    

    @override
    def buildSimulation(self):
        envCtrlName = type(self._environmentController).__name__
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._environmentController.build_scenario()
            self._red_ball_base = ("red_ball","world")
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._configuration.real:
                raise NotImplementedError()
            else:
                self._environmentController.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path("jumping_leg",
                                                                                                                          "gazebo/all_gazebo_xbot.launch"),
                                                           launch_file_args={"gui":"false"})
                self._red_ball_base = ("red_ball","sphere_link")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")

    @override
    def _destroySimulation(self):
        self._environmentController.destroy_scenario()




    # --------------------------------------------------------------------------------------------------------------------
    # State & Observation
    # --------------------------------------------------------------------------------------------------------------------
    @override
    def getUiRendering(self) -> tuple[np.ndarray | th.Tensor | None, float]:
        try:
            if isinstance(self._environmentController, BaseSimulationAdapter):
                body_state : LinkState = self._environmentController.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
                self._environmentController.setLinksStateDirect({self._configuration.ui_camera_link :
                                                                LinkState( position_xyz = th.tensor((body_state.pose.position[0],
                                                                                                    body_state.pose.position[1]+2.5,
                                                                                                    0.7), device=self._configuration.th_device),
                                                                            orientation_xyzw = th.tensor((0.,0.,-0.707,0.707), device=self._configuration.th_device),
                                                                            pos_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                            ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
            img, time = self._environmentController.getRenderings([self._configuration.ui_camera_name])[self._configuration.ui_camera_name]
            if img is None:
                time = -1
            return img, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return None, -1
    
    @override
    def getObservation(self, state) -> dict[Any, th.Tensor]:
        if not adarl.utils.tensor_trees.is_all_finite(state):
            ggLog.warn(f"Non-finite values in state {state}")
        self._last_obs = self._state_helper.observe(state)
        if th.any(th.abs(self._last_obs["vec"]) > 100):
            raise RuntimeError(f"Values over 100 in obs {self._last_obs}")
        if not adarl.utils.tensor_trees.is_all_finite(self._last_obs):
            raise RuntimeError(f"Non-finite values in obs {self._last_obs}")
        return self._last_obs

    @override
    def getState(self) -> dict[Any, th.Tensor]:
        return self._current_state
    

    @override
    def performStep(self):
        super().performStep()
        self._update_state()
        self._update_stats()
        self._last_step_simtime = self._environmentController.getEnvTimeFromReset()


    def _update_state(self):
        # ggLog.info(f"_stepCounter = {self._stepCounter}")
        
        jstates = self._environmentController.getJointsState(requestedJoints=self._configuration.controlled_joints)
        lstates = self._environmentController.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)
        body_vel_xyz = lstates[self._configuration.main_body_link].pos_velocity_xyz


        stats_minmaxavgstd_j_pve = self._environmentController.get_joints_state_step_stats()
        if not th.all(th.isfinite(stats_minmaxavgstd_j_pve)):
            raise RuntimeError(f"non finite values in joint stats: stats_minmaxavgstd_hipknee_pve = {stats_minmaxavgstd_j_pve}")

        internal_state = self._current_state[self.STATE_INTERNAL][0]
        step_count = internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        if step_count!=-1 and internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0:
            safety_triggered = True
        else:
            triggered_limits = th.logical_or(stats_minmaxavgstd_j_pve[0] < self._safe_limits_minmax_j_pve[0],
                                             stats_minmaxavgstd_j_pve[1] > self._safe_limits_minmax_j_pve[1])
            safety_triggered = th.any(triggered_limits)
            if safety_triggered:       
                elements = np.array([[f"{jn}_pos",f"{jn}_vel",f"{jn}_eff"] for jn in self._configuration.controlled_joints], dtype=object) #type: ignore
                triggered = []
                for i in np.ndindex(elements.shape):
                    if triggered_limits[i]:
                        triggered.append(elements[i])
                ggLog.info( f"SAFETY TRIGGERED:\n"
                            f"    triggered ({len(triggered)}) = {triggered}\n"
                            # f"    joints_minmax = \n{stats_minmaxavgstd_j_pve[:2]}\n"
                            # f"    j_safety_lims  = \n{self._safe_limits_minmax_j_pve} "
                            )



        tracking_error = th.linalg.norm(body_vel_xyz[:2]-self._current_episode_config.goal_velocity_xy).cpu().item()
        prev_tracking_error = internal_state[self.INTERNAL_FIELDS.SMOOTHED_TRACKING_ERROR]
        alpha = self._configuration.goal_err_exp_smoothing_1s**(self._configuration.stepLength_sec)
        if step_count > 0:
            smoothed_goal_dist = tracking_error*(1-alpha) + prev_tracking_error*alpha
        else:
            smoothed_goal_dist = 1

        new_internal_state = {
                        self.INTERNAL_FIELDS.REWARD_TORQUE_LIMIT_WEIGHT : self._configuration.reward_torque_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_POSITION_LIMIT_WEIGHT : self._configuration.reward_position_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_VELOCITY_LIMIT_WEIGHT : self._configuration.reward_velocity_limit_weight,
                        self.INTERNAL_FIELDS.REWARD_VELOCITY_WEIGHT : self._configuration.reward_velocity_weight,
                        self.INTERNAL_FIELDS.REWARD_ACCELERATION_WEIGHT : self._configuration.reward_acceleration_weight,
                        self.INTERNAL_FIELDS.REWARD_TRACKING_WEIGHT : self._configuration.reward_tracking_weight,
                        self.INTERNAL_FIELDS.REWARD_TORQUE_WEIGHT : self._configuration.reward_torque_weight,
                        self.INTERNAL_FIELDS.REWARD_TORQUEDIFF_WEIGHT : self._configuration.reward_torquediff_weight,
                        self.INTERNAL_FIELDS.SAFETY_TRIGGERED : 1 if safety_triggered else 0,
                        self.INTERNAL_FIELDS.SMOOTHED_TRACKING_ERROR : smoothed_goal_dist,
                        self.INTERNAL_FIELDS.STEP_COUNT : step_count+1,
                        self.INTERNAL_FIELDS.GOAL_VELOCITY_X : self._current_episode_config.goal_velocity_xy[0],
                        self.INTERNAL_FIELDS.GOAL_VELOCITY_Y : self._current_episode_config.goal_velocity_xy[1]}
        new_robot_state = {jn : th.concat([ jstates[jn].position[[0]],
                                            jstates[jn].rate[[0]],
                                            jstates[jn].effort[[0]],
                                            th.as_tensor(self._last_sent_pvesd[jn])])
                                for jn in self._configuration.controlled_joints}
        new_robot_stats_state = {jname : stats_minmaxavgstd_j_pve[:,i,:].flatten()
                                 for i,jname in enumerate(self._environmentController.get_monitored_joints())}
        if th.any(th.concat([new_robot_state[jn][6:] for jn in self._configuration.controlled_joints])<0):
            ggLog.warn(f"negative gains in new_robot_state = {new_robot_state}")
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_VEL_X : body_vel_xyz[0],
                                self.EXTRINSIC_FIELDS.BODY_VEL_Y : body_vel_xyz[1],
                                self.EXTRINSIC_FIELDS.BODY_VEL_Z : body_vel_xyz[2]}
        new_act_state = {self.ACT_FIELDS.ACTION : self._last_out_action}
        instantaneous_state = { self.STATE_EXTRINSIC    : new_extrinsic_state,
                                self.STATE_ACT          : new_act_state,
                                self.STATE_INTERNAL     : new_internal_state,
                                self.STATE_ROBOT        : new_robot_state,
                                self.STATE_ROBOT_STATS  : new_robot_stats_state}              
        
        
        if step_count <= 0:
            self._current_state = self._state_helper.reset_state(instantaneous_state)
        else:
            self._state_helper.update(instantaneous_state, state=self._current_state)
        
        map_tensor_tree(self._current_state, lambda t: t.detach().clone())



    def _update_stats(self):
        rew_dbg_info = {}
        self.computeReward( {},
                            self._current_state, 
                            th.tensor([]), 
                            env_conf=self.get_configuration(),
                            dbg_info=rew_dbg_info)
        body_vel_xyz = self._current_state[self.STATE_EXTRINSIC][0][0:3]
        tracking_error = th.linalg.norm(body_vel_xyz-self._current_episode_config.goal_velocity_xy)
        if self._stepCounter>0:
            self._stats.avg_tracking_error = (self._stats.avg_tracking_error*(self._stepCounter-1) + tracking_error.squeeze())/self._stepCounter
        self._stats.tracking_errors[self._stepCounter%len(self._stats.tracking_errors)] = tracking_error.cpu().item()
        self._stats.rewards = rew_dbg_info
        return rew_dbg_info
        
    @override
    def getInfo(self,state) -> dict[Any,Any]:
        i = super().getInfo(state=state)
        robot_state = state[self.STATE_ROBOT][0]
        internal_state = state[self.STATE_INTERNAL][0]
        i["goal_x"] = internal_state[self.INTERNAL_FIELDS.GOAL_VELOCITY_X]
        i["goal_y"] = internal_state[self.INTERNAL_FIELDS.GOAL_VELOCITY_Y]
        i.update(dataclasses.asdict(self._stats))
        i["avg_tracking_error"] = self._stats.avg_tracking_error
        i["avg10_tracking_errors"] = th.mean(self._stats.tracking_errors)
        i["step_count"] = self._stepCounter

        statenorm = self._state_helper.normalize(state)
        for substate in [self.STATE_ROBOT, self.STATE_EXTRINSIC, self.STATE_INTERNAL, self.STATE_ACT, self.STATE_ROBOT_STATS]:
            i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
            i["state_"+substate+"_labels"] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
            i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
            i["statenorm_"+substate+"_labels"] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
            
        i.update(self._stats.rewards)
        i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        i["safety_triggered"] = internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED]
        i["success"] = i["avg10_tracking_errors"] < 0.05
        i["vec_obs"] = self._last_obs["vec"]
        obslabels = [n.encode("utf-8").ljust(64)[:64] for n in self._state_helper.observation_names()["vec"]]
        i["vec_obs_labels"] = th.as_tensor(obslabels, dtype=th.uint8)
        return i

    @override
    def get_configuration(self):
        return dataclasses.asdict(self._configuration)
    
    @override
    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.as_tensor(False, device=self._configuration.th_device)
        r = state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        if r:
            ggLog.info(f"Terminated at step {self._stepCounter}")
        return r
    
    @override
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)