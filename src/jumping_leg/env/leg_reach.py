#!/usr/bin/env python3
from __future__ import annotations
import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List, Literal, TypeVar
import adarl.utils.dbg.ggLog as ggLog
import random
import adarl.utils.spaces as spaces

from adarl.envs.ControlledEnv import ControlledEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState, LinkState, MoveFailError, exc_to_str
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



_T = TypeVar('_T', float, th.Tensor)

class LegReachEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""

    metadata = {'render.modes': ['rgb_array']}
    STATE_BASE = "vec" # component of the state that is a vector and is always the same regardless of the configuration

    BASE_STATE_IDXS = IntEnum("BASE_STATE", [
                            "HIP_JOINT_POS",
                            "HIP_JOINT_VEL",
                            "HIP_JOINT_EFFORT",
                            "KNEE_JOINT_POS",
                            "KNEE_JOINT_VEL",
                            "KNEE_JOINT_EFFORT",
                            "HIP_POS_Z",
                            "HIP_VEL_Z",
                            "HIP_GOAL_Z",
                            "REWARD_TORQUE_LIMIT_WEIGHT",
                            "REWARD_POSITION_LIMIT_WEIGHT",
                            "REWARD_VELOCITY_WEIGHT",
                            "REWARD_TRACKING_WEIGHT",
                            "REWARD_TORQUE_WEIGHT",
                            "KNEE_TORQUE_CMD_SCALE",
                            "HIP_TORQUE_CMD_SCALE",
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
                            "SAFETY_TRIGGERED"], start=0)
    

    @dataclass
    class EpisodeConfiguration:
        hip_goal_z : th.Tensor

    @dataclass
    class EnvConfiguration:
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
        bstate_minmax : th.Tensor
        reward_scale : float
        stop_on_safety : bool
        max_stiffness : float
        max_damping : float

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
                    seed = 0,
                    th_device = th.device("cpu"),
                    reward_torque_limit_weight = 0.0,
                    reward_position_limit_weight = 1.0,
                    reward_velocity_weight = 0.0,
                    reward_tracking_weight = 1.0,
                    reward_torque_weight = 0.0,
                    reward_scale = 1.0,
                    real : bool = False,
                    step_precision_tolerance : float = 0.0,
                    stop_on_safety = True):


        
        self._knee_joint = ("leg","knee_joint_1")
        self._hip_joint = ("leg","hip_joint_1")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","tip1")
        self._thigh_base_link = ("leg", "thigh_link1")
        self._shin_base_link = ("leg", "shin_link1")
        self._thigh_com_link = ("leg", "thigh_link1_com")
        self._shin_com_link = ("leg", "shin_link1_com")
        self._rendering_cam_name = "simple_camera"



        self._th_device = th_device
        self._rendering_enabled = True
        self._start_height = 0.9
        self._show_goal = True
        self._rng = th.Generator(device=self._th_device)
        self._real = real
        self._obs_dtype = th.float32
        self._original_max_epsteps = maxStepsPerEpisode


        halflife_s = 0.05
        action_exp_smoothing_1s = 0.5**(1/halflife_s)
        self._configuration = LegReachEnv.EnvConfiguration( reward_position_limit_weight = reward_position_limit_weight,
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
                                                            bstate_minmax = th.empty((0,)),
                                                            reward_scale = reward_scale,
                                                            stop_on_safety = stop_on_safety,
                                                            max_stiffness=400,
                                                            max_damping=100)
        
        self._current_episode_config = LegReachEnv.EpisodeConfiguration(hip_goal_z=th.tensor(0))
        self._action_size = 4
        self._last_out_action = th.empty((0,))
        

        self._spawned = False
        self._last_step_got_state = -1
        self._cumulative_dist_to_goal = 0
        self._dists_to_goal = th.zeros(size=(int(maxStepsPerEpisode/10),), dtype=th.float32, device=self._th_device)
        
        vstate_min_max = {  self.BASE_STATE_IDXS.HIP_JOINT_POS : self._configuration.position_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_JOINT_VEL : self._configuration.velocity_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_JOINT_EFFORT : self._configuration.torque_phys_limits_hip,
                            self.BASE_STATE_IDXS.KNEE_JOINT_POS : self._configuration.position_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_JOINT_VEL : self._configuration.velocity_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT : self._configuration.torque_phys_limits_knee,
                            self.BASE_STATE_IDXS.HIP_POS_Z : [0,3],
                            self.BASE_STATE_IDXS.HIP_VEL_Z : [-100,100],
                            self.BASE_STATE_IDXS.HIP_GOAL_Z : [0,2],
                            self.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT : [0,10],
                            self.BASE_STATE_IDXS.KNEE_TORQUE_CMD_SCALE : [0,150],
                            self.BASE_STATE_IDXS.HIP_TORQUE_CMD_SCALE : [0,150],
                            self.BASE_STATE_IDXS.HIP_POS_REF : self._configuration.position_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_VEL_REF : self._configuration.velocity_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_EFFORT_REF : self._configuration.torque_phys_limits_hip,
                            self.BASE_STATE_IDXS.HIP_STIFFNESS : [0,self._configuration.max_stiffness],
                            self.BASE_STATE_IDXS.HIP_DAMPING : [0,self._configuration.max_damping],
                            self.BASE_STATE_IDXS.KNEE_POS_REF : self._configuration.position_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_VEL_REF : self._configuration.velocity_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_EFFORT_REF : self._configuration.torque_phys_limits_knee,
                            self.BASE_STATE_IDXS.KNEE_STIFFNESS : [0,self._configuration.max_stiffness],
                            self.BASE_STATE_IDXS.KNEE_DAMPING : [0,self._configuration.max_damping],
                            self.BASE_STATE_IDXS.SAFETY_TRIGGERED : [0,1]}        
        self._configuration.bstate_minmax = th.tensor([vstate_min_max[k] for k in self.BASE_STATE_IDXS], device = self._th_device)

        # Part of the BASE_STATE that gets stacked
        self._observed_state = th.as_tensor([   self.BASE_STATE_IDXS.HIP_JOINT_POS,
                                                self.BASE_STATE_IDXS.HIP_JOINT_VEL,
                                                self.BASE_STATE_IDXS.HIP_JOINT_EFFORT,
                                                self.BASE_STATE_IDXS.KNEE_JOINT_POS,
                                                self.BASE_STATE_IDXS.KNEE_JOINT_VEL,
                                                self.BASE_STATE_IDXS.KNEE_JOINT_EFFORT,
                                                self.BASE_STATE_IDXS.HIP_POS_Z,
                                                self.BASE_STATE_IDXS.HIP_VEL_Z,
                                                self.BASE_STATE_IDXS.HIP_POS_REF,
                                                self.BASE_STATE_IDXS.HIP_VEL_REF,
                                                self.BASE_STATE_IDXS.HIP_EFFORT_REF,
                                                self.BASE_STATE_IDXS.HIP_STIFFNESS,
                                                self.BASE_STATE_IDXS.HIP_DAMPING,
                                                self.BASE_STATE_IDXS.KNEE_POS_REF,
                                                self.BASE_STATE_IDXS.KNEE_VEL_REF,
                                                self.BASE_STATE_IDXS.KNEE_EFFORT_REF,
                                                self.BASE_STATE_IDXS.KNEE_STIFFNESS,
                                                self.BASE_STATE_IDXS.KNEE_DAMPING,
                                                self.BASE_STATE_IDXS.HIP_GOAL_Z], device=self._th_device)
        vec_obs_size = self._observed_state.size()[0]
        vec_obs_space_high = np.array( [1.0]*vec_obs_size)

        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        state_space = spaces.gym_spaces.Dict({  self.STATE_BASE: spaces.gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(len(LegReachEnv.BASE_STATE_IDXS),))})
        observation_space = spaces.gym_spaces.Dict({ self.STATE_BASE : vec_obs_space})     
            
        action_space_high = np.array([1]*self._action_size)
        action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high, seed=seed)


        hp_lim = self._configuration.position_phys_limits_hip
        he_lim = self._configuration.torque_command_scale_hip
        kp_lim = self._configuration.position_phys_limits_knee
        ke_lim = self._configuration.torque_command_scale_knee
        max_stiffness = self._configuration.max_stiffness
        max_damping   = self._configuration.max_damping
        self._minmax_hipknee_pvesd = th.tensor([[[hp_lim[0], -20, -he_lim, 0,             0],
                                                 [kp_lim[0], -20, -ke_lim, 0,             0]],
                                                [[hp_lim[1],  20,  he_lim, max_stiffness, max_damping],
                                                 [kp_lim[1],  20,  ke_lim, max_stiffness, max_damping]]])

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
        
        self._environmentController.startup()








    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    def _pvesd_to_action(self, cmds_pvesd):
        act_pvesd_interleaved = th.tensor([ cmds_pvesd[self._hip_joint][0],
                                cmds_pvesd[self._knee_joint][0],
                                cmds_pvesd[self._hip_joint][1],
                                cmds_pvesd[self._knee_joint][1],
                                cmds_pvesd[self._hip_joint][2],
                                cmds_pvesd[self._knee_joint][2],
                                cmds_pvesd[self._hip_joint][3],
                                cmds_pvesd[self._knee_joint][3],
                                cmds_pvesd[self._hip_joint][4],
                                cmds_pvesd[self._knee_joint][4]])
        act_pvesd_interleaved = self._normalize(act_pvesd_interleaved,
                                                self._minmax_hipknee_pvesd[[0,0],[0,1]].flatten(),
                                                self._minmax_hipknee_pvesd[[1,1],[1,0]].flatten())
        act = act_pvesd_interleaved[[0,1,6,7]]
        return act
    
    def _action_to_pvesd(self, action) -> dict[tuple[str,str],tuple[float,float,float,float,float]]:
        hip_pvesd =  th.tensor([action[0],0,0,action[2],0], dtype=self._obs_dtype, device=self._th_device)
        knee_pvesd = th.tensor([action[1],0,0,action[3],0], dtype=self._obs_dtype, device=self._th_device)
        hip_pvesd = self._unnormalize(hip_pvesd, min=self._minmax_hipknee_pvesd[0][0],
                                                 max=self._minmax_hipknee_pvesd[1][0])
        knee_pvesd = self._unnormalize(knee_pvesd,  min=self._minmax_hipknee_pvesd[0][1],
                                                    max=self._minmax_hipknee_pvesd[1][1])
        return {self._hip_joint :  tuple(hip_pvesd.tolist()),
                self._knee_joint:  tuple(knee_pvesd.tolist())}

    def submitAction(self, action : th.Tensor) -> None:
        # ggLog.info(f"Submitting action {action}")
        action = th.as_tensor(action).detach().cpu()
        super().submitAction(action)
        dt = self._configuration.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        prev_action = self._last_out_action
        if self._actionsCounter != 0:
            action = action*(1-alpha) + prev_action*alpha
        # action = th.clamp(action, min=prev_action-self._max_act_change, max=prev_action+self._max_act_change)
        action = th.clamp(action, min=-1, max=1)
        # action = th.tensor([0.,0.])
        self._last_out_action = action
        # action_l = action.tolist()
        jimp_pvesd = self._action_to_pvesd(action)
        self._last_sent_pvesd = jimp_pvesd
        self._environmentController.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_pvesd)
            






















    # --------------------------------------------------------------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------------------------------------------------------------


    def computeReward(self, previousState : Dict[str,th.Tensor],
                      state : Dict[str,th.Tensor],
                      action : th.Tensor,
                      env_conf,
                      sub_rewards : Optional[Dict[str,th.Tensor]] = None, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        vstate_norm = state[LegReachEnv.STATE_BASE][0]
        pvstate_norm = state[LegReachEnv.STATE_BASE][-1]

        normtorques = vstate_norm[[LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_EFFORT,LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_EFFORT]]
        normvelocities = vstate_norm[[LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_VEL,LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_VEL]]
        normpositions = vstate_norm[[LegReachEnv.BASE_STATE_IDXS.HIP_JOINT_POS,LegReachEnv.BASE_STATE_IDXS.KNEE_JOINT_POS]]
        # ntorques =       [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_EFFORT,LegJumpEnv.STATE.KNEE_JOINT_EFFORT]]
        # nvelocities =    [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_VEL,LegJumpEnv.STATE.KNEE_JOINT_VEL]]
        # npositions =     [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_POS,LegJumpEnv.STATE.KNEE_JOINT_POS]]
        max_r = 100
        torque_reward = - th.clamp(th.mean(th.pow(normtorques,4)),-max_r,max_r)
        torque_limit_reward = - th.clamp(th.mean(th.pow(normtorques,50)),-max_r,max_r)
        velocity_reward = - th.clamp(th.mean(th.pow(normvelocities,2)),-max_r,max_r)
        position_limit_reward = - th.clamp(th.mean(th.pow(normpositions,50)),-max_r,max_r)

        vstate_un = LegReachEnv._unnormalize(vstate_norm,env_conf["bstate_minmax"][:,0],env_conf["bstate_minmax"][:,1])

        goal_dist = th.abs(vstate_un[LegReachEnv.BASE_STATE_IDXS.HIP_GOAL_Z] - vstate_un[LegReachEnv.BASE_STATE_IDXS.HIP_POS_Z])
        # tracking_reward = 1 - goal_dist
        tracking_reward = 1/(1+goal_dist/0.05) # halves at 0.05m
        

        reward_scale = env_conf["reward_scale"]
        tracking_reward         = reward_scale * tracking_reward
        torque_reward           = reward_scale * torque_reward
        torque_limit_reward     = reward_scale * torque_limit_reward
        velocity_reward         = reward_scale * velocity_reward
        position_limit_reward   = reward_scale * position_limit_reward

        if sub_rewards is not None:
            sub_rewards["tracking_reward"] = tracking_reward
            sub_rewards["torque_reward"] = torque_reward
            sub_rewards["torque_limit_reward"] = torque_limit_reward
            sub_rewards["velocity_reward"] = velocity_reward
            sub_rewards["position_limit_reward"] = position_limit_reward

        torque_lim_weight = vstate_un[LegReachEnv.BASE_STATE_IDXS.REWARD_TORQUE_LIMIT_WEIGHT]
        position_lim_weight = vstate_un[LegReachEnv.BASE_STATE_IDXS.REWARD_POSITION_LIMIT_WEIGHT]
        velocity_weight = vstate_un[LegReachEnv.BASE_STATE_IDXS.REWARD_VELOCITY_WEIGHT]
        tracking_weight = vstate_un[LegReachEnv.BASE_STATE_IDXS.REWARD_TRACKING_WEIGHT]
        torque_weight = vstate_un[LegReachEnv.BASE_STATE_IDXS.REWARD_TORQUE_WEIGHT]

        return (tracking_weight*tracking_reward + 
                torque_lim_weight*torque_limit_reward + 
                velocity_weight*velocity_reward+
                position_lim_weight*position_limit_reward+
                torque_weight*torque_reward)


    def initializeEpisode(self, options = {}) -> None:
        self._current_state = {self.STATE_BASE   : th.zeros(len(self.BASE_STATE_IDXS), dtype=th.float32, device=self._th_device)}
        
        if not self._spawned and isinstance(self._environmentController, BaseSimulationAdapter):
            leg_model_name = "leg"
            cam_model_name = "camera"
            leg_pose = build_pose(0,0,0,0,0,0,1)
            self._spawned = True
            if isinstance(self._environmentController, PyBulletAdapter):
                leg_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")
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

        min_goal_z = 0.3
        max_goal_z = 0.7
        hip_goal_z = min_goal_z + th.rand(size=(1,), generator=self._rng, device=self._th_device)*(max_goal_z-min_goal_z) # uniform(0.4,1.2)

        maxStepsPerEpisode = self._original_max_epsteps
        # These override previous configs
        if options is not None:
            if "hip_goal_z" in options: hip_goal_z = options["hip_goal_z"]
            if "max_ep_steps" in options: maxStepsPerEpisode = options["max_ep_steps"]

        self._maxStepsPerEpisode = maxStepsPerEpisode
            
        #min 0.4, max support2_z+0.6
        self._current_episode_config = LegReachEnv.EpisodeConfiguration(hip_goal_z=hip_goal_z)
        if isinstance(self._environmentController, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            moved = False
            while not moved:
                ggLog.info(f"Cannot automatically initialize episode with non-simulated adapter. Lift up the robot and press ENTER. Be safe :)")
                input()
                if isinstance(self._environmentController, BaseJointPositionAdapter):
                    rpos, hpos, kpos = self._start_height, 3.4159/4,  3.14159/2
                    try:
                        self._environmentController.moveToJointPoseSync({self._hip_joint:  hpos,
                                                                        self._knee_joint: kpos})
                    except MoveFailError as e:
                        ggLog.warn(f"Failed to move to joint position. Error = {exc_to_str(e)}")
            # raise RuntimeError("")

    def _place_objects(self, goal_z=None):
        if not isinstance(self._environmentController, BaseSimulationAdapter):
            raise RuntimeError("Cannot place objects in the real")
        if goal_z is None:
            goal_z = self._current_episode_config.hip_goal_z
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
            self._place_objects(goal_z=10)
            rpos, hpos, kpos = self._start_height, 3.4159/4,  3.14159/2
            self._environmentController.setJointsStateDirect({  self._rail_joint: JointState(position = rpos, rate=0, effort=0),
                                                                self._hip_joint:  JointState(position = hpos, rate=0, effort=0),
                                                                self._knee_joint: JointState(position = kpos, rate=0, effort=0)})
            start_jimp : dict[tuple[str,str], tuple] = {self._hip_joint: (hpos,0,0,200,50),
                                                        self._knee_joint:(kpos,0,0,200,50)}         
            self._environmentController.setJointsImpedanceCommand(start_jimp)
            self._environmentController.apply_joint_impedances(start_jimp)
            self._environmentController.run(3.0) # let the leg fall
            self._place_objects(goal_z=self._current_episode_config.hip_goal_z)
            # jpos = {k:v.position for k,v in self._environmentController.getJointsState(requestedJoints=[self._rail_joint, self._hip_joint, self._knee_joint]).items()}
            # ggLog.info(f"Init: current jpos = {jpos}")
            self._last_sent_pvesd = start_jimp
            self._last_out_action = self._pvesd_to_action(start_jimp)
        else:
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")




    def buildSimulation(self):
        # ggLog.info("Building env")
        envCtrlName = type(self._environmentController).__name__
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._environmentController.build_scenario()

            self._red_ball_base = ("red_ball","world")
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._real:
                raise NotImplementedError()
            else:
                self._environmentController.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path("jumping_leg",
                                                                                                                          "gazebo/all_gazebo_xbot.launch"),
                                                           launch_file_args={"gui":"true"})

    def _destroySimulation(self):
        self._environmentController.destroy_scenario()





















    # --------------------------------------------------------------------------------------------------------------------
    # State & Observation
    # --------------------------------------------------------------------------------------------------------------------

        
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
        return {self.STATE_BASE : state[self.STATE_BASE][self._observed_state]}            
    
    def getState(self) -> Dict[Any, th.Tensor]:
        """Update and return the current state
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
                
                state = self._current_state[self.STATE_BASE]
                if len(self._current_state)!=0 and state[self.BASE_STATE_IDXS.SAFETY_TRIGGERED] > 0:
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



                state = th.tensor((jstates[self._hip_joint].position[0],
                                    jstates[self._hip_joint].rate[0],
                                    jstates[self._hip_joint].effort[0],
                                    jstates[self._knee_joint].position[0],
                                    jstates[self._knee_joint].rate[0],
                                    jstates[self._knee_joint].effort[0],
                                    hip_height,
                                    hip_vel_z,
                                    self._current_episode_config.hip_goal_z,
                                    self._configuration.reward_torque_limit_weight,
                                    self._configuration.reward_position_limit_weight,
                                    self._configuration.reward_velocity_weight,
                                    self._configuration.reward_tracking_weight,
                                    self._configuration.reward_torque_weight,
                                    self._configuration.torque_command_scale_knee,
                                    self._configuration.torque_command_scale_hip,
                                    self._last_sent_pvesd[self._hip_joint][0],
                                    self._last_sent_pvesd[self._hip_joint][1],
                                    self._last_sent_pvesd[self._hip_joint][2],
                                    self._last_sent_pvesd[self._hip_joint][3],
                                    self._last_sent_pvesd[self._hip_joint][4],
                                    self._last_sent_pvesd[self._knee_joint][0],
                                    self._last_sent_pvesd[self._knee_joint][1],
                                    self._last_sent_pvesd[self._knee_joint][2],
                                    self._last_sent_pvesd[self._knee_joint][3],
                                    self._last_sent_pvesd[self._knee_joint][4],
                                    1 if safety_triggered else 0),
                                dtype = self._obs_dtype)
                

                # ggLog.info(f"current_vstate = {current_vstate}")
                state = self._normalize(state,self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])
                                
                self._current_state = {self.STATE_BASE : state.detach().clone()}
                
            return self._current_state

    def _compute_dbg_info(self):
        vstate_unnorm = self._unnormalize(self._current_state[self.STATE_BASE][0],self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])
        self._cumulative_dist_to_goal += abs(vstate_unnorm[self.BASE_STATE_IDXS.HIP_GOAL_Z]-vstate_unnorm[self.BASE_STATE_IDXS.HIP_POS_Z])


    def performStep(self):
        super().performStep()
        self._compute_dbg_info()





    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        bstate_unnorm = self._unnormalize(state[self.STATE_BASE][0],self._configuration.bstate_minmax[:,0],self._configuration.bstate_minmax[:,1])
        i["hip_goal_z"] = bstate_unnorm[self.BASE_STATE_IDXS.HIP_GOAL_Z]
        i["avg_dist"] = self._cumulative_dist_to_goal/self._stepCounter if self._stepCounter!=0 else float("nan")
        i["step_count"] = self._stepCounter
        return i

    def get_configuration(self):
        return dataclasses.asdict(self._configuration)
    
        
    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.as_tensor(False, device=self._th_device)
        r = state[self.STATE_BASE][0][self.BASE_STATE_IDXS.SAFETY_TRIGGERED] > 0
        if r:
            ggLog.info(f"termination at step {self._stepCounter}")
        return r
    
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)


    @staticmethod
    def _unnormalize(v : _T, min : _T, max : _T) -> _T:
        return min+(v+1)/2*(max-min)
    
    @staticmethod        
    def _normalize(value : _T, min : _T, max : _T):
        return (value + (-min))/(max-min)*2-1
