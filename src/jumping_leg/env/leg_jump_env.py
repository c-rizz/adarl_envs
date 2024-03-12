#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""



import lr_gym.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List, Literal
import lr_gym.utils.dbg.ggLog as ggLog
import random
import lr_gym.utils.spaces as spaces

from lr_gym.envs.ControlledEnv import ControlledEnv
import lr_gym
from lr_gym.utils.utils import Pose, JointState, LinkState, quat_swing_twist_decomposition, quat_angle
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController
import torch as th
import lr_gym.utils.utils
from enum import IntEnum
from lr_gym.env_controllers.EnvironmentController import EnvironmentController
import dataclasses
from dataclasses import dataclass






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
                            "IMPULSES_SUM"], start=0)
    
    EPISODE_CONFIG = IntEnum("EPISODE_CONFIG", ["HIP_GOAL_Z",
                                                "SUPPORT1_POS_X",
                                                "SUPPORT1_POS_Z",
                                                "SUPPORT2_POS_X",
                                                "SUPPORT2_POS_Z",
                                                "REWARD_CONTACTS_WEIGHT"], start=0)


    @dataclass
    class EpisodeConfiguration:
        hip_goal_z : th.Tensor
        support1_pos_x : th.Tensor
        support1_pos_z : th.Tensor
        support2_pos_x : th.Tensor
        support2_pos_z : th.Tensor
        reward_contacts_weights : th.Tensor

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
        position_limits_hip : Tuple[float,float]
        position_limits_knee : Tuple[float,float]
        stepLength_sec : float
        torque_limits_hip : Tuple[float,float]
        torque_limits_knee : Tuple[float,float]
        torque_command_scale_hip : float
        torque_command_scale_knee : float
        velocity_limits_hip : Tuple[float,float]
        velocity_limits_knee : Tuple[float,float]
        velocity_command_scale_hip : float
        velocity_command_scale_knee : float
        vstate_minmax : Optional[th.Tensor]
        reward_scale : float
        platform_obs_noise_std : float

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
                    environmentController : EnvironmentController = None,
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
                    platform_randomization : Literal["none","single","double"] = "none"):
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
        environmentController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo


        """



        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = environmentController,
                         startSimulation = startSimulation,
                         simulationBackend = "")
        

        self._knee_joint = ("leg","knee_joint")
        self._hip_joint = ("leg","hip_joint")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","foot_link")
        self._thigh_base_link = ("leg", "thigh_link_base")
        self._shin_base_link = ("leg", "shin_link_base")
        self._thigh_com_link = ("leg", "thigh_link")
        self._shin_com_link = ("leg", "shin_link")
        self._rendering_cam_name = "simple_camera"
        self._use_velocity_control = False
        self._use_torque_control = False
        self._use_position_control = False
        self._use_full_impedance_control = False
        if control_mode.lower() == "velocity":            
            self._use_velocity_control = True
        elif control_mode.lower() == "position":
            self._use_position_control = True
        elif control_mode.lower() == "impedance":
            self._use_full_impedance_control = True
        elif control_mode.lower() == "torque":
            self._use_torque_control = True
        else:
            raise RuntimeError(f"Invalid control mode '{control_mode}'")
        self._platform_randomization = platform_randomization

        self._obs_only_vec = obs_only_vec
        self._obs_only_img = obs_only_img
        self._obs_img_height = obs_img_height
        self._obs_img_width = obs_img_width
        self._th_device = th_device
        self._rendering_enabled = True
        self._start_height = 0.55
        self._show_goal = True
        self._rng = th.Generator(device=self._th_device)
        self._wall_sim_speed = wall_sim_speed
        self._original_max_epsteps = maxStepsPerEpisode
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
        halflife_s = 0.01
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
                                                            position_limits_hip =  (-2.4, 2.4),
                                                            position_limits_knee = (-2.4, 2.4),
                                                            stepLength_sec=stepLength_sec,
                                                            torque_limits_hip =  (-112, 112),
                                                            torque_limits_knee = (-112, 112),
                                                            torque_command_scale_hip = 100,
                                                            torque_command_scale_knee = 100,
                                                            velocity_limits_hip =  (-20, 20),
                                                            velocity_limits_knee = (-20, 20),
                                                            velocity_command_scale_hip = 20,
                                                            velocity_command_scale_knee = 20,
                                                            vstate_minmax = None,
                                                            reward_scale = reward_scale,
                                                            platform_obs_noise_std = 0.002)
        action_len = 10 if self._use_full_impedance_control else 2
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=0,
                                                                       support1_pos_x=0,
                                                                       support1_pos_z=0,
                                                                       support2_pos_x=0,
                                                                       support2_pos_z=0,
                                                                       reward_contacts_weights=0)
        # max_dact_dt = 100 #max change in action, i.e. da/dt
        # self._max_act_change = th.tensor(max_dact_dt*stepLength_sec,dtype=th.float32, device=self._th_device)
        # self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)
        self._last_out_action = th.zeros((action_len,),dtype=th.float32, device=self._th_device)
        self._history_length = 2
        self._frame_stack_length = 1
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
        self._max_abs_impulses = 0
        self._last_external_work = 0
        self._last_state = None
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
        
        vstate_min_max = {  self.STATE.HIP_JOINT_POS : self._configuration.position_limits_hip,
                            self.STATE.HIP_JOINT_VEL : self._configuration.velocity_limits_hip,
                            self.STATE.HIP_JOINT_EFFORT : self._configuration.torque_limits_hip,
                            self.STATE.KNEE_JOINT_POS : self._configuration.position_limits_knee,
                            self.STATE.KNEE_JOINT_VEL : self._configuration.velocity_limits_knee,
                            self.STATE.KNEE_JOINT_EFFORT : self._configuration.torque_limits_knee,
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
                            self.STATE.IMPULSES_SUM : [0,100]}
        
        self._configuration.vstate_minmax = th.tensor([vstate_min_max[k] for k in self.STATE], device = self._th_device)

        self._stacked_part_len = self.STATE.HIP_VEL_Z+1
        self._1step_vec_obs_size = self.STATE.REWARD_IMPULSE_THRESHOLD+1
        self._vec_obs_size = self._stacked_part_len*self._frame_stack_length + (self._1step_vec_obs_size-self._stacked_part_len)



        vec_obs_space_high = np.array( [1.0]*self._vec_obs_size)
        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        
        self._img_channels = 3 if rgb else 1
        img_shape_chw = (self._img_channels,self._obs_img_height,self._obs_img_width)
        img_observation_space = spaces.gym_spaces.Box(low=0, high=255, shape=img_shape_chw, dtype=np.uint8)

        self.state_space = spaces.gym_spaces.Dict({self.VECTOR_PART: spaces.gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._history_length,len(LegJumpEnv.STATE),)),
                                                   self.IMAGE_PART: img_observation_space})
        
        if self._obs_only_vec:
            self.observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space})     
        elif self._obs_only_img:
            self.observation_space = spaces.gym_spaces.Dict({ self.IMAGE_PART  : img_observation_space})
        else:
            self.observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space,
                                                              self.IMAGE_PART  : img_observation_space})
            
        action_space_high = np.array([1]*action_len)
        self.action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high)



        self.seed(seed)
        self._environmentController.setJointsToObserve([self._knee_joint,self._hip_joint])
        self._environmentController.setLinksToObserve([self._foot_link, self._shin_com_link, self._thigh_com_link])
        self._environmentController.setCamerasToObserve(["camera"])
        self._environmentController.monitor_contacts([("leg",None,None,None)]) # Monitor the contacts between the leg and all the environment

        self._environmentController.startController()
        
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)

    @staticmethod
    def _unnormalize(v, min, max):
        return min+(v+1)/2*(max-min)
    
    @staticmethod        
    def _normalize(value, min, max):
        return (value + (-min))/(max-min)*2-1


    def submitAction(self, action : th.Tensor) -> None:
        
        super().submitAction(action)
        dt = self._configuration.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        action = th.as_tensor(action)
        action = action*(1-alpha) + self._last_out_action*alpha
        # action = th.clamp(action, min=self._last_out_action-self._max_act_change, max=self._last_out_action+self._max_act_change)
        # action = th.tensor([0.,0.])
        self._last_out_action = action
        if self._use_velocity_control:
            # hvel = self._unnormalize(action[0],self._configuration.velocity_command_scale_hip,self._configuration.velocity_limits_hip[0],self._configuration.velocity_limits_hip[1])
            # kvel = self._unnormalize(action[1],self._configuration.velocity_command_scale_knee,self._configuration.velocity_limits_knee[0],self._configuration.velocity_limits_knee[1])
            hvel = action[0]*self._configuration.velocity_command_scale_hip
            kvel = action[1]*self._configuration.velocity_command_scale_knee
            self._environmentController.setJointsVelocityCommand(jointVelocities = [(self._hip_joint,  hvel),
                                                                                    (self._knee_joint, kvel)])
        elif self._use_position_control:
            hpos = self._unnormalize(action[0],self._configuration.position_limits_hip[0],self._configuration.position_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_limits_knee[0],self._configuration.position_limits_knee[1])
            self._environmentController.setJointsImpedanceCommand(jointImpedances = 
                                                            [   (self._hip_joint,   (hpos,0,0,300,30)),
                                                                (self._knee_joint,  (kpos,0,0,300,30))])
        elif self._use_full_impedance_control:
            hpos = self._unnormalize(action[0],self._configuration.position_limits_hip[0],self._configuration.position_limits_hip[1])
            kpos = self._unnormalize(action[1],self._configuration.position_limits_knee[0],self._configuration.position_limits_knee[1])
            hvel = action[2]*self._configuration.velocity_command_scale_hip
            kvel = action[3]*self._configuration.velocity_command_scale_knee
            htorque = action[4]*self._configuration.torque_command_scale_hip
            ktorque = action[5]*self._configuration.torque_command_scale_knee
            hpgain = action[6]*500
            kpgain = action[7]*500
            hvgain = action[8]*100
            kvgain = action[9]*100
            
            self._environmentController.setJointsImpedanceCommand(jointImpedances = 
                                                            [   (self._hip_joint,   (hpos,hvel,htorque,hpgain,hvgain)),
                                                                (self._knee_joint,  (kpos,kvel,ktorque,kpgain,kvgain))])
        else:
            htorque = action[0]*self._configuration.torque_command_scale_hip
            ktorque = action[1]*self._configuration.torque_command_scale_knee
            self._environmentController.setJointsEffortCommand(jointTorques = [(self._hip_joint,  htorque),
                                                                            (self._knee_joint, ktorque)])
            



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
    def computeReward(previousState, state , action : int, env_conf, sub_rewards : Optional[Dict[str,th.Tensor]] = None, dbg_info = None) -> th.Tensor:

        # ggLog.info(f"computeReward state['vec'].size() = {state['vec'].size()}")

        vstate_norm = state[LegJumpEnv.VECTOR_PART][0]
        pvstate_norm = state[LegJumpEnv.VECTOR_PART][-1]

        
        ntorques =       [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_EFFORT,LegJumpEnv.STATE.KNEE_JOINT_EFFORT]]
        nvelocities =    [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_VEL,LegJumpEnv.STATE.KNEE_JOINT_VEL]]
        npositions =     [vstate_norm[k] for k in [LegJumpEnv.STATE.HIP_JOINT_POS,LegJumpEnv.STATE.KNEE_JOINT_POS]]
        torque_reward : th.Tensor = -(sum([t**4 for t in ntorques])/len(ntorques)) # type: ignore
        torque_limit_reward : th.Tensor =   -(sum([t**50 for t in ntorques])/len(ntorques)) # type: ignore # 0.0769 at 0.95
        velocity_reward : th.Tensor =       -(sum([t**2  for t in nvelocities])/len(nvelocities)) # type: ignore
        position_limit_reward : th.Tensor = -(sum([t**50 for t in npositions])/len(npositions)) # type: ignore # 0.0769 at 0.95

        vstate_un = LegJumpEnv._unnormalize(vstate_norm,env_conf["vstate_minmax"][:,0],env_conf["vstate_minmax"][:,1])
        pvstate_un = LegJumpEnv._unnormalize(pvstate_norm,env_conf["vstate_minmax"][:,0],env_conf["vstate_minmax"][:,1])

        goal_dist = th.abs(vstate_un[LegJumpEnv.STATE.HIP_GOAL_Z] - vstate_un[LegJumpEnv.STATE.HIP_POS_Z])
        # tracking_reward = 1 - goal_dist
        tracking_reward = 1/(1+goal_dist/0.05) # halves at 0.05m
        impulse_threshold = pvstate_un[LegJumpEnv.STATE.REWARD_IMPULSE_THRESHOLD]
        contacts_reward = th.clamp(-(vstate_un[LegJumpEnv.STATE.IMPULSES_SUM]/impulse_threshold)**10, min = -1)


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


        if not self._spawned and isinstance(self._environmentController, SimulatedEnvController):
            
            # supp1_pos = [-0.1,0.2]
            # supp2_pos = [-0.15,0.4]

            leg_model_name = "leg"
            cam_model_name = "camera"
            leg_pose = Pose(0,0,0,0,0,0,1)
            name = self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/leg_simple.urdf.xacro"),
                                                            model_name=leg_model_name,
                                                            pose=leg_pose,
                                                            model_format="urdf.xacro")
            self._spawned = True
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/simple_camera.sdf.xacro"),
                                                    model_name=cam_model_name,
                                                    pose=Pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707),
                                                    model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._intendedStepLength_sec},
                                                    model_format="sdf.xacro")
            # ggLog.info(f"Model spawned with name {name}")

            if self._show_goal:
                self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/red_intangible_ball.urdf.xacro"),
                                                        model_name="red_ball",
                                                        pose=leg_pose,
                                                        model_format="urdf.xacro")
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support1",
                                                    pose=Pose(-0.1-0.125, 0.3, 0.2, 0,0,0,1),
                                                    model_format="urdf.xacro")
            
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support2",
                                                    pose=Pose(-0.15-0.125, 0.3, 0.4, 0,0,0,1),
                                                    model_format="urdf.xacro")

        
        self._max_hip_height_reached = th.tensor(0)
        self._dists_to_goal = th.zeros(size=(int(self._maxStepsPerEpisode/10),), dtype=th.float32, device=self._th_device)
        self._cumulative_dist_to_goal = 0
        self._cumulative_knee_torque = 0
        self._cumulative_hip_torque = 0
        self._max_knee_torque = th.tensor(0)
        self._max_hip_torque = th.tensor(0)
        self._cumulated_abs_impulses = 0
        self._last_abs_impulses_sum = 0
        self._max_abs_impulses = 0
        self._last_external_work = 0
        self._last_step_got_state = -1

        if self._platform_randomization == "duoble":
            s1_area = th.tensor([[0.20, 0.30],  # minx, maxx
                                 [0.05, 0.40]], # miny, maxy
                                device=self._th_device)
            s1_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            s1_xz = s1_xz*(s1_area[:,1]-s1_area[:,0])+s1_area[:,0]
            # s1_pos = th.tensor([0.0,0.0])


            s2_area = th.tensor([[0.05, 0.30], # minx, maxx
                                [0.05, 0.45]], device=self._th_device) # miny, maxy
            s2_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            s2_xz = s1_xz + s2_xz*(s2_area[:,1]-s2_area[:,0])+s2_area[:,0]

            s1_xz[0] = s1_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
            s2_xz[0] = s2_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
        elif self._platform_randomization == "single":
            s1_xz = th.tensor([-0.1-0.125, -0.3]) # hide platform 
            s2_area = th.tensor([[0.20, 0.30],  # minx, maxx
                                 [0.05, 0.40]], # miny, maxy
                                device=self._th_device)
            s2_xz = th.rand(size=(2,), generator=self._rng, device=self._th_device)
            s2_xz = s2_xz*(s2_area[:,1]-s2_area[:,0])+s2_area[:,0]
            s2_xz[0] = s2_xz[0]*th.sign(th.rand((1,), generator=self._rng, device=self._th_device)-0.5)
        elif self._platform_randomization == "fixed":
            s1_xz = th.tensor([-0.1-0.125, 0.3])
            s2_xz = th.tensor([-0.15-0.125, 0.6])
        else:
            raise RuntimeError(f"Invalid platofrm_Randomization mode '{self._platform_randomization}'")

        hip_goal_z = 0.4 + th.rand(size=(1,), generator=self._rng, device=self._th_device)*(s2_xz[1]+0.8-0.4)

        self._current_episode_config.support1_pos_x = s1_xz[0]
        self._current_episode_config.support1_pos_z = s1_xz[1]
        self._current_episode_config.support2_pos_x = s2_xz[0]
        self._current_episode_config.support2_pos_z = s2_xz[1]
        self._current_episode_config.hip_goal_z = hip_goal_z
        reward_contacts_weights = self._sample(self._configuration.reward_contacts_weight,
                                                                                                self._rng,
                                                                                                self._th_device)
        maxStepsPerEpisode = self._original_max_epsteps
        # These override previous configs
        if "support1_pos_x" in options: s1_xz[0] = options["support1_pos_x"]
        if "support1_pos_z" in options: s1_xz[1] = options["support2_pos_z"]
        if "support2_pos_x" in options: s2_xz[0] = options["support2_pos_x"]
        if "support2_pos_z" in options: s2_xz[1] = options["support2_pos_z"]
        if "hip_goal_z" in options: hip_goal_z = options["hip_goal_z"]
        if "reward_contacts_weights" in options: reward_contacts_weights = options["reward_contacts_weights"]
        if "max_ep_steps" in options: maxStepsPerEpisode = options["max_ep_steps"]

        self._maxStepsPerEpisode = maxStepsPerEpisode
            
        #min 0.4, max support2_z+0.6
        self._current_episode_config = LegJumpEnv.EpisodeConfiguration(hip_goal_z=hip_goal_z,
                                                                       support1_pos_x=s1_xz[0],
                                                                       support1_pos_z=s1_xz[1],
                                                                       support2_pos_x=s2_xz[0],
                                                                       support2_pos_z=s2_xz[1],
                                                                       reward_contacts_weights=reward_contacts_weights)

        if isinstance(self._environmentController, SimulatedEnvController):
            if self._current_episode_config.support2_pos_x > 0:
                self._environmentController.setJointsStateDirect({self._rail_joint: JointState(position = [self._start_height], rate=[0], effort=[0]),
                                                                self._hip_joint:  JointState(position = [ 3.14159/4], rate=[0], effort=[0]),
                                                                self._knee_joint: JointState(position = [-3.14159/2], rate=[0], effort=[0])})
            else:
                self._environmentController.setJointsStateDirect({self._rail_joint: JointState(position = [self._start_height], rate=[0], effort=[0]),
                                                                self._hip_joint:  JointState(position = [-3.14159/4], rate=[0], effort=[0]),
                                                                self._knee_joint: JointState(position = [ 3.14159/2], rate=[0], effort=[0])})
        else:
            raise RuntimeError("Cannot reset joint state")
        self._environmentController.setJointsEffortCommand([(self._hip_joint,0),(self._knee_joint,0)])
        self._place_objects()

    def _place_objects(self):
        # ggLog.info(f"placing: _current_episode_config {self._current_episode_config}")
        self._environmentController.setLinksStateDirect({("support1","world") : 
                                                         LinkState( position_xyz = th.tensor((self._current_episode_config.support1_pos_x,
                                                                                              0.3,
                                                                                              self._current_episode_config.support1_pos_z)),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.)))})
        self._environmentController.setLinksStateDirect({("support2","world") :
                                                          LinkState(position_xyz = th.tensor((self._current_episode_config.support2_pos_x,
                                                                                              0.3,
                                                                                              self._current_episode_config.support2_pos_z)),
                                                                    orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                    pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                    ang_velocity_xyz = th.tensor((0.,0.,0.)))})
        if self._show_goal:
            self._environmentController.setLinksStateDirect({("red_ball","world") :
                                                             LinkState( position_xyz = th.tensor((0.,
                                                                                                 0.2,
                                                                                                 self._current_episode_config.hip_goal_z)),
                                                                        orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                                                                        pos_velocity_xyz = th.tensor((0.,0.,0)),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.)))})
        
    def getUiRendering(self) -> Tuple[Union[np.ndarray, th.Tensor], float]:
        try:
            img, time = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            if img is None:
                time = -1
            return img, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {lr_gym.utils.utils.exc_to_str(e)}")
            return None, -1


    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        if self._obs_only_vec:
            stacked_part =  state[self.VECTOR_PART][:self._frame_stack_length,:self._stacked_part_len].detach().clone()
            supp_pos = stacked_part[:,self.STATE.SUPPORT1_X:self.STATE.SUPPORT2_Z+1]
            noise = th.randn(size=supp_pos.size(), device = self._th_device, generator=self._rng)*self._configuration.platform_obs_noise_std
            stacked_part[:,self.STATE.SUPPORT1_X:self.STATE.SUPPORT2_Z+1] = supp_pos + noise
            stacked_part = stacked_part.flatten()
            constant_part = state[self.VECTOR_PART][0,self._stacked_part_len:self._1step_vec_obs_size]
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
                self._last_step_got_state = self._stepCounter
                
                jstates = self._environmentController.getJointsState(requestedJoints=[self._knee_joint, self._hip_joint])
                lstates : Dict[Tuple[str,str],LinkState] = self._environmentController.getLinksState(requestedLinks = [self._thigh_com_link,
                                                                                    self._shin_com_link,
                                                                                    self._thigh_base_link], use_com_frame = True)
                hip_height = lstates[self._thigh_base_link].pose.position[2]
                hip_vel_z = lstates[self._thigh_base_link].pos_velocity_xyz[2]
                self._max_hip_height_reached = th.maximum(self._max_hip_height_reached,hip_height)

                # n = '\n'
                # ggLog.info(f"contacts == {n.join([str(c) for c in contacts])}")
                thigh_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
                thigh_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
                thigh_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([0.0,0.0,1.0], device=self._th_device))[1])
                shin_ang_pos_x = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([1.0,0.0,0.0], device=self._th_device))[1])
                shin_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
                shin_ang_pos_z = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_wxyz,
                                                                                    th.tensor([0.0,0.0,1.0], device=self._th_device))[1])

                contacts = self._environmentController.get_contacts()
                impulses = []
                for simsteps_contacts in contacts:
                    impulses += [contact[3]*contact[4] for contact in simsteps_contacts]
                abs_impulses = [abs(i) for i in impulses]


                # ggLog.info(f"jstates = {jstates}")


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
                                    sum(abs_impulses)),
                                dtype = th.float32,
                                device = self._th_device)
                
                current_vstate = self._normalize(current_vstate,self._configuration.vstate_minmax[:,0],self._configuration.vstate_minmax[:,1])
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
        contacts = self._environmentController.get_contacts()
        abs_impulses = []
        for simsteps_contacts in contacts:
            abs_impulses += [abs(contact[3]*contact[4]) for contact in simsteps_contacts]
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

        if len(abs_impulses)>0:
            self._max_abs_impulses = max(self._max_abs_impulses, max(abs_impulses))
        return rew_dbg_info

    def performStep(self):
        super().performStep()
        self._dbg_info = self._compute_dbg_info()



    def buildSimulation(self, backend):
        # ggLog.info("Building env")
        envCtrlName = type(self._environmentController).__name__
        if envCtrlName == "PyBulletJointImpedanceController":
            self._environmentController.build_scenario(None)
            self._rendering_cam_name = "simple_camera"
        # elif envCtrlName in ["GazeboController", "GazeboControllerNoPlugin"]:
        #     # ggLog.info(f"sim_img_width  = {sim_img_width}")
        #     # ggLog.info(f"sim_img_height = {sim_img_height}")
        #     if not self._rendering_enabled:
        #         worldpath = "\"$(find lr_gym_ros)/worlds/ground_plane_world_plugin.world\""
        #     else:
        #         worldpath = "\"$(find lr_gym_ros)/worlds/fixed_camera_world_plugin.world\""
        #     self._environmentController.build_scenario( launch_file_pkg_and_path=("lr_gym_ros","/launch/gazebo_server.launch"),
        #                                                 launch_file_args={  "gui":"false",
        #                                                                     "paused":"true",
        #                                                                     "physics_engine":"bullet",
        #                                                                     "limit_sim_speed":"false",
        #                                                                     "world_name":worldpath,
        #                                                                     "gazebo_seed":f"{self._envSeed}",
        #                                                                     "wall_sim_speed":f"{self._wall_sim_speed}"})
        #     self._rendering_cam_name = "camera"
        # elif envCtrlName == "GzController":
        #     self._environmentController.build_scenario(sdf_file = ("lr_gym_ros2","/worlds/empty_cams.sdf"))
        #     # self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/simple_camera.sdf.xacro"),
        #     #                                         model_name=None,
        #     #                                         pose=Pose(0,2,0.5,0,0.0,-0.707,0.707),
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
        i["max_abs_impulse"] = self._max_abs_impulses
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
        i["vstate"] = current_vstate_unnorm
        i.update(self._dbg_info)
        # i["config"] = dataclasses.asdict(self._configuration)
        # i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i

    def get_configuration(self):
        return dataclasses.asdict(self._configuration)