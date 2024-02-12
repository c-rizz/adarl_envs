#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""



import lr_gym.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List
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
                            "HIP_GOAL_Z",
                            "THIGH_VEL_X",
                            "THIGH_VEL_Z",
                            "THIGH_ANG_VEL_Y",
                            "SHIN_VEL_X",
                            "SHIN_VEL_Z",
                            "SHIN_ANG_VEL_Y",
                            "THIGH_POS_Z",
                            "THIGH_ANG_POS_Y",
                            "SHIN_POS_Z",
                            "SHIN_ANG_POS_Y",
                            "IMPULSES_SUM"], start=0)

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
                    use_velocity_control = False):
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
        self._use_velocity_control = use_velocity_control

        self._obs_only_vec = obs_only_vec
        self._obs_only_img = obs_only_img
        self._obs_img_height = obs_img_height
        self._obs_img_width = obs_img_width
        self._th_device = th_device
        self._rendering_enabled = True
        self._max_hip_height_reached = th.tensor(0)
        self._hip_torque_scale = 100
        self._knee_torque_scale = 100
        self._velocity_scale = {self._knee_joint : 1,
                                self._hip_joint  : 1}
        max_dact_dt = 10 #max change in action, i.e. da/dt
        self._max_act_change = th.tensor(max_dact_dt*stepLength_sec,dtype=th.float32, device=self._th_device)
        self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)
        self._last_clamped_action = th.zeros((2,),dtype=th.float32, device=self._th_device)

        self._configuration = dict( reward_torque_limit_weight = reward_torque_limit_weight,
                                    reward_position_limit_weight = reward_position_limit_weight,
                                    reward_velocity_weight = reward_velocity_weight,
                                    reward_energy_weight = reward_energy_weight,
                                    reward_tracking_weight = reward_tracking_weight,
                                    reward_torque_weight = reward_torque_weight,
                                    hip_torque_scale = self._hip_torque_scale,
                                    knee_torque_scale = self._knee_torque_scale,
                                    reward_contacts_weight = reward_contacts_weight,
                                    max_impulse = 4)

        self._torque_limits =   {self._knee_joint : [-112,112],
                                 self._hip_joint : [-112,112]}
        self._velocity_limits = {self._knee_joint : [-20,20],
                                 self._hip_joint : [-20,20]}
        self._position_limits = {self._knee_joint : [-2.4,2.4],
                                 self._hip_joint : [-2.4,2.4]}

        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._rng = th.Generator(device=self._th_device)
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
        

        self._start_height = 0.55

        self._show_goal = True


        if self._obs_only_vec:
            vec_obs_size = 9
        else:
            vec_obs_size = self.STATE.HIP_GOAL_Z+1
        vec_obs_space_high = np.array( [1.0]*vec_obs_size)
        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        
        self._img_channels = 3 if rgb else 1
        img_shape_chw = (self._img_channels,self._obs_img_height,self._obs_img_width)
        img_observation_space = spaces.gym_spaces.Box(low=0, high=255, shape=img_shape_chw, dtype=np.uint8)

        self.state_space = spaces.gym_spaces.Dict({self.VECTOR_PART: spaces.gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(len(LegJumpEnv.STATE),)),
                                                   self.IMAGE_PART: img_observation_space})
        
        if self._obs_only_vec:
            self.observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space})     
        elif self._obs_only_img:
            self.observation_space = spaces.gym_spaces.Dict({ self.IMAGE_PART  : img_observation_space})
        else:
            self.observation_space = spaces.gym_spaces.Dict({ self.VECTOR_PART : vec_obs_space,
                                                              self.IMAGE_PART  : img_observation_space})
            
        action_space_high = np.array([1, 1])
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
    def _unnorm(v, scale, min, max):
        return min+v*scale*(max-min)


    def submitAction(self, action : th.Tensor) -> None:
        super().submitAction(action)
        action = th.as_tensor(action)
        # action = th.clamp(action, min=self._last_clamped_action-self._max_act_change, max=self._last_clamped_action+self._max_act_change)
        self._last_clamped_action = action
        if self._use_velocity_control:

            hvel = self._unnorm(action[0],self._velocity_scale[self._hip_joint],self._velocity_limits[self._hip_joint][0],self._velocity_limits[self._hip_joint][1])
            kvel = self._unnorm(action[1],self._velocity_scale[self._knee_joint],self._velocity_limits[self._knee_joint][0],self._velocity_limits[self._knee_joint][1])
            self._environmentController.setJointsVelocityCommand(jointVelocities = [(self._hip_joint,  hvel),
                                                                                    (self._knee_joint, kvel)])

        else:
            htorque = action[0]*self._hip_torque_scale
            ktorque = action[1]*self._knee_torque_scale
            self._environmentController.setJointsEffortCommand(jointTorques = [(self._hip_joint,  htorque),
                                                                            (self._knee_joint, ktorque)])



    @staticmethod
    def _kinetic_energy_2d(mass, inertia_moment, vel_x, vel_z, ang_vel_y):
        return 0.5*mass*(vel_x**2 + vel_z**2) + 0.5*inertia_moment*ang_vel_y**2


    @staticmethod
    def _compute_mechanical_energies(vstate):
        thigh_mass = 3.37
        thigh_length = 0.3
        shin_mass = 1.3
        shin_length = 0.45
        slider_mass = 8
        g = 9.8
        
        thigh_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = thigh_mass,
                                                        inertia_moment=1/12*thigh_mass*thigh_length**2,
                                                        vel_x=vstate[LegJumpEnv.STATE.THIGH_VEL_X],
                                                        vel_z=vstate[LegJumpEnv.STATE.THIGH_VEL_Z],
                                                        ang_vel_y=vstate[LegJumpEnv.STATE.THIGH_ANG_VEL_Y])
        shin_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = shin_mass,
                                                        inertia_moment=1/12*shin_mass*shin_length**2,
                                                        vel_x=vstate[LegJumpEnv.STATE.SHIN_VEL_X],
                                                        vel_z=vstate[LegJumpEnv.STATE.SHIN_VEL_Z],
                                                        ang_vel_y=vstate[LegJumpEnv.STATE.SHIN_ANG_VEL_Y])
        slider_kin_energy = LegJumpEnv._kinetic_energy_2d(mass = slider_mass,
                                                        inertia_moment=0,
                                                        vel_x=0,
                                                        vel_z=vstate[LegJumpEnv.STATE.HIP_VEL_Z],
                                                        ang_vel_y=0)
        thigh_pot_energy = thigh_mass*g*vstate[LegJumpEnv.STATE.THIGH_POS_Z]
        shin_pot_energy = shin_mass*g*vstate[LegJumpEnv.STATE.SHIN_POS_Z]
        slider_pot_energy = slider_mass*g*vstate[LegJumpEnv.STATE.HIP_POS_Z]
        return thigh_kin_energy+thigh_pot_energy, shin_kin_energy+shin_pot_energy, slider_kin_energy+slider_pot_energy

    @staticmethod
    def computeReward(previousState, state , action : int, env_conf, sub_rewards : Optional[Dict[str,th.Tensor]] = None, dbg_info = None) -> th.Tensor:

        vstate = state[LegJumpEnv.VECTOR_PART]
        prev_vstate = previousState[LegJumpEnv.VECTOR_PART]
        torque_lim_weight = env_conf["reward_torque_limit_weight"]
        position_lim_weight = env_conf["reward_position_limit_weight"]
        velocity_weight = env_conf["reward_velocity_weight"]
        tracking_weight = env_conf["reward_tracking_weight"]
        energy_weight = env_conf["reward_energy_weight"]
        torque_weight =  env_conf["reward_torque_weight"]
        contacts_weight = env_conf["reward_contacts_weight"]

        goal_dist = th.abs(vstate[LegJumpEnv.STATE.HIP_GOAL_Z] - vstate[LegJumpEnv.STATE.HIP_POS_Z])
        tracking_reward = 1 - goal_dist
        # tracking_reward = 1/(1+(goal_dist/0.1)**2) # halves at 0.1m

        ktorque = vstate[LegJumpEnv.STATE.KNEE_JOINT_EFFORT]*env_conf["knee_torque_scale"]
        htorque = vstate[LegJumpEnv.STATE.HIP_JOINT_EFFORT]*env_conf["hip_torque_scale"]
        shin_rotation = vstate[LegJumpEnv.STATE.SHIN_ANG_POS_Y] - prev_vstate[LegJumpEnv.STATE.SHIN_ANG_POS_Y]
        thigh_rotation = vstate[LegJumpEnv.STATE.THIGH_ANG_POS_Y] - prev_vstate[LegJumpEnv.STATE.THIGH_ANG_POS_Y]
        joint_efforts = [LegJumpEnv.STATE.HIP_JOINT_EFFORT,LegJumpEnv.STATE.KNEE_JOINT_EFFORT]
        joint_velocities = [LegJumpEnv.STATE.HIP_JOINT_VEL,LegJumpEnv.STATE.KNEE_JOINT_VEL]
        joint_positions = [LegJumpEnv.STATE.HIP_JOINT_POS,LegJumpEnv.STATE.KNEE_JOINT_POS]
        torques =       [vstate[k] for k in joint_efforts] # normalized torques
        velocities =    [vstate[k] for k in joint_velocities]
        positions =     [vstate[k] for k in joint_positions]
        torque_reward : th.Tensor = -(sum([t**4 for t in torques])/len(torques)) # type: ignore
        torque_limit_reward : th.Tensor =   -(sum([t**50 for t in torques])/len(torques)) # type: ignore # 0.0769 at 0.95
        velocity_reward : th.Tensor =       -(sum([t**2 for t in velocities])/len(velocities)) # type: ignore
        position_limit_reward : th.Tensor = -(sum([t**50 for t in positions])/len(positions)) # type: ignore # 0.0769 at 0.95
        contacts_reward = th.clamp(-(state[LegJumpEnv.VECTOR_PART][LegJumpEnv.STATE.IMPULSES_SUM]/env_conf["max_impulse"])**50, min = -1)

        new_thigh_energy, new_shin_energy, new_slider_energy = LegJumpEnv._compute_mechanical_energies(vstate)
        old_thigh_energy, old_shin_energy, old_slider_energy = LegJumpEnv._compute_mechanical_energies(prev_vstate)
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

        if sub_rewards is not None:
            sub_rewards["tracking_reward"] = tracking_reward
            sub_rewards["torque_reward"] = torque_reward
            sub_rewards["torque_limit_reward"] = torque_limit_reward
            sub_rewards["velocity_reward"] = velocity_reward
            sub_rewards["position_limit_reward"] = position_limit_reward
            sub_rewards["energy_reward"] = global_energy_reward
            sub_rewards["contacts_reward"] = contacts_reward
        return (tracking_weight*tracking_reward + 
                torque_lim_weight*torque_limit_reward + 
                velocity_weight*velocity_reward+
                position_lim_weight*position_limit_reward+
                energy_weight*global_energy_reward+
                torque_weight*torque_reward+
                contacts_weight * contacts_reward )


    def initializeEpisode(self) -> None:

        if not self._spawned and isinstance(self._environmentController, SimulatedEnvController):
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
            ggLog.info(f"Model spawned with name {name}")

            if self._show_goal:
                self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/red_intangible_ball.urdf.xacro"),
                                                        model_name="red_ball",
                                                        pose=leg_pose,
                                                        model_format="urdf.xacro")
            
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support",
                                                    pose=Pose(-0.1-0.125,0.3,0.2,0,0,0,1),
                                                    model_format="urdf.xacro")
            
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("jumping_leg","models/support.urdf.xacro"),
                                                    model_name="support2",
                                                    pose=Pose(-0.15-0.125,0.3,0.4,0,0,0,1),
                                                    model_format="urdf.xacro")

        
        if isinstance(self._environmentController, SimulatedEnvController):
            self._environmentController.setJointsStateDirect({self._rail_joint: JointState(position = [self._start_height], rate=[0], effort=[0]),
                                                              self._hip_joint:  JointState(position = [-3.14159/4], rate=[0], effort=[0]), 
                                                              self._knee_joint: JointState(position = [ 3.14159/2], rate=[0], effort=[0])})
        else:
            raise RuntimeError("Cannot reset joint state")
        self._environmentController.setJointsEffortCommand([(self._hip_joint,0),(self._knee_joint,0)])
        self._max_hip_height_reached = th.tensor(0)
        self._hip_goal_z = 0.4 + th.rand(size=(1,), generator=self._rng, device=self._th_device)*1.0
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

        if self._show_goal:
            ls = LinkState( position_xyz = th.tensor((0.,0.2,self._hip_goal_z)),
                            orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                            pos_velocity_xyz = th.tensor((0.,0.,0)),
                            ang_velocity_xyz = th.tensor((0.,0.,0.)))
            self._environmentController.setLinksStateDirect({("red_ball","world") : ls})
        
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
            vec_obs = state[self.VECTOR_PART][:self.STATE.HIP_GOAL_Z+1]
            return {self.VECTOR_PART : vec_obs}
        else:
            vec_obs = state[self.VECTOR_PART][:-2]
            img_obs = state[self.IMAGE_PART]
            if self._obs_only_img:
                return {self.IMAGE_PART : img_obs}
            else:
                return {self.IMAGE_PART : img_obs,
                        self.VECTOR_PART : vec_obs}
    @staticmethod        
    def _normalize(value, min_max):
        min,max = min_max
        return (value + (-min))/(max-min)  

    def getState(self) -> Dict[Any, th.Tensor]:
        """Get an observation of the environment.
        """

        jstates = self._environmentController.getJointsState(requestedJoints=[self._knee_joint, self._hip_joint])
        lstates : Dict[Tuple[str,str],LinkState] = self._environmentController.getLinksState(requestedLinks = [self._thigh_com_link,
                                                                              self._shin_com_link,
                                                                              self._thigh_base_link], use_com_frame = True)
        hip_height = lstates[self._thigh_base_link].pose.position[2]
        hip_vel_z = lstates[self._thigh_base_link].pos_velocity_xyz[2]
        self._max_hip_height_reached = th.maximum(self._max_hip_height_reached,hip_height)

        # n = '\n'
        # ggLog.info(f"contacts == {n.join([str(c) for c in contacts])}")
        thigh_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._thigh_com_link].pose.orientation_wxyz,
                                                                            th.tensor([0.0,1.0,0.0], device=self._th_device))[1])
        shin_ang_pos_y = quat_angle(quat_swing_twist_decomposition(lstates[self._shin_com_link].pose.orientation_wxyz,
                                                                            th.tensor([0.0,1.0,0.0], device=self._th_device))[1])

        contacts = self._environmentController.get_contacts()
        impulses = []
        for simsteps_contacts in contacts:
            impulses += [contact[3]*contact[4] for contact in simsteps_contacts]
        abs_impulses = [abs(i) for i in impulses]

        vstate = th.tensor((self._normalize(jstates[self._hip_joint].position[0],   self._position_limits[self._hip_joint]),
                            self._normalize(jstates[self._hip_joint].rate[0],       self._velocity_limits[self._hip_joint]),
                            self._normalize(jstates[self._hip_joint].effort[0],     self._torque_limits[self._hip_joint]),
                            self._normalize(jstates[self._knee_joint].position[0],  self._position_limits[self._knee_joint]),
                            self._normalize(jstates[self._knee_joint].rate[0],      self._velocity_limits[self._knee_joint]),
                            self._normalize(jstates[self._knee_joint].effort[0],    self._torque_limits[self._knee_joint]),
                            hip_height,
                            hip_vel_z,
                            self._hip_goal_z,
                            lstates[self._thigh_com_link].pos_velocity_xyz[0],
                            lstates[self._thigh_com_link].pos_velocity_xyz[2],
                            lstates[self._thigh_com_link].ang_velocity_xyz[1],
                            lstates[self._shin_com_link].pos_velocity_xyz[0],
                            lstates[self._shin_com_link].pos_velocity_xyz[2],
                            lstates[self._shin_com_link].ang_velocity_xyz[1],
                            lstates[self._thigh_com_link].pose.position[2],
                            thigh_ang_pos_y,
                            lstates[self._shin_com_link].pose.position[2],
                            shin_ang_pos_y,
                            sum(abs_impulses)),
                           dtype = th.float32,
                           device = self._th_device)
        # ggLog.info(f"vstate = {vstate}")
        # ggLog.info(f"thigh = {lstates}")
        if not self._obs_only_vec:
            istate, t = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            istate = th.tensor(istate, dtype = th.uint8, device = self._th_device)
        else:
            istate = th.empty(size=(0,), dtype = th.uint8, device = self._th_device)

        state = {self.VECTOR_PART : vstate,
                 self.IMAGE_PART : istate}
          
        return state

    def _compute_dbg_info(self):
        prev_state = self._last_state
        self._last_state = self.getState()
        if prev_state is None:
            prev_state = self._last_state
        rew_dbg_info = {}
        sub_rewards = {}
        r = self.computeReward(prev_state,
                               self._last_state, 
                               None, 
                               env_conf=self.get_configuration(),
                               dbg_info=rew_dbg_info,
                               sub_rewards=sub_rewards)
        contacts = self._environmentController.get_contacts()
        abs_impulses = []
        for simsteps_contacts in contacts:
            abs_impulses += [abs(contact[3]*contact[4]) for contact in simsteps_contacts]
        vstate = self._last_state[self.VECTOR_PART]
        goal_dist = abs(vstate[self.STATE.HIP_GOAL_Z]-vstate[self.STATE.HIP_POS_Z])
        self._cumulative_dist_to_goal += goal_dist
        self._cumulative_knee_torque += abs(vstate[self.STATE.KNEE_JOINT_EFFORT])
        self._cumulative_hip_torque += abs(vstate[self.STATE.HIP_JOINT_EFFORT])
        self._max_knee_torque = th.maximum(self._max_knee_torque, abs(vstate[self.STATE.KNEE_JOINT_EFFORT]))
        self._max_hip_torque = th.maximum(self._max_hip_torque, th.abs(vstate[self.STATE.HIP_JOINT_EFFORT]))
        self._last_abs_impulses_sum = vstate[self.STATE.IMPULSES_SUM]
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
        if envCtrlName in ["GazeboController", "GazeboControllerNoPlugin"]:
            # ggLog.info(f"sim_img_width  = {sim_img_width}")
            # ggLog.info(f"sim_img_height = {sim_img_height}")
            if not self._rendering_enabled:
                worldpath = "\"$(find lr_gym_ros)/worlds/ground_plane_world_plugin.world\""
            else:
                worldpath = "\"$(find lr_gym_ros)/worlds/fixed_camera_world_plugin.world\""
            self._environmentController.build_scenario( launch_file_pkg_and_path=("lr_gym_ros","/launch/gazebo_server.launch"),
                                                        launch_file_args={  "gui":"false",
                                                                            "paused":"true",
                                                                            "physics_engine":"bullet",
                                                                            "limit_sim_speed":"false",
                                                                            "world_name":worldpath,
                                                                            "gazebo_seed":f"{self._envSeed}",
                                                                            "wall_sim_speed":f"{self._wall_sim_speed}"})
            self._rendering_cam_name = "camera"
        elif envCtrlName == "GzController":
            self._environmentController.build_scenario(sdf_file = ("lr_gym_ros2","/worlds/empty_cams.sdf"))
            # self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/simple_camera.sdf.xacro"),
            #                                         model_name=None,
            #                                         pose=Pose(0,2,0.5,0,0.0,-0.707,0.707),
            #                                         model_kwargs={"camera_width":"1920","camera_height":"1080","frame_rate":1/self._intendedStepLength_sec},
            #                                         model_format="sdf.xacro")
            self._rendering_cam_name = "simple_camera"
        elif envCtrlName == "PyBulletController":
            self._environmentController.build_scenario(None)
            self._rendering_cam_name = "simple_camera"
        else:
            raise NotImplementedError("environmentController "+envCtrlName+" not supported")





    def _destroySimulation(self):
        self._environmentController.destroy_scenario()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        # ggLog.info(f"getInfo(): {self._stepCounter}")
        # i["step_count"] = self._stepCounter
        i["hip_goal_z"] = self._hip_goal_z
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
        i["thigh_vel_x_z"] = state[self.VECTOR_PART][[self.STATE.THIGH_VEL_X,self.STATE.THIGH_VEL_Z]]
        i["shin_vel_x_z"] = state[self.VECTOR_PART][[self.STATE.SHIN_VEL_X,self.STATE.SHIN_VEL_Z]]
        i["thigh_pos_z"] = state[self.VECTOR_PART][[self.STATE.THIGH_POS_Z]]
        i["shin_pos_z"] = state[self.VECTOR_PART][[self.STATE.SHIN_POS_Z]]
        i["vstate"] = state[self.VECTOR_PART]
        i.update(self._dbg_info)
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i

    def get_configuration(self):
        return self._configuration