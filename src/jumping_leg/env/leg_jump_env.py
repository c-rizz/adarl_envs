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
from lr_gym.utils.utils import Pose, JointState, LinkState
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController
import torch as th
import lr_gym.utils.utils


class LegJumpEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""

    metadata = {'render.modes': ['rgb_array']}

    VECTOR_PART = "vec"
    IMAGE_PART = "img"
    HIP_JOINT_POS = 0
    HIP_JOINT_VEL = 1
    HIP_JOINT_EFFORT = 2
    KNEE_JOINT_POS = 3
    KNEE_JOINT_VEL = 4
    KNEE_JOINT_EFFORT = 5
    HIP_POS_Z = 6
    HIP_GOAL_Z = 7
    THIGH_VEL_X = 8
    THIGH_VEL_Z = 9
    THIGH_ANG_VEL_Y = 10
    SHIN_VEL_X = 11
    SHIN_VEL_Z = 12
    SHIN_ANG_VEL_Y = 13

    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    stepLength_sec : float = 0.01,
                    environmentController = None,
                    startSimulation : bool = True,
                    wall_sim_speed = False,
                    seed = 0,
                    obs_only_vec = False,
                    obs_only_img = False,
                    obs_img_height = 64,
                    obs_img_width = 64,
                    rgb = True,
                    th_device = th.device("cpu")):
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

        self._knee_joint = ("leg","knee_joint")
        self._hip_joint = ("leg","hip_joint")
        self._rail_joint = ("leg","rail_joint")
        self._foot_link = ("leg","foot_link")
        self._thigh_link_base = ("leg", "thigh_link_base")
        self._shin_link_base = ("leg", "shin_link_base")
        self._thigh_link = ("leg", "thigh_link")
        self._shin_link = ("leg", "shin_link")
        self._rendering_cam_name = "simple_camera"

        self._obs_only_vec = obs_only_vec
        self._obs_only_img = obs_only_img
        self._obs_img_height = obs_img_height
        self._obs_img_width = obs_img_width
        self._th_device = th_device
        self._rendering_enabled = True
        self._max_hip_height_reached = 0
        self._hip_torque_scale = 100
        self._knee_torque_scale = 100
        self._hip_goal_z = th.tensor(0.5,dtype=th.float32, device=self._th_device)

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
        self._cumulative_knee_torque = 0
        self._cumulative_hip_torque = 0
        self._max_knee_torque = 0
        self._max_hip_torque = 0

        self._show_goal = True

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = environmentController,
                         startSimulation = startSimulation,
                         simulationBackend = "")
        if self._obs_only_vec:
            vec_obs_size = 8
        else:
            vec_obs_size = 6
        vec_obs_space_high = np.array( [1.0]*vec_obs_size)
        vec_obs_space = spaces.gym_spaces.Box(-vec_obs_space_high,vec_obs_space_high)
        
        self._img_channels = 3 if rgb else 1
        img_shape_chw = (self._img_channels,self._obs_img_height,self._obs_img_width)
        img_observation_space = spaces.gym_spaces.Box(low=0, high=255, shape=img_shape_chw, dtype=np.uint8)

        self.state_space = spaces.gym_spaces.Dict({self.VECTOR_PART: spaces.gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(14,)),
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
        self._environmentController.setLinksToObserve([self._foot_link, self._shin_link, self._thigh_link])
        self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()
        
    def seed(self, seed : int) -> None:
        super().seed(seed)
        self._rng = self._rng.manual_seed(seed)



    def submitAction(self, action : th.Tensor) -> None:
        super().submitAction(action)
        htorque = action[0]*self._hip_torque_scale
        ktorque = action[1]*self._knee_torque_scale
        self._environmentController.setJointsEffortCommand(jointTorques = [(self._hip_joint,  htorque),
                                                                           (self._knee_joint, ktorque)])



    def checkEpisodeEnded(self, previousState : Tuple[float,float,float,float, np.ndarray], state : Tuple[float,float,float,float, np.ndarray]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True
        return False

    @staticmethod
    def compute_energies(vstate):
        thigh_mass = 3.37
        thigh_length = 0.3
        thigh_inertia_moment = 1/12*thigh_mass*thigh_length**2
        thigh_vel_sq = vstate[LegJumpEnv.THIGH_VEL_X]**2 + vstate[LegJumpEnv.THIGH_VEL_Z]**2
        thigh_ang_vel_sq = vstate[LegJumpEnv.THIGH_ANG_VEL_Y]**2
        thigh_energy = 0.5*thigh_mass*thigh_vel_sq + 0.5*thigh_inertia_moment*thigh_ang_vel_sq
        shin_mass = 3.37
        shin_length = 0.3
        shin_inertia_moment = 1/12*shin_mass*shin_length**2
        shin_vel_sq = vstate[LegJumpEnv.SHIN_VEL_X]**2 + vstate[LegJumpEnv.SHIN_VEL_Z]**2
        shin_ang_vel_sq = vstate[LegJumpEnv.SHIN_ANG_VEL_Y]**2
        shin_energy = 0.5*shin_mass*shin_vel_sq + 0.5*shin_inertia_moment*shin_ang_vel_sq
        return thigh_energy, shin_energy

    @staticmethod
    def computeReward(previousState, state , action : int, env_conf = None, sub_rewards : Optional[Dict[str,th.Tensor]] = None) -> th.Tensor:
        vstate = state[LegJumpEnv.VECTOR_PART]
        torque_weight = 0.1 #0.000001
        position_weight = 1.0 #0.000001
        velocity_weight = 0.0 #0.00001
        tracking_weight = 1.0
        energy_weight = 0.001
        goal_dist = th.abs(vstate[LegJumpEnv.HIP_GOAL_Z] - vstate[LegJumpEnv.HIP_POS_Z])
        # tracking_reward = (1 - th.abs(vstate[LegJumpEnv.HIP_GOAL_Z] - vstate[LegJumpEnv.HIP_POS_Z]))
        tracking_reward = 1/(1+(goal_dist/0.1)**2) # halves at 0.1m
        torques = [vstate[k] for k in [LegJumpEnv.HIP_JOINT_EFFORT,LegJumpEnv.KNEE_JOINT_EFFORT]]
        velocities = [vstate[k] for k in [LegJumpEnv.HIP_JOINT_VEL,LegJumpEnv.KNEE_JOINT_VEL]]
        positions = [vstate[k] for k in [LegJumpEnv.HIP_JOINT_POS,LegJumpEnv.KNEE_JOINT_POS]]
        torque_reward : th.Tensor = -(sum([t**2 + 10*t**50 for t in torques])/len(torques)) # type: ignore
        velocity_reward : th.Tensor = -(sum([t**2 for t in velocities])/len(velocities)) # type: ignore
        position_reward : th.Tensor = -(sum([t**10 for t in positions])/len(positions)) # type: ignore

        new_thigh_energy, new_shin_energy = LegJumpEnv.compute_energies(vstate)
        old_thigh_energy, old_shin_energy = LegJumpEnv.compute_energies(previousState[LegJumpEnv.VECTOR_PART])
        thigh_energy_diff = new_thigh_energy-old_thigh_energy
        shin_energy_diff = new_shin_energy-old_shin_energy
        energy_reward = - (thigh_energy_diff*thigh_energy_diff + shin_energy_diff*shin_energy_diff)

        if sub_rewards is not None:
            sub_rewards["tracking_reward"] = tracking_reward
            sub_rewards["torque_reward"] = torque_reward
            sub_rewards["velocity_reward"] = velocity_reward
            sub_rewards["position_reward"] = position_reward
            sub_rewards["energy_reward"] = energy_reward
        return (tracking_weight*tracking_reward + 
                torque_weight*torque_reward + 
                velocity_weight*velocity_reward+
                position_weight*position_reward+
                energy_weight*energy_reward)


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
                                                    pose=Pose(0,2,0.5, 0.0,0.0,-0.707,0.707),
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
                                                    pose=Pose(-0.1-0.125,0.3,0.4,0,0,0,1),
                                                    model_format="urdf.xacro")

        
        if isinstance(self._environmentController, SimulatedEnvController):
            self._environmentController.setJointsStateDirect({self._rail_joint: JointState(position = [0.55], rate=[0], effort=[0]),
                                                              self._hip_joint:  JointState(position = [-3.14159/4], rate=[0], effort=[0]), 
                                                              self._knee_joint: JointState(position = [ 3.14159/2], rate=[0], effort=[0])})
        else:
            raise RuntimeError("Cannot reset joint state")
        self._environmentController.setJointsEffortCommand([(self._hip_joint,0),(self._knee_joint,0)])
        self._max_hip_height_reached = 0
        self._hip_goal_z = 0.3 + th.rand(size=(1,), generator=self._rng, device=self._th_device)*1.0
        self._last_step_got_state = -1
        self._cumulative_dist_to_goal = 0
        self._cumulative_knee_torque = 0
        self._cumulative_hip_torque = 0
        self._max_knee_torque = 0
        self._max_hip_torque = 0

        if self._show_goal:
            ls = LinkState( position_xyz = th.tensor((0.,0.2,self._hip_goal_z)),
                            orientation_xyzw = th.tensor((0.,0.,0.,1.0)),
                            pos_velocity_xyz = th.tensor((0.,0.,0)),
                            ang_velocity_xyz = th.tensor((0.,0.,0.)))
            self._environmentController.setLinksStateDirect({("red_ball","world") : ls})
        
    def getUiRendering(self) -> Tuple[Union[np.ndarray, th.Tensor], float]:
        try:
            img, t = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            # return imgs[0]
            # img = self._environmentController.getRenderings(["box::simple_camera_link::simple_camera"])["box::simple_camera_link::simple_camera"]
            if img is None:
                time = -1
            else:
                time = t
            return img, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {lr_gym.utils.utils.exc_to_str(e)}")
            return None, 0


    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        if self._obs_only_vec:
            vec_obs = state[self.VECTOR_PART][:self.THIGH_VEL_X]
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
        lstates = self._environmentController.getLinksState(requestedLinks = [self._thigh_link,
                                                                              self._shin_link,
                                                                              self._thigh_link_base], use_com_frame = True)
        hip_height = lstates[self._thigh_link_base].pose.position[2]
        self._max_hip_height_reached = max(self._max_hip_height_reached,hip_height)
        vstate = th.tensor((self._normalize(jstates[self._hip_joint].position[0],   self._position_limits[self._hip_joint]),
                            self._normalize(jstates[self._hip_joint].rate[0],       self._velocity_limits[self._hip_joint]),
                            self._normalize(jstates[self._hip_joint].effort[0],     self._torque_limits[self._hip_joint]),
                            self._normalize(jstates[self._knee_joint].position[0],  self._position_limits[self._knee_joint]),
                            self._normalize(jstates[self._knee_joint].rate[0],      self._velocity_limits[self._knee_joint]),
                            self._normalize(jstates[self._knee_joint].effort[0],    self._torque_limits[self._knee_joint]),
                            hip_height,
                            self._hip_goal_z,
                            lstates[self._thigh_link].pos_velocity_xyz[0],
                            lstates[self._thigh_link].pos_velocity_xyz[2],
                            lstates[self._thigh_link].ang_velocity_xyz[1],
                            lstates[self._shin_link].pos_velocity_xyz[0],
                            lstates[self._shin_link].pos_velocity_xyz[2],
                            lstates[self._shin_link].ang_velocity_xyz[1]),
                           dtype = th.float32,
                           device = self._th_device)
        # ggLog.info(f"vstate = {vstate}")
        # ggLog.info(f"thigh = {lstates}")
        if not self._obs_only_vec:
            istate, t = self._environmentController.getRenderings([self._rendering_cam_name])[self._rendering_cam_name]
            istate = th.tensor(istate, dtype = th.uint8, device = self._th_device)
        else:
            istate = th.empty(size=(0,), dtype = th.uint8, device = self._th_device)

        if self._last_step_got_state < self._stepCounter:
            self._cumulative_dist_to_goal += abs(vstate[self.HIP_GOAL_Z]-vstate[self.HIP_POS_Z])
            self._cumulative_knee_torque += abs(vstate[self.KNEE_JOINT_EFFORT])
            self._cumulative_hip_torque += abs(vstate[self.HIP_JOINT_EFFORT])
            self._max_knee_torque = max(self._max_knee_torque, abs(vstate[self.KNEE_JOINT_EFFORT]))
            self._max_hip_torque = max(self._max_hip_torque, abs(vstate[self.HIP_JOINT_EFFORT]))

        self._last_step_got_state = self._stepCounter
        return {self.VECTOR_PART : vstate,
                self.IMAGE_PART : istate}


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
        i["hip_goal_z"] = self._hip_goal_z
        if self._stepCounter:
            i["avg_dist"] = self._cumulative_dist_to_goal/self._stepCounter
            i["avg_dist"] = self._cumulative_dist_to_goal/self._stepCounter
            i["avg_knee_torque"] = self._cumulative_knee_torque/self._stepCounter
            i["avg_hip_torque"] = self._cumulative_hip_torque/self._stepCounter
        else:
            i["avg_dist"] = float("+inf")
            i["avg_dist"] = float("+inf")
            i["avg_knee_torque"] = float("+inf")
            i["avg_hip_torque"] = float("+inf")
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i
