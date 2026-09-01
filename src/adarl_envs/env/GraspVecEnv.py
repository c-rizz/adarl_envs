from __future__ import annotations

from adarl_envs.env.env_utils import joint_penalty_reward, norm_penalty, smoothclip_flattener
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.utils.utils import (LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, vector_projection, isinstance_noimport,
                               quat_xyzw_between_vecs_py, masked_assign, quat_mul_xyzw, quat_angle_xyzw, ros_rpy_to_quaternion_xyzw_th,
                               average_two_quaternions, dataclass2dict, pure_yaw_quaternion_xyzw_th)
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
from adarl_envs.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS, DistributionDefTh, DistributionDef, RobotVecEnvInitArgs, RobotAdapterData
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
import adarl.utils.tensor_trees
import traceback
import pprint
import dataclasses
from pathlib import Path
from adarl_envs.env.env_utils import flattened_joint_penalty_reward
from adarl.utils.dbg.dbg_checks import dbg_check_finite
from adarl_envs.env.env_utils import double_bell_reward
from adarl.utils.base_utils import record_time, print_recorded_times, record_region_start, record_region_end, clear_recorded_times, trace_malloc_diffs

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
def ramp_reward(error : th.Tensor, zero_rew_dist : th.Tensor):
    return 1-error/zero_rew_dist

@dataclass
class GrapVecEnvInitArgs():
    robot_init_args : RobotVecEnvInitArgs
    manipulator_tip_links : list[tuple[str,str]]
    manipulator_all_links : list[tuple[str,str]]
    reward_health_weight : float
    reward_joint_actacc_weight : float
    reward_joint_actdiff_weight : float
    reward_joint_position_limit_weight : float
    reward_joint_position_weight : float
    reward_joint_power_weight : float
    reward_joint_torque_weight : float
    reward_safety_weight : float
    reward_scale : float
    reward_object_pose_weight : float
    reward_gripper_pose_weight : float
    reward_height_position_weight : float
    reward_pitchnroll_weight : float
    reward_velocity_tracking_weight : float
    reward_yaw_vel_track_weight : float
    reward_feet_linvel_weight : float
    target_object_link : tuple[str,str]
    table_height : DistributionDef
    feet_contact_links : list[tuple[str,str]]
    feet_bottom_links : list[tuple[str,str]]
    """ Links representing the contact point of the feet, should be on the bottom of the foot, it is used for position-related logic"""
    neutral_body_height : float
    max_reach_height : float
    observe_object_pose : bool = False
    observe_camera : bool = False
    observe_initial_object_pose : bool = False
    gripper_link_transforms : list[tuple[float,float,float,float,float,float,float]] | None = None
    """Per gripper-link transform applied when computing the overall gripper pose.

    One transform per entry in ``gripper_links``, each expressed in the corresponding
    gripper link frame as ``(x, y, z, qx, qy, qz, qw)``. The transform is applied to the
    link pose before the link positions are averaged into the overall gripper position,
    so it can be used to move the reference point e.g. to the middle of the fingertips
    instead of the link origin reported by the robot description.
    If ``None``, an identity transform is used for every gripper link.
    """
    

@dataclass
class GraspAdapterData:
    """Raw per-step data read from the adapter by GraspVecEnv, wrapping the RobotVecEnv data."""
    robot_data : RobotAdapterData
    current_object_pose : th.Tensor
    current_gripper_poses : th.Tensor
    current_object_linvel_angvel : th.Tensor
    current_gripper_linvel_angvel : th.Tensor
    current_feetbottom_linvel_angvel : th.Tensor
    feet_touching_ground : th.Tensor
    obs_camera_image : th.Tensor | None

class GraspVecEnv(RobotVecEnv):
    STATE_GRASPING = "grasp"
    STATE_GRASPING_VELOCITIES = "grasp_velocities"
    STATE_FEET = "feet"
    STATE_CAMERA = "camera"

    @dataclass
    class GraspingConfiguration:
        obs_camera_render_resolution_hw : tuple[int,int]
        obs_camera_resolution_hw : tuple[int,int]
        ui_camera_resolution_hw : tuple[int,int]
        init_obj_area_minmax_xy : th.Tensor
        goal_obj_area_minmax_xy : th.Tensor
        table_link : tuple[str,str]
        use_head_cam_as_ui_camera : bool
        use_external_head_camera : bool
        split_rewards : bool
        gripper_link_transforms : th.Tensor
        show_gripper_marker : bool
        show_goal_marker : bool
        grasping_init_args : GrapVecEnvInitArgs
        table_height : DistributionDefTh

    @dataclass
    class SubRewards:
        health : th.Tensor
        joint_actacc : th.Tensor
        joint_actdiff : th.Tensor
        joint_power : th.Tensor
        joint_torque : th.Tensor
        joint_position : th.Tensor
        joint_position_limit : th.Tensor
        safety_triggered : th.Tensor
        object_pose : th.Tensor
        gripper_pose : th.Tensor
        height_position : th.Tensor
        pitchnroll : th.Tensor
        velocity_tracking : th.Tensor
        yaw_vel_track : th.Tensor
        feet_linvel : th.Tensor


    @dataclass
    class EpisodeGraspingConfiguration:
        initial_object_pose : th.Tensor
        goal_object_pose : th.Tensor
        table_height : th.Tensor

    GRASPING_POSES = IntEnum(       "GRASPING_POSES", [
                                    "GOAL_POSE",
                                    "OBJECT_POSE",
                                    "GRIPPER_POSE",
                                    "INITIAL_OBJ_POSE"
                                    ], start=0)
    GRASPING_VELOCITIES = IntEnum(  "GRASPING_VELOCITIES", [
                                    "OBJECT_LINVEL",
                                    "OBJECT_ANGVEL",
                                    "GRIPPER_LINVEL",
                                    "GRIPPER_ANGVEL"
                                    ], start=0)
    FEET = IntEnum("FEET", [        "FEET_VEL_X",
                                    "FEET_VEL_Y",
                                    "FEET_ON_GROUND"], start=0)

    CAMERA_FIELDS = IntEnum("CAMERA_FIELDS",   ["IMAGE"], start=0)


    def __init__(self,  grasp_init_args : GrapVecEnvInitArgs):
        robot_init_args = grasp_init_args.robot_init_args
        adapter = robot_init_args.adapter
        num_envs = adapter.vec_size()
        th_device = robot_init_args.th_device
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((num_envs,), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((num_envs,), device=th_device, dtype=th.bool)
        self._unit_3d_vector = self._thtens([1.0, 0.0, 0.0])
        self._unit_quaternion = self._thtens([0.0, 0.0, 0.0, 1.0])
        self._zero = self._thtens([0.0])
        # ITU-R 601 luma weights, folded with the uint8->[0,1] scale; precomputed once (avoids per-step H2D)
        self._rgb_to_gray_w = self._thtens([0.299, 0.587, 0.114])
        self._head_camera_name = "head_camera"
        self._grasp_ui_camera_name = "ui_camera"
        self._cube_size = 0.04
        manipulation_area_minmax_xy = [[ 0.45,  -0.15],
                                        [ 0.65,   0.15]]

        num_gripper_links = len(grasp_init_args.manipulator_tip_links)
        if grasp_init_args.gripper_link_transforms is None:
            gripper_link_transforms = th.zeros((num_gripper_links, 7), device=th_device, dtype=th.float32)
            gripper_link_transforms[:, 6] = 1.0 # identity quaternion (qw = 1)
        else:
            gripper_link_transforms = self._thtens(grasp_init_args.gripper_link_transforms)
            if gripper_link_transforms.shape != (num_gripper_links, 7):
                raise ValueError(f"gripper_link_transforms must have shape ({num_gripper_links}, 7) "
                                 f"(one (x,y,z,qx,qy,qz,qw) transform per gripper link), "
                                 f"got {tuple(gripper_link_transforms.shape)}")
        obs_cam_render_height = 64
        self._grasping_conf = GraspVecEnv.GraspingConfiguration(
                        obs_camera_render_resolution_hw = (obs_cam_render_height, int(obs_cam_render_height*16/10)),
                        obs_camera_resolution_hw = (64,64),
                        ui_camera_resolution_hw = (480,480),
                        init_obj_area_minmax_xy = self._thtens(manipulation_area_minmax_xy),
                        goal_obj_area_minmax_xy = self._thtens(manipulation_area_minmax_xy),
                        table_link = ("table","cube"),
                        use_head_cam_as_ui_camera = False,
                        use_external_head_camera = False,
                        split_rewards = False,
                        gripper_link_transforms = gripper_link_transforms,
                        show_gripper_marker = True, # spawn a small cube and place it at the computed gripper pose when rendering
                        show_goal_marker = True, # spawn a small cube and place it at the object goal pose when rendering
                        grasping_init_args = grasp_init_args,
                        table_height = self._distr_to_tensor(grasp_init_args.table_height, size=(num_envs,)))

        self._observation_camera = self._head_camera_name if grasp_init_args.observe_camera else None
        self._ui_camera = self._head_camera_name if self._grasping_conf.use_head_cam_as_ui_camera else self._grasp_ui_camera_name

        self._sub_rewards_weights2 = GraspVecEnv.SubRewards(
                health = self._thtens(grasp_init_args.reward_health_weight),
                joint_actacc = self._thtens(grasp_init_args.reward_joint_actacc_weight),
                joint_actdiff = self._thtens(grasp_init_args.reward_joint_actdiff_weight),
                joint_power = self._thtens(grasp_init_args.reward_joint_power_weight),
                joint_torque = self._thtens(grasp_init_args.reward_joint_torque_weight),
                joint_position_limit = self._thtens(grasp_init_args.reward_joint_position_limit_weight),
                joint_position = self._thtens(grasp_init_args.reward_joint_position_weight),
                safety_triggered = self._thtens(grasp_init_args.reward_safety_weight),
                object_pose = self._thtens(grasp_init_args.reward_object_pose_weight),
                gripper_pose = self._thtens(grasp_init_args.reward_gripper_pose_weight),
                height_position = self._thtens(grasp_init_args.reward_height_position_weight),
                pitchnroll = self._thtens(grasp_init_args.reward_pitchnroll_weight),
                velocity_tracking = self._thtens(grasp_init_args.reward_velocity_tracking_weight),
                yaw_vel_track = self._thtens(grasp_init_args.reward_yaw_vel_track_weight),
                feet_linvel = self._thtens(grasp_init_args.reward_feet_linvel_weight)
        )
        self._sub_rewards_enabled = {k:v for k,v in dataclasses.asdict(self._sub_rewards_weights2).items() if v!=0.0}
        self._sub_rewards_enabled_weights_th = self._thtens([v for v in self._sub_rewards_enabled.values()])

        self._grasping_episode_config = GraspVecEnv.EpisodeGraspingConfiguration(
                                                        initial_object_pose = self._thzeros((num_envs, 7)),
                                                        goal_object_pose = self._thzeros((num_envs, 7)),
                                                        table_height = self._thzeros((num_envs,)))
        if robot_init_args.enable_link_collisions is None:
            robot_init_args.enable_link_collisions = []
        cube_colliding_links = [self._grasping_conf.table_link]
        cube_colliding_links += self._grasping_conf.grasping_init_args.manipulator_all_links
        robot_init_args.enable_link_collisions.append((self._grasping_conf.grasping_init_args.target_object_link, cube_colliding_links))
        # Enable physical collisions between each foot and the ground so the feet (e.g. the legs,
        # when spawned) rest on the floor instead of passing through it.
        for foot in grasp_init_args.feet_contact_links:
            robot_init_args.enable_link_collisions.append((foot, [robot_init_args.ground_link]))
        # Keep the base at its homing spot (zero-width spawn box = no body-position randomization)
        _homing_xyz = tuple(robot_init_args.homing_body_pose_xyz_xyzw[:3])
        robot_init_args.randomized_homing_body_position_minmax_xyz=(_homing_xyz, _homing_xyz)
        robot_init_args.noise_abs_obs_linacc_ep_mustd_step_std=(0.0,0.0,0.0)
        super().__init__(robot_init_args)

    @override
    def _build_stats(self):
        self._stats = {}
        self._buff_sizes = int(self._configuration.init_args.maxStepsPerEpisode/10)
        self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))

    @override
    def _build(self):
        # Monitored collision pairs must be registered before the scenario is built (super()._build()).
        # They provide the feet-vs-ground contact signal used to gate the feet-slip penalty.
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            feet_ground_collision_pairs = [(foot, self._configuration.init_args.ground_link)
                                           for foot in self._grasping_conf.grasping_init_args.feet_contact_links]
            self._adapter.set_monitored_collision_pairs(feet_ground_collision_pairs)
        super()._build()
        self._adapter.set_monitored_links(self._adapter.get_monitored_links() +
                                          [self._grasping_conf.grasping_init_args.target_object_link] + 
                                          self._grasping_conf.grasping_init_args.manipulator_tip_links + 
                                          self._grasping_conf.grasping_init_args.feet_bottom_links)
        self._object_link_id = self._adapter.get_monitored_links_ids([self._grasping_conf.grasping_init_args.target_object_link])
        self._gripper_link_ids = self._adapter.get_monitored_links_ids(self._grasping_conf.grasping_init_args.manipulator_tip_links)
        self._feet_bottom_link_ids = self._adapter.get_monitored_links_ids(self._grasping_conf.grasping_init_args.feet_bottom_links)
        self._obj_and_gripper_and_feetbottom_link_ids = self._adapter.get_monitored_links_ids([self._grasping_conf.grasping_init_args.target_object_link]+
                                                                                              self._grasping_conf.grasping_init_args.manipulator_tip_links+
                                                                                              self._grasping_conf.grasping_init_args.feet_bottom_links)
        monitored_cameras = []
        if self._grasping_conf.grasping_init_args.observe_camera or self._grasping_conf.use_head_cam_as_ui_camera:
            monitored_cameras.append(self._head_camera_name)
        if not self._grasping_conf.use_head_cam_as_ui_camera:
            monitored_cameras.append(self._grasp_ui_camera_name)
        ggLog.info(f"setting monitored_cameras to {monitored_cameras}")
        self._adapter.set_monitored_cameras(monitored_cameras)

        # Hide the debug markers (axes + goal marker) from the observation (head) camera while other
        # cameras (e.g. the UI camera) still see them. Whitelist semantics: tell the observation
        # camera to see every model EXCEPT those markers. Needs observation and UI to be distinct
        # cameras (use_head_cam_as_ui_camera=False).
        if (self._grasping_conf.grasping_init_args.observe_camera
                and not self._grasping_conf.use_head_cam_as_ui_camera
                and hasattr(self._adapter, "set_body_camera_visibility")):
            hidden_models = {"goal_axes", "goal_marker_cube"}
            visible_models = sorted({l[0] for l in self._adapter.get_detected_links()} - hidden_models)
            self._adapter.set_body_camera_visibility([(self._head_camera_name, visible_models)])
    



    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        base_observable_fields=[ self.GRASPING_POSES.GOAL_POSE,
                            self.GRASPING_POSES.GRIPPER_POSE]
        if self._grasping_conf.grasping_init_args.observe_object_pose:
            base_observable_fields.append(self.GRASPING_POSES.OBJECT_POSE)
        if self._grasping_conf.grasping_init_args.observe_initial_object_pose:
            base_observable_fields.append(self.GRASPING_POSES.INITIAL_OBJ_POSE)
        privileged_observable_fields = [    self.GRASPING_POSES.GOAL_POSE,
                                            self.GRASPING_POSES.GRIPPER_POSE,
                                            self.GRASPING_POSES.OBJECT_POSE,
                                            self.GRASPING_POSES.INITIAL_OBJ_POSE]
        grasping_state_helper = ThBoxStateHelper( field_names=[e for e in self.GRASPING_POSES],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(7,),
                                                    fields_minmax={ 
                                                        self.GRASPING_POSES.GOAL_POSE :         [-10, 10],
                                                        self.GRASPING_POSES.OBJECT_POSE :       [-10, 10],
                                                        self.GRASPING_POSES.INITIAL_OBJ_POSE :  [-10, 10],
                                                        self.GRASPING_POSES.GRIPPER_POSE :      [-10, 10]},
                                                    vec_size=adapter.vec_size(),
                                                    history_length=2,
                                                    observation_definitions={
                                                        "base": ThBoxStateHelper.SimpleObsDef(
                                                            obs_history_length=1,
                                                            observable_fields=base_observable_fields,
                                                            observable_subfields=None),
                                                        "privileged": ThBoxStateHelper.SimpleObsDef(
                                                            obs_history_length=1,
                                                            observable_fields=privileged_observable_fields,
                                                            observable_subfields=None),
                                                            })
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_GRASPING,
                                                            grasping_state_helper,
                                                        obs_defs={"base":{"observable":True,"concatenate":True,"noise":None},
                                                                  "privileged":{"observable":True,"concatenate":True,"noise":None}})

        velocities_state_helper = ThBoxStateHelper( field_names=[e for e in self.GRASPING_VELOCITIES],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(3,),
                                                    fields_minmax={ self.GRASPING_VELOCITIES.OBJECT_LINVEL :  [-10, 10],
                                                                    self.GRASPING_VELOCITIES.OBJECT_ANGVEL :  [-10, 10],
                                                                    self.GRASPING_VELOCITIES.GRIPPER_LINVEL : [-10, 10],
                                                                    self.GRASPING_VELOCITIES.GRIPPER_ANGVEL : [-10, 10]},
                                                    vec_size=adapter.vec_size(),
                                                    history_length=1,
                                                    observation_definitions={
                                                        "base": ThBoxStateHelper.SimpleObsDef(
                                                            obs_history_length=1,
                                                            observable_fields=None,
                                                            observable_subfields=None),
                                                        "privileged": ThBoxStateHelper.SimpleObsDef(
                                                            obs_history_length=1,
                                                            observable_fields=None,
                                                            observable_subfields=None),
                                                            })
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_GRASPING_VELOCITIES,
                                                             velocities_state_helper,
                                                             obs_defs={
                                                                "base":{"observable":True,"concatenate":True,"noise":None},
                                                                "privileged":{"observable":True,"concatenate":True,"noise":None}})

        num_feet = len(self._grasping_conf.grasping_init_args.feet_bottom_links)
        feet_state_helper = ThBoxStateHelper( field_names=[e for e in self.FEET],
                                              dtype=self._obs_dtype,
                                              th_device=self._th_device,
                                              field_size=(num_feet,),
                                              fields_minmax={ self.FEET.FEET_VEL_X :    [-10, 10],
                                                              self.FEET.FEET_VEL_Y :    [-10, 10],
                                                              self.FEET.FEET_ON_GROUND: [0, 1]},
                                              vec_size=adapter.vec_size(),
                                              history_length=1,
                                              observation_definitions={
                                                "base": ThBoxStateHelper.SimpleObsDef(
                                                        obs_history_length=1,
                                                        observable_fields=None,
                                                        observable_subfields=None),
                                                "privileged": ThBoxStateHelper.SimpleObsDef(
                                                        obs_history_length=1,
                                                        observable_fields=None,
                                                        observable_subfields=None)})
        # Feet horizontal velocity and ground contact are tracked only to drive the feet-slip penalty, so they are not observed.
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_FEET,
                                                             feet_state_helper,
                                                             obs_defs={"base":{"observable":False,"concatenate":False,"noise":None},
                                                                       "privileged":{"observable":False,"concatenate":False,"noise":None}})
        
        



        if self._grasping_conf.grasping_init_args.observe_camera:
            camera_state_helper = ThBoxStateHelper( field_names=[e for e in self.CAMERA_FIELDS],
                                                    dtype=th.uint8,
                                                    normalization_range=(0,255),
                                                    th_device=self._th_device,
                                                    field_size=self._grasping_conf.obs_camera_resolution_hw,
                                                    fields_minmax={ self.CAMERA_FIELDS.IMAGE : [0,255]},
                                                    vec_size=adapter.vec_size(),
                                                    observation_definitions={"base":
                                                                            ThBoxStateHelper.SimpleObsDef(  obs_history_length=1,
                                                                                                            observable_fields=None,
                                                                                                            observable_subfields=None,
                                                                                                            skip_history_dim=True)})
            self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_CAMERA,
                                                                camera_state_helper,
                                                                obs_defs={  "base":{
                                                                                "observable":self._grasping_conf.grasping_init_args.observe_camera,
                                                                                "concatenate":False,
                                                                                "noise":None},
                                                                            # "privileged":{
                                                                            #     "observable":self._grasping_conf.grasping_init_args.observe_camera,
                                                                            #     "concatenate":False,
                                                                            #     "noise":None}
                                                                        }
                                                                                )

        self._grav_xy_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                                                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y))
        self._abs_linvel_idx = self._state_helper.sub_helpers[self.STATE_EXTRINSIC].field_idx((self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X,
                                                                                            self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y,
                                                                                            self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z))
        
        ggLog.info(f"Built state/obs/action helpers")


    def _get_adapter_data_raw(self):
        super_adapter_data =  super()._get_adapter_data_raw()
        poses = self._adapter.getLinksState(self._obj_and_gripper_and_feetbottom_link_ids, use_com_pose=False)
        # poses is ordered as [target_object] + manipulator_tip_links + feet_bottom_links (see _build)
        num_gripper_links = len(self._grasping_conf.grasping_init_args.manipulator_tip_links)
        feet_start = 1 + num_gripper_links
        current_object_pose = poses[:,0, :7]
        current_gripper_poses = poses[:,1:feet_start, :7]

        current_object_linvel_angvel = poses[:,0, 7:13]
        current_gripper_linvel_angvel = poses[:,1:feet_start, 7:13]
        current_feetbottom_linvel_angvel = poses[:,feet_start:, 7:13]

        # Feet-vs-ground contact, used to gate the feet-slip penalty. check_colliding_links returns one column
        # per monitored pair, ordered as feet_contact_links, which must align 1:1 with feet_bottom_links.
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            feet_touching_ground = self._adapter.check_colliding_links() # (vec_size, nfeet)
        else:
            feet_touching_ground = self._thzeros((self.num_envs, len(self._grasping_conf.grasping_init_args.feet_contact_links)))
        # ggLog.info(f"feet_touching_ground = {feet_touching_ground}")
        # ggLog.info(f"current_feetbottom_linvel_angvel = {current_feetbottom_linvel_angvel}")

        if self._grasping_conf.grasping_init_args.observe_camera:
            obs_camera_images, img_times = self._adapter.getRenderings([self._observation_camera])
            obs_camera_image = obs_camera_images[0]
            if not obs_camera_image.dtype.is_floating_point:
                obs_camera_image = obs_camera_image.to(th.float32) / 255.0
            obs_camera_image = obs_camera_image.permute(0, 3, 1, 2) # to (vec, C, H, W)
            obs_camera_image = th.nn.functional.interpolate(obs_camera_image,
                                                size=self._grasping_conf.obs_camera_resolution_hw,
                                                mode="bilinear",
                                                align_corners=False,
                                                antialias=True)
            obs_camera_image = obs_camera_image.permute(0, 2, 3, 1) # back to (vec, H, W, C)                
            # getRenderings gives HWC uint8 RGB (vec, H, W, 3); the CAMERA field is a single 64x64 plane,
            # so collapse RGB -> one grayscale channel scaled to [0,1]: (vec, H, W, 3) -> (vec, H, W).
            obs_camera_image = ((obs_camera_image[..., :3] @ self._rgb_to_gray_w)*255).to(self._obs_dtype)  # (vec, H, W) in [0,1]
        else:
            obs_camera_image = None

        return GraspAdapterData(robot_data = super_adapter_data,
                                current_object_pose = current_object_pose,
                                current_gripper_poses = current_gripper_poses,
                                current_object_linvel_angvel = current_object_linvel_angvel,
                                current_gripper_linvel_angvel = current_gripper_linvel_angvel,
                                current_feetbottom_linvel_angvel = current_feetbottom_linvel_angvel,
                                feet_touching_ground = feet_touching_ground,
                                obs_camera_image = obs_camera_image)

    @override
    def _get_new_instantaneous_state(self, adapter_data):
        super_adapter_data                = adapter_data.robot_data
        current_object_pose               = adapter_data.current_object_pose
        current_gripper_poses             = adapter_data.current_gripper_poses
        current_object_linvel_angvel      = adapter_data.current_object_linvel_angvel
        current_gripper_linvel_angvel     = adapter_data.current_gripper_linvel_angvel
        current_feetbottom_linvel_angvel  = adapter_data.current_feetbottom_linvel_angvel
        feet_touching_ground              = adapter_data.feet_touching_ground
        obs_camera_image                  = adapter_data.obs_camera_image
        new_inst_state = super()._get_new_instantaneous_state(super_adapter_data)

        # Apply the per-link transforms (expressed in each gripper link frame) before
        # averaging the link positions. This allows moving the reference point, e.g. to
        # the middle of the fingertips instead of the link origin reported by the robot.
        vec_size = current_gripper_poses.shape[0]
        num_gripper_links = current_gripper_poses.shape[1]
        link_pos = current_gripper_poses[:,:,:3]
        link_quat = current_gripper_poses[:,:,3:7]
        transforms = self._grasping_conf.gripper_link_transforms.unsqueeze(0).expand(vec_size, num_gripper_links, 7)
        transform_trans = transforms[:,:,:3]
        transform_quat = transforms[:,:,3:7]
        link_offset_world = th_quat_rotate(transform_trans, link_quat) # transform translation expressed in world frame
        transformed_link_pos = link_pos + link_offset_world
        transformed_link_quat = quat_mul_xyzw(link_quat, transform_quat)

        gripper_pos = transformed_link_pos.mean(dim=1)
        # gripper_quat = average_quaternions(transformed_link_quat)
        gripper_quat = transformed_link_quat[:,0] # for now just take the first gripper link's orientation as the gripper orientation, averaging can be weird if the gripper is closed and the fingers are in contact with the object, resulting in noisy orientations
        current_gripper_pose = th.cat([gripper_pos, gripper_quat], dim=1)
        # Velocity of the transformed (rigidly-attached) point: v_point = v_link + omega x r.
        # The angular velocity is unchanged by a rigid offset.
        link_linvel = current_gripper_linvel_angvel[:,:,:3]
        link_angvel = current_gripper_linvel_angvel[:,:,3:6]
        transformed_link_linvel = link_linvel + th.linalg.cross(link_angvel, link_offset_world, dim=-1)
        gripper_linvel = transformed_link_linvel.mean(dim=1)
        gripper_angvel = link_angvel.mean(dim=1)

        current_object_pose = current_object_pose.clamp(min=self._thtens([float('-inf'), float('-inf'), 0.0, 
                                                                          float('-inf'), float('-inf'), float('-inf'), float('-inf')]))

        # ggLog.info(f"current_object_pose = {current_object_pose}")
        # ggLog.info(f"current_gripper_pose = {current_gripper_pose}")
        # Express the object/goal/gripper poses in the robot's body reference frame (the same frame
        # used for the extrinsic body-relative quantities), so they are seen relative to the robot:
        # the position is rotated into the body frame and the orientation is made relative to it.
        # vec_bodystates_13 (body pose+vel) comes from the base RobotVecEnv adapter data.
        # The same rigid transform is applied to all three, so the reward distances are unchanged.
        body_state_13 = super_adapter_data.vec_bodystates_13
        body_pos = body_state_13[:, :3]
        conj_body_quat = th_quat_conj(body_state_13[:, 3:7])
        def _pose_to_body_frame(pose_v7):
            pos_body  = th_quat_rotate(pose_v7[:, :3] - body_pos, conj_body_quat)
            quat_body = quat_mul_xyzw(conj_body_quat, pose_v7[:, 3:7])
            return th.cat([pos_body, quat_body], dim=1)
        goal_pose    = _pose_to_body_frame(self._grasping_episode_config.goal_object_pose)
        object_pose  = _pose_to_body_frame(current_object_pose)
        gripper_pose = _pose_to_body_frame(current_gripper_pose)
        new_grasping_state = {self.GRASPING_POSES.GOAL_POSE   : goal_pose.expand(self.num_envs,7),
                              self.GRASPING_POSES.OBJECT_POSE : object_pose.expand(self.num_envs,7),
                              self.GRASPING_POSES.INITIAL_OBJ_POSE : self._grasping_episode_config.initial_object_pose.expand(self.num_envs,7),
                              self.GRASPING_POSES.GRIPPER_POSE : gripper_pose.expand(self.num_envs,7)}
        new_inst_state[self.STATE_GRASPING] = new_grasping_state

        new_grasping_velocities = {self.GRASPING_VELOCITIES.OBJECT_LINVEL: current_object_linvel_angvel[:, :3].expand(self.num_envs,3),
                                   self.GRASPING_VELOCITIES.OBJECT_ANGVEL: current_object_linvel_angvel[:, 3:6].expand(self.num_envs,3),
                                   self.GRASPING_VELOCITIES.GRIPPER_LINVEL: gripper_linvel.expand(self.num_envs,3),
                                   self.GRASPING_VELOCITIES.GRIPPER_ANGVEL: gripper_angvel.expand(self.num_envs,3)}
        new_inst_state[self.STATE_GRASPING_VELOCITIES] = new_grasping_velocities

        # World-frame horizontal feet velocity plus the ground-contact flag. The slip penalty uses only the XY
        # components (matching LocomotionVecEnv), and their norm is frame-invariant so no body-frame transform is needed.
        feet_linvel_xy = current_feetbottom_linvel_angvel[:, :, :2] # (vec_size, nfeet, 2)
        new_inst_state[self.STATE_FEET] = {self.FEET.FEET_VEL_X:     feet_linvel_xy[:, :, 0].expand(self.num_envs, -1),
                                           self.FEET.FEET_VEL_Y:     feet_linvel_xy[:, :, 1].expand(self.num_envs, -1),
                                           self.FEET.FEET_ON_GROUND: feet_touching_ground.to(self._obs_dtype).expand(self.num_envs, -1)}

        if self._grasping_conf.grasping_init_args.observe_camera:
            new_inst_state[self.STATE_CAMERA] = {self.CAMERA_FIELDS.IMAGE: obs_camera_image}
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
                                sub_rewards_return : dict[str,th.Tensor] | None = None) -> th.Tensor:
        if sub_rewards_return is None:
            sub_rewards_return = {}
        record_region_start("GraspVecEnv.compute_rewards")
        rewards, sub_rewards_dict = self._compute_rewards(state) # Avoid input mutation for compiled function
        sub_rewards_return.update(sub_rewards_dict)
        record_region_end("GraspVecEnv.compute_rewards")
        return rewards

    @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune",
                                      fullgraph=True,
                                    #   skip_eval_unsafe_warmup=100, skip_eval_unsafe_manual_arg_guard=0,
                                      disable=disable_compile)
    def _compute_rewards(self,   state : dict[str,th.Tensor]) -> tuple[th.Tensor, dict[str,th.Tensor]]:
        record_time("GraspVecEnv.compute_rewards: start")
        sub_rewards_return = {}
        max_rew = self._configuration.reward_penalties_max
        current_state_internal = state[self.STATE_INTERNAL][:, 0,:,0]
        curr_state_extr_vec =    state[self.STATE_EXTRINSIC][:, 0,:,0]
        state_action_raw_vec = state[self.STATE_ACT_RAW_HIST]
        state_stats_v_h_j_minmaxavgstd_pvaeep = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 6)
        last_step_dt = current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT].view((self.num_envs,))
        record_time("GraspVecEnv.compute_rewards: state unpack")

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_position, lims[0,:,0], lims[1,:,0])
        state_robot = state[self.STATE_ROBOT]
        state_robot_norm        = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state_robot, warn_limits_violation=False)
        longterm_stats_pos_norm = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)
        joints_num = state_robot_norm.size()[2]
        norm_posstathomingdiff    = longterm_stats_pos_norm[:,0,0] - normhoming
        actdiff             = th.flatten((state_action_raw_vec[:,0] - state_action_raw_vec[:,1])/2, start_dim=1) # divide by 2 to keep it in [-1,1]
        prev_actdiff        = th.flatten((state_action_raw_vec[:,1] - state_action_raw_vec[:,2])/2, start_dim=1)
        act_acc             = (actdiff - prev_actdiff)/2
        state_robot_safenorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state_robot, self._safety_limits, warn_limits_violation=False)
        state_robot_ctrlnorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state_robot, self._ctrl_limits, warn_limits_violation=False)
        position_ctrlnorm   = state_robot_ctrlnorm[:,0,:,0]

        record_time("GraspVecEnv.compute_rewards: state normalization")

        # ---------------- JOINT-LEVEL PENALTIES ----------------

        reward_position         = joint_penalty_reward(norm_posstathomingdiff,max_rew=max_rew, exponent=2.0)
        reward_actdiff          = joint_penalty_reward(actdiff,max_rew=1, exponent=2, presquash_factor=10)
        reward_actacc           = joint_penalty_reward(act_acc,max_rew=1, exponent=2, presquash_factor=100)
        reward_position_limit   = joint_penalty_reward(position_ctrlnorm,max_rew=1,exponent=50)

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

        record_time("GraspVecEnv.compute_rewards: joint-level penalties")

        # ---------------- SAFETY TRIGGERED REWARD ----------------
        # This is a penalty for triggering safety mechanisms
        safety_triggered = th.logical_or(state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED,0],
                                         state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED,0])
        reward_safety_triggered = -1*safety_triggered

        # # FAILURE SCALING
        # failed = (curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] < 0)
        # if self._configuration.fail_on_safety:
        #     failed = th.logical_or(failed, safety_triggered)
        record_time("GraspVecEnv.compute_rewards: safety-triggered reward")

        # ----------------- GRASPING REWARDS ----------------

        obj_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.OBJECT_POSE,:3]
        goal_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GOAL_POSE,:3]
        gripper_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GRIPPER_POSE,:3]
        obj2goal_dist = th.linalg.norm(obj_position - goal_position, dim = -1)
        obj2hand_dist = th.linalg.norm(obj_position - gripper_position, dim = -1)
        min_gripper_dist = 0.025 # Below this distance, the gripper is considered to be "touching" the object, and the reward is saturated
        obj2hand_dist = obj2hand_dist.clamp(min=min_gripper_dist)-min_gripper_dist

        max_dist = 1.0
        # reward_gripper_pose = 1-th.tanh(obj2hand_dist/max_dist)
        # reward_object_pose = 1-th.tanh(obj2goal_dist/max_dist)
        reward_gripper_pose = double_bell_reward(obj2hand_dist,
                                                 bell_width_a=0.5,
                                                 bell_width_b=0.03,
                                                 bell_b_weight=0.25)
        reward_object_pose = double_bell_reward(obj2goal_dist,
                                                 bell_width_a=0.4,
                                                 bell_width_b=0.1,
                                                 bell_b_weight=0.8)

        record_time("GraspVecEnv.compute_rewards: grasping rewards")
        # ------------------ BODY POSTURE REWARDS ----------------
        
        # ---- Height ----
        # Height of the body
        height_err = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z]-self._grasping_conf.grasping_init_args.neutral_body_height
        reward_height_position = -height_err**2
        record_time("GraspVecEnv.compute_rewards: body posture rewards: height")
        # ---- Pitch and Roll ----
        # Pitch and Roll of the body
        pitchnroll_err = th.linalg.norm(curr_state_extr_vec[:,self._grav_xy_idx], dim=-1)
        reward_pitchnroll = -pitchnroll_err**2
        record_time("GraspVecEnv.compute_rewards: body posture rewards: pitch and roll")

        # ---- Linear Velocity ----
        # Linear velocity of the body
        linvel = th.linalg.norm(curr_state_extr_vec[:, self._abs_linvel_idx], dim=-1)
        reward_velocity_tracking = -linvel**2
        record_time("GraspVecEnv.compute_rewards: body posture rewards: linear velocity")

        # ---- Yaw Velocity Tracking ----
        # Yaw velocity of the pelvis
        yawvel = curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z]
        reward_yaw_vel_track = -yawvel**2
        record_time("GraspVecEnv.compute_rewards: body posture rewards: yaw velocity")

        # ---- Feet Slip ----
        # Penalize feet sliding while in contact with the ground (same as LocomotionVecEnv's slip reward):
        # only the horizontal velocity counts, and only for feet that are touching the ground.
        feet_state = state[self.STATE_FEET][:,0] # vec_size*history*fields*nfeet -> vec_size*fields*nfeet
        feet_linvels_xy = th.stack([feet_state[:,self.FEET.FEET_VEL_X],
                                    feet_state[:,self.FEET.FEET_VEL_Y]], dim=1) # vec_size*2*nfeet
        feet_speeds = th.linalg.norm(feet_linvels_xy, dim=1) # vec_size*2*nfeet -> vec_size*nfeet
        feet_on_ground = feet_state[:,self.FEET.FEET_ON_GROUND] # vec_size*nfeet
        feet_sliding_speed = feet_speeds*feet_on_ground
        reward_feet_linvel = joint_penalty_reward(feet_sliding_speed, max_rew=1, exponent=2)
        # reward_feet_linvel = th.zeros_like(reward_yaw_vel_track)
        record_time("GraspVecEnv.compute_rewards: body posture rewards: feet sliding")


        raw_rewards = GraspVecEnv.SubRewards(
            health = th.ones_like(reward_position),
            joint_actacc = reward_actacc,
            joint_actdiff = reward_actdiff,
            joint_power = reward_power,
            joint_torque = reward_cmdtorque,
            safety_triggered = reward_safety_triggered,
            object_pose = reward_object_pose,
            gripper_pose = reward_gripper_pose,
            joint_position_limit = reward_position_limit,
            joint_position = reward_position,
            height_position = reward_height_position,
            pitchnroll = reward_pitchnroll,
            velocity_tracking = reward_velocity_tracking,
            yaw_vel_track = reward_yaw_vel_track,
            feet_linvel = reward_feet_linvel
        )
        sub_rew_unscaled = th.stack([dataclass2dict(raw_rewards)[k] for k in self._sub_rewards_enabled], dim=1)
        sub_rew_scaled = sub_rew_unscaled*self._sub_rewards_enabled_weights_th.unsqueeze(0)*self._grasping_conf.grasping_init_args.reward_scale

        record_time("GraspVecEnv.compute_rewards: sub-rewards scaling")

        sub_rewards_return.update({k:sub_rew_scaled[:,i] for i,k in enumerate(self._sub_rewards_enabled.keys())})
        if self._grasping_conf.split_rewards:
            reward = sub_rew_scaled
            dbg_check_size(reward, (self._adapter.vec_size(),len(sub_rewards_return)), f"Unexpected reward size")
        else:
            reward = th.sum(sub_rew_scaled, dim =1, keepdim=True)
            dbg_check_size(reward, (self._adapter.vec_size(),1), f"Unexpected reward size")
        reward = th.clamp(reward, -self._configuration.reward_clamp, self._configuration.reward_clamp)

        dbg_check_finite(sub_rewards_return, async_assert=True, assert_msg="Nonfinite sub rewards detected")
        record_time("GraspVecEnv.compute_rewards: end")

        return reward, sub_rewards_return




    def _update_stats(self, state):
        super()._update_stats(state)
        step_counts = state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT,0].to(th.long)
        starting_eps = step_counts==0
        self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))
        obj_pose  =      state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.OBJECT_POSE]
        prev_obj_pose  = state[self.STATE_GRASPING][:,1,self.GRASPING_POSES.OBJECT_POSE]
        obj_travel = th.linalg.norm(obj_pose[:,:3]-prev_obj_pose[:,:3], dim = -1)
        self._stats["ep_obj_travel"]             = (self._stats["ep_obj_travel"] + obj_travel) # Elements with step_count == 0 will be inf
        masked_assign(self._stats["ep_obj_travel"],         starting_eps, obj_travel)

        # self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        # self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        # self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
        # self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))

        # goal_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GOAL_POSE]
        # gripper_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GRIPPER_POSE]
        # obj2goal_dist = th.linalg.norm(obj_pose[:,:3]-goal_pose[:,:3], dim = -1)
        # obj2hand_dist = th.linalg.norm(obj_pose[:,:3]-gripper_pose[:,:3], dim = -1)

        # dbg_check_size(step_counts, (self._adapter.vec_size(),))

        # # Update episode averages
        # self._stats["ep_obj2hand_dist"]          = (self._stats["ep_obj2hand_dist"]*(step_counts-1) + obj2hand_dist)/step_counts # Elements with step_count == 0 will be inf
        # self._stats["ep_obj2goal_dist"]          = (self._stats["ep_obj2goal_dist"]*(step_counts-1) + obj2goal_dist)/step_counts # Elements with step_count == 0 will be inf
        # # Correct the episode averages for episodes that have just started
        # masked_assign(self._stats["ep_obj2hand_dist"],      starting_eps, obj2hand_dist)
        # masked_assign(self._stats["ep_obj2goal_dist"],      starting_eps, obj2goal_dist)
        # # Fill the buffers for episodes that have just staretd
        # masked_assign(self._stats["obj2hand_dist"],     step_counts==0, obj2hand_dist.unsqueeze(1).expand(-1, self._buff_sizes))
        # masked_assign(self._stats["obj2goal_dist"],     step_counts==0, obj2goal_dist.unsqueeze(1).expand(-1, self._buff_sizes))

        # # Update the buffers
        # # idxs = step_counts%self._buff_sizes
        # idxs = step_counts%self._stats["obj2hand_dist"].size()[1]
        # # print(f"torch.is_grad_enabled()) = {th.is_grad_enabled()}")
        # # print(f"idx.size() = {idxs.size()}, idx = {idxs}")
        # # print(f"vel_error_vec.size() = {vel_error_vec.size()}, {vel_error_vec}")
        # self._stats["obj2hand_dist"][:,idxs] = obj2hand_dist
        # self._stats["obj2goal_dist"][:,idxs] = obj2goal_dist

    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        i = super().get_infos(state=state, labels=labels)

        obj_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.OBJECT_POSE,:3]
        goal_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GOAL_POSE,:3]
        gripper_position = state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GRIPPER_POSE,:3]
        obj2goal_dist = th.linalg.norm(obj_position - goal_position, dim = -1)
        obj2hand_dist = th.linalg.norm(obj_position - gripper_position, dim = -1)
        i["obj2hand_dist"] = obj2hand_dist
        i["obj2goal_dist"] = obj2goal_dist

        sub_rews = {}
        self.compute_rewards(state, sub_rews)
        i["rewards"] = th.stack(list(sub_rews.values()), dim = 1)
        # ggLog.info(f"i['rewards'] = {i['rewards'].size()}")
        if labels is not None:
            labels["rewards"] = to_string_tensor(list(sub_rews.keys()))

        if self._configuration.init_args.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_GRASPING]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())

        return i

    @override
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        table_height = self._sample_distr((self.num_envs), self._grasping_conf.table_height)
        # record_time("GraspVecEnv._set_current_ep_config: sampled table height")
        min_obj_z = table_height + self._cube_size/2 + 0.005
        init_obj_minmax_z = min_obj_z.view(self.num_envs,1,1).expand(self.num_envs,2,1)
        init_obj_area_minmax_xyz = th.cat([self._grasping_conf.init_obj_area_minmax_xy.expand(self.num_envs,2,2),
                                           init_obj_minmax_z], dim=2)
        if "goal_pose" in reset_options:
            sampled_goal_obj_pose_v_xyzxyzw = self._thtens(reset_options["goal_pose"])
            # record_time("GraspVecEnv._set_current_ep_config: using provided goal pose")
        else:
            max_obj_z = self._thtens(self._grasping_conf.grasping_init_args.max_reach_height).expand(self.num_envs,1)
            goal_obj_minmax_z = th.stack([min_obj_z.view(self.num_envs,1),
                                        max_obj_z], dim=1).unsqueeze(2).view(self.num_envs,2,1)
            goal_obj_area_minmax_xyz = th.cat([self._grasping_conf.goal_obj_area_minmax_xy.expand(self.num_envs,2,2),
                                                goal_obj_minmax_z], dim=2)
            sampled_goal_obj_pose_v_xyz = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                                            goal_obj_area_minmax_xyz[:,0],
                                                            goal_obj_area_minmax_xyz[:,1])
            goal_obj_quat = ros_rpy_to_quaternion_xyzw_th(self._thrand((self.num_envs,3))*math.pi*2)
            sampled_goal_obj_pose_v_xyzxyzw = th.cat([sampled_goal_obj_pose_v_xyz, goal_obj_quat], dim=1).view((self.num_envs,7))
            # record_time("GraspVecEnv._set_current_ep_config: sampled goal pose")
        sampled_initial_object_position = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                                      init_obj_area_minmax_xyz[:,0],
                                                      init_obj_area_minmax_xyz[:,1])
        # record_time("GraspVecEnv._set_current_ep_config: sampled initial object position")
        sampled_initial_object_yaw = self._thrand((self.num_envs,))*math.pi*2
        # record_time("GraspVecEnv._set_current_ep_config: sampled initial object yaw")
        sampled_initial_obj_quat = pure_yaw_quaternion_xyzw_th(sampled_initial_object_yaw)
        # record_time("GraspVecEnv._set_current_ep_config: sampled initial object quat")
        sampled_initial_obj_pose = th.cat([sampled_initial_object_position,
                                   sampled_initial_obj_quat], dim=1).view((self.num_envs,7))
        # record_time("GraspVecEnv._set_current_ep_config: sampled initial object pose")
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
        masked_assign(self._grasping_episode_config.table_height, vec_mask, table_height)
        masked_assign(self._grasping_episode_config.initial_object_pose, vec_mask, sampled_initial_obj_pose)
        # masked_assign(self._grasping_episode_config.goal_object_pose, vec_mask, sampled_goal_obj_pose_v_xyzxyzw)
        self.set_max_episode_steps(reset_options.get("reset_options",self._current_episode_config.vec_max_ep_steps))
        self.set_goals(sampled_goal_obj_pose_v_xyzxyzw, vec_mask=vec_mask)

    def set_goals(self, goal_object_pose_vec_xyzxyzw : th.Tensor, reference_framce : Literal["absolute","relative"] = "absolute",
                  vec_mask : th.Tensor | None = None):
        if vec_mask is None:
            vec_mask = self._all_vecs
        if reference_framce == "relative":
            goal_object_pose_vec_xyzxyzw[:,:3] = goal_object_pose_vec_xyzxyzw[:,:3] + self._grasping_episode_config.goal_object_pose[:,:3]
            goal_object_pose_vec_xyzxyzw[:,3:7] = quat_mul_xyzw(goal_object_pose_vec_xyzxyzw[:,3:7], self._grasping_episode_config.goal_object_pose[:,3:7])
        # self._grasping_episode_config.goal_object_pose = goal_object_pose_vec_xyzxyzw
        masked_assign(self._grasping_episode_config.goal_object_pose, vec_mask, goal_object_pose_vec_xyzxyzw)

    def get_goals(self):
        return self._grasping_episode_config.goal_object_pose

    @override
    def are_states_terminal(self, states) -> th.Tensor:
        return super().are_states_terminal(states)

    def _set_gripper_marker_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter) and self._grasping_conf.show_gripper_marker:
            gripper_pose_body = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_POSES.GRIPPER_POSE]
            body_pose = self._adapter.getLinksState(self._main_body_mon_link_ids, use_com_pose=False)[:,0,:7]
            body_pos = body_pose[:, :3]
            body_quat = body_pose[:, 3:7]
            gripper_pos_world  = body_pos + th_quat_rotate(gripper_pose_body[:, :3], body_quat)
            gripper_quat_world = quat_mul_xyzw(body_quat, gripper_pose_body[:, 3:7])
            gripper_pose = th.cat([gripper_pos_world, gripper_quat_world], dim=1)
            self._adapter.setLinksStateDirect(  link_names=[self._gripper_marker_link],
                                                link_states_pose_vel=th.cat([gripper_pose,
                                                                             self._thzeros((self.num_envs,6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)

    def _set_goal_cube_marker_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter) and self._grasping_conf.show_goal_marker:
            self._adapter.setLinksStateDirect(  link_names=[self._goal_cube_marker_link],
                                                link_states_pose_vel=th.cat([self._grasping_episode_config.goal_object_pose,
                                                                             self._thzeros((self.num_envs,6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)

    @override
    def _simulation_initialization(self, vec_mask : th.Tensor):
        super()._simulation_initialization(vec_mask = vec_mask)
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            table_state = self._thzeros((self.num_envs,1,13))
            table_state[:,0,0] = 0.8
            table_state[:,0,1] = 0.0
            table_state[:,0,2] = self._grasping_episode_config.table_height-0.5
            table_state[:,0,6] = 1.0
            self._adapter.setLinksStateDirect(link_names=[self._grasping_conf.table_link],
                                              link_states_pose_vel=table_state,
                                              vec_mask=vec_mask)
            
            obj_state = self._thzeros((self.num_envs,1,13))
            obj_state[:,0,:7] = self._grasping_episode_config.initial_object_pose[:]
            self._adapter.setLinksStateDirect(link_names=[self._grasping_conf.grasping_init_args.target_object_link],
                                              link_states_pose_vel=obj_state,
                                              vec_mask=vec_mask)
        else:
            raise RuntimeError(f"Cannot run simulation initialization on non-simulated adapter")

    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        self.set_cam_pose((2.5, 30/180*3.14159, -200/180*3.14159))
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._set_gripper_marker_pose(vec_mask=self._all_vecs)
            self._set_goal_cube_marker_pose(vec_mask=self._all_vecs)
        if th.any(vec_mask[1:]):
            raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
        try:
            head_imgs, head_times = self._adapter.getRenderings([self._ui_camera], vec_mask=vec_mask)
            return head_imgs, head_times
            external_imgs, external_times = super().get_ui_renderings(vec_mask=vec_mask)
            imgs = head_imgs + external_imgs
            times = th.cat([head_times,external_times], dim = -1)
            return imgs, times
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return [], th.empty((0,))

    @override
    def _get_spawn_defs(self):
        spawn_defs = super()._get_spawn_defs()
        if isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
            subadapters = self._adapter.sub_adapters()
            is_pybullet = adarl.utils.utils.isinstance_noimport(subadapters[0], ("PyBulletJointImpedanceAdapter"))
        else:
            is_pybullet = False
        if not hasattr(self,"_cube_spawn_def"):
            self._cube_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/cube.urdf.xacro")).read_text(),
                                            name="cube",
                                            pose=None,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet),
                                                    "size" : self._cube_size})
        spawn_defs.append(self._cube_spawn_def)
        # if not hasattr(self,"_axes_spawn_def"):
        #     self._axes_spawn_def = ModelSpawnDef(   definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/axes.urdf.xacro")).read_text(),
        #                                             name="goal_axes",
        #                                             pose=None,
        #                                             format="urdf.xacro",
        #                                             kwargs={"add_world_link":str(is_pybullet),
        #                                                     "size" : 0.2})
        #     if not is_pybullet:
        #         self._goal_marker_base_link = ("goal_axes","root")
        #     else:
        #         self._goal_marker_base_link = ("goal_axes","world")
        # spawn_defs.append(self._axes_spawn_def)
        if self._grasping_conf.show_gripper_marker:
            if not hasattr(self,"_gripper_marker_spawn_def"):
                self._gripper_marker_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/cube.urdf.xacro")).read_text(),
                                                name="gripper_marker",
                                                pose=None,
                                                format="urdf.xacro",
                                                kwargs={"add_world_link":str(is_pybullet),
                                                        "size" :  0.01,
                                                        "red" :   0.0,
                                                        "green" : 0.0,
                                                        "blue" :  1.0,
                                                        "has_collision" : False})
                self._gripper_marker_link = ("gripper_marker","cube")
            spawn_defs.append(self._gripper_marker_spawn_def)
        if self._grasping_conf.show_goal_marker:
            if not hasattr(self,"_goal_cube_marker_spawn_def"):
                self._goal_cube_marker_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/cube.urdf.xacro")).read_text(),
                                                name="goal_marker_cube",
                                                pose=None,
                                                format="urdf.xacro",
                                                kwargs={"add_world_link":str(is_pybullet),
                                                        "size" :  self._cube_size,
                                                        "red" :   0.0,
                                                        "green" : 1.0,
                                                        "blue" :  0.0,
                                                        "has_collision" : False})
                self._goal_cube_marker_link = ("goal_marker_cube","cube")
            spawn_defs.append(self._goal_cube_marker_spawn_def)
        if not hasattr(self,"_table_spawn_def"):
            self._table_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/cube.urdf.xacro")).read_text(),
                                            name="table",
                                            pose=None,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet),
                                                    "size" :  1.0,
                                                    "red" :   0.5,
                                                    "green" : 0.5,
                                                    "blue" :  0.5,
                                                    "add_floating_joint" : False,
                                                    "add_fixed_joint" : True,
                                                    "fixed_joint_xyz" : f"0.0 0.0 0.0",
                                                    "fixed_joint_rpy" : "0 0 0"},)
        spawn_defs.append(self._table_spawn_def)
        if adarl.utils.utils.isinstance_noimport(self._adapter, ("MjxAdapter", "MujocoAdapter")):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:
            cam_file = "models/simple_camera.sdf.xacro"
        full_view_cam_xyz =   "2.0   0      2.0"
        full_view_cam_wxyz =  "0.0  -0.383  0       0.924"
        table_view_xyz =  "1.0   0      1.25"
        table_view_wxyz = "0.0  -0.5  0.0     0.866"
        if not hasattr(self,"_head_camera"):
            if self._grasping_conf.use_external_head_camera:
                attachment_link = None
                obs_cam_kwargs={"position_xyz": table_view_xyz,
                                "orientation_wxyz": table_view_wxyz}
            elif self._configuration.init_args.robot_name == "centauro":
                attachment_link = ("centauro","D435_head_camera_link")
                obs_cam_kwargs : dict = {}
            elif self._configuration.init_args.robot_name == "kyon":
                attachment_link = ("kyon","zed_front_down_mount_link")
                obs_cam_kwargs = {"position_xyz": "0.0 0.03 0.015"} # +x moves the camera forward
            else:
                attachment_link = None
                obs_cam_kwargs={"position_xyz": full_view_cam_xyz,
                                "orientation_wxyz": full_view_cam_wxyz}
            obs_cam_kwargs.update({ "camera_width":self._grasping_conf.obs_camera_render_resolution_hw[1],
                                    "camera_height":self._grasping_conf.obs_camera_render_resolution_hw[0],
                                    "frame_rate":1/self._intendedStepLength_sec,
                                    "camera_name": self._head_camera_name,
                                    "horizontal_fov_degrees":78.0})
            self._obs_cam_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                            name=self._head_camera_name,
                                            pose=None,
                                            format="sdf.xacro",
                                            kwargs=obs_cam_kwargs,
                                            attachment_link=attachment_link)
        spawn_defs.append(self._obs_cam_spawn_def)
        self._ui_cam_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                        name=self._grasp_ui_camera_name,
                                        pose=None,
                                        format="sdf.xacro",
                                        kwargs={"camera_width":self._grasping_conf.ui_camera_resolution_hw[0],
                                                "camera_height":self._grasping_conf.ui_camera_resolution_hw[1],
                                                "frame_rate":1/self._intendedStepLength_sec,
                                                "camera_name": self._grasp_ui_camera_name,
                                                "position_xyz": full_view_cam_xyz,
                                                "orientation_wxyz": full_view_cam_wxyz},)
        spawn_defs.append(self._ui_cam_spawn_def)
        return spawn_defs
