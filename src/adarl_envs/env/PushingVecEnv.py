#!/usr/bin/env python3
"""Vectorized port of the simplified (robot-less) panda pushing environment.

This is the adarl re-implementation of ``panda_pushing.PandaPushingEnv2DOF`` in its "simplified"
mode: there is no arm, just a cylindrical end effector floating over the operating plane, carried by
two prismatic joints along the world x and y axes. The agent commands a planar displacement of the
end effector and has to push a cube onto a goal position.

The end effector is position-controlled through a :class:`BaseVecCartesianPositionAdapter` (for MJX,
:class:`Mjx2DofCartesianAdapter`), whose step is blocking: a step of this environment lasts as long
as the commanded movement takes, like it did in the original environment.

The state is described by a :class:`DictStateHelper` built in :meth:`_build_state_helper`, like in
:class:`RobotVecEnv`: each substate declares its fields, their limits, and which of them each
observation ("base" for the policy, "privileged" for the critic) can see.
"""
from __future__ import annotations

from adarl.adapters.BaseVecCartesianPositionAdapter import BaseVecCartesianPositionAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
from adarl.utils.dbg.dbg_checks import dbg_check, dbg_check_finite
from adarl.utils.spaces import ThBox
from adarl.utils.tensor_trees import space_from_tree
from adarl.utils.utils import to_string_tensor, th_quat_rotate, ros_rpy_to_quaternion_xyzw
from adarl.utils.vec_state_helper import ThBoxStateHelper, DictStateHelper
from enum import IntEnum
from pathlib import Path
from typing import Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import math
import numpy as np
import torch as th


class PushingVecEnv(ControlledVecEnv):
    """Simplified planar pushing environment, vectorized, MJX-first."""

    STATE_POSE = "pose"
    STATE_INTERNAL = "internal"
    STATE_CAMERA = "camera"

    POSE_FIELDS = IntEnum("POSE_FIELDS", ["TIP_X",
                                          "TIP_Y",
                                          "CUBE_X",
                                          "CUBE_Y",
                                          "CUBE_YAW_COS",
                                          "CUBE_YAW_SIN",
                                          "GOAL_X",
                                          "GOAL_Y"], start=0)

    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", ["STEP_COUNT",
                                                  "TIME",
                                                  "CUBE_DISPLACEMENT",
                                                  "EE_TRACKING_ERROR"], start=0)

    CAMERA_FIELDS = IntEnum("CAMERA_FIELDS", ["IMAGE"], start=0)

    def __init__(self,
                 adapter : BaseVecCartesianPositionAdapter,
                 th_device : th.device,
                 max_episode_steps : int = 50,
                 step_duration_sec : float = 0.5,
                 seed : int = 0,
                 observe_camera : bool = True,
                 history_length : int = 1,
                 frame_stack_length : int = 1,
                 obs_camera_resolution_hw : tuple[int,int] = (64,64),
                 obs_camera_render_resolution_hw : tuple[int,int] = (144,256),
                 img_crop_ltrb : tuple[float,float,float,float] = (220/848, 0.0, 708/848, 420/480),
                 ui_camera_resolution_hw : tuple[int,int] = (240,426),
                 operating_area_xy : tuple[tuple[float,float],tuple[float,float]] = ((0.2975,-0.225),(0.7475,0.225)),
                 goal_tolerance : float = 0.05,
                 random_goal : bool = False,
                 goal_position_xy : tuple[float,float] = (0.45,-0.15),
                 max_position_change : float = 0.025,
                 ee_height : float = 0.025,
                 ee_diameter : float = 0.035,
                 cube_size : float = 0.06,
                 cube_mass : float = 0.1,
                 prevent_ee_out : bool = True,
                 terminate_on_success : bool = False,
                 sparse_reward : bool = False,
                 reward_cube_pos_weight : float = 1.0,
                 reward_tip_pos_weight : float = 0.0,
                 reward_cube_move_weight : float = 0.0,
                 reward_scale : float = 0.1,
                 goal_spawn_border_dist : float = 0.01,
                 cube_spawn_border_dist : float = 0.08,
                 ee_spawn_border_dist : float = 0.02,
                 allow_successful_initial_cube_position : bool = True,
                 spawn_rejection_candidates : int = 16,
                 cube_color : tuple[float,float,float] = (0.175, 0.175, 0.175),
                 camera_position_xyz : tuple[float,float,float] = (0.9935, -0.0225, 0.6),
                 camera_orientation_rpy : tuple[float,float,float] = (0.0, 55/180*math.pi, math.pi),
                 camera_offset_xyz : tuple[float,float,float] = (0.0, 0.0, 0.0),
                 camera_offset_rpy : tuple[float,float,float] = (0.0, 0.0, 0.0),
                 enable_ui_camera : bool = True):
        """
        Parameters
        ----------
        adapter : BaseVecCartesianPositionAdapter
            The adapter driving the end effector. It must also be a BaseVecSimulationAdapter.
        max_episode_steps : int
            Episode length, in agent steps.
        step_duration_sec : float
            Nominal duration of a step, i.e. ControlledVecEnv's step length. With a blocking
            (position-controlled) adapter a step lasts as long as the commanded movement takes, so
            this value is not enforced: step_precision_tolerance is set to infinity and the adapter
            decides the actual duration.
        observe_camera : bool
            If True the "base" observation carries the camera image, and the cube pose is left out of
            its vector part, i.e. the policy can only see the cube through the image, like the
            original environment's default mode. If False the cube pose is in the vector instead.
            The "privileged" observation always sees the full state either way.
        history_length : int
            How many past steps the state keeps. 1 means only the current one.
        frame_stack_length : int
            How many past steps the "privileged" observation sees. Must be <= history_length.
        random_goal : bool
            If True a goal position is sampled inside the operating area at every episode. If False
            (the default, as in the original experiments) the goal is fixed at goal_position_xy.
        goal_position_xy : tuple[float,float]
            The goal position used when random_goal is False. The default is the one the original
            setup used, and is also where the old simplified model's goal marker was fixed.
        max_position_change : float
            Maximum end-effector displacement commanded by a single action, in meters.
        """
        adapter_types = (BaseVecCartesianPositionAdapter, BaseVecSimulationAdapter)
        for t in adapter_types:
            if not isinstance(adapter, t):
                raise AttributeError(f"{type(self).__name__} requires an adapter implementing {t.__name__}, "
                                     f"but got a {type(adapter).__name__}")

        area_min = np.array(operating_area_xy[0], dtype=np.float64)
        area_max = np.array(operating_area_xy[1], dtype=np.float64)
        area_ranges = area_max - area_min
        if abs(area_ranges[0] - area_ranges[1]) > 1e-9:
            raise AttributeError(f"The operating area must be square (the scene model uses a single "
                                 f"area_size), but got ranges {area_ranges}")
        self._area_size = float(area_ranges[0])
        self._area_center_xy = (area_min + area_max)/2
        self._operating_area_np = np.stack([area_min, area_max])

        self._observe_camera = observe_camera
        self._history_length = history_length
        self._frame_stack_length = frame_stack_length
        self._obs_camera_resolution_hw = obs_camera_resolution_hw
        self._obs_camera_render_resolution_hw = obs_camera_render_resolution_hw
        self._img_crop_ltrb = img_crop_ltrb
        self._ui_camera_resolution_hw = ui_camera_resolution_hw
        self._enable_ui_camera = enable_ui_camera
        self._goal_tolerance = goal_tolerance
        self._max_position_change = max_position_change
        self._ee_height = ee_height
        self._ee_diameter = ee_diameter
        self._cube_size = cube_size
        self._cube_mass = cube_mass
        self._prevent_ee_out = prevent_ee_out
        self._terminate_on_success = terminate_on_success
        self._sparse_reward = sparse_reward
        self._reward_cube_pos_weight = reward_cube_pos_weight
        self._reward_tip_pos_weight = reward_tip_pos_weight
        self._reward_cube_move_weight = reward_cube_move_weight
        self._reward_scale = reward_scale
        self._goal_spawn_border_dist = goal_spawn_border_dist
        self._cube_spawn_border_dist = cube_spawn_border_dist
        self._ee_spawn_border_dist = ee_spawn_border_dist
        self._allow_successful_initial_cube_position = allow_successful_initial_cube_position
        self._random_goal = random_goal
        self._goal_position_xy = goal_position_xy
        self._spawn_rejection_candidates = spawn_rejection_candidates
        self._cube_color = cube_color
        self._camera_position_xyz = tuple(np.array(camera_position_xyz) + np.array(camera_offset_xyz))
        self._camera_orientation_rpy = tuple(np.array(camera_orientation_rpy) + np.array(camera_offset_rpy))

        self._ee_link = ("pushing", "pushing_ee")
        self._cube_link = ("cube", "cube")
        self._goal_marker_link = ("pushing", "goal")
        self._xjoint = ("pushing", "ee_x_slider")
        self._yjoint = ("pushing", "ee_y_slider")
        self._obs_camera_name = "obs_camera"
        self._ui_camera_name = "ui_camera"

        self._build_state_helper(adapter=adapter, th_device=th_device)

        act_max = np.array([1.0, 1.0])
        super().__init__(th_device=th_device,
                         seed=seed,
                         obs_dtype=th.float32,
                         single_action_space=ThBox(-act_max, act_max, torch_device=th_device),
                         single_observation_space=self._state_helper.get_single_obs_space(),
                         single_state_space=self._state_helper.get_single_space(),
                         single_reward_space=ThBox(low=float("-inf"), high=float("+inf"),
                                                   shape=tuple(), torch_device=th_device),
                         info_space=None, # type: ignore : set below, it needs the env to be initialized
                         step_duration_sec=step_duration_sec,
                         adapter=adapter, # type: ignore : it is also a BaseVecAdapter
                         max_episode_steps=max_episode_steps,
                         # The adapter's step lasts as long as the commanded movement takes, so the
                         # step duration cannot be checked against the nominal one.
                         step_precision_tolerance=float("+inf"))

        # Cached field index tensors, so that indexing groups of fields does not build a tensor (and
        # sync) on every step.
        pose_helper = self._state_helper.sub_helpers[self.STATE_POSE]
        self._idx_tip_xy = pose_helper.field_idx((self.POSE_FIELDS.TIP_X, self.POSE_FIELDS.TIP_Y))
        self._idx_cube_xy = pose_helper.field_idx((self.POSE_FIELDS.CUBE_X, self.POSE_FIELDS.CUBE_Y))
        self._idx_goal_xy = pose_helper.field_idx((self.POSE_FIELDS.GOAL_X, self.POSE_FIELDS.GOAL_Y))

        self._area_min_xy = th.as_tensor(area_min, dtype=th.float32, device=th_device)
        self._area_max_xy = th.as_tensor(area_max, dtype=th.float32, device=th_device)
        self._fixed_goal_xy = th.as_tensor(goal_position_xy, dtype=th.float32, device=th_device)
        self._goal_xy = self._thzeros((self.num_envs, 2))
        self._prev_cube_xy = self._thzeros((self.num_envs, 2))
        self._cached_tip_xy = self._thzeros((self.num_envs, 2))
        self._cached_cube_xy = self._thzeros((self.num_envs, 2))
        self._success = th.zeros((self.num_envs,), dtype=th.bool, device=th_device)
        self._ep_min_cube2goal_dist = self._thfull(float("+inf"), (self.num_envs,))
        self._ep_min_cube2tip_dist = self._thfull(float("+inf"), (self.num_envs,))
        self._ep_cube2goal_dist_sum = self._thzeros((self.num_envs,))
        self._ep_cube2tip_dist_sum = self._thzeros((self.num_envs,))
        self._ep_cube_travel = self._thzeros((self.num_envs,))
        self._ep_steps_before_success = self._thfull(float(max_episode_steps+1), (self.num_envs,))
        self._no_success_steps = self._thfull(float(max_episode_steps+1), (self.num_envs,))
        self._rgb_to_gray_w = self._thtens([0.299, 0.587, 0.114])
        self._current_state = self._state_helper.reset_state()

        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done after super().__init__

        self._build()
        self._adapter.startup()
        self.initialize_episodes()

    # ------------------------------------------------------------------ state definition

    def _build_state_helper(self, adapter : BaseVecCartesianPositionAdapter, th_device : th.device):
        """Builds the state helper, which defines how the state is represented and observed."""

        vsize_dev_type = dict(dtype=th.float32, th_device=th_device, vec_size=adapter.vec_size())
        area_min, area_max = self._operating_area_np[0], self._operating_area_np[1]
        # Positions are allowed a margin outside the operating area: the cube can be pushed past its
        # edge, and the limits are what normalize() maps into [-1,1].
        margin = 0.15
        x_minmax = [float(area_min[0])-margin, float(area_max[0])+margin]
        y_minmax = [float(area_min[1])-margin, float(area_max[1])+margin]

        # :::::::::::::::::::::::::::::::::::::::: POSE STATE ::::::::::::::::::::::::::::::::::::::::

        # The cube pose is privileged whenever the policy is supposed to read it off the camera
        # image, which reproduces the original environment's default observation.
        base_pose_observable_fields = [self.POSE_FIELDS.TIP_X,
                                       self.POSE_FIELDS.TIP_Y,
                                       self.POSE_FIELDS.GOAL_X,
                                       self.POSE_FIELDS.GOAL_Y]
        if not self._observe_camera:
            base_pose_observable_fields += [self.POSE_FIELDS.CUBE_X,
                                            self.POSE_FIELDS.CUBE_Y,
                                            self.POSE_FIELDS.CUBE_YAW_COS,
                                            self.POSE_FIELDS.CUBE_YAW_SIN]
        pose_state_helper = ThBoxStateHelper(
                field_names=[e for e in self.POSE_FIELDS],
                field_size=(1,),
                fields_minmax={
                        self.POSE_FIELDS.TIP_X : x_minmax,
                        self.POSE_FIELDS.TIP_Y : y_minmax,
                        self.POSE_FIELDS.CUBE_X : x_minmax,
                        self.POSE_FIELDS.CUBE_Y : y_minmax,
                        self.POSE_FIELDS.CUBE_YAW_COS : [-1.0, 1.0],
                        self.POSE_FIELDS.CUBE_YAW_SIN : [-1.0, 1.0],
                        self.POSE_FIELDS.GOAL_X : x_minmax,
                        self.POSE_FIELDS.GOAL_Y : y_minmax},
                history_length=self._history_length,
                **vsize_dev_type, # type: ignore
                observation_definitions={
                        "base":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=base_pose_observable_fields,
                                obs_history_length=1,
                                observable_subfields=None),
                        "privileged":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=[e for e in self.POSE_FIELDS],
                                obs_history_length=self._frame_stack_length,
                                observable_subfields=None)})

        # :::::::::::::::::::::::::::::::::::::::: INTERNAL STATE ::::::::::::::::::::::::::::::::::::::::

        internal_state_helper = ThBoxStateHelper(
                field_names=[e for e in self.INTERNAL_FIELDS],
                field_size=(1,),
                fields_minmax={
                        self.INTERNAL_FIELDS.STEP_COUNT : [0, 100_000],
                        self.INTERNAL_FIELDS.TIME : [-1.0, 1.0],
                        self.INTERNAL_FIELDS.CUBE_DISPLACEMENT : [0.0, 0.5],
                        self.INTERNAL_FIELDS.EE_TRACKING_ERROR : [0.0, 0.5]},
                history_length=self._history_length,
                **vsize_dev_type, # type: ignore
                observation_definitions={
                        "base":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=[self.INTERNAL_FIELDS.TIME],
                                obs_history_length=1,
                                observable_subfields=None),
                        "privileged":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=[self.INTERNAL_FIELDS.TIME,
                                                   self.INTERNAL_FIELDS.CUBE_DISPLACEMENT],
                                obs_history_length=1,
                                observable_subfields=None)})

        # :::::::::::::::::::::::::::::::::::::::: STATE AGGREGATION ::::::::::::::::::::::::::::::::::::::::

        vec_substates = [self.STATE_POSE, self.STATE_INTERNAL]
        self._state_helper = DictStateHelper(
                {self.STATE_POSE : pose_state_helper,
                 self.STATE_INTERNAL : internal_state_helper},
                obs_definitions={
                        "base" : DictStateHelper.SimpleDictObsDef(
                                observable_substates=list(vec_substates),
                                concatenable_substates=list(vec_substates),
                                concatenated_part_name="vec",
                                noise_generators={}),
                        "privileged" : DictStateHelper.SimpleDictObsDef(
                                observable_substates=list(vec_substates),
                                concatenable_substates=list(vec_substates),
                                concatenated_part_name="vec",
                                noise_generators={})})

        # :::::::::::::::::::::::::::::::::::::::: CAMERA STATE ::::::::::::::::::::::::::::::::::::::::

        # Kept out of the concatenated vector part: it is an image, with its own dtype and range.
        if self._observe_camera:
            camera_state_helper = ThBoxStateHelper(
                    field_names=[e for e in self.CAMERA_FIELDS],
                    field_size=self._obs_camera_resolution_hw,
                    fields_minmax={self.CAMERA_FIELDS.IMAGE : [0, 255]},
                    dtype=th.uint8,
                    normalization_range=(0, 255),
                    th_device=th_device,
                    vec_size=adapter.vec_size(),
                    history_length=self._history_length,
                    observation_definitions={
                            "base":ThBoxStateHelper.SimpleObsDef(
                                    observable_fields=None,
                                    obs_history_length=1,
                                    observable_subfields=None,
                                    skip_history_dim=True),
                            # not_observable() would set observable_subfields=[], which the helper
                            # rejects for multi-dimensional fields such as an image.
                            "privileged":ThBoxStateHelper.SimpleObsDef(
                                    observable_fields=[],
                                    obs_history_length=1,
                                    observable_subfields=None,
                                    skip_history_dim=True)})
            self._state_helper = self._state_helper.add_substate(
                    self.STATE_CAMERA,
                    camera_state_helper,
                    obs_defs={"base" :       {"observable":True,  "concatenate":False, "noise":None},
                              "privileged" : {"observable":False, "concatenate":False, "noise":None}})

    # ------------------------------------------------------------------ scenario

    def _get_spawn_defs(self) -> list[ModelSpawnDef]:
        pushing_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path(
                                            "adarl_envs", "models/pushing_simplified.mjcf.xacro")).read_text(),
                                    name="pushing",
                                    pose=None,
                                    format="mjcf.xacro",
                                    kwargs={"area_size" : self._area_size,
                                            "area_center_x" : float(self._area_center_xy[0]),
                                            "area_center_y" : float(self._area_center_xy[1]),
                                            "ee_diameter" : self._ee_diameter,
                                            "ee_ground_distance" : self._ee_height,
                                            "goal_radius" : self._goal_tolerance})
        cube_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path(
                                            "adarl_envs", "models/cube.urdf.xacro")).read_text(),
                                 name="cube",
                                 pose=None,
                                 format="urdf.xacro",
                                 kwargs={"size" : self._cube_size,
                                         "mass" : self._cube_mass,
                                         "red" : self._cube_color[0],
                                         "green" : self._cube_color[1],
                                         "blue" : self._cube_color[2],
                                         "add_floating_joint" : "true"})
        spawn_defs = [pushing_def, cube_def]
        camera_model = Path(adarl.utils.utils.pkgutil_get_path("adarl", "models/simple_camera.mjcf.xacro")).read_text()
        cam_quat_xyzw = ros_rpy_to_quaternion_xyzw(self._camera_orientation_rpy)
        cam_quat_wxyz = f"{cam_quat_xyzw[3]} {cam_quat_xyzw[0]} {cam_quat_xyzw[1]} {cam_quat_xyzw[2]}"
        cam_pos = " ".join(str(v) for v in self._camera_position_xyz)
        cameras = []
        if self._observe_camera:
            cameras.append((self._obs_camera_name, self._obs_camera_render_resolution_hw))
        if self._enable_ui_camera:
            cameras.append((self._ui_camera_name, self._ui_camera_resolution_hw))
        for cam_name, res_hw in cameras:
            spawn_defs.append(ModelSpawnDef(definition_string=camera_model,
                                            name=cam_name,
                                            pose=None,
                                            format="mjcf.xacro",
                                            kwargs={"camera_name" : cam_name,
                                                    "camera_width" : res_hw[1],
                                                    "camera_height" : res_hw[0],
                                                    "position_xyz" : cam_pos,
                                                    "orientation_wxyz" : cam_quat_wxyz}))
        return spawn_defs

    @override
    def _build(self):
        if not adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            raise NotImplementedError(f"Adapter {type(self._adapter).__name__} is not supported yet")
        self._adapter.build_scenario(models=self._get_spawn_defs())
        self._adapter.set_monitored_joints([self._xjoint, self._yjoint])
        self._adapter.set_monitored_links([self._ee_link, self._cube_link, self._goal_marker_link])
        monitored_cameras = []
        if self._observe_camera:
            monitored_cameras.append(self._obs_camera_name)
        if self._enable_ui_camera:
            monitored_cameras.append(self._ui_camera_name)
        self._adapter.set_monitored_cameras(monitored_cameras)
        self._mon_link_ids = self._adapter.get_monitored_links_ids([self._ee_link, self._cube_link])

    @override
    def close(self):
        self._adapter.destroy_scenario()

    # ------------------------------------------------------------------ actions

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        dbg_check_finite(actions, assert_msg="Non-finite actions submitted to the environment",
                         async_assert=True)
        action_xy = th.clamp(actions, -1, 1)*self._max_position_change
        # As in the original environment, the displacement is applied to the *measured* end effector
        # position, not to the previous command.
        target_xy = self._cached_tip_xy + action_xy
        if self._prevent_ee_out:
            eps = 0.005
            target_xy = th.clamp(target_xy, min=self._area_min_xy+eps, max=self._area_max_xy-eps)
        poses = self._thzeros((self.num_envs, 1, 7))
        poses[:,0,0:2] = target_xy
        poses[:,0,2] = self._ee_height
        poses[:,0,6] = 1.0
        self._adapter.setCartesianPoseCommand(link_names=(self._ee_link,), link_poses_xyz_xyzw=poses)

    # ------------------------------------------------------------------ state

    def _cube_yaw_from_quat(self, cube_quat_xyzw : th.Tensor) -> th.Tensor:
        """Port of PandaPushingEnv2DOF._cube_quat2Yaw.

        Returns the (unsigned) angle between a reference axis and its image through the cube
        orientation, using the x axis unless it ended up pointing mostly up, in which case the z axis
        is used instead. It is not a yaw in the usual sense, but it is what the original environment
        fed to the policy (as its cosine and sine), so it is reproduced as-is.
        """
        xaxis = self._thtens([1.0, 0.0, 0.0]).expand(self.num_envs, 3)
        zaxis = self._thtens([0.0, 0.0, 1.0]).expand(self.num_envs, 3)
        rotated_xaxis = th_quat_rotate(xaxis, cube_quat_xyzw)
        rotated_zaxis = th_quat_rotate(zaxis, cube_quat_xyzw)
        points_mostly_up = rotated_xaxis[:,2] > th.linalg.vector_norm(rotated_xaxis[:,0:2], dim=-1)
        angle_x = th.acos(th.clamp(rotated_xaxis[:,0], -1.0, 1.0)) # dot(xaxis, rotated_xaxis)
        angle_z = th.acos(th.clamp(rotated_zaxis[:,2], -1.0, 1.0)) # dot(zaxis, rotated_zaxis)
        return th.where(points_mostly_up, angle_z, angle_x)

    def _read_instantaneous_state(self) -> dict[str, dict[Any, th.Tensor]]:
        """Read the current state from the adapter, as one tensor per state field."""
        # Cloned: getLinksState hands out tensors backed by simulator memory, which MJX donates on
        # the next step, and the end effector position is kept across the step (submit_actions uses it).
        link_states = self._adapter.getLinksState(self._mon_link_ids)
        tip_xy = link_states[:,0,0:2].clone()
        cube_xy = link_states[:,1,0:2].clone()
        cube_yaw = self._cube_yaw_from_quat(link_states[:,1,3:7])
        self._cached_tip_xy = tip_xy
        self._cached_cube_xy = cube_xy
        step_count = self.get_ep_step_counter().to(self._obs_dtype)
        instantaneous_state : dict[str, dict[Any, th.Tensor]] = {
                self.STATE_POSE : {
                        self.POSE_FIELDS.TIP_X : tip_xy[:,0],
                        self.POSE_FIELDS.TIP_Y : tip_xy[:,1],
                        self.POSE_FIELDS.CUBE_X : cube_xy[:,0],
                        self.POSE_FIELDS.CUBE_Y : cube_xy[:,1],
                        self.POSE_FIELDS.CUBE_YAW_COS : th.cos(cube_yaw),
                        self.POSE_FIELDS.CUBE_YAW_SIN : th.sin(cube_yaw),
                        self.POSE_FIELDS.GOAL_X : self._goal_xy[:,0],
                        self.POSE_FIELDS.GOAL_Y : self._goal_xy[:,1]},
                self.STATE_INTERNAL : {
                        self.INTERNAL_FIELDS.STEP_COUNT : step_count,
                        self.INTERNAL_FIELDS.TIME : step_count/self.get_max_episode_steps()*2-1,
                        self.INTERNAL_FIELDS.CUBE_DISPLACEMENT : th.linalg.vector_norm(cube_xy - self._prev_cube_xy, dim=-1),
                        self.INTERNAL_FIELDS.EE_TRACKING_ERROR : self._adapter.get_cartesian_position_error()[:,0].clone()}}
        if self._observe_camera:
            instantaneous_state[self.STATE_CAMERA] = {self.CAMERA_FIELDS.IMAGE : self._render_observation_image()}
        dbg_check(lambda: th.isfinite(self._state_helper.sub_helpers[self.STATE_POSE]._mapping_to_tensor(
                                        instantaneous_state[self.STATE_POSE])).all(),
                  lambda: f"Non-finite values in pose state: {instantaneous_state[self.STATE_POSE]}")
        return instantaneous_state

    @override
    def get_states(self) -> dict[str, th.Tensor]:
        return self._current_state

    def _render_observation_image(self) -> th.Tensor:
        renderings = self._adapter.get_camera_images([self._obs_camera_name], rgb=True, depth=False)
        img_vhwc = renderings.rgb[0] # (vec, H, W, C)
        if not img_vhwc.dtype.is_floating_point:
            img_vhwc = img_vhwc.to(th.float32)/255.0
        h, w = img_vhwc.shape[1], img_vhwc.shape[2]
        l, t, r, b = self._img_crop_ltrb
        img_vhwc = img_vhwc[:, int(t*h):int(b*h), int(l*w):int(r*w), :]
        img = img_vhwc.permute(0, 3, 1, 2) # to (vec, C, H, W)
        img = th.nn.functional.interpolate(img,
                                           size=self._obs_camera_resolution_hw,
                                           mode="bilinear",
                                           align_corners=False,
                                           antialias=True)
        img = img.permute(0, 2, 3, 1) # back to (vec, H, W, C)
        gray = (img[..., :3] @ self._rgb_to_gray_w)*255 # (vec, H, W)
        return gray.to(th.uint8)

    @override
    def get_observations(self, states : dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        return self._state_helper.observe(states)

    @override
    def pre_step(self):
        self._prev_cube_xy = self._cached_cube_xy
        return super().pre_step()

    @override
    def post_step(self):
        instantaneous_state = self._read_instantaneous_state()
        self._current_state = self._state_helper.update(instantaneous_state,
                                                        state=self._current_state,
                                                        inplace=False) # rolls down the history and adds the current state
        cube2goal = th.linalg.vector_norm(self._cached_cube_xy - self._goal_xy, dim=-1)
        cube2tip = th.linalg.vector_norm(self._cached_cube_xy - self._cached_tip_xy, dim=-1)
        self._ep_min_cube2goal_dist = th.minimum(self._ep_min_cube2goal_dist, cube2goal)
        self._ep_min_cube2tip_dist = th.minimum(self._ep_min_cube2tip_dist, cube2tip)
        self._ep_cube2goal_dist_sum += cube2goal
        self._ep_cube2tip_dist_sum += cube2tip
        self._ep_cube_travel += th.linalg.vector_norm(self._cached_cube_xy - self._prev_cube_xy, dim=-1)
        newly_successful = th.logical_and(cube2goal < self._goal_tolerance, th.logical_not(self._success))
        self._ep_steps_before_success = th.where(newly_successful,
                                                 self.get_ep_step_counter().to(self._obs_dtype),
                                                 self._ep_steps_before_success)
        self._success = th.logical_or(self._success, cube2goal < self._goal_tolerance)
        return super().post_step()

    # ------------------------------------------------------------------ reward and termination

    def _pose_xy(self, states : dict[str, th.Tensor], field_idx : th.Tensor) -> th.Tensor:
        """Most recent value of a pair of pose fields, as a (vec_size, 2) tensor."""
        return states[self.STATE_POSE][:,0,field_idx,0]

    def _internal(self, states : dict[str, th.Tensor], field : IntEnum) -> th.Tensor:
        """Most recent value of an internal field, as a (vec_size,) tensor."""
        return states[self.STATE_INTERNAL][:,0,field,0]

    def _reward_terms(self, cube2goal_dist : th.Tensor, tip2cube_dist : th.Tensor,
                      cube_displacement : th.Tensor) -> th.Tensor:
        """Port of PandaPushingEnv2DOF.reward_func (without the gate term, there is no gate here)."""
        cube_position_reward = 100/0.5*(0.5-cube2goal_dist)  # 100 at distance 0, 0 at distance 0.5, linear
        tip_position_reward = 100/0.5*(0.5-tip2cube_dist)    # 100 at distance 0, 0 at distance 0.5, linear
        cube_displacement_reward = cube_displacement*100*20  # 50 if pushed 2.5cm (the default maximum push)
        return (self._reward_cube_pos_weight*cube_position_reward +
                self._reward_tip_pos_weight*0.5*tip_position_reward +
                self._reward_cube_move_weight*cube_displacement_reward)

    @override
    def compute_rewards(self, states : dict[str, th.Tensor],
                        sub_rewards_return : dict[str, th.Tensor] | None = None) -> th.Tensor:
        cube_xy = self._pose_xy(states, self._idx_cube_xy)
        tip_xy = self._pose_xy(states, self._idx_tip_xy)
        goal_xy = self._pose_xy(states, self._idx_goal_xy)
        cube2goal_dist = th.linalg.vector_norm(cube_xy - goal_xy, dim=-1)
        tip2cube_dist = th.linalg.vector_norm(cube_xy - tip_xy, dim=-1)
        cube_displacement = self._internal(states, self.INTERNAL_FIELDS.CUBE_DISPLACEMENT)
        step = self._internal(states, self.INTERNAL_FIELDS.STEP_COUNT)
        succeeded = cube2goal_dist < self._goal_tolerance

        if sub_rewards_return is not None:
            sub_rewards_return["cube2goal_dist"] = cube2goal_dist
            sub_rewards_return["tip2cube_dist"] = tip2cube_dist
            sub_rewards_return["cube_displacement"] = cube_displacement
            sub_rewards_return["succeeded"] = succeeded.to(self._obs_dtype)

        if self._sparse_reward:
            return succeeded.to(self._obs_dtype)

        base_reward = self._reward_terms(cube2goal_dist, tip2cube_dist, cube_displacement)
        max_steps = self.get_max_episode_steps().to(self._obs_dtype)
        if self._terminate_on_success:
            # As if the agent stayed here up to the timeout, plus a bonus (which roughly triples the reward)
            success_reward = self._reward_terms(th.zeros_like(cube2goal_dist),
                                                th.zeros_like(tip2cube_dist),
                                                th.full_like(cube_displacement, 0.025)) * (max_steps - step + 1)
        else:
            success_reward = base_reward + 200/self._goal_tolerance*(self._goal_tolerance-cube2goal_dist)
        reward = th.where(succeeded, success_reward, base_reward)
        dbg_check(lambda: th.isfinite(reward).all(),
                  lambda: f"Non-finite reward: {reward}")
        return reward*self._reward_scale

    @override
    def are_states_terminal(self, states : dict[str, th.Tensor]) -> th.Tensor:
        cube2goal_dist = th.linalg.vector_norm(self._pose_xy(states, self._idx_cube_xy) -
                                               self._pose_xy(states, self._idx_goal_xy), dim=-1)
        terminal = th.zeros((self.num_envs,), dtype=th.bool, device=self._th_device)
        if not self._prevent_ee_out:
            # Note that the end effector's joint limits coincide with the operating area, so in this
            # simplified setup it cannot actually leave it, exactly as in the original PyBullet one.
            tip_xy = self._pose_xy(states, self._idx_tip_xy)
            out_of_area = th.logical_or(th.any(tip_xy < self._area_min_xy, dim=-1),
                                        th.any(tip_xy > self._area_max_xy, dim=-1))
            terminal = th.logical_or(terminal, out_of_area)
        if self._terminate_on_success:
            terminal = th.logical_or(terminal, cube2goal_dist < self._goal_tolerance)
        return terminal

    @override
    def are_states_timedout(self, states : dict[str, th.Tensor]) -> th.Tensor:
        return self._internal(states, self.INTERNAL_FIELDS.STEP_COUNT) >= self.get_max_episode_steps()

    # ------------------------------------------------------------------ episode initialization

    def _sample_in_area(self, border_dist : float, is_valid, fallback_xy : th.Tensor) -> th.Tensor:
        """Rejection-sample one xy point per environment inside the operating area.

        ``is_valid`` takes a (candidates, vec_size, 2) tensor and returns a (candidates, vec_size)
        boolean mask. A fixed number of candidates is drawn (instead of looping until they are all
        valid, like the original environment did) so that the sampling cost does not depend on the
        data; environments for which no candidate is valid fall back to ``fallback_xy``.
        """
        n = self._spawn_rejection_candidates
        low = self._area_min_xy + border_dist
        high = self._area_max_xy - border_dist
        candidates = self._thrand((n, self.num_envs, 2))*(high-low) + low
        valid = is_valid(candidates)
        any_valid = valid.any(dim=0)
        first_valid = th.argmax(valid.to(th.uint8), dim=0)
        chosen = candidates[first_valid, th.arange(self.num_envs, device=self._th_device)]
        return th.where(any_valid.unsqueeze(-1), chosen, fallback_xy)

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options : dict = {}) -> None:
        if vec_mask is None:
            vec_mask = th.ones((self.num_envs,), dtype=th.bool, device=self._th_device)
        area_center = (self._area_min_xy + self._area_max_xy)/2

        if self._random_goal:
            goal_xy = self._sample_in_area(border_dist=self._goal_spawn_border_dist,
                                           is_valid=lambda c: th.ones(c.shape[:2], dtype=th.bool, device=self._th_device),
                                           fallback_xy=area_center.expand(self.num_envs, 2))
        else:
            goal_xy = self._fixed_goal_xy.expand(self.num_envs, 2)
        self._goal_xy = th.where(vec_mask.unsqueeze(-1), goal_xy, self._goal_xy)

        def cube_is_valid(candidates : th.Tensor) -> th.Tensor:
            if self._allow_successful_initial_cube_position:
                return th.ones(candidates.shape[:2], dtype=th.bool, device=self._th_device)
            dist2goal = th.linalg.vector_norm(candidates - self._goal_xy.unsqueeze(0), dim=-1)
            return dist2goal >= self._goal_tolerance + 0.005
        cube_xy = self._sample_in_area(border_dist=self._cube_spawn_border_dist,
                                       is_valid=cube_is_valid,
                                       fallback_xy=(area_center + self._thtens([-0.05, 0.0])).expand(self.num_envs, 2))

        # minimum distance between the center of the cube and the center of the end effector
        ee_min_dist = self._cube_size*1.25/2 + self._ee_diameter/2 + 0.05
        def ee_is_valid(candidates : th.Tensor) -> th.Tensor:
            # the original environment excluded a square around the cube, not a circle
            return th.any(th.abs(candidates - cube_xy.unsqueeze(0)) > ee_min_dist, dim=-1)
        ee_xy = self._sample_in_area(border_dist=self._ee_spawn_border_dist,
                                     is_valid=ee_is_valid,
                                     fallback_xy=area_center.expand(self.num_envs, 2))

        cube_state = self._thzeros((self.num_envs, 1, 13))
        cube_state[:,0,0:2] = cube_xy
        cube_state[:,0,2] = self._cube_size/2 + 0.002
        cube_state[:,0,6] = 1.0
        self._adapter.setLinksStateDirect(link_names=[self._cube_link],
                                          link_states_pose_vel=cube_state,
                                          vec_mask=vec_mask)
        goal_state = self._thzeros((self.num_envs, 1, 13))
        goal_state[:,0,0:2] = self._goal_xy
        goal_state[:,0,2] = 0.001
        goal_state[:,0,6] = 1.0
        self._adapter.setLinksStateDirect(link_names=[self._goal_marker_link],
                                          link_states_pose_vel=goal_state,
                                          vec_mask=vec_mask)
        joint_states_pve = self._thzeros((self.num_envs, 2, 3))
        joint_states_pve[:,0,0] = ee_xy[:,0] # the two prismatic joints move along the world x and y axes
        joint_states_pve[:,1,0] = ee_xy[:,1]
        self._adapter.setJointsStateDirect(joint_names=[self._xjoint, self._yjoint],
                                           joint_states_pve=joint_states_pve,
                                           vec_mask=vec_mask)
        # Do not let the controller drag the end effector back to the previous episode's target
        self._adapter.reset_cartesian_command_to_current(vec_mask=vec_mask) # type: ignore : Mjx2DofCartesianAdapter

        zeros = self._thzeros((self.num_envs,))
        infs = self._thfull(float("+inf"), (self.num_envs,))
        self._success = th.where(vec_mask, th.zeros_like(self._success), self._success)
        self._ep_min_cube2goal_dist = th.where(vec_mask, infs, self._ep_min_cube2goal_dist)
        self._ep_min_cube2tip_dist = th.where(vec_mask, infs, self._ep_min_cube2tip_dist)
        self._ep_cube2goal_dist_sum = th.where(vec_mask, zeros, self._ep_cube2goal_dist_sum)
        self._ep_cube2tip_dist_sum = th.where(vec_mask, zeros, self._ep_cube2tip_dist_sum)
        self._ep_cube_travel = th.where(vec_mask, zeros, self._ep_cube_travel)
        self._ep_steps_before_success = th.where(vec_mask, self._no_success_steps, self._ep_steps_before_success)

        # The cube has not moved yet, so read it once before building the state, to get a zero displacement
        self._prev_cube_xy = th.where(vec_mask.unsqueeze(-1),
                                      self._adapter.getLinksState(self._mon_link_ids)[:,1,0:2],
                                      self._prev_cube_xy)
        instantaneous_state = self._read_instantaneous_state()
        # This repeats the instantaneous state across the history dimension
        self._current_state = self._state_helper.reset_state(instantaneous_state,
                                                             vec_mask=vec_mask,
                                                             old_state=self._current_state)

    # ------------------------------------------------------------------ rendering and infos

    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if not self._enable_ui_camera:
            return [], th.empty((0,))
        try:
            return self._adapter.getRenderings([self._ui_camera_name], vec_mask=vec_mask)
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return [], th.empty((0,))

    @override
    def get_infos(self, states : dict[str, th.Tensor],
                  labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        sub_rewards : dict[str, th.Tensor] = {}
        reward = self.compute_rewards(states, sub_rewards)
        step_count = self._internal(states, self.INTERNAL_FIELDS.STEP_COUNT)
        steps_done = th.clamp(step_count, min=1.0)
        info = {"success" : self._success.to(self._obs_dtype),
                "cube2goal_dist" : sub_rewards["cube2goal_dist"],
                "cube2tip_dist" : sub_rewards["tip2cube_dist"],
                "min_cube_dist" : self._ep_min_cube2goal_dist,
                "min_tip_dist" : self._ep_min_cube2tip_dist,
                "avg_cube_dist" : self._ep_cube2goal_dist_sum/steps_done,
                "avg_tip_dist" : self._ep_cube2tip_dist_sum/steps_done,
                "total_cube_travel" : self._ep_cube_travel,
                "steps_before_success" : self._ep_steps_before_success,
                "ee_tracking_error" : self._internal(states, self.INTERNAL_FIELDS.EE_TRACKING_ERROR),
                "ep_step_count" : step_count,
                "reward" : reward}
        info.update({"reward_"+k : v for k,v in sub_rewards.items()})
        obs = self.get_observations(states)
        info["obs"] = obs
        if labels is not None:
            obs_names = self._state_helper.observation_names()
            obs_labels = {}
            for k in obs:
                subobs_names = obs_names[k]
                if len(subobs_names.shape) == 1:
                    obs_labels[k] = to_string_tensor([str(n) for n in subobs_names])
            if obs_labels:
                labels["obs"] = obs_labels
        return info

    def get_configuration(self) -> dict[str, Any]:
        return {"reward_cube_pos_weight" : self._reward_cube_pos_weight,
                "reward_tip_pos_weight" : self._reward_tip_pos_weight,
                "reward_cube_move_weight" : self._reward_cube_move_weight,
                "reward_scale" : self._reward_scale,
                "goal_tolerance" : self._goal_tolerance,
                "max_steps" : int(self.get_max_episode_steps().max()),
                "terminate_on_success" : self._terminate_on_success,
                "sparse_reward" : self._sparse_reward,
                "max_position_change" : self._max_position_change,
                "operating_area" : self._operating_area_np.tolist()}
