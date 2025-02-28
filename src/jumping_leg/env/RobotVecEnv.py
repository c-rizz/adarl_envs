from __future__ import annotations
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.adapters.MjxAdapter import MjxAdapter
from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
from adarl.envs.vec.BaseVecEnv import Observation
from adarl.utils.robot_helpers import Robot
from adarl.utils.utils import to_string_tensor, th_quat_rotate_py, th_quat_conj, ros_rpy_to_quaternion_xyzw_th, quat_mul_xyzw
from adarl.utils.vec_state_helper import    JointImpedanceActionHelper, ThBoxStateHelper,\
                                        RobotStateHelper, RobotStatsStateHelper,\
                                        StateNoiseGenerator, DictStateHelper, unnormalize, normalize
from adarl.utils.tensor_trees import map_tensor_tree, flatten_tensor_tree, map2_tensor_tree, space_from_tree
from adarl.utils.utils import build_pose, JointState, Pose, LinkState, isinstance_noimport, masked_assign, masked_assign_sc
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check, dbg_run
from dataclasses import dataclass
from gymnasium import Space
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.tensor_trees
import adarl.utils.utils
from adarl.utils.spaces import ThBox
import dataclasses
import numpy as np
import torch as th
import time
from pathlib import Path


def hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))

JOINT_FILTERS = Enum("JOINT_FILTERS",["ALL_REVOLUTE",
                                         "ALL"])

class RobotVecEnv(ControlledVecEnv[BaseVecJointImpedanceAdapter, Observation]):

    @dataclass
    class Configuration:
        action_delay_mustd : th.Tensor
        action_exp_smoothing_1s : float
        action_noise_mustd : th.Tensor
        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor]
        control_mode : JointImpedanceActionHelper.CONTROL_MODES
        controlled_joints : Sequence[tuple[str,str]]
        enable_dbg_checks : bool
        frame_stack_length : int
        goal_err_exp_smoothing_1s : float
        history_length : int
        homing_body_pose_xyz_xyzw : th.Tensor
        homing_ctrl_joints_pvesd : th.Tensor
        homing_nonctrl_joints_position : dict[tuple[str,str],th.Tensor]
        init_on_reset_ratio : float
        initial_pose_randomization : float
        joint_physical_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_damping : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_stiffness : dict[tuple[str,str],th.Tensor]
        main_body_link : tuple[str,str]
        model_urdf_string : str
        noise_angvel_ep_mustdstd : th.Tensor
        noise_gravity_ep_mustdstd : th.Tensor
        noise_joints_pve_mustdstd : th.Tensor
        noise_linvel_ep_mustdstd : th.Tensor
        noise_posz_ep_mustdstd : th.Tensor
        obs_dtype : th.dtype
        observe_body_state : bool
        original_max_epsteps : int
        quiet : bool
        real : bool
        robot_is_floating : bool
        robot_name : str
        robot_root_joint : str
        robot_root_link : tuple[str,str]
        safe_damping : float
        safe_stiffness : float
        seed : int
        show_goal : bool
        spawn_root_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float]
        stepLength_sec : float
        stop_on_safety : bool
        th_device : th.device
        ui_camera_link : tuple[str,str]
        ui_camera_name : str
        ui_camera_resolution_hw : tuple[int,int]
        ui_rel_camera_pose_dist_pitch_yaw : th.Tensor
        vec_jimp_cmd_size : tuple[int,int,int]
        vec_size : int
        verbose_infos : bool


    metadata = {'render.modes': ['rgb_array']}
    # STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT = "action"
    STATE_ROBOT = "robot"
    STATE_ROBOT_STATS = "robot_stats"
    STATE_EXTRINSIC = "extrinsic"
    STATE_INTERNAL = "internal"
    
    
    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", [  "SAFETY_TRIGGERED",
                                                    "STEP_COUNT"], start=0)

    EXTRINSIC_FIELDS = IntEnum("EXTRINSIC_FIELS", ["BODY_REL_LINVEL_X",
                                                   "BODY_REL_LINVEL_Y",
                                                   "BODY_REL_LINVEL_Z",
                                                   "BODY_REL_ANGVEL_X",
                                                   "BODY_REL_ANGVEL_Y",
                                                   "BODY_REL_ANGVEL_Z",
                                                   "BODY_ABS_LINVEL_X",
                                                   "BODY_ABS_LINVEL_Y",
                                                   "BODY_ABS_LINVEL_Z",
                                                   "BODY_ABS_POS_Z",
                                                   "BODY_REL_GRAVITY_X",
                                                   "BODY_REL_GRAVITY_Y",
                                                   "BODY_REL_GRAVITY_Z"], start=0)
    ACT_FIELDS = IntEnum("ACT_FIELDS", ["ACTION"], start=0)
    
    
    joint_filters = {JOINT_FILTERS.ALL : lambda joint_name, robot_model: True,
                     JOINT_FILTERS.ALL_REVOLUTE : lambda joint_name, robot_model: robot_model.get_joint_properties([joint_name])[joint_name]["type"] == Robot.JOINT_TYPES.REVOLUTE}

    @dataclass
    class EpisodeConfiguration:
        vec_initial_ctrl_joint_pose : th.Tensor
        vec_max_ep_steps : th.Tensor
        vec_init_on_reset : th.Tensor

    @dataclass
    class Statistics:
        tracking_errors : th.Tensor
        avg_tracking_error : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        rewards : dict = dataclasses.field(default_factory=lambda: {})

    def  __init__(self, action_delay_mustd : tuple[float,float],
                        action_noise_mustd : Sequence[float] | th.Tensor, 
                        action_smoothing_halflife_sec : float,
                        adapter: BaseVecJointImpedanceAdapter,
                        control_mode : Literal["impedance","impedance_no_gains","position_and_torques", "position_and_gains","torque","velocity","position"],
                        controlled_joints : Sequence[str | JOINT_FILTERS],
                        goal_err_smoothing_halflife_sec : float,
                        maxStepsPerEpisode : int,
                        minmax_damping : dict[str,tuple[float,float]] | tuple[float,float],
                        minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float],
                        robot_main_body_link : str,
                        robot_root_link : str,
                        robot_name : str,
                        robot_urdf_string : str,
                        safe_damping : float,
                        safe_stiffness : float,
                        safety_limits_factor : float,
                        seed : int,
                        stepLength_sec,
                        step_precision_tolerance : float,
                        stop_on_safety : bool,
                        th_device : th.device,
                        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float],
                        homing_joint_pose : dict[tuple[str,str], float],
                        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor],
                        observe_body_velocity : bool,
                        frame_stack_length : int,
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
                        ui_camera_resolution_hw : tuple[int,int] = (144,256),
                        enable_link_collisions : list[tuple[tuple[str,str],list[tuple[str,str]]]] | None = []
                        ):
        self._main_seed = seed
        # self._rng_get_count = 0
        self._rng = th.Generator(device=th_device)
        self._rng.manual_seed(seed)
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._robot_model = Robot(adarl.utils.utils.compile_xacro_string(  model_definition_string=robot_urdf_string))
        root_joint_name = self._robot_model.get_parent_joint(robot_root_link)
        is_floating = self._robot_model.get_joint_properties([root_joint_name])[root_joint_name]["type"] == Robot.JOINT_TYPES.FLOATING
        self._build_new_instantaneous_state = th.vmap(self._build_new_instantaneous_state_single)
        ggLog.info("Properties:"+("\n".join([str(jp) for jp in self._robot_model.get_joint_properties(self._robot_model.get_joint_names()).items()])))
        # exit()
        controlled_joints_str = []
        for j in controlled_joints:
            if isinstance(j, str):
                controlled_joints_str.append(j)
            elif j in self.joint_filters:
                for jn in self._robot_model.get_joint_names():
                    if self.joint_filters[j](jn,self._robot_model):
                        controlled_joints_str.append(jn)
            else:
                raise RuntimeError(f"Unexpected controlled joint request {j} of type {type(j)} (self.joint_filters = {self.joint_filters})")

        controlled_joints_rn : list[tuple[str,str]] = [(robot_name,jn) for jn in controlled_joints_str]
        phys_limits_minmax_pve = {(robot_name,k):self._thtens(l) 
                                    for k,l in self._robot_model.get_joint_limits(controlled_joints_str).items()}
        safe_limits_minmax_pve = {k:(lims_minmax-0.5*(lims_minmax[1]+lims_minmax[0]))*safety_limits_factor+0.5*(lims_minmax[1]+lims_minmax[0])
                                    for k,lims_minmax in phys_limits_minmax_pve.items()}

        for jn in safe_limits_minmax_pve.keys():
            if jn not in control_limits_minmax_pve:
                control_limits_minmax_pve[jn] = safe_limits_minmax_pve[jn]

        if isinstance(minmax_stiffness, tuple):
            minmax_stiffness_thdict = {k:self._thtens(minmax_stiffness) for k in phys_limits_minmax_pve.keys()}
        else:
            minmax_stiffness_thdict = {(robot_name,k):self._thtens(minmax) for k,minmax in minmax_stiffness.items()}
        if isinstance(minmax_damping, tuple):
            minmax_damping_thdict = {k:self._thtens(minmax_damping) for k in phys_limits_minmax_pve.keys()}
        else:
            minmax_damping_thdict = {(robot_name,k):self._thtens(minmax) for k,minmax in minmax_damping.items()}
        action_exp_smoothing_1s = 0.5**(1/action_smoothing_halflife_sec) if action_smoothing_halflife_sec>0 else 0.0
        goal_err_exp_smoothing_1s = 0.5**(1/goal_err_smoothing_halflife_sec) if goal_err_smoothing_halflife_sec>0 else 0.0
        default_homing_joint_pose = {jn: unnormalize(0.0, safe_limits_minmax_pve[jn][0,0].item(), safe_limits_minmax_pve[jn][1,0].item())
                                     for jn in controlled_joints_rn}
        for jn in homing_joint_pose:
            if jn not in controlled_joints_rn:
                ggLog.warn(f"homing_joint_pose contains non-controlled joint {jn}")
        for jn in controlled_joints_rn:
            if jn not in homing_joint_pose:
                homing_joint_pose[jn] = default_homing_joint_pose[jn]

        if not quiet:
            ggLog.info(f"phys_limits_minmax_pve = \n"+"\n".join([str(jn_lim) for jn_lim in phys_limits_minmax_pve.items()]))
            ggLog.info(f"safe_limits_minmax_pve = \n"+"\n".join([str(jn_lim) for jn_lim in safe_limits_minmax_pve.items()]))
            ggLog.info(f"control_limits_minmax_pve = \n"+"\n".join([str(jn_lim) for jn_lim in control_limits_minmax_pve.items()]))
            ggLog.info(f"controlled_joints_rn = \n"+"\n".join([str(jn) for jn in controlled_joints_rn]))
            ggLog.info(f"homing_joint_pose = "+"\n".join([f"{jn}:{p}" for jn,p in homing_joint_pose.items()]))

        homing_ctrl_joints_pvesd = self._thtens([(homing_joint_pose[jn], 0, 0, safe_stiffness, safe_damping)
                                                    for jn in controlled_joints_rn])
        homing_nonctrl_joints_position = {jn:self._thtens(p) for jn,p in homing_joint_pose.items() if jn not in controlled_joints_rn}
        self._configuration = self.Configuration(   action_delay_mustd = self._thtens(action_delay_mustd),
                                                    action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                    action_noise_mustd = self._thtens(action_noise_mustd),
                                                    control_limits_minmax_pve = control_limits_minmax_pve,
                                                    control_mode = JointImpedanceActionHelper.CONTROL_MODES[control_mode.upper()],
                                                    controlled_joints = controlled_joints_rn,
                                                    frame_stack_length = frame_stack_length,
                                                    goal_err_exp_smoothing_1s = goal_err_exp_smoothing_1s,
                                                    history_length = max(2,frame_stack_length),
                                                    homing_body_pose_xyz_xyzw = self._thtens(homing_body_pose_xyz_xyzw),
                                                    homing_ctrl_joints_pvesd = homing_ctrl_joints_pvesd,
                                                    homing_nonctrl_joints_position = homing_nonctrl_joints_position,
                                                    joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                    joint_safe_limits_minmax_damping = minmax_damping_thdict,
                                                    joint_safe_limits_minmax_pve = safe_limits_minmax_pve,
                                                    joint_safe_limits_minmax_stiffness = minmax_stiffness_thdict,
                                                    main_body_link=(robot_name,robot_main_body_link),
                                                    robot_root_link=(robot_name,robot_root_link),
                                                    model_urdf_string=robot_urdf_string,
                                                    obs_dtype = self._obs_dtype,
                                                    observe_body_state = observe_body_velocity,
                                                    original_max_epsteps = maxStepsPerEpisode,
                                                    initial_pose_randomization = initial_pose_randomization,
                                                    real = False,
                                                    robot_name = robot_name,
                                                    robot_is_floating = is_floating,
                                                    robot_root_joint = root_joint_name,
                                                    safe_damping = safe_damping,
                                                    safe_stiffness = safe_stiffness,
                                                    seed = seed,
                                                    show_goal = True,
                                                    stepLength_sec = stepLength_sec,
                                                    stop_on_safety = stop_on_safety,
                                                    th_device = th_device,
                                                    ui_camera_link = ("simple_camera", "simple_camera_link"),
                                                    ui_camera_name="simple_camera",
                                                    verbose_infos = verbose_infos,
                                                    quiet=quiet,
                                                    spawn_root_pose_xyz_xyzw = (0,0,0,0,0,0,1),
                                                    init_on_reset_ratio=init_on_reset_ratio,
                                                    noise_joints_pve_mustdstd = self._thtens(obs_noise_joints_pve_ep_mustd_step_std),
                                                    noise_linvel_ep_mustdstd =  self._thtens(obs_noise_linvel_ep_mustd_step_std),
                                                    noise_angvel_ep_mustdstd =  self._thtens(obs_noise_angvel_ep_mustd_step_std),
                                                    noise_posz_ep_mustdstd =    self._thtens(obs_noise_posz_ep_mustd_step_std),
                                                    noise_gravity_ep_mustdstd = self._thtens(obs_noise_gravity_ep_mustd_step_std),
                                                    ui_rel_camera_pose_dist_pitch_yaw = self._thtens([2.5, 30/180*3.14159, -90/180*3.14159]),
                                                    ui_camera_resolution_hw = ui_camera_resolution_hw,
                                                    vec_size=adapter.vec_size(),
                                                    vec_jimp_cmd_size=(adapter.vec_size(), len(controlled_joints_rn), 5),
                                                    enable_dbg_checks = enable_dbg_checks
                                                    )
        self._current_episode_config = RobotVecEnv.EpisodeConfiguration(
                                                    vec_initial_ctrl_joint_pose = self._configuration.homing_ctrl_joints_pvesd[:,0].expand(adapter.vec_size(), len(self._configuration.controlled_joints)).clone(),
                                                    vec_init_on_reset = th.ones(size=(adapter.vec_size(),), device=th_device, dtype=th.bool),
                                                    vec_max_ep_steps = th.full(fill_value=maxStepsPerEpisode, size=(adapter.vec_size(),), device=th_device, dtype=th.int64))
        self._last_sent_v_j_pvesd = homing_ctrl_joints_pvesd.repeat(adapter.vec_size(), 1, 1)


        self._always_present_collisions : set[tuple[str,str]] = set()

        self._safe_limits_minmax_j_pve = th.stack([safe_limits_minmax_pve[jn] for jn in controlled_joints_rn], dim=1)
        self._action_helper = JointImpedanceActionHelper(
                                vec_size=adapter.vec_size(),
                                control_mode=self._configuration.control_mode,
                                joints=controlled_joints_rn,
                                joints_minmax_pvesd={jn:th.cat([control_limits_minmax_pve[jn],
                                                                minmax_stiffness_thdict[jn].unsqueeze(1),
                                                                minmax_damping_thdict[jn].unsqueeze(1)], dim=1) 
                                                        for jn in controlled_joints_rn},
                                safe_stiffness=self._thtens([self._configuration.safe_stiffness]).repeat(len(controlled_joints_rn)),
                                safe_damping=self._thtens([self._configuration.safe_damping]).repeat(len(controlled_joints_rn)),
                                th_device=self._configuration.th_device,
                                generator=self._rng)
        ggLog.info(f"Built action helper")

        self._build_state_helper(adapter)
        self._current_state = self._state_helper.reset_state()
        
        self._safety_limits = self._state_helper.sub_helpers[self.STATE_ROBOT].build_robot_limits(joint_limit_minmax_pve=self._configuration.joint_safe_limits_minmax_pve,
                                                                    stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                                                    damping_minmax=self._configuration.joint_safe_limits_minmax_damping)
        ggLog.info(f"Built safety limits")
        
        self._build_stats()
        ggLog.info(f"Built stats")

        super().__init__(max_episode_steps=maxStepsPerEpisode,
                         step_duration_sec=stepLength_sec,
                         adapter=adapter,
                         single_state_space=self._state_helper.get_single_space(),
                         single_observation_space=self._state_helper.get_single_obs_space(),
                         single_action_space=self._action_helper.get_single_action_space(),
                         single_reward_space=ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=th_device),
                         info_space=None,
                         step_precision_tolerance = step_precision_tolerance,
                         th_device = self._th_device,
                         obs_dtype = self._obs_dtype,
                         seed = seed)
        self._build()
        ggLog.info(f"enable_link_collisions = {enable_link_collisions}")
        if isinstance(self._adapter, BaseVecSimulationAdapter) and enable_link_collisions is not None:
            self._adapter.set_body_collisions(enable_link_collisions)
        self.initialize_episodes()
        ggLog.info(f"Built scenario")
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__
        ggLog.info(f"Built info helper")

        self.set_seeds(th.as_tensor(seed))
        self._adapter.set_monitored_links([self._configuration.main_body_link])
        self._adapter.startup()

    # @property
    # def _rng(self):
    #     import traceback
    #     ggLog.info(f"Getting rng {self._rng_get_count} {hash_tensor(self._rng_v.get_state())} at {''.join(traceback.format_list(traceback.extract_stack(limit=3)))}")
    #     self._rng_get_count += 1
    #     return self._rng_v

    def _build_stats(self):
        self._stats = {}

    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        robot_state_helper = RobotStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                              stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                              damping_minmax=self._configuration.joint_safe_limits_minmax_damping,
                                              obs_dtype=self._configuration.obs_dtype,
                                              th_device=self._configuration.th_device,
                                              history_length=self._configuration.history_length,
                                              obs_history_length = self._configuration.frame_stack_length,
                                              vec_size=adapter.vec_size())
        robot_stats_state_helper = RobotStatsStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                                        obs_dtype=self._configuration.obs_dtype,
                                                        th_device=self._configuration.th_device,
                                                        vec_size=adapter.vec_size())
        internal_state_helper =   ThBoxStateHelper( field_names=[e for e in self.INTERNAL_FIELDS],
                                                    obs_dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    fields_minmax={   self.INTERNAL_FIELDS.SAFETY_TRIGGERED : [0,1],
                                                                        self.INTERNAL_FIELDS.STEP_COUNT : [-1,1000_000_000]},
                                                    observable_fields=[],
                                                    vec_size=adapter.vec_size())
        extrinsic_state_helper =  ThBoxStateHelper(field_names=[e for e in self.EXTRINSIC_FIELDS],
                                                    obs_dtype=th.float32,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    fields_minmax={ self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : [-100,100],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : [-100,100],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : [-100,100],
                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : [-10,10],
                                                                    self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : [-1,1],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : [-1,1],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : [-1,1],
                                                                    self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : [-1,1]},
                                                    observable_fields=[self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z,
                                                                        self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                                                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z],
                                                    history_length=self._configuration.history_length,
                                                    obs_history_length = self._configuration.frame_stack_length,
                                                    vec_size=adapter.vec_size())
        act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                    obs_dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(self._action_helper.single_action_len(),),
                                                    fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                    history_length=2,
                                                    vec_size=adapter.vec_size())
        robot_state_noise =  StateNoiseGenerator(robot_state_helper,
                                                self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                                episode_mu_std = self._configuration.noise_joints_pve_mustdstd[:2],
                                                step_std = self._configuration.noise_joints_pve_mustdstd[2])
        extrinsic_state_noise =  StateNoiseGenerator(extrinsic_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = th.cat([   self._configuration.noise_linvel_ep_mustdstd[:2].expand(3,2),
                                                                        self._configuration.noise_angvel_ep_mustdstd[:2].expand(3,2),
                                                                        self._configuration.noise_linvel_ep_mustdstd[:2].expand(3,2),
                                                                        self._configuration.noise_posz_ep_mustdstd[:2].expand(1,2),
                                                                        self._configuration.noise_gravity_ep_mustdstd[:2].expand(3,2)]).permute(1,0).unsqueeze(-1),
                                            step_std = th.cat([ self._configuration.noise_linvel_ep_mustdstd[2].expand(3),
                                                                self._configuration.noise_angvel_ep_mustdstd[2].expand(3),
                                                                self._configuration.noise_linvel_ep_mustdstd[2].expand(3),
                                                                self._configuration.noise_posz_ep_mustdstd[2].expand(1),
                                                                self._configuration.noise_gravity_ep_mustdstd[2].expand(3)]).unsqueeze(-1))
        if self._configuration.observe_body_state:
            observable_fields = [   self.STATE_ROBOT,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_INTERNAL]
        else:
            observable_fields = [   self.STATE_ROBOT,
                                    self.STATE_INTERNAL]
        statehelpers : dict[str,ThBoxStateHelper] = {self.STATE_ROBOT : robot_state_helper,
                        self.STATE_ROBOT_STATS : robot_stats_state_helper,
                        self.STATE_EXTRINSIC : extrinsic_state_helper,
                        self.STATE_INTERNAL : internal_state_helper,
                        self.STATE_ACT: act_history_state_helper}
        # ggLog.info("\n".join([f"{k} : state={s._state_space.shape}  obs ={s._obs_space.shape}" for k,s in statehelpers.items()]))

        self._state_helper = DictStateHelper(statehelpers,
                                              observable_fields=observable_fields,
                                              noise = {
                                                    self.STATE_ROBOT : robot_state_noise,
                                                    self.STATE_EXTRINSIC : extrinsic_state_noise},
                                              flatten_in_obs=[   self.STATE_ROBOT,
                                                                self.STATE_EXTRINSIC,
                                                                self.STATE_INTERNAL],
                                              flattened_part_name="vec")        

    
    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    # @th.jit.script
    def _preproc_acts(self, actions : th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        dt = self._configuration.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        prev_actions = self._current_state[self.STATE_ACT][:,0,self.ACT_FIELDS.ACTION].detach().to(device=self._configuration.th_device)
        actions = actions*(1-alpha) + prev_actions*alpha
        actions = th.clamp(actions, min=-1, max=1)
        n = th.randn(size=(self._adapter.vec_size(),),
                    generator=self._rng,
                    dtype=self._configuration.obs_dtype,
                    device=self._configuration.th_device)
        action_delay = th.clamp(self._configuration.action_delay_mustd[0] + self._configuration.action_delay_mustd[1]*n, min = 0.0)
        return actions, action_delay

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        with th.no_grad():
            actions = self._thtens(actions).detach()
            dbg_check_size(actions, (self._adapter.vec_size(), self._action_helper.single_action_len()))
            actions, action_delay = self._preproc_acts(actions)
            self._last_out_actions = actions
            v_j_pvesd = self._action_helper.action_to_pvesd(actions)
            # do this better, avoid this if, whatever, put it in the helper
            if self._configuration.control_mode in [JointImpedanceActionHelper.CONTROL_MODES.POSITION, JointImpedanceActionHelper.CONTROL_MODES.POSITION_AND_STIFFNESS, JointImpedanceActionHelper.CONTROL_MODES.POSITION_AND_TORQUES] :
                v_j_pvesd[:,:,1] = th.clamp((v_j_pvesd[:,:,0] - self._last_sent_v_j_pvesd[:,:,0])/self._intendedStepLength_sec, 
                                            min=self._safe_limits_minmax_j_pve[0,:,1], 
                                            max=self._safe_limits_minmax_j_pve[1,:,1]) # set velocity reference
            self._last_sent_v_j_pvesd = v_j_pvesd# ggLog.info(f"sending jimp: {self._last_sent_v_j_pvesd}")
            self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = self._last_sent_v_j_pvesd,
                                                    delay_sec=action_delay)
            





    





    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------

    def _get_spawn_defs(self):
        if not hasattr(self, "_spawn_defs"):
            robot_pose = None #build_pose(*self._configuration.spawn_root_pose_xyz_xyzw)
            arrow_pose = robot_pose
            camera_pose = None #build_pose(0,0.0,0.0,    0.0, 0.0, 0.0,   1.0)
            robot_spawn_def = ModelSpawnDef(definition_string=self._configuration.model_urdf_string,
                                            name=self._configuration.robot_name,
                                            pose=robot_pose,
                                            format="urdf",
                                            kwargs={})
            if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
                cam_file = "models/simple_camera.mjcf.xacro"
            else:            
                cam_file = "models/simple_camera.sdf.xacro"
            is_pybullet = False
            is_ros = False
            if isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
                if adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("PyBulletJointImpedanceAdapter")):
                    is_pybullet= True
                elif adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("RosXbotAdapter", "RosXbotGazeboAdapter")):
                    is_ros = True
            camera_spawn_def = ModelSpawnDef(   definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                                name="simple_camera",
                                                pose=camera_pose,
                                                format="sdf.xacro",
                                                kwargs={"camera_width":self._configuration.ui_camera_resolution_hw[1],
                                                        "camera_height":self._configuration.ui_camera_resolution_hw[0],
                                                        "frame_rate":1/self._intendedStepLength_sec})
            arrow_spawn_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("jumping_leg","models/red_arrow.urdf.xacro")).read_text(),
                                            name="arrow",
                                            pose=arrow_pose,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
            axes_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("jumping_leg","models/axes.urdf.xacro")).read_text(),
                                            name="axes",
                                            pose=None,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
            self._spawn_defs = [robot_spawn_def,
                                    camera_spawn_def]
            if self._configuration.show_goal:
                self._spawn_defs.append(arrow_spawn_def)
                self._spawn_defs.append(axes_spawn_def)
        return self._spawn_defs
        


    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        # ggLog.info(f"_initialize_episodes({vec_mask})")
        if vec_mask is None:
            vec_mask = th.ones((self.num_envs,), device=self._configuration.th_device, dtype=th.bool)
        # ggLog.info(f"initializing episodes {vec_mask}")
        resetted_state = self._state_helper.reset_state()
        map2_tensor_tree(self._current_state, resetted_state,
                        lambda l1, l2: masked_assign(l1, vec_mask, l2)) # should not be necessary, just for safety
        self._current_state[self.STATE_INTERNAL][vec_mask,0,self.INTERNAL_FIELDS.STEP_COUNT] = th.tensor(-1.) # all other fields will be overwritten accordingly in state_update
        self._last_obs = self._state_helper.observe(self._current_state)
        self._set_current_ep_config(reset_options = options, vec_mask=vec_mask)
        
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._simulation_initialization(vec_mask=vec_mask)
        else:
            self._realworld_initialization(vec_mask=vec_mask)
        self._last_out_actions = th.clamp(self._action_helper.pvesd_to_action(self._last_sent_v_j_pvesd), min=-1, max=1)
        # ggLog.info(f"initial action {self._last_out_action}, pvesd = {self._last_sent_pvesd}")

        if isinstance(self._adapter, MjxAdapter):
            self._adapter.alter_model_rel(link_masses = (self._main_body_link_ids,
                                                         self._thrandn(size=(self.num_envs, 1))*0.1))

        self._update_state()
        self._update_stats()

        # ggLog.info(f"initialzed")
        # ggLog.info(f"jstate {self._adapter.getJointsState()}")
        # ggLog.info(f"lstate {self._adapter.getLinksState([self._configuration.main_body_link])}")
        # time.sleep(10)
        # ggLog.info(f"initialzed, slept")


    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        maxStepsPerEpisode = reset_options.get("max_ep_steps", self._configuration.original_max_epsteps)           
        if vec_mask is not None:
            selected_vecs_num = int(th.count_nonzero(vec_mask).item())
        else:
            selected_vecs_num = self.num_envs
        if selected_vecs_num == 0:
            return
        
        original_collision_pairs = self._robot_model.get_enabled_collision_pairs()
        self._robot_model.set_collision_pairs("all")
        self._robot_model.remove_collision_pairs(self._always_present_collisions)
        homing_pos = self._configuration.homing_ctrl_joints_pvesd[:,0]
        jp_dict = {k:v for k,v in self._configuration.homing_nonctrl_joints_position.items()}
        if self._configuration.initial_pose_randomization > 0:
            limits_minmax = th.stack([self._configuration.joint_safe_limits_minmax_pve[jn][:,0] for jn in self._configuration.controlled_joints], dim = 1)
            founds = [False]*selected_vecs_num
            initial_jposes = th.zeros(  size = (selected_vecs_num, len(self._configuration.controlled_joints)),
                                        dtype=th.float32).to(device=self._configuration.th_device, non_blocking=True)
            for v in range(selected_vecs_num): # TODO: this may be sloooooow, can I parallelize it?
                for i in range(1000):
                    normpos = (self._thrand(size=(len(self._configuration.controlled_joints),))*2-1)*self._configuration.initial_pose_randomization
                    # initial_joint_pose = unnormalize(((npos)),limits_minmax[0],limits_minmax[1])                
                    initial_joint_pose = ((normpos>=0)*((limits_minmax[1]-homing_pos)*normpos + homing_pos) + 
                                          (normpos< 0)*((homing_pos-limits_minmax[0])*normpos + homing_pos))
                    jp_dict.update({jn:initial_joint_pose[i] for i,jn in enumerate(self._configuration.controlled_joints)})
                    self._robot_model.set_joint_pose_by_names({jn[1]:jp.cpu().numpy() for jn,jp in jp_dict.items()})
                    if self._configuration.robot_is_floating:
                        self._robot_model.set_joint_pose_by_names({self._configuration.robot_root_joint:self._configuration.homing_body_pose_xyz_xyzw.cpu().numpy()})
                    collisions = self._robot_model.get_all_collisions()
                    if len(collisions) == 0:
                        founds[v] = True
                        initial_jposes[v] = initial_joint_pose
                        break
                if not founds[v]:
                    initial_jposes[v] = homing_pos
                    ggLog.warn(f"Failed to find initial joint configuration. Last collisions = {collisions}, always present collisions = {self._always_present_collisions}")
        else:
            initial_jposes = homing_pos.expand(selected_vecs_num, len(self._configuration.controlled_joints))
        if  self._configuration.init_on_reset_ratio<1.0 and self._init_counter_since_reset>1:
            vec_init_on_reset = self._thrand((selected_vecs_num,)) < self._configuration.init_on_reset_ratio
        else:
            vec_init_on_reset = th.ones((selected_vecs_num,), dtype=th.bool, device=self._configuration.th_device)
        # ggLog.info(f"initial_jpose = {initial_joint_pose}, homing = {homing}")
        self._robot_model.set_collision_pairs(original_collision_pairs)
        masked_assign(self._current_episode_config.vec_initial_ctrl_joint_pose, vec_mask, initial_jposes)
        masked_assign(self._current_episode_config.vec_init_on_reset, vec_mask, vec_init_on_reset)
        masked_assign(self._current_episode_config.vec_max_ep_steps, vec_mask, maxStepsPerEpisode)
        self.set_max_episode_steps(self._current_episode_config.vec_max_ep_steps)
        # ggLog.info(f"_current_episode_config = {self._current_episode_config}")

    def _realworld_initialization(self, vec_mask : th.Tensor):
        raise NotImplementedError()
    
    def _simulation_initialization(self, vec_mask : th.Tensor):
        if not isinstance(self._adapter, BaseVecSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
                
        # ggLog.info(f"simulation init ({vec_mask}) (count={self._tot_init_counter},{self._init_counter_since_reset})")
        # time.sleep(5)
        if self._configuration.homing_body_pose_xyz_xyzw is not None and self._configuration.robot_is_floating:
            # ggLog.info(f"setting body pose ({self._current_episode_config.vec_init_on_reset})")
            self._adapter.setLinksStateDirect(link_names=[self._configuration.main_body_link],
                                              link_states_pose_vel=th.cat([self._configuration.homing_body_pose_xyz_xyzw,
                                                                           th.zeros((6,), device=self._configuration.th_device, dtype=th.float32)])
                                                                           .expand(self._adapter.vec_size(), 1, 13),
                                              vec_mask=th.logical_and(self._current_episode_config.vec_init_on_reset, vec_mask))
        vjpose = self._current_episode_config.vec_initial_ctrl_joint_pose
        initial_cmd_vec_j_pvesd = th.stack([vjpose,
                                    th.zeros_like(vjpose),
                                    th.zeros_like(vjpose),
                                    th.full_like(vjpose, self._configuration.safe_stiffness),
                                    th.full_like(vjpose, self._configuration.safe_damping)], dim = 2)
        # initial_state_pve = th.zeros(size=(self.num_envs, len(self._configuration.controlled_joints), 3))
        not_resetting_sims = th.logical_not(self._current_episode_config.vec_init_on_reset)
        # if th.any(not_resetting_sims):
        # ggLog.info(f"initial_cmd_vec_j_pvesd.device = {initial_cmd_vec_j_pvesd.device}, self._last_sent_v_j_pvesd.deive = {self._last_sent_v_j_pvesd.device} not_resetting_sims.device={not_resetting_sims.device}")
        masked_assign(initial_cmd_vec_j_pvesd, not_resetting_sims, self._last_sent_v_j_pvesd)
        # initial_cmd_vec_j_pvesd[not_resetting_sims] = self._last_sent_v_j_pvesd[not_resetting_sims]
        # ggLog.info(f"Set joint state>")
        # time.sleep(5)
        self._adapter.setJointsStateDirect(joint_names=self._configuration.controlled_joints,
                                           joint_states_pve=initial_cmd_vec_j_pvesd[:,:,:3],
                                           vec_mask=th.logical_and(self._current_episode_config.vec_init_on_reset, vec_mask))
        uncontrolled_joints_states_pve = th.zeros(size=(self._adapter.vec_size(),len(self._configuration.homing_nonctrl_joints_position),3), device=vjpose.device, dtype=vjpose.dtype)
        uncontrolled_joints_states_pve[:,:,0] = self._thtens(list(self._configuration.homing_nonctrl_joints_position.values()))
        self._adapter.setJointsStateDirect(joint_names=list(self._configuration.homing_nonctrl_joints_position.keys()),
                                           joint_states_pve=uncontrolled_joints_states_pve,
                                           vec_mask=th.logical_and(self._current_episode_config.vec_init_on_reset, vec_mask))
        # ggLog.info(f"Set imp cmd>")        
        # time.sleep(5)
        self._adapter.setJointsImpedanceCommand(initial_cmd_vec_j_pvesd, vec_mask=vec_mask)
        # ggLog.info(f"Set current jimp>")
        # time.sleep(5)
        self._adapter.set_current_joint_impedance_command(initial_cmd_vec_j_pvesd, vec_mask=vec_mask)
        masked_assign(self._last_sent_v_j_pvesd, vec_mask, initial_cmd_vec_j_pvesd)

    @override
    def _build(self):
        envCtrlName = type(self._adapter).__name__
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            self._adapter.build_scenario(models = self._get_spawn_defs())
            self._arrow_base = ("arrow","arrow_link")
        elif isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
            if adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("PyBulletJointImpedanceAdapter")):
                self._adapter.build_scenario(models = self._get_spawn_defs())
                self._arrow_base = ("arrow","world")
            elif adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("RosXbotAdapter", "RosXbotGazeboAdapter")):
                if self._configuration.real:
                    raise NotImplementedError()
                else:
                    self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path( "jumping_leg",
                                                                                                                "gazebo/all_gazebo_xbot.launch"),
                                                launch_file_args={"gui":"false"})
                    self._arrow_base = ("arrow","arrow_link")
            else:
                raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        
        self._main_body_link_ids = self._adapter.get_links_ids([self._configuration.main_body_link])
        self._controlled_joints_ids = self._adapter.get_joints_ids(self._configuration.controlled_joints)

        ggLog.info(f"Detecting always present self collisions...")
        self._robot_model.disable_tree_self_collisions(root_frame=self._configuration.robot_root_link[1])
        # self._robot_model.remove_collision_pairs([("rail_link_0","slider_link_0")])            
        self._ground_co_id = self._robot_model.add_collision_box(   pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
                                                                    collision_box_size_xyz=(100,100,1),
                                                                    collision_obj_id="ground_collision")
        self._always_present_collisions : set[tuple[str,str]] = self._robot_model.detect_always_present_collisions(
            moving_joints=[jn[1] for jn in self._configuration.controlled_joints],
            fixed_joints_pose={self._configuration.robot_root_joint : self._configuration.homing_body_pose_xyz_xyzw.cpu().numpy()}
                                            if self._configuration.robot_is_floating else {},
                                            samples=100)
        ggLog.info(f"Always present self collisions = {self._always_present_collisions}")
        self._adapter.set_monitored_joints(self._configuration.controlled_joints)
        self._adapter.set_impedance_controlled_joints(self._configuration.controlled_joints)
        # ggLog.info("Initialized RobotVecEnv scenario")





    @override
    def close(self):
        self._adapter.destroy_scenario()

    def set_cam_pose(self, pose_dist_pitch_roll : tuple[float,float,float] | th.Tensor):
        self._configuration.ui_rel_camera_pose_dist_pitch_yaw = self._thtens(pose_dist_pitch_roll)

    def get_cam_pose(self):
        return self._configuration.ui_rel_camera_pose_dist_pitch_yaw    
    
    def _get_cam_pose_xyz_xyzw(self):
        cam_rel_pos_dist_pitch_yaw = self._configuration.ui_rel_camera_pose_dist_pitch_yaw
        cam_rel_pos  = self._thtens([-cam_rel_pos_dist_pitch_yaw[0], 0.0, 0.0])
        cam_rel_rpy  = self._thtens([0.0, cam_rel_pos_dist_pitch_yaw[1], cam_rel_pos_dist_pitch_yaw[2]])
        cam_rel_quat = ros_rpy_to_quaternion_xyzw_th(cam_rel_rpy)
        # ggLog.info(f"cam pos0 = {cam_rel_pos}")
        return th.cat([th_quat_rotate_py(cam_rel_pos, cam_rel_quat), cam_rel_quat])

    # --------------------------------------------------------------------------------------------------------------------
    # State & Observation
    # --------------------------------------------------------------------------------------------------------------------
    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        # camera by default looks down the x axis
        cam_link_state = th.zeros((13,), device=self._configuration.th_device, dtype=th.float32)
        cam_link_state[:7] = self._get_cam_pose_xyz_xyzw()
        # if isinstance_noimport(self._adapter, "MjxAdapter"):
        #      # MJ uses "-Z forward, +X right, +Y up, PyBullet (and others) use +X forward +Y left, +Z up"
        #     cam_link_state[3:7] = quat_mul_xyzw(cam_link_state[3:7], self._thtens([-0.5,0.5,0.5,-0.5]))


        # cam_link_state[:7] = th.as_tensor([0.0, -2.0, 0.5, 0.0, 0.0, 0.0, 1.0], device=self._configuration.th_device, dtype=th.float32)
        if th.any(vec_mask[1:]):
            raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
        # ggLog.info(f"cam pos = {cam_rel_pos}")
        # ggLog.info(f"cam quat = {cam_rel_quat}")
        try:
            if isinstance(self._adapter, BaseVecSimulationAdapter):
                body_states13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_frame = False)[:,0,:]
                cam_link_state[:3] += body_states13[0,:3] #body_states13[:,:,:3] # Camera is on a fixed link, so it must be set to the same pose across all links
                cam_link_state = cam_link_state.expand(self._adapter.vec_size(),1,13)
                # cam_link_state[:,:,:3] += body_states13[:,:,:3] # Camera is on a fixed link, so it must be set to the same pose across all sims
                self._adapter.setLinksStateDirect(link_names=[self._configuration.ui_camera_link],
                                                  link_states_pose_vel=cam_link_state,
                                                  vec_mask=None)
            imgs, times = self._adapter.getRenderings([self._configuration.ui_camera_name], vec_mask=vec_mask)
            return imgs, times
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return [], th.empty((0,))
    
    @override
    def get_observations(self, state) -> dict[Any, th.Tensor]:
        self._last_obs = self._state_helper.observe(state)
        if self._configuration.enable_dbg_checks:
            if not adarl.utils.tensor_trees.is_all_finite(state):
                ggLog.warn(f"Non-finite values in state {state}")
            if not adarl.utils.tensor_trees.is_all_finite(self._last_obs):
                ggLog.warn(f"Non-finite values in obs {self._last_obs}")
            if th.any(th.abs(self._last_obs["vec"]) > 100):
                ggLog.warn(f"Values over 100 in obs {self._last_obs}")
        return self._last_obs


    @override
    def get_states(self) -> dict[Any, th.Tensor]:
        return self._current_state


    @override
    def on_step(self):
        # t0 = time.monotonic()
        self._update_state()
        # t1 = time.monotonic()
        self._update_stats()
        # tf = time.monotonic()
        # ggLog.info(f"update_state: {t1-t0} update_stats: {tf-t1}")
        self._last_step_simtime = self._adapter.getEnvTimeFromReset()
        # ggLog.info(f"on_step(): {self._current_state[self.STATE_ROBOT][0,0]}")


    def _get_new_instantaneous_state(self):
        # ggLog.info(f"_stepCounter = {self._stepCounter}")
        # t0 = time.monotonic()
        jstates_v_j_pve = self._adapter.getJointsState(requestedJoints=self._controlled_joints_ids)
        # ggLog.info(f"jstates_v_j_pve = {jstates_v_j_pve}")
        # th.cuda.synchronize()
        # t1 = time.monotonic()
        bstates_v_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_frame = False)[:,0,:]
        # ggLog.info(f"axes pose = {self._adapter.getLinksState(requestedLinks = self._adapter.get_links_ids([('axes','root')]), use_com_frame = False)[:,0,:]}")
        # ggLog.info(f"bstates_v_13 = {bstates_v_13}")
        # th.cuda.synchronize()
        # t2 = time.monotonic()
        internal_states = self._current_state[self.STATE_INTERNAL][:,0]
        vec_stats_minmaxavgstd_j_pvae = self._adapter.get_joints_state_step_stats()
        # th.cuda.synchronize()
        # t3 = time.monotonic()
        # ggLog.info(f"vec_stats_minmaxavgstd_j_pvae = {vec_stats_minmaxavgstd_j_pvae}")
        # ggLog.info(f"bstates_v_13 = {bstates_v_13}")
        # ggLog.info(f"internal_states = {internal_states}")
        # ggLog.info(f"jstates_v_j_pve.device = {jstates_v_j_pve.device}")
        # ggLog.info(f"self._last_sent_v_j_pvesd.device = {self._last_sent_v_j_pvesd.device}")
        dbg_check(lambda: th.all(th.isfinite(vec_stats_minmaxavgstd_j_pvae)),
                  lambda: f"non finite values in joint stats: {vec_stats_minmaxavgstd_j_pvae}")
        dbg_check(lambda: th.all(th.isfinite(bstates_v_13)),
                  lambda: f"non finite values in body link state: {bstates_v_13}")
        # bstates_v_13 = th.zeros(size=(1,13), dtype=th.float32, device=self._adapter._th_device)
        new_inst_state = self._build_new_instantaneous_state_vec(   bstates_v_13,
                                                                    internal_states,
                                                                    vec_stats_minmaxavgstd_j_pvae,
                                                                    jstates_v_j_pve,
                                                                    self._last_sent_v_j_pvesd)
        # ggLog.info(f"insta_state sizes = "+str(map_tensor_tree(new_inst_state,lambda t: t.size())))
        new_inst_state[self.STATE_ACT] = {self.ACT_FIELDS.ACTION : self._last_out_actions}
        # th.cuda.synchronize()
        # t4 = time.monotonic()
        # if not th.all(th.isfinite(new_inst_state[self.STATE_ROBOT_STATS])):
        #     ggLog.info(f"nonfinite vals in new_robot_stats_state = {new_inst_state[self.STATE_ROBOT_STATS]}")
        dbg_check(lambda: th.all(new_inst_state[self.STATE_ROBOT][:,:,6:]>=0), lambda: f"negative gains in new_robot_state") #type: ignore
        # th.cuda.synchronize()
        # t5 = time.monotonic()
        # ggLog.info(f"getJoints={t1-t0:.6f} getlinks={t2-t1:.6f} getstats={t3-t2:.6f} build={t4-t3:.6f} check={t5-t4:.6f} tot={t5-t0}")
        return new_inst_state

    def _build_new_instantaneous_state_single(self, body_state_13 : th.Tensor,
                                                    single_internal_state : th.Tensor,
                                                    stats_minmaxavgstd_j_pvae : th.Tensor,
                                                    jstates_j_pve : th.Tensor,
                                                    last_sent_j_pvesd : th.Tensor):
        body_abs_linvel_xyz = body_state_13[7:10]
        body_angvel_xyz = body_state_13[10:13]
        body_position_xyz = body_state_13[0:3]
        conj_body_quat_xyzw = th_quat_conj(body_state_13[3:7])
        # th.as_tensor([0.0,0.0,-1.0]).expand(conj_body_quat_xyzw[...,:3].size())
        gdir = th.zeros_like(conj_body_quat_xyzw[...,:3])
        gdir[...,2] = -1
        # ggLog.info(f"body_state_13[3:7] = {body_state_13[3:7]}")
        gravity_vec         = th_quat_rotate_py(gdir, conj_body_quat_xyzw)
        body_rel_linvel_xyz = th_quat_rotate_py(body_abs_linvel_xyz,     conj_body_quat_xyzw)
        body_rel_angvel_xyz = th_quat_rotate_py(body_angvel_xyz,     conj_body_quat_xyzw)


        step_count = single_internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        safety_triggered = single_internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        # ggLog.info(f"stats_minmaxavgstd_j_pvae.device = {stats_minmaxavgstd_j_pvae.device}   self._safe_limits_minmax_j_pve[0].device = {self._safe_limits_minmax_j_pve[0].device}")
        pveidx = th.as_tensor([0,1,3]).to(device=stats_minmaxavgstd_j_pvae.device, non_blocking=True)
        triggered_limits = th.logical_or(   stats_minmaxavgstd_j_pvae[0, :, pveidx] < self._safe_limits_minmax_j_pve[0],
                                            stats_minmaxavgstd_j_pvae[1, :, pveidx] > self._safe_limits_minmax_j_pve[1])
        safety_triggered = th.any(triggered_limits)
        safety_triggered = th.logical_and(safety_triggered, step_count>=1)
        safety_triggered = th.logical_or(safety_triggered, safety_triggered)

        # if step_count!=-1 and single_internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0:
        #     safety_triggered = True
        # elif step_count>=1: # stats are not valid at step 0
        #     triggered_limits = th.logical_or(stats_minmaxavgstd_j_pvae[0, :, [0,1,3]] < self._safe_limits_minmax_j_pve[0],
        #                                      stats_minmaxavgstd_j_pvae[1, :, [0,1,3]] > self._safe_limits_minmax_j_pve[1])
        #     safety_triggered = th.any(triggered_limits, dim = 1)
        #     if safety_triggered:       
        #         elements = np.array([[f"{jn[1]}_pos",f"{jn[1]}_vel",f"{jn[1]}_eff"] for jn in self._configuration.controlled_joints], dtype=object) #type: ignore
        #         triggered = []
        #         for i in np.ndindex(elements.shape):
        #             if triggered_limits[i]:
        #                 triggered.append(elements[i])
        #         if not self._configuration.quiet:
        #             ggLog.info( f"SAFETY TRIGGERED (step {step_count.item()}):"
        #                         f"\n    triggered ({len(triggered)}) = {triggered}"
        #                         # f"\n    joints_minmax = \n{stats_minmaxavgstd_j_pve[:2]}"
        #                         # f"\n    j_safety_lims  = \n{self._safe_limits_minmax_j_pve} "
        #                         )
        # else:
        #     safety_triggered = False


        new_internal_state = {  self.INTERNAL_FIELDS.SAFETY_TRIGGERED : safety_triggered.to(dtype=th.float32),
                                self.INTERNAL_FIELDS.STEP_COUNT : step_count+1}
        new_robot_state = th.cat([jstates_j_pve, last_sent_j_pvesd], dim = 1)
        # build stats:
        # with permute the first dimension becomes the joint (ordered as in set_monitored_joints)
        # with flatten the second dimension becomes minp,minv,mina,mmine,maxp,maxv,...
        new_robot_stats_state = stats_minmaxavgstd_j_pvae.permute(1,0,2).flatten(start_dim=1)
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : body_rel_linvel_xyz[0].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : body_rel_linvel_xyz[1].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : body_rel_linvel_xyz[2].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : body_rel_angvel_xyz[0].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : body_rel_angvel_xyz[1].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : body_rel_angvel_xyz[2].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : body_abs_linvel_xyz[0].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : body_abs_linvel_xyz[1].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : body_abs_linvel_xyz[2].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : body_position_xyz[2].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : gravity_vec[0].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : gravity_vec[1].unsqueeze(0),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : gravity_vec[2].unsqueeze(0)}
        return {    self.STATE_EXTRINSIC    : new_extrinsic_state,
                    self.STATE_INTERNAL     : new_internal_state,
                    self.STATE_ROBOT        : new_robot_state,
                    self.STATE_ROBOT_STATS  : new_robot_stats_state}
        

    def _build_new_instantaneous_state_vec(self, vec_body_state_13 : th.Tensor,
                                                 vec_internal_state : th.Tensor,
                                                 vec_stats_minmaxavgstd_j_pvae : th.Tensor,
                                                 vec_jstates_j_pve : th.Tensor,
                                                 vec_last_sent_j_pvesd : th.Tensor):
        body_abs_linvel_xyz_vec = vec_body_state_13[:,7:10]
        body_angvel_xyz_vec = vec_body_state_13[:,10:13]
        body_position_xyz_vec = vec_body_state_13[:,0:3]
        conj_body_quat_xyzw_vec = th_quat_conj(vec_body_state_13[:,3:7])
        # th.as_tensor([0.0,0.0,-1.0]).expand(conj_body_quat_xyzw[...,:3].size())
        gdir = th.zeros_like(conj_body_quat_xyzw_vec[:,:3])
        gdir[:,2] = -1
        # ggLog.info(f"body_state_13[3:7] = {body_state_13[3:7]}")
        gravity_dir_vec     = th_quat_rotate_py(gdir, conj_body_quat_xyzw_vec)
        body_rel_linvel_xyz_vec = th_quat_rotate_py(body_abs_linvel_xyz_vec,     conj_body_quat_xyzw_vec)
        body_rel_angvel_xyz_vec = th_quat_rotate_py(body_angvel_xyz_vec,     conj_body_quat_xyzw_vec)


        vec_step_count = vec_internal_state[:,self.INTERNAL_FIELDS.STEP_COUNT]
        prev_safety_triggered_vec = vec_internal_state[:,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        # ggLog.info(f"stats_minmaxavgstd_j_pvae.device = {stats_minmaxavgstd_j_pvae.device}   self._safe_limits_minmax_j_pve[0].device = {self._safe_limits_minmax_j_pve[0].device}")
        pveidx = th.as_tensor([0,1,3]).to(device=vec_stats_minmaxavgstd_j_pvae.device, non_blocking=True)
        vec_triggered_limits = th.logical_or(   vec_stats_minmaxavgstd_j_pvae[:, 0, :, pveidx] < self._safe_limits_minmax_j_pve[0],
                                                vec_stats_minmaxavgstd_j_pvae[:, 1, :, pveidx] > self._safe_limits_minmax_j_pve[1])
        vec_safety_triggered = th.any(vec_triggered_limits, dim=(1,2))
        vec_safety_triggered = th.logical_and(vec_safety_triggered, vec_step_count.view((self.num_envs,))>=1)
        vec_safety_triggered = th.logical_or(vec_safety_triggered, prev_safety_triggered_vec.view((self.num_envs,)))

        # if step_count!=-1 and single_internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0:
        #     safety_triggered = True
        # elif step_count>=1: # stats are not valid at step 0
        #     triggered_limits = th.logical_or(stats_minmaxavgstd_j_pvae[0, :, [0,1,3]] < self._safe_limits_minmax_j_pve[0],
        #                                      stats_minmaxavgstd_j_pvae[1, :, [0,1,3]] > self._safe_limits_minmax_j_pve[1])
        #     safety_triggered = th.any(triggered_limits, dim = 1)
        #     if safety_triggered:       
        #         elements = np.array([[f"{jn[1]}_pos",f"{jn[1]}_vel",f"{jn[1]}_eff"] for jn in self._configuration.controlled_joints], dtype=object) #type: ignore
        #         triggered = []
        #         for i in np.ndindex(elements.shape):
        #             if triggered_limits[i]:
        #                 triggered.append(elements[i])
        #         if not self._configuration.quiet:
        #             ggLog.info( f"SAFETY TRIGGERED (step {step_count.item()}):"
        #                         f"\n    triggered ({len(triggered)}) = {triggered}"
        #                         # f"\n    joints_minmax = \n{stats_minmaxavgstd_j_pve[:2]}"
        #                         # f"\n    j_safety_lims  = \n{self._safe_limits_minmax_j_pve} "
        #                         )
        # else:
        #     safety_triggered = False


        new_internal_state = {  self.INTERNAL_FIELDS.SAFETY_TRIGGERED : vec_safety_triggered.to(dtype=th.float32).view(self.num_envs,1),
                                self.INTERNAL_FIELDS.STEP_COUNT : (vec_step_count+1).view(self.num_envs,1)}
        new_robot_state = th.cat([vec_jstates_j_pve, vec_last_sent_j_pvesd], dim = -1)
        # build stats:
        # with permute the first dimension becomes the joint (ordered as in set_monitored_joints)
        # with flatten the second dimension becomes minp,minv,mina,mmine,maxp,maxv,...
        new_robot_stats_state = vec_stats_minmaxavgstd_j_pvae.permute(0,2,1,3).flatten(start_dim=2)
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : body_rel_linvel_xyz_vec[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : body_rel_linvel_xyz_vec[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : body_rel_linvel_xyz_vec[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : body_rel_angvel_xyz_vec[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : body_rel_angvel_xyz_vec[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : body_rel_angvel_xyz_vec[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : body_abs_linvel_xyz_vec[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : body_abs_linvel_xyz_vec[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : body_abs_linvel_xyz_vec[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : body_position_xyz_vec[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : gravity_dir_vec[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : gravity_dir_vec[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : gravity_dir_vec[:,2].view(self.num_envs,1)}
        return {    self.STATE_EXTRINSIC    : new_extrinsic_state,
                    self.STATE_INTERNAL     : new_internal_state,
                    self.STATE_ROBOT        : new_robot_state,
                    self.STATE_ROBOT_STATS  : new_robot_stats_state}
        


    def _update_state(self):
        # th.cuda.synchronize()
        # t0 = time.monotonic()
        instantaneous_state : dict[str,dict[Any,th.Tensor]]= self._get_new_instantaneous_state()
        # th.cuda.synchronize()
        # t01 = time.monotonic()
        self._state_helper.check_size(instantaneous_state=instantaneous_state)
        # dbg_run(lambda: self._state_helper.check_size(instantaneous_state=instantaneous_state))
        # sizes = map_tensor_tree(flatten_tensor_tree(instantaneous_state), lambda t: t.size())
        # # {k:v.size() for k,v in instantaneous_state.items()}
        # n = "\n"
        # ggLog.info(f"Got instantaneous state with sizes: {n.join([str(kv) for kv in sizes.items()])}")
        # t1 = time.monotonic()
        new_step_counts = instantaneous_state[self.STATE_INTERNAL][self.INTERNAL_FIELDS.STEP_COUNT][0] # all env have the same step count
        dbg_check(lambda: th.all(new_step_counts == new_step_counts[0]),
                  lambda: "asynchronous terminations are not supported yet")
        new_step_count = new_step_counts[0]
        if new_step_count == 0:
            self._current_state = self._state_helper.reset_state(instantaneous_state) # fills up history with current instantaneous state
        else:
            self._state_helper.update(instantaneous_state, state=self._current_state) # rolls down the history and adds current state
        # ss = {k:t.size() for k,t in self._current_state.items()}
        # ggLog.info(f"state sizes = {ss}")
        dbg_check(lambda: th.all(self._current_state[self.STATE_INTERNAL][0,0,self.INTERNAL_FIELDS.STEP_COUNT] >= 0),
                  lambda: f"Negative step_counts {self._current_state[self.STATE_INTERNAL][0,0,self.INTERNAL_FIELDS.STEP_COUNT]}")
        # map_tensor_tree(self._current_state, lambda t: t.detach().clone())
        tf = time.monotonic()
        # print(f"newinst = {t01-t0}, check = {t1-t01}, map = {tf-t1}, tot = {tf-t0}")
        self._current_state = {k:t.detach().clone() for k,t in self._current_state.items()} # TODO: remove, this shouldn't be necessary, just here out of caution



    def _update_stats(self):
        sub_rewards = {}
        self.compute_rewards(self._current_state, 
                                sub_rewards_return=sub_rewards)
        self._stats["rewards"] = sub_rewards
        
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        # i = super().get_infos(states=state)
        i : dict[str, th.Tensor] = {}
        i.update(self._stats)
        i["ep_step_count"] = self._ep_step_counter
        i["ep_count"] = self._ep_counter
        # i["tot_step_count"] = th.as_tensor(self._tot_step_counter)
        # i["tot_init_count"] = th.as_tensor(self._tot_init_counter)
        i["joint_homing_dist"] = state[self.STATE_ROBOT][:,0,:,0] - self._configuration.homing_ctrl_joints_pvesd[:,0]
        if labels is not None:
            labels["joint_homing_dist"] = to_string_tensor([jn[1] for jn in self._configuration.controlled_joints])

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_ROBOT, self.STATE_EXTRINSIC, self.STATE_INTERNAL, self.STATE_ACT, self.STATE_ROBOT_STATS]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["vec_obs"] = to_string_tensor([n for n in self._state_helper.observation_names()["vec"]])
            i["vec_obs"] = self._last_obs["vec"]
            if labels is not None:
                labels["vec_obs"] = to_string_tensor([n for n in self._state_helper.observation_names()["vec"]])
        sub_rewards = {}
        self.compute_rewards(state, sub_rewards)
        i.update({f"sub_reward_{k}":r for k,r in sub_rewards.items()})
            
        i.update({"ep_config."+k:v for k,v in dataclasses.asdict(self._current_episode_config).items()})
        i["safety_triggered"] = state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED]
        
        return i
    
    @override
    def are_states_terminal(self, states) -> th.Tensor:
        if not self._configuration.stop_on_safety:
            return th.zeros((self.num_envs,), dtype=th.bool, device=self._configuration.th_device)
        # if th.any(r):
        #     term_idxs = th.nonzero(r)
        #     ggLog.info(f"Env {term_idxs} terminated at step {self._ep_step_counter[term_idxs]}")
        return (states[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED,0] > 0).view((self.num_envs,))
    
    @override
    def are_states_timedout(self, states) -> th.Tensor:
        sinternal = states[self.STATE_INTERNAL]
        # r = sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT] >= self.get_max_episode_steps()
        # ggLog.info(f"sinternal.size() = {states[self.STATE_INTERNAL].size()}")
        # ggLog.info(f" sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT] = { sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT].size()}")
        # ggLog.info(f"self.get_max_episode_steps() = {self.get_max_episode_steps().size()}")
        # ggLog.info(f"r.size() = {r.size()}")
        return (sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT,0] >= self.get_max_episode_steps()).view((self.num_envs,))



    @override
    def set_seeds(self, seeds : th.Tensor):
        super().set_seeds(seeds)
        self._main_seed = int(th.sum(seeds).item())
        self._rng.manual_seed(self._main_seed)
        self.vec_action_space.seed(self._main_seed)
        self.vec_observation_space.seed(self._main_seed)
        self.vec_state_space.seed(self._main_seed)
        self.single_action_space.seed(self._main_seed)
        self.single_observation_space.seed(self._main_seed)
        self.single_state_space.seed(self._main_seed)

    def _warn_out_of_bounds(self, robot_state_norm):
        if not adarl.utils.tensor_trees.is_all_bounded(robot_state_norm, -10, 10):
            flatstate = flatten_tensor_tree(robot_state_norm)
            violations = {k:(th.abs(flatstate[k])>10).nonzero() for k in flatstate.keys() if not adarl.utils.tensor_trees.is_leaf_bounded(flatstate[k],min=-10,max=10)}
            ggLog.warn(f"robot_state is 10X out of bounds: violations at \n{violations} \n robot_state_norm = \n {robot_state_norm}")

    @override
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        # reward_health = th.ones((self.num_envs,), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
        # sub_rewards_return["health"] = reward_health

        max_rew = 100
        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_pvesd[:,0], lims[0,:,0], lims[1,:,0])

        robot_state_norm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)
        dbg_run(lambda: self._warn_out_of_bounds(robot_state_norm))
        
        normposhomingdiff = robot_state_norm[:,0,:,0] - normhoming
        normvelocities =    robot_state_norm[:,0,:,1]
        normtorques =       robot_state_norm[:,0,:,2]
        normaccelerations = (robot_state_norm[:,0,:,1] - robot_state_norm[:,1,:,1])/self._configuration.stepLength_sec
        
        reward_position     = - th.clamp(th.mean(th.pow(normposhomingdiff,2),   dim = 1), -max_rew,max_rew)
        reward_torque       = - th.clamp(th.mean(th.pow(normtorques,2),         dim = 1), -max_rew,max_rew)
        reward_velocity     = - th.clamp(th.mean(th.pow(normvelocities,2),      dim = 1), -max_rew,max_rew)
        reward_acceleration = - th.clamp(th.mean(th.pow(normaccelerations,2),   dim = 1), -max_rew,max_rew)
        
        
        sub_rewards_return["position"] = reward_position
        sub_rewards_return["torque"] = reward_torque
        sub_rewards_return["velocity"] = reward_velocity
        sub_rewards_return["acceleration"] = reward_acceleration

        weights = { "torque" : 0.01,
                    "velocity" : 1.0,
                    "acceleration" : 0.1,
                    "position" : 1.0
                    }

        reward = th.sum(th.stack([sub_rewards_return[k]*weights[k] for k in sub_rewards_return]), dim=0)
        return reward
