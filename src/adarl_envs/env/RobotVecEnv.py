from __future__ import annotations
from adarl.adapters.BaseVecAdapter import JointType
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
from adarl.envs.vec.BaseVecEnv import Observation
from adarl.utils.robot_helpers import Robot
from adarl.utils.robot_helpers import find_poses
from adarl.utils.utils import expand_default_dict
from adarl.utils.vec_state_helper import    JointImpedanceActionHelper, ThBoxStateHelper,\
                                        JointStateHelper, RobotStatsStateHelper,\
                                        StateNoiseGenerator, DictStateHelper, unnormalize, normalize
from adarl.utils.tensor_trees import space_from_tree
import adarl.utils.utils
from adarl.utils.utils import (isinstance_noimport, masked_assign, quat_conj_xyzw_np, quat_mul_xyzw_np,
                               DistributionDef, DistributionDefTh, sample_distr, distr_to_tensor, distr_is_constant,
                               to_string_tensor, th_quat_rotate_py, th_quat_conj, ros_rpy_to_quaternion_xyzw_th, )
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check,  dbg_check_bounded, dbg_check_finite
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Sequence, Literal, Any, Callable
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.spaces import ThBox
import dataclasses
import numpy as np
import torch as th
import time
from pathlib import Path
import pprint
from gymnasium.vector.utils import batch_space
import os
from adarl.utils.spaces import get_space_labels
from adarl.utils.base_utils import record_time, record_region_start, record_region_end, DelayStats
import gc

disable_compile = bool(os.environ.get("DISABLE_ENV_TH_COMPILE", False))
unsafe_realworld_init = False


JOINT_FILTERS = Enum("JOINT_FILTERS",["ALL_REVOLUTE",
                                         "ALL"])
LINK_FILTERS = Enum("LINK_FILTERS",["ALL","ALL_ROBOT"])

def _is_joint_revolute(joint_name : str, robot_model : Robot) -> bool:
    """Check if a joint is revolute"""
    if joint_name not in robot_model.get_joint_names():
        return False
    joint_properties = robot_model.get_joint_properties([joint_name])
    # ggLog.info(f"joint_properties = {joint_properties}")
    return joint_properties[joint_name]["type"] == Robot.JOINT_TYPES.REVOLUTE

@dataclass
class RobotVecEnvInitArgs():
    noise_action_delay_mustd_std : tuple[float,float,float]
    noise_action_mustd : Sequence[float] | th.Tensor
    action_smoothing_halflife_sec : float
    adapter: BaseVecJointImpedanceAdapter
    control_limits_center : dict[tuple[str,str], float]
    control_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float]
    control_mode : Literal["velocity", "torque","position","pvesd","pve","pt","ps"]
    controlled_joints : Sequence[str | JOINT_FILTERS]
    enable_dbg_checks : bool
    posref_err_history_length : int
    fail_on_safety : bool
    frame_stack_length : int
    free_joints : Sequence[str]
    goal_err_smoothing_halflife_sec : float
    ground_link : tuple[str,str]
    held_joints_damping : float | dict[tuple[str,str] | str, float]
    held_joints_stiffness : float | dict[tuple[str,str] | str, float]
    homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float]
    homing_joint_position : dict[tuple[str,str], float]
    history_length_action_smoothed : int
    history_length_action_raw : int
    init_on_reset_ratio : float
    randomization_initial_height_range_meters : float
    randomization_initial_joint_pose_range : float | dict[tuple[str,str] | str, float]
    maxStepsPerEpisode : int
    minmax_damping : dict[str,tuple[float,float]] | tuple[float,float]
    minmax_stiffness : dict[str,tuple[float,float]] | tuple[float,float]
    noise_abs_obs_angvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    noise_abs_obs_gravity_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    noise_abs_obs_joints_pve_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    noise_abs_obs_linacc_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    noise_abs_obs_linvel_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    noise_abs_obs_posz_ep_mustd_step_std : tuple[float,float,float] |  th.Tensor
    observe_actor_safety_state : bool
    quiet : bool
    robot_description_format : Literal["urdf", "sdf", "mjcf"]
    robot_description_string : str
    robot_main_body_link : str
    robot_name : str
    robot_root_link : str
    ctrl_joints_damping : float | dict[tuple[str,str] | str, float]
    ctrl_joints_stiffness : float | dict[tuple[str,str] | str, float]
    safety_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float]
    seed : int
    step_precision_tolerance : float
    stepLength_sec : float
    terminate_on_safety : bool
    th_device : th.device
    verbose_infos : bool
    observe_linvel_nonprivileged : bool
    control_limits_minmax_pve : dict[tuple[str,str], th.Tensor] | None = None
    control_mode_position_delta_max : float | dict[tuple[str,str] | str, float] | None = None
    enable_limits_safety : bool = True
    enable_link_collisions : list[tuple[tuple[str,str],list[tuple[str,str]]]] | None = dataclasses.field(default_factory=list)
    enable_posref_safety : bool = True
    extrinsics_only_privileged : bool = False
    homing_joint_position_references : dict[tuple[str,str], float] | None = None
    impulse_duration_minmax : tuple[float,float ]= (0.01, 5.0)
    impulse_mean_std : tuple[float,float ]= (50.0, 50.0)
    impulse_probability_per_sec : float = 0.0
    just_health_reward : bool = False
    longterm_states_decimation_time : float = 0.0001
    merge_privileged : bool = False
    minimal_infos : bool = False
    no_infos : bool = False
    observe_full_robot_state : bool = False
    offset_envs_ep_starts : bool = False
    posref_safety_period : float = 0.001
    randomized_com_links : list[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_com_xyz_diff_distribution : DistributionDef = ("uniform", ([-0.01,-0.01,-0.01],[0.01, 0.01,0.01]))
    randomized_dof_armature_joints : Sequence[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_dof_armature_ratios : DistributionDef = ("uniform", (0.9, 1.1))
    randomized_dof_damping_joints : Sequence[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_dof_damping_ratios : DistributionDef = ("uniform", (0.9, 1.1))
    randomized_dof_frictionloss_joints : Sequence[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_dof_frictionloss_ratios : DistributionDef = ("uniform", (0.9, 1.1))
    randomized_friction_links : list[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_friction_slide_spin_roll_ratios : DistributionDef = ("uniform", ([0.9,0.9,0.9],[1.1,1.1,1.1]))
    randomized_gains_damping_ratio_epstd : float = 0.0
    """ Standard deviation of the Normal randomization applied to the impedance control damping at the start of each episode """
    randomized_gains_stiffness_ratio_epstd : float = 0.0
    """ Standard deviation of the Normal randomization applied to the impedance control stiffness at the start of each episode """
    randomized_mass_links : list[tuple[str,str]] = dataclasses.field(default_factory=list)
    randomized_mass_ratios_distr : DistributionDef = ("normal", (1.0, 0.05))
    randomization_recycle_model_alterations : bool = False
    randomized_reference_filter_distribution : DistributionDef | None = None
    randomization_recycle_init_pose : bool = False
    saturate_jimp_posref_limits : bool = False
    single_reward_space : ThBox | None = None
    ui_camera_resolution_hw : tuple[int,int] = (144,256)

class RobotVecEnv(ControlledVecEnv[BaseVecJointImpedanceAdapter, Observation]):

    @dataclass
    class Configuration:
        action_delay_epmustd_ststd : th.Tensor
        action_exp_smoothing_1s : float
        noise_action_mustd : th.Tensor
        joint_ctrl_limits_minmax_pve : dict[tuple[str,str], th.Tensor]
        control_mode : JointImpedanceActionHelper.CONTROL_MODES
        goal_err_exp_smoothing_1s : float
        history_length : int
        homing_body_pose_xyz_xyzw : th.Tensor
        homing_ctrl_joints_position : th.Tensor
        """Actual homing positions for agent-controlled joints (may differ from references in homing_ctrl_joints_pvesd)"""
        homing_ctrl_joints_pvesd : th.Tensor
        homing_held_joints_position : dict[tuple[str,str],th.Tensor]
        homing_held_joints_pvesd : th.Tensor
        homing_nonctrl_joints_position : dict[tuple[str,str],th.Tensor]
        init_args : RobotVecEnvInitArgs
        joint_physical_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_damping : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_stiffness : dict[tuple[str,str],th.Tensor]
        joints_agent_controlled : Sequence[tuple[str,str]]
        joints_all_env_controlled : Sequence[tuple[str,str]]
        joints_env_held : Sequence[tuple[str,str]]
        longterm_exp_smoothing_1s : float
        main_body_link : tuple[str,str]
        noise_angvel_ep_mustdstd : th.Tensor
        noise_gravity_ep_mustdstd : th.Tensor
        noise_joints_pve_mustdstd : th.Tensor
        noise_linacc_ep_mustdstd : th.Tensor
        noise_linvel_ep_mustdstd : th.Tensor
        noise_posz_ep_mustdstd : th.Tensor
        obs_dtype : th.dtype
        observe_held_joints : bool
        randomized_com_links : tuple[tuple[str,str], ...]
        """Center of mass randomization, randomized links"""
        randomized_com_xyz_diff_distribution : DistributionDefTh
        """Center of mass randomization ranges. The position is randomized by summing to it a 3-vector sampled from this distribution"""
        randomized_dof_armature_joints : tuple[tuple[str,str],...]
        randomized_dof_armature_ratios : DistributionDefTh
        randomized_dof_damping_joints : tuple[tuple[str,str],...]
        randomized_dof_damping_ratios : DistributionDefTh
        randomized_dof_frictionloss_joints : tuple[tuple[str,str],...]
        randomized_dof_frictionloss_ratios : DistributionDefTh
        randomized_friction_links : tuple[tuple[str,str],...]
        """Friction randomization, randomized links"""
        randomized_friction_slide_spin_roll_ratios : DistributionDefTh
        """Friction randomization ratio for each link in randomized_friction_slide_spin_roll_ratios"""
        randomized_mass_links : tuple[tuple[str,str],...]
        """Mass randomization, randomized links"""
        randomized_mass_ratio_distribution : DistributionDefTh
        """Mass randomization ratio for each link in randomized_mass_links. The mass is randomized by multiplying by a factor sampled from this distribution"""
        randomized_reference_filter_distribution : DistributionDefTh | None
        """If not None, the reference filter cutoff frequency is randomized at each episode start by sampling from this distribution""" 
        real : bool
        reward_clamp : th.Tensor
        reward_penalties_max : float
        robot_is_floating : bool
        robot_root_joint : str
        robot_root_link : tuple[str,str]
        show_goal : bool
        spawn_root_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float]
        ui_camera_link : tuple[str,str]
        ui_camera_name : str
        ui_rel_camera_pose_dist_pitch_yaw : th.Tensor
        vec_jimp_cmd_size : tuple[int,int,int]
        vec_size : int
        velref_from_posref : bool


    metadata = {'render.modes': ['rgb_array']}
    # STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT_PREPROC = "action"
    STATE_ACT_RAW_HIST = "action_raw"
    STATE_LAST_ACT_RAW = "last_action_raw"
    STATE_ROBOT = "robot"
    STATE_HELD_JOINTS = "held_joints"
    STATE_JOINT_STEP_STATS = "joint_step_stats"
    STATE_JOINT_LONGTERM_STATS = "joint_longterm_stats"
    STATE_EXTRINSIC = "extrinsic"
    STATE_INTERNAL = "internal"
    STATE_RANDOMIZATIONS = "randomizations"
    STATE_POS_REF_ERR = "pos_ref_err"
    
    
    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", [  "SAFETY_LIMITS_TRIGGERED",
                                                    "SAFETY_POSREF_TRIGGERED",
                                                    "SAFETY_POSREF_VIOLATION_COUNT",
                                                    "STEP_COUNT",
                                                    "SIM_TIME",
                                                    "LAST_STEP_DT"], start=0)

    EXTRINSIC_FIELDS = IntEnum("EXTRINSIC_FIELDS", ["BODY_REL_LINVEL_X",
                                                   "BODY_REL_LINVEL_Y",
                                                   "BODY_REL_LINVEL_Z",
                                                   "BODY_REL_ANGVEL_X",
                                                   "BODY_REL_ANGVEL_Y",
                                                   "BODY_REL_ANGVEL_Z",
                                                   "BODY_ABS_LINVEL_X",
                                                   "BODY_ABS_LINVEL_Y",
                                                   "BODY_ABS_LINVEL_Z",
                                                   "BODY_ABS_ANGVEL_X",
                                                   "BODY_ABS_ANGVEL_Y",
                                                   "BODY_ABS_ANGVEL_Z",
                                                   "BODY_ABS_POS_Z",
                                                   "BODY_REL_GRAVITY_X",
                                                   "BODY_REL_GRAVITY_Y",
                                                   "BODY_REL_GRAVITY_Z",
                                                   "BODY_REL_LINACC_X",
                                                   "BODY_REL_LINACC_Y",
                                                   "BODY_REL_LINACC_Z"], start=0)
    ACT_FIELDS = IntEnum("ACT_FIELDS", ["ACTION"], start=0)

    JOINT_LONGTERM_STATS_FIELDS = IntEnum("LONGTERM_STATS_FIELDS", ["AVG_POS"])

    RANDOMIZATIONS_FIELDS = IntEnum("RANDOMIZATIONS_FIELDS", [  "FULL_RANDOMIZATION_STATE"]) # Right now put everything together as there are randomizations of different dimensionality

    POS_REF_ERR_FIELDS = IntEnum("POS_REF_ERRS", ["POS_ERR"], start=0)

    joint_filters = {JOINT_FILTERS.ALL : lambda joint_name, robot_model: True,
                     JOINT_FILTERS.ALL_REVOLUTE : _is_joint_revolute}
    link_filters  = {LINK_FILTERS.ALL : lambda link_name, robot_model: True}

    @dataclass
    class EpisodeConfiguration:
        vec_initial_ctrl_joint_pose : th.Tensor
        vec_max_ep_steps : th.Tensor
        vec_init_on_reset : th.Tensor
        randomized_stiffness_factor : th.Tensor
        """Shape (envs_num, ctrl_joints)"""
        randomized_damping_factor : th.Tensor
        """Shape (envs_num, ctrl_joints)"""
        action_delay_mu : th.Tensor
        """Mean of the action delay in these episodes, shape (envs_num,)"""
        link_masses_ratios : th.Tensor
        link_frictions_ratios : th.Tensor
        link_coms_diffs : th.Tensor
        joint_armatures_ratios : th.Tensor
        joint_dampings_ratios : th.Tensor
        joint_frictionlosses_ratios : th.Tensor
        joint_reference_filter_freqs : th.Tensor

    @dataclass
    class PrecomputedSimInitData:
        all_joints_names : Sequence[tuple[str,str]] # all the joints to be directly set in sim init, in the correct order
        all_joints_states : th.Tensor # the corresponding states to be directly set in sim init
        initial_cmd_vec_j_pvesd : th.Tensor # the initial impedance command to be set in sim init (excluding the homing-held joints)
        full_cmd_vec_j_pvesd : th.Tensor # the full initial impedance command to be set in sim init (including the homing-held joints)


    @dataclass
    class Statistics:
        tracking_errors : th.Tensor
        avg_tracking_error : th.Tensor = dataclasses.field(default_factory=lambda: th.tensor(-1.0))
        rewards : dict = dataclasses.field(default_factory=lambda: {})


    def  __init__(self, init_args : RobotVecEnvInitArgs):
        self._main_seed = init_args.seed
        # self._rng_get_count = 0
        self._rng = th.Generator(device=init_args.th_device)
        self._rng.manual_seed(init_args.seed)
        self._th_device = init_args.th_device
        self._obs_dtype = th.float32
        self._robot_model = Robot(robot_description_string=init_args.robot_description_string,
                                    robot_description_format=init_args.robot_description_format)
        ggLog.info(f"urdf: {init_args.robot_description_string}")
        ggLog.info(f"Robot has links: {self._robot_model.get_frame_names()}")
        ggLog.info(f"Robot has joints: {self._robot_model.get_joint_names()}")
        root_joint_name = self._robot_model.get_parent_joint(init_args.robot_root_link)
        is_floating = self._robot_model.get_joint_properties([root_joint_name])[root_joint_name]["type"] == Robot.JOINT_TYPES.FLOATING
        ggLog.info(f"Robot root joint {root_joint_name} is floating: {is_floating}")
        # self._build_new_instantaneous_state = th.vmap(self._build_new_instantaneous_state_single)
        # ggLog.info("Properties:"+("\n".join([str(jp) for jp in self._robot_model.get_joint_properties(self._robot_model.get_joint_names()).items()])))
        # exit()
        self._obs2act_timings = DelayStats(maxlen=500,
                                           track_obj_type_growth=False)
        
        action_exp_smoothing_1s = 0.5**(1/init_args.action_smoothing_halflife_sec) if init_args.action_smoothing_halflife_sec>0 else 0.0
        goal_err_exp_smoothing_1s = 0.5**(1/init_args.goal_err_smoothing_halflife_sec) if init_args.goal_err_smoothing_halflife_sec>0 else 0.0
        longterm_exp_smoothing_1s = 0.1**(1/init_args.longterm_states_decimation_time) if init_args.longterm_states_decimation_time>0 else 0.0

        (phys_limits_minmax_pve,
        safe_limits_minmax_pve,
        control_limits_minmax_pve,
        controlled_joints_rn,
        held_joints,
        free_joints_rn,
        all_controlled_joints,
        homing_ctrl_joints_position,
        homing_ctrl_joints_pvesd,
        homing_held_joints_pvesd,
        homing_held_joints_position,
        homing_nonctrl_joints_position
        ) = self._build_joint_limits(   robot_name=init_args.robot_name,
                                        controlled_joints = init_args.controlled_joints,
                                        control_limits_center = init_args.control_limits_center,
                                        control_limits_ratios_minmax_pve = init_args.control_limits_ratios_minmax_pve,
                                        control_limits_minmax_pve = init_args.control_limits_minmax_pve,
                                        safety_limits_ratios_minmax_pve = init_args.safety_limits_ratios_minmax_pve,
                                        free_joints=init_args.free_joints,
                                        homing_joint_pose = init_args.homing_joint_position,
                                        homing_references = init_args.homing_joint_position_references,
                                        ctrl_joints_stiffness = init_args.ctrl_joints_stiffness,
                                        ctrl_joints_damping = init_args.ctrl_joints_damping,
                                        held_joints_stiffness=init_args.held_joints_stiffness,
                                        held_joints_damping=init_args.held_joints_damping)
        if isinstance(init_args.minmax_stiffness, tuple):
            minmax_stiffness_thdict = {k:self._thtens(init_args.minmax_stiffness) for k in phys_limits_minmax_pve.keys()}
        else:
            if not all(k in init_args.minmax_stiffness for k in phys_limits_minmax_pve.keys()):
                raise ValueError(f"minmax_stiffness dict is missing keys {[k for k in phys_limits_minmax_pve.keys() if k not in init_args.minmax_stiffness]}."
                                 "If specified as a dict, minmax_stiffness must contain an entry for each joint")
            minmax_stiffness_thdict = {(init_args.robot_name,k):self._thtens(minmax) for k,minmax in init_args.minmax_stiffness.items()}
        if isinstance(init_args.minmax_damping, tuple):
            minmax_damping_thdict = {k:self._thtens(init_args.minmax_damping) for k in phys_limits_minmax_pve.keys()}
        else:
            if not all(k in init_args.minmax_damping for k in phys_limits_minmax_pve.keys()):
                raise ValueError(f"minmax_damping dict is missing keys {[k for k in phys_limits_minmax_pve.keys() if k not in init_args.minmax_damping]}."
                                 "If specified as a dict, minmax_damping must contain an entry for each joint")
            minmax_damping_thdict = {(init_args.robot_name,k):self._thtens(minmax) for k,minmax in init_args.minmax_damping.items()}


        self._configuration = self.Configuration(   init_args=init_args,
                                                    action_delay_epmustd_ststd = self._thtens(init_args.noise_action_delay_mustd_std),
                                                    action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                    noise_action_mustd = self._thtens(init_args.noise_action_mustd),
                                                    joint_ctrl_limits_minmax_pve = control_limits_minmax_pve,
                                                    control_mode = JointImpedanceActionHelper.CONTROL_MODES[init_args.control_mode.upper()],
                                                    goal_err_exp_smoothing_1s = goal_err_exp_smoothing_1s,
                                                    history_length = max(2,init_args.frame_stack_length),
                                                    homing_body_pose_xyz_xyzw = self._thtens(init_args.homing_body_pose_xyz_xyzw),
                                                    homing_ctrl_joints_position = homing_ctrl_joints_position,
                                                    homing_ctrl_joints_pvesd = homing_ctrl_joints_pvesd,
                                                    homing_held_joints_position = homing_held_joints_position,
                                                    homing_held_joints_pvesd = homing_held_joints_pvesd,
                                                    homing_nonctrl_joints_position = homing_nonctrl_joints_position,
                                                    joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                    joint_safe_limits_minmax_damping = minmax_damping_thdict,
                                                    joint_safe_limits_minmax_pve = safe_limits_minmax_pve,
                                                    joint_safe_limits_minmax_stiffness = minmax_stiffness_thdict,
                                                    joints_all_env_controlled = all_controlled_joints,
                                                    joints_agent_controlled = controlled_joints_rn,
                                                    joints_env_held = held_joints,
                                                    longterm_exp_smoothing_1s = longterm_exp_smoothing_1s, # alpha so that the contribution of a sample longterm_states_decimation_time seconds ago is 0.1
                                                    main_body_link=(init_args.robot_name,init_args.robot_main_body_link),
                                                    noise_angvel_ep_mustdstd =  self._thtens(init_args.noise_abs_obs_angvel_ep_mustd_step_std),
                                                    noise_gravity_ep_mustdstd = self._thtens(init_args.noise_abs_obs_gravity_ep_mustd_step_std),
                                                    noise_joints_pve_mustdstd = self._thtens(init_args.noise_abs_obs_joints_pve_ep_mustd_step_std),
                                                    noise_linacc_ep_mustdstd =  self._thtens(init_args.noise_abs_obs_linacc_ep_mustd_step_std),
                                                    noise_linvel_ep_mustdstd =  self._thtens(init_args.noise_abs_obs_linvel_ep_mustd_step_std),
                                                    noise_posz_ep_mustdstd =    self._thtens(init_args.noise_abs_obs_posz_ep_mustd_step_std),
                                                    obs_dtype = self._obs_dtype,
                                                    observe_held_joints = False,
                                                    randomized_dof_armature_joints =                None, # Will fill up later
                                                    randomized_dof_armature_ratios =                None, # Will fill up later
                                                    randomized_dof_damping_joints =                 None, # Will fill up later
                                                    randomized_dof_damping_ratios =                 None, # Will fill up later
                                                    randomized_dof_frictionloss_joints =            None, # Will fill up later
                                                    randomized_dof_frictionloss_ratios =            None, # Will fill up later
                                                    randomized_com_links =                          None, # Will fill up later
                                                    randomized_com_xyz_diff_distribution =          None, # Will fill up later
                                                    randomized_friction_links =                     None, # Will fill up later
                                                    randomized_friction_slide_spin_roll_ratios =    None, # Will fill up later
                                                    randomized_mass_links =                         None, # Will fill up later
                                                    randomized_mass_ratio_distribution =            None, # Will fill up later
                                                    randomized_reference_filter_distribution =      None, # Will fill up later
                                                    real = False,
                                                    reward_clamp = self._thtens(100.0),
                                                    reward_penalties_max = 100.0,
                                                    robot_is_floating = is_floating,
                                                    robot_root_joint = root_joint_name,
                                                    robot_root_link=(init_args.robot_name,init_args.robot_root_link),
                                                    show_goal = True,
                                                    spawn_root_pose_xyz_xyzw = (0,0,0,0,0,0,1),
                                                    ui_camera_link = ("simple_camera", "simple_camera_link"),
                                                    ui_camera_name="simple_camera",
                                                    ui_rel_camera_pose_dist_pitch_yaw = self._thtens([2.5, 30/180*3.14159, -90/180*3.14159]),
                                                    vec_jimp_cmd_size=(init_args.adapter.vec_size(), len(controlled_joints_rn), 5),
                                                    vec_size=init_args.adapter.vec_size(),
                                                    velref_from_posref=False,
                                                    )
        jrand = self._configuration.init_args.randomization_initial_joint_pose_range
        has_joint_randomization = (isinstance(jrand, dict) and any(r > 0 for r in jrand.values())) or (isinstance(jrand, float) and jrand > 0)
        has_height_randomization = self._configuration.init_args.randomization_initial_height_range_meters > 0
        self._initial_pose_randomization_enabled = has_joint_randomization or has_height_randomization
        self._last_pose_randomization : th.Tensor | None = None
        self._last_sent_v_j_pvesd = homing_ctrl_joints_pvesd.repeat(init_args.adapter.vec_size(), 1, 1)
        self._excluded_collision_pairs : set[tuple[str,str]] = set()
        self._safe_limits_minmax_j_pve = th.stack([safe_limits_minmax_pve[jn] for jn in controlled_joints_rn], dim=1)
        self._posref_safety_minmmax_diff = self._safe_limits_minmax_j_pve[:,:,1]*self._configuration.init_args.stepLength_sec
        self._posref_saturation_minmmax_diff = self._posref_safety_minmmax_diff*0.999
        self._impulse_disturbances_enabled = init_args.impulse_probability_per_sec > 0
        self._homing_held_joints_vec_pvesd = homing_held_joints_pvesd.repeat(init_args.adapter.vec_size(), 1, 1)
        # ggLog.info(f"homing_ctrl_joints_pvesd = {homing_ctrl_joints_pvesd}")
        # ggLog.info(f"self._held_joints_cmd_vec_j_pvesd = {self._configuration.homing_held_joints_pvesd}")
        # ggLog.info(f"internally_controlled_joints = {self._configuration.all_controlled_joints}")

        _ctrl_pdm = self._configuration.init_args.control_mode_position_delta_max
        if isinstance(_ctrl_pdm, dict):
            _ctrl_pdm_dict = expand_default_dict(_ctrl_pdm, controlled_joints_rn)
            ctrl_position_delta_max : th.Tensor | float | None = self._thtens([_ctrl_pdm_dict[jn] for jn in controlled_joints_rn])
        else:
            ctrl_position_delta_max = _ctrl_pdm
        self._action_helper = JointImpedanceActionHelper(
                                vec_size=init_args.adapter.vec_size(),
                                control_mode=self._configuration.control_mode,
                                joints=controlled_joints_rn,
                                joints_minmax_pvesd={jn:th.cat([control_limits_minmax_pve[jn],
                                                                minmax_stiffness_thdict[jn].unsqueeze(1),
                                                                minmax_damping_thdict[jn].unsqueeze(1)], dim=1) 
                                                        for jn in controlled_joints_rn},
                                center_position = homing_ctrl_joints_pvesd[:,0],
                                safe_stiffness=homing_ctrl_joints_pvesd[:,3].clone(),
                                safe_damping=homing_ctrl_joints_pvesd[:,4].clone(),
                                th_device=self._configuration.init_args.th_device,
                                generator=self._rng,
                                position_delta_max=ctrl_position_delta_max,)
        
        self._build_stats()

        if init_args.single_reward_space is None:
            init_args.single_reward_space = ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=init_args.th_device)
        
        super().__init__(max_episode_steps=init_args.maxStepsPerEpisode,
                         step_duration_sec=init_args.stepLength_sec,
                         adapter=init_args.adapter,
                         single_state_space=None,
                         single_observation_space=None,
                         single_action_space=self._action_helper.get_single_action_space(),
                         single_reward_space=init_args.single_reward_space,
                         info_space=None,
                         step_precision_tolerance = init_args.step_precision_tolerance,
                         th_device = self._th_device,
                         obs_dtype = self._obs_dtype,
                         seed = init_args.seed,
                         max_possible_episode_steps = self._configuration.init_args.maxStepsPerEpisode)
        self._build()

        # Randomizations
        init_args.randomized_mass_links =               self._find_links(init_args.randomized_mass_links)
        init_args.randomized_com_links =                self._find_links(init_args.randomized_com_links)
        init_args.randomized_friction_links =           self._find_links(init_args.randomized_friction_links)        
        init_args.randomized_dof_armature_joints =      self._find_joints(init_args.randomized_dof_armature_joints)
        init_args.randomized_dof_damping_joints =       self._find_joints(init_args.randomized_dof_damping_joints)
        init_args.randomized_dof_frictionloss_joints =  self._find_joints(init_args.randomized_dof_frictionloss_joints)
        self._configuration.randomized_mass_links = tuple(init_args.randomized_mass_links)
        self._configuration.randomized_com_links = tuple(init_args.randomized_com_links)
        self._configuration.randomized_friction_links = tuple(init_args.randomized_friction_links)
        self._configuration.randomized_dof_armature_joints = tuple(init_args.randomized_dof_armature_joints)
        self._configuration.randomized_dof_damping_joints = tuple(init_args.randomized_dof_damping_joints)
        self._configuration.randomized_dof_frictionloss_joints = tuple(init_args.randomized_dof_frictionloss_joints)
        self._model_randomization_enabled = (
                any([len(self._configuration.randomized_mass_links)>0               and not distr_is_constant(init_args.randomized_mass_ratios_distr),
                     len(self._configuration.randomized_com_links)>0                and not distr_is_constant(init_args.randomized_com_xyz_diff_distribution),
                     len(self._configuration.randomized_friction_links)>0           and not distr_is_constant(init_args.randomized_friction_slide_spin_roll_ratios),
                     len(self._configuration.randomized_dof_armature_joints)>0      and not distr_is_constant(init_args.randomized_dof_armature_ratios),
                     len(self._configuration.randomized_dof_damping_joints)>0       and not distr_is_constant(init_args.randomized_dof_damping_ratios),
                     len(self._configuration.randomized_dof_frictionloss_joints)>0  and not distr_is_constant(init_args.randomized_dof_frictionloss_ratios)]) 
                )
        self._configuration.randomized_mass_ratio_distribution          = self._distr_to_tensor(init_args.randomized_mass_ratios_distr,         size=(len(init_args.randomized_mass_links),))
        self._configuration.randomized_com_xyz_diff_distribution        = self._distr_to_tensor(init_args.randomized_com_xyz_diff_distribution, size=(len(init_args.randomized_com_links), 3))
        self._configuration.randomized_friction_slide_spin_roll_ratios  = self._distr_to_tensor(init_args.randomized_friction_slide_spin_roll_ratios, size=(len(init_args.randomized_friction_links), 3))
        self._configuration.randomized_dof_armature_ratios              = self._distr_to_tensor(init_args.randomized_dof_armature_ratios,       size=(len(init_args.randomized_dof_armature_joints),))
        self._configuration.randomized_dof_damping_ratios               = self._distr_to_tensor(init_args.randomized_dof_damping_ratios,        size=(len(init_args.randomized_dof_damping_joints),))
        self._configuration.randomized_dof_frictionloss_ratios          = self._distr_to_tensor(init_args.randomized_dof_frictionloss_ratios,   size=(len(init_args.randomized_dof_frictionloss_joints),))
        self._configuration.randomized_reference_filter_distribution    = self._distr_to_tensor(init_args.randomized_reference_filter_distribution, size=(1,)) if init_args.randomized_reference_filter_distribution is not None else None
        self._filters_randomization_enabled = self._configuration.randomized_reference_filter_distribution != None and not distr_is_constant(self._configuration.randomized_reference_filter_distribution)


        self._state_helper : DictStateHelper
        self._build_state_helper(init_args.adapter)
        self._safety_limits = self._state_helper.sub_helpers[self.STATE_ROBOT].build_robot_limits(
                                                    joint_limit_minmax_pve={jn:self._configuration.joint_safe_limits_minmax_pve[jn] for jn in self._configuration.joints_agent_controlled},
                                                    stiffness_minmax={jn: self._configuration.joint_safe_limits_minmax_stiffness[jn] for jn in self._configuration.joints_agent_controlled},
                                                    damping_minmax={jn: self._configuration.joint_safe_limits_minmax_damping[jn] for jn in self._configuration.joints_agent_controlled})
        self._ctrl_limits = self._state_helper.sub_helpers[self.STATE_ROBOT].build_robot_limits(
                                                    joint_limit_minmax_pve={jn:self._configuration.joint_ctrl_limits_minmax_pve[jn] for jn in self._configuration.joints_agent_controlled},
                                                    stiffness_minmax={jn: self._configuration.joint_safe_limits_minmax_stiffness[jn] for jn in self._configuration.joints_agent_controlled},
                                                    damping_minmax={jn: self._configuration.joint_safe_limits_minmax_damping[jn] for jn in self._configuration.joints_agent_controlled})

        self._current_episode_config = RobotVecEnv.EpisodeConfiguration(
                                                    vec_initial_ctrl_joint_pose = th.stack([self._configuration.homing_ctrl_joints_position,
                                                                                             self._configuration.homing_ctrl_joints_pvesd[:,0]
                                                                                            ], dim=-1).unsqueeze(0).expand(init_args.adapter.vec_size(), -1, -1).clone(),
                                                    vec_init_on_reset = th.ones(size=(init_args.adapter.vec_size(),), dtype=th.bool).to(device=init_args.th_device, non_blocking=init_args.th_device.type=="cuda"),
                                                    vec_max_ep_steps = self._thfull(size=(init_args.adapter.vec_size(),), fill_value=self._configuration.init_args.maxStepsPerEpisode, dtype=th.long),
                                                    randomized_damping_factor=self._thtens(1.0).expand(init_args.adapter.vec_size(),len(self._configuration.joints_agent_controlled)).clone(),
                                                    randomized_stiffness_factor=self._thtens(1.0).expand(init_args.adapter.vec_size(),len(self._configuration.joints_agent_controlled)).clone(),
                                                    action_delay_mu = self._thzeros((init_args.adapter.vec_size(),)),
                                                    link_masses_ratios = self._thones((init_args.adapter.vec_size(), len(self._configuration.randomized_mass_links))),
                                                    link_frictions_ratios = self._thones((init_args.adapter.vec_size(), len(self._configuration.randomized_friction_links), 3)),
                                                    link_coms_diffs = self._thzeros((init_args.adapter.vec_size(), len(self._configuration.randomized_com_links), 3)),
                                                    joint_armatures_ratios = self._thones((init_args.adapter.vec_size(), len(self._configuration.randomized_dof_armature_joints))),
                                                    joint_dampings_ratios = self._thones((init_args.adapter.vec_size(), len(self._configuration.randomized_dof_damping_joints))),
                                                    joint_frictionlosses_ratios = self._thones((init_args.adapter.vec_size(), len(self._configuration.randomized_dof_frictionloss_joints))),
                                                    joint_reference_filter_freqs = self._thzeros((init_args.adapter.vec_size(),))
                                                    )

        # preallocate some things
        self._randomized_mass_link_ids = self._adapter.get_links_ids(self._configuration.randomized_mass_links)
        self._randomized_com_links_ids = self._adapter.get_links_ids(self._configuration.randomized_com_links)
        self._randomized_friction_links_ids = self._adapter.get_links_ids(self._configuration.randomized_friction_links)
        self._randomized_dof_armature_joints_ids = self._adapter.get_joints_ids(self._configuration.randomized_dof_armature_joints)
        self._randomized_dof_damping_joints_ids = self._adapter.get_joints_ids(self._configuration.randomized_dof_damping_joints)
        self._randomized_dof_frictionloss_joints_ids = self._adapter.get_joints_ids(self._configuration.randomized_dof_frictionloss_joints)

        self.single_state_space=self._state_helper.get_single_space()
        self.single_observation_space=self._state_helper.get_single_obs_space()
        self.vec_state_space = batch_space(self.single_state_space, self._adapter.vec_size())
        self.vec_observation_space = batch_space(self.single_observation_space, self._adapter.vec_size())

        self._abs_gravity_dir = self._thtens([0.0,0.0,-1.0])
        self._eps_start_stime = self._thzeros(size=(self.num_envs,))
        self._reset_state_full()
        self._set_current_ep_config(reset_options = {}, vec_mask=self._all_envs)
        self._last_obs = self._state_helper.observe(self._current_state)
        self._last_raw_actions = self._thzeros(self._action_helper.get_vec_action_space().shape)
        self._last_preprocessed_actions = self._thzeros(self._action_helper.get_vec_action_space().shape)


        if isinstance(self._adapter, BaseVecSimulationAdapter) and init_args.enable_link_collisions is not None:
            self._adapter.set_body_collisions(init_args.enable_link_collisions)
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels)
        self.set_seeds(th.as_tensor(init_args.seed))
        ggLog.info(f"Starting up adapter....")
        self._adapter.startup()
        ggLog.info(f"Adapter started.")
        self.initialize_episodes()
        
    def _build_joint_limits(self,   robot_name : str,
                                    controlled_joints : Sequence[str | JOINT_FILTERS],
                                    free_joints : Sequence[str],
                                    control_limits_center : dict[tuple[str,str], float],
                                    safety_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float], 
                                    control_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float], 
                                    control_limits_minmax_pve : dict[tuple[str,str], th.Tensor] | None,
                                    homing_joint_pose : dict[tuple[str,str], float],
                                    homing_references : dict[tuple[str,str], float] | None,
                                    ctrl_joints_stiffness : float | dict[tuple[str,str] | str, float],
                                    ctrl_joints_damping : float | dict[tuple[str,str] | str, float],
                                    held_joints_stiffness : float | dict[tuple[str,str] | str, float],
                                    held_joints_damping : float | dict[tuple[str,str] | str, float]
                                    ):
        
        # ----- DEFINE JOINT GROUPS ------------------------------------------

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

        self.link_filters[LINK_FILTERS.ALL_ROBOT] = lambda link_name, robot_model: link_name[0]==robot_name
        
        controllable_joints = [(robot_name,jn) for jn,p in self._robot_model.get_joint_properties().items() if p["type"] in [Robot.JOINT_TYPES.REVOLUTE, Robot.JOINT_TYPES.PRISMATIC,Robot.JOINT_TYPES.CONTINUOUS]]
        agent_controlled_joints_rn : list[tuple[str,str]] = [(robot_name,jn) for jn in controlled_joints_str]
        free_joints_rn = [(robot_name,jn) for jn in free_joints]        
        # Held joints will be still be controlled with a joint impedance adapter, but are not exposed to the outside
        # they will be kept at a fixed position
        held_joints = [jn for jn in controllable_joints if (jn not in agent_controlled_joints_rn and jn not in free_joints_rn)]
        all_controlled_joints = agent_controlled_joints_rn+held_joints

        # ----- DEFINE PHYSICAL LIMITS ------------------------------------------

        phys_limits_minmax_pve = {(robot_name,k):self._thtens(l) 
                                    for k,l in self._robot_model.get_joint_limits([jn[1] for jn in all_controlled_joints]).items()}
        default_effort_lim = 1000.0
        default_vel_lim = 100.0
        for jn, minmax_pve in phys_limits_minmax_pve.items():
            if not th.isfinite(minmax_pve).all():
                 ggLog.warn(f"Physical limits for joint {jn} contains non-finite values {minmax_pve}, using defaults (effort_lim={default_effort_lim}, vel_lim={default_vel_lim})")
            minmax_pve[0,2][minmax_pve[0,2]==float("-inf")] = -default_effort_lim
            minmax_pve[1,2][minmax_pve[1,2]==float("inf")] = default_effort_lim
            minmax_pve[0,1][minmax_pve[0,1]==float("-inf")] = -default_vel_lim
            minmax_pve[1,1][minmax_pve[1,1]==float("inf")] = default_vel_lim

        # ----- DEFINE SAFETY LIMITS ------------------------------------------

        if isinstance(safety_limits_ratios_minmax_pve, dict):
            safety_limits_dict_ratios_minmax_pve = safety_limits_ratios_minmax_pve
        else:
            safety_limits_dict_ratios_minmax_pve = {k:safety_limits_ratios_minmax_pve for k in phys_limits_minmax_pve}
        safe_limits_ratios_minmax_pve_th = {k:self._thtens(v).expand((2,3,)) for k,v in safety_limits_dict_ratios_minmax_pve.items()}
        def scale_limit(jn, minmax_pve : th.Tensor, minmax_scaling : th.Tensor):
            ranges_pve = minmax_pve[1] - minmax_pve[0]
            center_pve = (minmax_pve[1] + minmax_pve[0])/2
            scaled_ranges = ranges_pve.expand(2,3)*minmax_scaling
            lims = th.stack([center_pve - scaled_ranges[0]/2, center_pve + scaled_ranges[1]/2], dim=0)
            return lims
        safe_limits_minmax_pve = {jn: scale_limit(  jn,
                                                    lim_minmax_pve,
                                                    safe_limits_ratios_minmax_pve_th[jn])
                                    for jn,lim_minmax_pve in phys_limits_minmax_pve.items()}
        for jn in safe_limits_minmax_pve.keys():
            if th.any(safe_limits_minmax_pve[jn][0] < phys_limits_minmax_pve[jn][0]) or th.any(phys_limits_minmax_pve[jn][1] < safe_limits_minmax_pve[jn][1]):
                ggLog.warn( f"Safe limits exceeds physical limits for joint {jn},\n"
                            f" safe=\n{safe_limits_minmax_pve[jn]}\n"
                            f" physical=\n{phys_limits_minmax_pve[jn]}\n"
                            f"The limit will be clamped.") 
        safe_limits_minmax_pve = {jn:lims.clamp(min=phys_limits_minmax_pve[jn][0].expand(2,-1), 
                                                max=phys_limits_minmax_pve[jn][1].expand(2,-1)) 
                                  for jn,lims in safe_limits_minmax_pve.items()}

        # ----- DEFINE CONTROL LIMITS ------------------------------------------

        if control_limits_minmax_pve is None:
            if control_limits_ratios_minmax_pve is None:
                control_limits_ratios_minmax_pve = safety_limits_ratios_minmax_pve
            if isinstance(control_limits_ratios_minmax_pve, dict):
                control_limits_dict_ratios_minmax_pve = control_limits_ratios_minmax_pve
            else:
                control_limits_dict_ratios_minmax_pve = {k:control_limits_ratios_minmax_pve for k in phys_limits_minmax_pve}
            control_limits_ratios_minmax_pve_th = {k:self._thtens(v).expand((2,3,)) for k,v in control_limits_dict_ratios_minmax_pve.items()}
            def scale_limit_centered(jn, minmax_pve : th.Tensor, minmax_scaling : th.Tensor, center_pos : float | None):
                ranges_pve = minmax_pve[1] - minmax_pve[0]
                if center_pos is not None:
                    center_pve = (minmax_pve[1] + minmax_pve[0]) / 2
                    center_pve = center_pve.clone()
                    center_pve[0] = center_pos  # override position center
                else:
                    center_pve = (minmax_pve[1] + minmax_pve[0]) / 2
                scaled_ranges = ranges_pve.expand(2,3) * minmax_scaling
                lims = th.stack([center_pve - scaled_ranges[0]/2, center_pve + scaled_ranges[1]/2], dim=0)
                return lims
            control_limits_minmax_pve = {jn: scale_limit_centered(jn,
                                                        phys_limits_minmax_pve[jn],
                                                        control_limits_ratios_minmax_pve_th[jn],
                                                        control_limits_center.get(jn, None))
                                        for jn in phys_limits_minmax_pve.keys()}
        else:
            if control_limits_ratios_minmax_pve is not None:
                raise RuntimeError("Both control_limits_minmax_pve and control_limits_ratios_minmax_pve are provided")
            if control_limits_center is not None:
                raise RuntimeError("Both control_limits_minmax_pve and control_limits_center are provided, control_limits_center can only be used in combination with control_limits_ratios_minmax_pve")
            for jn in safe_limits_minmax_pve.keys():
                if jn not in control_limits_minmax_pve:
                    ggLog.warn(f"control_limits_minmax_pve does not contain joint {jn} that is present in safe_limits_minmax_pve, using the safe limit for that joint as control limit")
                    control_limits_minmax_pve[jn] = safe_limits_minmax_pve[jn]
                l = control_limits_minmax_pve[jn]
                dbg_check_size(l, (2,3), msg=f"control_limits_minmax_pve[{jn}] has shape {control_limits_minmax_pve[jn].shape}, should have shape (2,3) representing min/max for position, velocity and effort")
                control_limits_minmax_pve[jn] = self._thtens(l)
        control_limits_minmax_pve = {k:t.to(self._th_device) for k,t in control_limits_minmax_pve.items()}
        dbg_check_finite(control_limits_minmax_pve, assert_msg="control_limits_minmax_pve contains non-finite values")
        # Ensure control limits are within safe limits
        for jn in safe_limits_minmax_pve.keys():
            if jn not in control_limits_minmax_pve:
                control_limits_minmax_pve[jn] = safe_limits_minmax_pve[jn]
            if th.any(control_limits_minmax_pve[jn][0] < safe_limits_minmax_pve[jn][0]) or th.any(safe_limits_minmax_pve[jn][1] < control_limits_minmax_pve[jn][1]):
                ggLog.warn( f"Control limits exceeds safe limits for joint {jn},\n"
                            f" ctrl=\n{control_limits_minmax_pve[jn]}\n"
                            f" safe=\n{safe_limits_minmax_pve[jn]}\n"
                            f"The limit will be clamped.") 
                control_limits_minmax_pve[jn] = control_limits_minmax_pve[jn].clamp(min=safe_limits_minmax_pve[jn][0].expand(2,-1), 
                                                                                    max=safe_limits_minmax_pve[jn][1].expand(2,-1))

        # ----- DEFINE HOMING POSE ------------------------------------------
        ggLog.info(f"Input homing_joint_pose = {homing_joint_pose}")
        default_homing_joint_pose = {jn: unnormalize(0.0, safe_limits_minmax_pve[jn][0,0].item(), safe_limits_minmax_pve[jn][1,0].item())
                                     for jn in all_controlled_joints}
        ggLog.info(f"default homing_joint_pose = {default_homing_joint_pose}")
        
        for jn in homing_joint_pose:
            if jn not in all_controlled_joints:
                ggLog.warn(f"homing_joint_pose contains non-controlled joint {jn}")
        for jn in all_controlled_joints:
            if jn not in homing_joint_pose:
                homing_joint_pose[jn] = default_homing_joint_pose[jn]

        # If homing_references is provided, use it for position references in impedance commands (the p in pvesd).
        # This allows specifying references that differ from actual positions, e.g. to account for gravity.
        homing_ref = dict(homing_joint_pose)
        if homing_references is not None:
            for jn in homing_references:
                if jn not in homing_ref:
                    ggLog.warn(f"homing_references contains unknown joint {jn}")
            homing_ref.update(homing_references)

        ctrl_joints_stiffness_dict = expand_default_dict(ctrl_joints_stiffness, agent_controlled_joints_rn)
        ctrl_joints_damping_dict = expand_default_dict(ctrl_joints_damping, agent_controlled_joints_rn)
        held_joints_stiffness_dict = expand_default_dict(held_joints_stiffness, held_joints)
        held_joints_damping_dict = expand_default_dict(held_joints_damping, held_joints)
        homing_ctrl_joints_position = self._thtens([homing_joint_pose[jn] for jn in agent_controlled_joints_rn])
        homing_ctrl_joints_pvesd = self._thtens([(homing_ref[jn], 0, 0, ctrl_joints_stiffness_dict[jn], ctrl_joints_damping_dict[jn])
                                                    for jn in agent_controlled_joints_rn]).view(-1,5)
        homing_held_joints_pvesd = self._thtens([(homing_ref[jn], 0, 0, held_joints_stiffness_dict[jn], held_joints_damping_dict[jn])
                                                    for jn in held_joints]).view(-1,5)
        homing_held_joints_position = {jn:self._thtens(p) for jn,p in homing_joint_pose.items() if jn in held_joints}
        homing_nonctrl_joints_position = {jn:self._thtens(p) for jn,p in homing_joint_pose.items() if jn not in agent_controlled_joints_rn}
        
        # ----- RECAP -------------------------------------------------------

        def format_dict(d : dict):
            longest_key = max(len(str(k)) for k in d.keys())
            col = longest_key + 6
            s = " "*col
            return "\n".join([f"'{k}':".ljust(col)+
                              "\n".join([sv if i == 0 else s+sv for i,sv in enumerate(f"{v}".split("\n"))])
                             for k,v in d.items()])
        
        ggLog.info(f"homing_nonctrl_joints_position = {homing_nonctrl_joints_position}")
        ggLog.info(f"homing_ctrl_joints_pvesd = {homing_ctrl_joints_pvesd}")
        ggLog.info(f"phys_limits_minmax_pve = \n"+format_dict(phys_limits_minmax_pve))
        ggLog.info(f"safe_limits_minmax_pve = \n"+format_dict(safe_limits_minmax_pve))
        ggLog.info(f"control_limits_minmax_pve = \n"+format_dict(control_limits_minmax_pve))
        ggLog.info(f"agent_controlled_joints = \n"+pprint.pformat(agent_controlled_joints_rn)) 
        ggLog.info(f"held_joints = \n"+pprint.pformat(held_joints)) # Joints that are controlled by the env to be held at a fixed position
        ggLog.info(f"free_joints = \n"+pprint.pformat(free_joints_rn)) # Joints that are not controlled at all
        ggLog.info(f"all_controlled_joints = \n"+pprint.pformat(all_controlled_joints)) # All the joints that the env controls (env held + agent controleld)
        ggLog.info(f"homing_joint_pose = \n"+pprint.pformat(homing_joint_pose)) # Homing position, it can include agent controlled, held and free joints
        if homing_references is not None:
            ggLog.info(f"homing_references = \n"+pprint.pformat(homing_references)) # Position references for impedance commands, may differ from actual positions

        return (phys_limits_minmax_pve,
                safe_limits_minmax_pve,
                control_limits_minmax_pve,
                agent_controlled_joints_rn,
                held_joints,
                free_joints_rn,
                all_controlled_joints,
                homing_ctrl_joints_position,
                homing_ctrl_joints_pvesd,
                homing_held_joints_pvesd,
                homing_held_joints_position,
                homing_nonctrl_joints_position)

    def _find_links(self, links : Sequence[tuple[str,str]]) -> list[tuple[str,str]]:
        actual_links = []
        for l in links:
            if isinstance(l, tuple):
                actual_links.append(l)
            elif isinstance(l,Callable):
                for ln in self._robot_model.get_frame_names():
                    if l(ln,self._robot_model):
                        actual_links.append(ln)
            elif l in self.link_filters:
                for ln in self._adapter.get_detected_links():
                    if self.link_filters[l](ln,self._robot_model):
                        actual_links.append(ln)
            else:
                raise RuntimeError(f"Unexpected randomized link {l} of type {type(l)} (self.link_filters = {self.link_filters})")
        return actual_links
    
    def _find_joints(self, joints : Sequence[tuple[str,str] | JOINT_FILTERS | Callable]) -> list[tuple[str,str]]:
        robot_model = self._robot_model
        robot_name = self._configuration.init_args.robot_name
        joint_filters = self.joint_filters
        filtered_joints = []
        for j in joints:
            if isinstance(j, tuple):
                filtered_joints.append(j)
            elif isinstance(j,Callable):
                for jn in robot_model.get_joint_names():
                    if j(jn,robot_model):
                        filtered_joints.append((robot_name,jn))
            elif j in joint_filters:
                for jn in robot_model.get_joint_names():
                    if joint_filters[j](jn,robot_model):
                        filtered_joints.append((robot_name,jn))
            else:
                raise RuntimeError(f"Unexpected randomized joint {j} of type {type(j)} (self.joint_filters = {self.joint_filters})")
        return filtered_joints
    

    def _build_stats(self):
        self._stats = {}

    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        """Builds the state helper, which defines how the state is represented and observed."""

        vsize_dev_type = dict(dtype=th.float32, th_device=self._th_device, vec_size=adapter.vec_size())
        if self._configuration.init_args.observe_full_robot_state:
            observable_robot_state = ["pos","vel","cmdeff","refpos","refvel","refeff","stiff","damp"] 
        else:
            observable_robot_state = ["pos"]

        
        # :::::::::::::::::::::::::::::::::::::::: ROBOT JOINT STATE ::::::::::::::::::::::::::::::::::::::::

        # Controlled joints state:

        robot_state_helper = JointStateHelper(
                joint_limit_minmax_pveae={jn:self._configuration.joint_physical_limits_minmax_pve[jn] for jn in self._configuration.joints_agent_controlled},
                stiffness_minmax={jn:self._configuration.joint_safe_limits_minmax_stiffness[jn] for jn in self._configuration.joints_agent_controlled},
                damping_minmax={jn:self._configuration.joint_safe_limits_minmax_damping[jn] for jn in self._configuration.joints_agent_controlled},
                obs_dtype=self._configuration.obs_dtype,
                th_device=self._configuration.init_args.th_device,
                history_length=self._configuration.history_length,
                vec_size=adapter.vec_size(),
                observation_definitions={ 
                        "base":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=None,
                                obs_history_length = 1,
                                observable_subfields=observable_robot_state),
                        "privileged":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=None,
                                obs_history_length = self._configuration.init_args.frame_stack_length,
                                observable_subfields=observable_robot_state)})
        robot_state_noise =  StateNoiseGenerator(
                robot_state_helper,
                self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.init_args.th_device,
                episode_mu_std = self._configuration.noise_joints_pve_mustdstd[:2],
                step_std = self._configuration.noise_joints_pve_mustdstd[2])
        
        # Position Reference Errors:

        joints_range = robot_state_helper.get_limits()[1,:,0] - robot_state_helper.get_limits()[0,:,0]
        max_error = th.stack([-joints_range*2, joints_range*2], dim=0) 
        joint_pos_ref_error_state_helper = ThBoxStateHelper(
                field_names=[e for e in self.POS_REF_ERR_FIELDS],
                field_size=(self._action_helper.single_action_len(),),
                fields_minmax = {self.POS_REF_ERR_FIELDS.POS_ERR : max_error},
                history_length=5,
                flatten_observation=True,
                **vsize_dev_type, # type: ignore
                observation_definitions={
                        "privileged" : ThBoxStateHelper.SimpleObsDef.fully_observable(self._configuration.init_args.posref_err_history_length),
                        "base" : ThBoxStateHelper.SimpleObsDef.fully_observable(self._configuration.init_args.posref_err_history_length)
                })
        joint_pos_error_noise = StateNoiseGenerator(
                joint_pos_ref_error_state_helper,
                self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.init_args.th_device,
                episode_mu_std = self._configuration.noise_joints_pve_mustdstd[:2],
                step_std = self._configuration.noise_joints_pve_mustdstd[2])

        # Held joints state:

        held_joints_state_helper = JointStateHelper(
                joint_limit_minmax_pveae={jn:self._configuration.joint_physical_limits_minmax_pve[jn] for jn in self._configuration.joints_env_held},
                stiffness_minmax={jn:self._configuration.joint_safe_limits_minmax_stiffness[jn] for jn in self._configuration.joints_env_held},
                damping_minmax={jn:self._configuration.joint_safe_limits_minmax_damping[jn] for jn in self._configuration.joints_env_held},
                obs_dtype=self._configuration.obs_dtype,
                th_device=self._configuration.init_args.th_device,
                history_length=self._configuration.history_length,
                vec_size=adapter.vec_size(),
                observation_definitions={ 
                        "base":ThBoxStateHelper.SimpleObsDef(
                                    observable_fields=None,
                                    obs_history_length = 1,
                                    observable_subfields=["pos"])
                                if self._configuration.observe_held_joints else 
                                    ThBoxStateHelper.SimpleObsDef.not_observable(),
                        "privileged":ThBoxStateHelper.SimpleObsDef(
                                    observable_fields=None,
                                    obs_history_length = self._configuration.init_args.frame_stack_length,
                                    observable_subfields=["pos"])
                                if self._configuration.observe_held_joints else 
                                    ThBoxStateHelper.SimpleObsDef.not_observable()})
        
        # Joints step stats state:

        joint_step_stats_state_helper = RobotStatsStateHelper(  
                joint_limit_minmax_pve={jn:self._configuration.joint_physical_limits_minmax_pve[jn] for jn in self._configuration.joints_agent_controlled},
                **vsize_dev_type, # type: ignore
                include_senseff_and_power=True,
                flatten_observation=True,
                observation_definitions={
                        "privileged": ThBoxStateHelper.SimpleObsDef.not_observable(),
                        "base":       ThBoxStateHelper.SimpleObsDef.not_observable()}
                )
        
        # Joints long-term stats state:

        joint_longterm_stats_helper = ThBoxStateHelper( 
                field_names=[e for e in self.JOINT_LONGTERM_STATS_FIELDS],
                field_size=(len(self._configuration.joints_agent_controlled),),
                fields_minmax={self.JOINT_LONGTERM_STATS_FIELDS.AVG_POS : 
                                th.stack([self._configuration.joint_physical_limits_minmax_pve[jn][:,0]
                                            for jn in self._configuration.joints_agent_controlled],
                                        dim = 1)},
                **vsize_dev_type) # type: ignore
        
        # :::::::::::::::::::::::::::::::::::::::: ROBOT EXTRINSIC STATE ::::::::::::::::::::::::::::::::::::::::
        
        privileged_extrinsic_observable_fields = [
                                    self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X, self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y, self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z,
                                    self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_X, self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Y, self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z,
                                    self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X, self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y, self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z,
                                    self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X, self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y, self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z,
                                    self.EXTRINSIC_FIELDS.BODY_REL_LINACC_X, self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Y, self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Z,
                                    self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z
                                    ]
        
        if self._configuration.init_args.merge_privileged:
            base_extrinsic_observable_fields = privileged_extrinsic_observable_fields
        else:
            base_extrinsic_observable_fields = [
                                    self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X, self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y, self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z,
                                    self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X, self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y, self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z,
                                    ]
            if self._configuration.init_args.observe_linvel_nonprivileged:
                base_extrinsic_observable_fields += [self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X, self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y, self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z]
        if self._configuration.init_args.extrinsics_only_privileged:
            base_extrinsic_observable_fields = []
        extrinsic_state_helper =  ThBoxStateHelper(
                field_names=[e for e in self.EXTRINSIC_FIELDS],
                field_size=(1,),
                fields_minmax={ 
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : [-5,5],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_X : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Y : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z : [-20,20],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_X : [-1000,1000],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Y : [-1000,1000],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Z : [-1000,1000],
                        self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : [-1,1],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : [-1,1],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : [-1,1],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : [-1,1]},
                history_length=self._configuration.history_length,
                **vsize_dev_type, # type: ignore
                observation_definitions={  
                        "base":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=base_extrinsic_observable_fields,
                                obs_history_length = 1,
                                observable_subfields=None),
                        "privileged":ThBoxStateHelper.SimpleObsDef(
                                observable_fields=privileged_extrinsic_observable_fields,
                                obs_history_length = self._configuration.init_args.frame_stack_length,
                                observable_subfields=None)}
                )
        extrinsic_state_noise =  StateNoiseGenerator(
                extrinsic_state_helper,
                self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.init_args.th_device,
                episode_mu_std = {
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : self._configuration.noise_linvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_X : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Y : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z : self._configuration.noise_angvel_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_X : self._configuration.noise_linacc_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Y : self._configuration.noise_linacc_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Z : self._configuration.noise_linacc_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z    : self._configuration.noise_posz_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : self._configuration.noise_gravity_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : self._configuration.noise_gravity_ep_mustdstd[:2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : self._configuration.noise_gravity_ep_mustdstd[:2]},
                step_std = {
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : self._configuration.noise_linvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_X : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Y : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z : self._configuration.noise_angvel_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_X : self._configuration.noise_linacc_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Y : self._configuration.noise_linacc_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Z : self._configuration.noise_linacc_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z    : self._configuration.noise_posz_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : self._configuration.noise_gravity_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : self._configuration.noise_gravity_ep_mustdstd[2],
                        self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : self._configuration.noise_gravity_ep_mustdstd[2]})
        
        # :::::::::::::::::::::::::::::::::::::::: INTERNAL STATE ::::::::::::::::::::::::::::::::::::::::
        
        internal_obsdef = ThBoxStateHelper.SimpleObsDef(observable_fields=[
                                                            self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED,
                                                            self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED
                                                                            ]
                                                            if self._configuration.init_args.observe_actor_safety_state else
                                                            [],
                                                        observable_subfields=None,
                                                        obs_history_length=1)
        internal_state_helper =   ThBoxStateHelper( 
                field_names=[e for e in self.INTERNAL_FIELDS],
                field_size=(1,),
                fields_minmax={ 
                        self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED : [0,1],
                        self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED : [0,1],
                        self.INTERNAL_FIELDS.SAFETY_POSREF_VIOLATION_COUNT : [0,1000_000],
                        self.INTERNAL_FIELDS.STEP_COUNT : [-1,100_000],
                        self.INTERNAL_FIELDS.SIM_TIME : [-1,1000_000],
                        self.INTERNAL_FIELDS.LAST_STEP_DT : [-1,1]},
                **vsize_dev_type,  # type: ignore
                observation_definitions={
                        "privileged" : internal_obsdef,
                        "base" : internal_obsdef})
        
        # :::::::::::::::::::::::::::::::::::::::: ACTION HISTORY STATE ::::::::::::::::::::::::::::::::::::::::
        
        # Smoothed actions state:

        acthistory_obsdef = ThBoxStateHelper.SimpleObsDef(
                observable_fields=None,
                observable_subfields=None,
                obs_history_length=self._configuration.init_args.history_length_action_smoothed
                )
        act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                    field_size=(self._action_helper.single_action_len(),),
                                                    fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                    history_length=5,
                                                    flatten_observation=True,
                                                    observation_definitions={
                                                        "privileged" :acthistory_obsdef,
                                                        "base" : acthistory_obsdef},
                                                    **vsize_dev_type) # type: ignore
        
        # Raw actions state:

        rawactihostory_obsdef = ThBoxStateHelper.SimpleObsDef(
                observable_fields=None,
                observable_subfields=None,
                obs_history_length=self._configuration.init_args.history_length_action_raw)
        raw_act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                        field_size=(self._action_helper.single_action_len(),),
                                                        fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                        history_length=5,
                                                        flatten_observation=True,
                                                        observation_definitions={
                                                            "privileged" :rawactihostory_obsdef,
                                                            "base" : rawactihostory_obsdef
                                                        },
                                                        **vsize_dev_type) # type: ignore
        
        # Last action state

        lastrawact_obsdef = ThBoxStateHelper.SimpleObsDef(  observable_fields=None,
                                                            observable_subfields=None,
                                                            obs_history_length=1)
        last_raw_act_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                        field_size=(self._action_helper.single_action_len(),),
                                                        fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                        history_length=5,
                                                        flatten_observation=True,
                                                        observation_definitions={
                                                            "privileged" :lastrawact_obsdef,
                                                            "base" : lastrawact_obsdef
                                                        },
                                                        **vsize_dev_type) # type: ignore
        

        # :::::::::::::::::::::::::::::::::::::::: STATE AGGREGATION ::::::::::::::::::::::::::::::::::::::::

        if not self._configuration.init_args.merge_privileged:
            obs_definitions={
                    "base" : DictStateHelper.SimpleDictObsDef(
                            observable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_POS_REF_ERR,
                                    self.STATE_ACT_PREPROC,
                                    self.STATE_HELD_JOINTS,
                                    ], # this will only take the "base" obs inside these
                            concatenable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_POS_REF_ERR,
                                    self.STATE_ACT_PREPROC,
                                    self.STATE_HELD_JOINTS,
                                    # self.STATE_LAST_ACT_RAW,
                                    # self.STATE_JOINT_LONGTERM_STATS,
                                    # self.STATE_JOINT_STEP_STATS
                                    ],
                            concatenated_part_name="vec",
                            noise_generators={
                                    self.STATE_ROBOT       : robot_state_noise,
                                    self.STATE_EXTRINSIC   : extrinsic_state_noise,
                                    self.STATE_POS_REF_ERR : joint_pos_error_noise
                                    }),
                    "privileged" : DictStateHelper.SimpleDictObsDef(
                            observable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_ACT_PREPROC,
                                    self.STATE_HELD_JOINTS,
                                    #  self.STATE_JOINT_STEP_STATS,
                                    ],
                            concatenable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_INTERNAL,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_ACT_PREPROC,
                                    self.STATE_HELD_JOINTS
                                    # self.STATE_JOINT_STEP_STATS,
                                    ],
                            concatenated_part_name="vec",
                            noise_generators={})}
        else:
            obs_definitions={
                    "base" : DictStateHelper.SimpleDictObsDef(
                            observable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_HELD_JOINTS
                                    # self.STATE_LAST_ACT_RAW,
                                    # self.STATE_ACT_PREPROC,
                                    # self.STATE_JOINT_LONGTERM_STATS,
                                    # self.STATE_JOINT_STEP_STATS
                                    ],
                            concatenable_substates=[
                                    self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_ACT_RAW_HIST,
                                    self.STATE_HELD_JOINTS,
                                    # self.STATE_LAST_ACT_RAW,
                                    # self.STATE_ACT_PREPROC,
                                    # self.STATE_JOINT_LONGTERM_STATS,
                                    # self.STATE_JOINT_STEP_STATS,
                                    ],
                            concatenated_part_name="vec",
                            noise_generators={  
                                    self.STATE_ROBOT     : robot_state_noise,
                                    self.STATE_EXTRINSIC : extrinsic_state_noise
                                    })}
        self._state_helper = DictStateHelper({  
                self.STATE_EXTRINSIC : extrinsic_state_helper,
                self.STATE_ROBOT : robot_state_helper,
                self.STATE_POS_REF_ERR : joint_pos_ref_error_state_helper,
                self.STATE_LAST_ACT_RAW : last_raw_act_state_helper,
                self.STATE_JOINT_STEP_STATS : joint_step_stats_state_helper,
                self.STATE_INTERNAL : internal_state_helper,
                self.STATE_ACT_PREPROC: act_history_state_helper,
                self.STATE_ACT_RAW_HIST : raw_act_history_state_helper,
                self.STATE_JOINT_LONGTERM_STATS : joint_longterm_stats_helper,
                self.STATE_HELD_JOINTS : held_joints_state_helper},
                obs_definitions=obs_definitions)

    
    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    def _preproc_acts(self, actions : th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # @th_compile_ext(just_graphit=True)
        # def _preproc(actions : th.Tensor):
        dt = self._configuration.init_args.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        prev_actions = self._current_state[self.STATE_ACT_PREPROC][:,0,self.ACT_FIELDS.ACTION].detach().to(device=self._configuration.init_args.th_device)
        actions = actions*(1-alpha) + prev_actions*alpha
        # actions = th.clamp(actions, min=-1, max=1)
        n = self._thrandn(size=(self._adapter.vec_size(),))
        action_delay = th.clamp(self._current_episode_config.action_delay_mu + self._configuration.action_delay_epmustd_ststd[2]*n, min = 0.0)
        return actions, action_delay
        # return _preproc(actions)

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        with th.no_grad():
            # actions = self._thtens(actions).detach()
            actions = actions.to(device=self._configuration.init_args.th_device)
            self._last_raw_actions = actions
            dbg_check_finite(actions, async_assert=True, assert_msg="Actions contains non-finite values")
            actions = th.clamp(actions, min=-1, max=1)
            # dbg_check_size(actions, (self._adapter.vec_size(), self._action_helper.single_action_len()))
            actions, action_delay = self._preproc_acts(actions)
            self._last_preprocessed_actions = actions
            actions = th.clamp(actions + self._thrandn(size=actions.shape)*self._configuration.noise_action_mustd[1], min = -1, max = 1)
            v_j_pvesd = self._action_helper.action_to_pvesd(actions, self._last_sent_v_j_pvesd[:,:,0])
            # do this better, avoid this if condition, put it in the helper
            if self._configuration.init_args.saturate_jimp_posref_limits:
                v_j_pvesd[:,:,:3] = th.clamp(v_j_pvesd[:,:,:3], min=self._safe_limits_minmax_j_pve[0], max=self._safe_limits_minmax_j_pve[1])
                posref_diff = v_j_pvesd[:,:,0] - self._last_sent_v_j_pvesd[:,:,0]
                posref_diff = th.clamp(posref_diff, min=self._posref_saturation_minmmax_diff[0], max=self._posref_saturation_minmmax_diff[1])
                v_j_pvesd[:,:,0] = self._last_sent_v_j_pvesd[:,:,0] + posref_diff
            # if self._configuration.control_mode in [JointImpedanceActionHelper.CONTROL_MODES.POSITION, JointImpedanceActionHelper.CONTROL_MODES.PS, JointImpedanceActionHelper.CONTROL_MODES.PT] :
            #     if self._configuration.velref_from_posref:
            #         refvel = (v_j_pvesd[:,:,0] - self._last_sent_v_j_pvesd[:,:,0])/self._intendedStepLength_sec
            #         # current_pos = self._current_state[self.STATE_ROBOT][:,0,:,0]
            #         # refvel = (v_j_pvesd[:,:,0] - current_pos)/self._intendedStepLength_sec
            #         # refvel = th.zeros_like(refvel)
            #         v_j_pvesd[:,:,1] = th.clamp(refvel,
            #                                     min=self._safe_limits_minmax_j_pve[0,:,1], 
            #                                     max=self._safe_limits_minmax_j_pve[1,:,1]) # set velocity reference

            v_j_pvesd[:,:,3]*=self._current_episode_config.randomized_stiffness_factor
            v_j_pvesd[:,:,4]*=self._current_episode_config.randomized_damping_factor
            self._last_sent_v_j_pvesd = v_j_pvesd
            full_cmd_vec_j_pvesd = th.concat([v_j_pvesd, self._homing_held_joints_vec_pvesd], dim = 1)
            self._submit_time = time.monotonic()
            self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = full_cmd_vec_j_pvesd,
                                                    delay_sec=action_delay)
            self._obs2act_timings.mark_end()
            # ggLog.info(f"obs-act delay: \n{pprint.pformat(self._obs2act_timings.get_stats())}")
    





    





    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------

    def _get_spawn_defs(self):
        if not hasattr(self, "_spawn_defs"):
            robot_pose = None #build_pose(*self._configuration.spawn_root_pose_xyz_xyzw)
            arrow_pose = robot_pose
            camera_pose = None #build_pose(0,0.0,0.0,    0.0, 0.0, 0.0,   1.0)
            robot_spawn_def = ModelSpawnDef(definition_string=self._configuration.init_args.robot_description_string,
                                            name=self._configuration.init_args.robot_name,
                                            pose=robot_pose,
                                            format=self._configuration.init_args.robot_description_format,
                                            kwargs={})
            if adarl.utils.utils.isinstance_noimport(self._adapter, ("MjxAdapter", "MujocoAdapter", "GenesisAdapter")):
                cam_file = "models/simple_camera.mjcf.xacro"
                cam_format = "mjcf.xacro"
            else:            
                cam_file = "models/simple_camera.sdf.xacro"
                cam_format = "sdf.xacro"
            is_pybullet = False
            is_ros = False
            if isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
                subadapters = self._adapter.sub_adapters()
                if len(subadapters) != 1:
                    raise RuntimeError(f"Expected exactly one subadapter in VecSimJointImpedanceAdapterWrapper, got {len(subadapters)}")
                subadapter = subadapters[0]
                if adarl.utils.utils.isinstance_noimport(subadapter, ("PyBulletJointImpedanceAdapter")):
                    is_pybullet= True
                elif adarl.utils.utils.isinstance_noimport(subadapter, ("RosXbotAdapter", "RosXbotGazeboAdapter")):
                    is_ros = True
            camera_spawn_def = ModelSpawnDef(   definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                                name="simple_camera",
                                                pose=camera_pose,
                                                format=cam_format,
                                                kwargs={"camera_width":self._configuration.init_args.ui_camera_resolution_hw[1],
                                                        "camera_height":self._configuration.init_args.ui_camera_resolution_hw[0],
                                                        "frame_rate":1/self._intendedStepLength_sec})
            arrow_spawn_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/red_arrow.urdf.xacro")).read_text(),
                                            name="arrow",
                                            pose=arrow_pose,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
            arrow_yellow_spawn_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/red_arrow.urdf.xacro")).read_text(),
                                            name="arrow_yellow",
                                            pose=arrow_pose,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet),
                                                    "color_rgba":"1 1 0 1"})
            axes_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/axes.urdf.xacro")).read_text(),
                                            name="axes",
                                            pose=None,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
            self._spawn_defs = [robot_spawn_def,
                                    camera_spawn_def]
            if self._configuration.show_goal:
                self._spawn_defs.append(arrow_spawn_def)
                self._spawn_defs.append(arrow_yellow_spawn_def)
            self._spawn_defs.append(axes_spawn_def)
        return self._spawn_defs


    def _reset_state_full(self):
        resetted_state = self._state_helper.reset_state()
        if not hasattr(self, "_current_state") or self._current_state is None:
            self._current_state = resetted_state
        self._current_state = resetted_state
        self._current_state[self.STATE_INTERNAL][:,:,self.INTERNAL_FIELDS.STEP_COUNT] = -1
        self._current_state[self.STATE_INTERNAL][:,:,self.INTERNAL_FIELDS.LAST_STEP_DT] = self._configuration.init_args.stepLength_sec
        self._current_state[self.STATE_EXTRINSIC][:,:,self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z] = -1 # Should always have norm 1


    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        # ggLog.info(f"_initialize_episodes({vec_mask})")
        record_time("RobotVecEnv.initialize_episodes")
        if vec_mask is None:
            vec_mask = self._all_envs

        if not isinstance_noimport(self._adapter, ("BaseVecSimulationAdapter")):
            gc.enable() # enable garbage collection
            gc.unfreeze() # unfreezes whatevwer is already frozen
            gc.collect(2) # collects whatever nedds to be collected
            gc.freeze() # freeze currently allocated objects, so it is ignored in future collections, to make the fast
            gc.disable() # disable automatic garbage collection


        self._set_current_ep_config(reset_options = options, vec_mask=vec_mask)
        record_time("RobotVecEnv.initialize_episodes: setted episode config")
        
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._simulation_initialization(vec_mask=vec_mask)
        else:
            self._realworld_initialization(vec_mask=vec_mask)
        record_time("RobotVecEnv.initialize_episodes: initialized")
            
        last_sent_actions = th.clamp(self._action_helper.pvesd_to_action(self._last_sent_v_j_pvesd, self._last_sent_v_j_pvesd[:,:,0]), min=-1, max=1)
        masked_assign(self._last_preprocessed_actions,  vec_mask, last_sent_actions)
        masked_assign(self._last_raw_actions,           vec_mask, last_sent_actions)
        masked_assign(self._eps_start_stime,            vec_mask, self._adapter.getEnvTimeFromStartup())

        record_time("RobotVecEnv.initialize_episodes: setted things")
        adapter_data = self._get_adapter_data()
        record_time("RobotVecEnv.initialize_episodes: got adapter data")
        self._reinit_state(vec_mask=vec_mask, adapter_data=adapter_data)
        record_time("RobotVecEnv.initialize_episodes: reinitialized state")

        self._update_stats()
        record_time("RobotVecEnv.initialize_episodes: updated stats")

        self._last_obs = self._state_helper.observe(self._current_state)
        record_time("RobotVecEnv.initialize_episodes: observed")



    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        if self._configuration.init_args.offset_envs_ep_starts and self._init_counter_since_reset == 1 and self._vstep_counter_since_reset == 0:
            # randomize max episode steps for the first episode to decorrelate initial randomizations from episode length, which can be important for some learning algorithms, e.g. RNN training with truncated backpropagation through time, to learn better from the initial randomizations
            # We env just got resetted
            max_steps = th.randint(1,
                                self._configuration.init_args.maxStepsPerEpisode+1,
                                size=(self.num_envs,),
                                generator=self._rng,
                                device=self._th_device,
                                dtype=th.int64)
            maxStepsPerEpisode = reset_options.get("max_ep_steps", max_steps)
        else:
            maxStepsPerEpisode = reset_options.get("max_ep_steps", self._configuration.init_args.maxStepsPerEpisode)
            # ggLog.info(f"Randomizing maxStepsPerEpisode = {maxStepsPerEpisode}, vec_mask = {vec_mask}")
        
        homing_pos = self._configuration.homing_ctrl_joints_position
        t0 = time.monotonic()
        recomputed_pose_randomization = False
        if self._initial_pose_randomization_enabled:
            never_sampled_poses_yet = self._last_pose_randomization is None
            not_recycling = not self._configuration.init_args.randomization_recycle_init_pose
            at_least_one_episode_passed = self._configuration.init_args.maxStepsPerEpisode == 0 # approximately, just to avoid doing it too often
            if (not_recycling and at_least_one_episode_passed) or never_sampled_poses_yet:
                jp_dict = {k:v for k,v in self._configuration.homing_nonctrl_joints_position.items()}
                jp_dict.update({k:v for k,v in self._configuration.homing_held_joints_position.items()})
                recomputed_pose_randomization = True
                ggLog.info(f"Randomizing initial joint poses for {vec_mask.sum().item()} envs...")
                _rand_poses = find_poses(   root_joint = self._configuration.robot_root_joint,
                                            homing_body_pose_xyzxyzw = self._pinocchio_corrected_homing_body_pose_xyzxyzw,
                                            controlled_joints = self._configuration.joints_agent_controlled,
                                            initial_pose_randomization_range = self._configuration.init_args.randomization_initial_joint_pose_range,
                                            initial_height_randomization_range = self._configuration.init_args.randomization_initial_height_range_meters,
                                            limits_minmax = th.stack([self._configuration.joint_safe_limits_minmax_pve[jn][:,0] for jn in self._configuration.joints_agent_controlled], dim = 1).cpu().numpy(),
                                            homing_pos = homing_pos.cpu().numpy(),
                                            noncontrolled_jointpos = {k:v.cpu().numpy() for k,v in jp_dict.items()},
                                            robot_model = self._robot_model,
                                            is_floating_base = self._configuration.robot_is_floating,
                                            seed = th.randint(0, 2**31-1, (1,), generator=self._rng, device=self._rng.device).item(), # type: ignore
                                            excluded_collision_pairs = self._excluded_collision_pairs,
                                            num_envs=self.num_envs).to(device=self._configuration.init_args.th_device, non_blocking=True)
                ggLog.info(f"Randomized initial joint poses for {vec_mask.sum().item()} envs")
                ggLog.info(f"Randomized initial joint poses = {_rand_poses}")
                # For randomized poses, set references equal to positions (for now)
                self._last_pose_randomization = th.stack([_rand_poses, _rand_poses], dim=2)
        else:
            homing_ref = self._configuration.homing_ctrl_joints_pvesd[:,0]
            self._last_pose_randomization = th.stack([homing_pos, homing_ref], dim=-1).unsqueeze(0).expand(self.num_envs, -1, -1)
        initial_ctrl_jposes = self._last_pose_randomization
        rand_time = time.monotonic()-t0
        if rand_time > 1.0:
            ggLog.info(f"pose randomization took {rand_time:.6f}s")
        
        if self._tot_init_counter <= 1 or recomputed_pose_randomization:
            self._precompute_init_tensors(initial_ctrl_jposes)
        if  self._configuration.init_args.init_on_reset_ratio<1.0 and self._init_counter_since_reset>1:
            vec_init_on_reset = self._thrand((self.num_envs,)) < self._configuration.init_args.init_on_reset_ratio
        else:
            vec_init_on_reset = th.ones((self.num_envs,), dtype=th.bool).to(device=self._th_device, non_blocking=self._th_device.type=="cuda")
        # ggLog.info(f"initial_jpose = {initial_joint_pose}, homing = {homing}")
        masked_assign(self._current_episode_config.vec_initial_ctrl_joint_pose, vec_mask, initial_ctrl_jposes)
        masked_assign(self._current_episode_config.vec_init_on_reset,           vec_mask, vec_init_on_reset)
        masked_assign(self._current_episode_config.vec_max_ep_steps,            vec_mask, maxStepsPerEpisode)
        ctrl_joints_num = len(self._configuration.joints_agent_controlled)
        
        damping_ratios =    self._thrandn_truncnorm((self.num_envs,ctrl_joints_num),0,1,-3,+3)*self._configuration.init_args.randomized_gains_damping_ratio_epstd+1
        stiffness_ratios =  self._thrandn_truncnorm((self.num_envs,ctrl_joints_num),0,1,-3,+3)*self._configuration.init_args.randomized_gains_stiffness_ratio_epstd+1
        masked_assign(self._current_episode_config.randomized_damping_factor,   vec_mask, damping_ratios)
        masked_assign(self._current_episode_config.randomized_stiffness_factor, vec_mask, stiffness_ratios)
        # ggLog.info(f"_current_episode_config = {self._current_episode_config}")
        if self._model_randomization_enabled:
            # ggLog.info(f"self._mass_randomized_link_ids = {self._mass_randomized_link_ids}")
            link_masses_ratios =            self._sample_distr((self.num_envs, len(self._configuration.randomized_mass_links)), self._configuration.randomized_mass_ratio_distribution)
            link_coms_diffs =               self._sample_distr((self.num_envs, len(self._configuration.randomized_com_links),3), self._configuration.randomized_com_xyz_diff_distribution)
            joint_armatures_ratios =        self._sample_distr((self.num_envs, len(self._configuration.randomized_dof_armature_joints)), self._configuration.randomized_dof_armature_ratios)
            joint_dampings_ratios =         self._sample_distr((self.num_envs, len(self._configuration.randomized_dof_damping_joints)), self._configuration.randomized_dof_damping_ratios)
            joint_frictionlosses_ratios =   self._sample_distr((self.num_envs, len(self._configuration.randomized_dof_frictionloss_joints)), self._configuration.randomized_dof_frictionloss_ratios)
            link_frictions_ratios =         self._sample_distr((self.num_envs, len(self._configuration.randomized_friction_links), 3), self._configuration.randomized_friction_slide_spin_roll_ratios)
            masked_assign(self._current_episode_config.link_masses_ratios,          vec_mask, link_masses_ratios)
            masked_assign(self._current_episode_config.link_frictions_ratios,       vec_mask, link_frictions_ratios)
            masked_assign(self._current_episode_config.joint_armatures_ratios,      vec_mask, joint_armatures_ratios)
            masked_assign(self._current_episode_config.joint_dampings_ratios,       vec_mask, joint_dampings_ratios)
            masked_assign(self._current_episode_config.joint_frictionlosses_ratios, vec_mask, joint_frictionlosses_ratios)
            masked_assign(self._current_episode_config.link_coms_diffs,             vec_mask, link_coms_diffs)
        new_filters_freqs = self._sample_distr((self.num_envs,), self._configuration.randomized_reference_filter_distribution)
        masked_assign(self._current_episode_config.joint_reference_filter_freqs, vec_mask, new_filters_freqs)
        
        delay_mu, delay_std = self._configuration.action_delay_epmustd_ststd[:2]
        action_delay_mu = th.clamp(self._thrandn_clamp(size=(self.num_envs,), min=-5, max=5)*delay_std+delay_mu, min=0)
        masked_assign(self._current_episode_config.action_delay_mu, vec_mask, action_delay_mu)

        self.set_max_episode_steps(self._current_episode_config.vec_max_ep_steps)
        

    def _realworld_robot_init_move(self, vec_mask : th.Tensor):
        if isinstance_noimport(self._adapter,["VecRosXBotAdapterWrapper","VecZmqXbotAdapter"]):
            vjposref = self._current_episode_config.vec_initial_ctrl_joint_pose[:,:,1]  # references
            initial_cmd_vec_j_pvesd = th.stack([vjposref,
                                        th.zeros_like(vjposref),
                                        th.zeros_like(vjposref),
                                        self._configuration.homing_ctrl_joints_pvesd[:,3].to(dtype=vjposref.dtype, device=vjposref.device).expand_as(vjposref),
                                        self._configuration.homing_ctrl_joints_pvesd[:,4].to(dtype=vjposref.dtype, device=vjposref.device).expand_as(vjposref)], dim = 2)
            # initial_state_pve = th.zeros(size=(self.num_envs, len(self._configuration.controlled_joints), 3))
            not_resetting_sims = th.logical_not(self._current_episode_config.vec_init_on_reset)
            # if th.any(not_resetting_sims):
            # ggLog.info(f"initial_cmd_vec_j_pvesd.device = {initial_cmd_vec_j_pvesd.device}, self._last_sent_v_j_pvesd.deive = {self._last_sent_v_j_pvesd.device} not_resetting_sims.device={not_resetting_sims.device}")
            masked_assign(initial_cmd_vec_j_pvesd, not_resetting_sims, self._last_sent_v_j_pvesd)
            full_cmd_vec_j_pvesd = th.concat([initial_cmd_vec_j_pvesd, self._homing_held_joints_vec_pvesd], dim = 1)
            ggLog.info(f"Moving robot...")
            try:
                self._adapter.moveToJointPoseSync(  joint_names = self._configuration.joints_all_env_controlled,
                                                    positions = full_cmd_vec_j_pvesd[:,:,0],
                                                    velocity_scaling = 0.1,
                                                    acceleration_scaling = 0.1,
                                                    joint_position_tolerance = 0.01,
                                                    max_time_s = 60)
                ggLog.info(f"Moved robot.")
            except adarl.utils.utils.MoveFailError as e:
                ggLog.warn(f"Timed out reaching position: {adarl.utils.utils.exc_to_str(e)}")
            time.sleep(1)
            self._adapter.setJointsImpedanceCommand(full_cmd_vec_j_pvesd, vec_mask=vec_mask)
            time.sleep(1)
            self._adapter.set_current_joint_impedance_command(full_cmd_vec_j_pvesd, vec_mask=vec_mask)
            masked_assign(self._last_sent_v_j_pvesd, vec_mask, initial_cmd_vec_j_pvesd)
        else:
            raise NotImplementedError(f"Unsupported real-world initialization for adapter of type '{type(self._adapter)}'")

    def _realworld_initialization(self, vec_mask : th.Tensor):
        while True:
            self._adapter.set_reference_filter(self._thtens(20.0).expand(self._adapter.vec_size(),), vec_mask=vec_mask)
            print(f"Episode Initialization:\n"
                  f"Will move the robot joints into the homing pose and set the initial joint impedance command.")
            if unsafe_realworld_init:
                r = "move"
            else:
                r = input("Enter 'move' to move the robot or 'skip' to skip the robot pose initialization > ")
            if r == "move":
                self._realworld_robot_init_move(vec_mask)
            elif r == "skip":
                masked_assign(self._last_sent_v_j_pvesd,
                              vec_mask,
                              self._adapter.get_current_joint_impedance_command()[:,:len(self._configuration.joints_agent_controlled)])
                pass
            else:
                print(f"Invalid answer '{r}'")
                continue
            
            if unsafe_realworld_init:
                r = "start"
            else:
                r = input("Please ensure the robot is in a suitable pose and type 'start' to start episode > ")
            if r == "start":
                self._adapter.set_reference_filter(self._current_episode_config.joint_reference_filter_freqs)
                return
        raise NotImplementedError()
    
    def _precompute_init_tensors(self, vec_initial_ctrl_joint_posposref):
        """vec_initial_ctrl_joint_posref: (envs, joints, 2) where [:,:,0]=positions, [:,:,1]=references"""
        vec_initial_ctrl_joint_pose = vec_initial_ctrl_joint_posposref[:,:,0]
        initial_ctrl_posref = vec_initial_ctrl_joint_posposref[:,:,1]
        initial_cmd_vec_j_pvesd = th.stack([initial_ctrl_posref,
                                    th.zeros_like(initial_ctrl_posref),
                                    th.zeros_like(initial_ctrl_posref),
                                    self._configuration.homing_ctrl_joints_pvesd[:,3].to(dtype=initial_ctrl_posref.dtype, device=initial_ctrl_posref.device).expand_as(initial_ctrl_posref),
                                    self._configuration.homing_ctrl_joints_pvesd[:,4].to(dtype=initial_ctrl_posref.dtype, device=initial_ctrl_posref.device).expand_as(initial_ctrl_posref)], dim = 2)
        full_cmd_vec_j_pvesd = th.concat([initial_cmd_vec_j_pvesd, self._homing_held_joints_vec_pvesd], dim = 1)

        # Sim state uses actual positions, not references
        num_ctrl = len(self._configuration.joints_agent_controlled)
        ctrl_joint_states = full_cmd_vec_j_pvesd[:,:,:3].clone()
        ctrl_joint_states[:,:num_ctrl,0] = vec_initial_ctrl_joint_pose
        # Held joints: use actual positions from homing_held_joints_position
        for i, jn in enumerate(self._configuration.joints_env_held):
            if jn in self._configuration.homing_held_joints_position:
                ctrl_joint_states[:,num_ctrl+i,0] = self._configuration.homing_held_joints_position[jn]


        
        non_ctrl_homing_pos = {k:v for k,v in self._configuration.homing_nonctrl_joints_position.items() if k not in self._configuration.joints_all_env_controlled}
        nonctrl_joints_states = th.zeros(size=(self._adapter.vec_size(),len(non_ctrl_homing_pos),3),
                                         device=vec_initial_ctrl_joint_pose.device,
                                         dtype=vec_initial_ctrl_joint_pose.dtype)
        nonctrl_joints_states[:,:,0] = self._thtens(list(non_ctrl_homing_pos.values()))
        
        all_joints_states = th.cat([ctrl_joint_states, nonctrl_joints_states], dim=1)
        all_joints_names = list(self._configuration.joints_agent_controlled)+list(self._configuration.joints_env_held)+list(non_ctrl_homing_pos.keys())
        self._precomputed_sim_init = self.PrecomputedSimInitData(
            all_joints_names=all_joints_names,
            all_joints_states=all_joints_states,
            initial_cmd_vec_j_pvesd=initial_cmd_vec_j_pvesd,
            full_cmd_vec_j_pvesd=full_cmd_vec_j_pvesd
        )
        ggLog.info(f"Precomputed sim init data: {self._precomputed_sim_init}")

    def _simulation_initialization(self, vec_mask : th.Tensor):
        if not isinstance(self._adapter, BaseVecSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        
        record_region_start("RobotVecEnv._simulation_initialization")
        reinit_vecs = th.logical_and(self._current_episode_config.vec_init_on_reset, vec_mask)
        use_mjx_command_sequence = isinstance_noimport(self._adapter, "MjxJointImpedanceAdapter")
                
        if use_mjx_command_sequence:
            from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter, SetCurrentJointImpedanceCommand
            from adarl.adapters.MjxAdapter import SetJointsStateDirectCommand, SetLinksStateDirectCommand, AlterModelCommand
            mjx_adapter : MjxAdapter = self._adapter # type: ignore
            do_main_link_homing = self._configuration.homing_body_pose_xyz_xyzw is not None and self._configuration.robot_is_floating
            command_sequence = [
                SetLinksStateDirectCommand(
                        link_names=[self._configuration.main_body_link],
                        link_states_pose_vel=th.cat([
                            self._configuration.homing_body_pose_xyz_xyzw,
                            th.zeros((6,), device=self._configuration.init_args.th_device, dtype=th.float32),
                        ]).expand(self._adapter.vec_size(), 1, 13),
                        vec_mask=reinit_vecs,
                    ) if do_main_link_homing else None,
                SetCurrentJointImpedanceCommand(
                    joint_impedances_pvesd=self._precomputed_sim_init.full_cmd_vec_j_pvesd,
                    vec_mask=reinit_vecs,
                ),
                SetJointsStateDirectCommand(
                    joint_names=self._precomputed_sim_init.all_joints_names,
                    joint_states_pve=self._precomputed_sim_init.all_joints_states,
                    vec_mask=reinit_vecs,
                )
            ]
            if self._model_randomization_enabled and not (self._tot_init_counter > 1 and self._configuration.init_args.randomization_recycle_model_alterations):
                command_sequence.append(AlterModelCommand(
                        link_masses=(self._randomized_mass_link_ids, self._current_episode_config.link_masses_ratios) if len(self._randomized_mass_link_ids) > 0 else None,
                        link_frictions=(self._randomized_friction_links_ids, self._current_episode_config.link_frictions_ratios) if len(self._randomized_friction_links_ids) > 0 else None,
                        joint_armature_ratios=(self._randomized_dof_armature_joints_ids, self._current_episode_config.joint_armatures_ratios) if len(self._randomized_dof_armature_joints_ids) > 0 else None,
                        joint_damping_ratios=(self._randomized_dof_damping_joints_ids, self._current_episode_config.joint_dampings_ratios) if len(self._randomized_dof_damping_joints_ids) > 0 else None,
                        joint_frictionloss_ratios=(self._randomized_dof_frictionloss_joints_ids, self._current_episode_config.joint_frictionlosses_ratios) if len(self._randomized_dof_frictionloss_joints_ids) > 0 else None,
                        com_position_diffs=(self._randomized_com_links_ids, self._current_episode_config.link_coms_diffs) if len(self._randomized_com_links_ids) > 0 else None,
                        com_quatxyzw_diffs=None,
                        vec_mask=reinit_vecs,
                ))
            mjx_adapter.run_command_sequence(command_sequence)
        else:
            if self._configuration.homing_body_pose_xyz_xyzw is not None and self._configuration.robot_is_floating:
                # ggLog.info(f"setting body pose ({self._current_episode_config.vec_init_on_reset})")
                self._adapter.setLinksStateDirect(link_names=[self._configuration.main_body_link],
                                                  link_states_pose_vel=th.cat([self._configuration.homing_body_pose_xyz_xyzw,
                                                                               th.zeros((6,), device=self._configuration.init_args.th_device, dtype=th.float32)])
                                                                               .expand(self._adapter.vec_size(), 1, 13),
                                                  vec_mask=reinit_vecs)
            record_time(f"RobotVecEnv._simulation_initialization: setted main body pose")
            self._adapter.set_current_joint_impedance_command(self._precomputed_sim_init.full_cmd_vec_j_pvesd, vec_mask=reinit_vecs)
            record_time(f"RobotVecEnv._simulation_initialization: setted current joint impedance command")
            self._adapter.setJointsStateDirect(joint_names=self._precomputed_sim_init.all_joints_names,
                                               joint_states_pve=self._precomputed_sim_init.all_joints_states,
                                               vec_mask=reinit_vecs)
            record_time(f"RobotVecEnv._simulation_initialization: setted joint states")
            if self._model_randomization_enabled and not (self._tot_init_counter > 1 and self._configuration.init_args.randomization_recycle_model_alterations):
                mjx_adapter = self._adapter # type: ignore
                mjx_adapter.alter_model(link_masses =
                                            (self._randomized_mass_link_ids, self._current_episode_config.link_masses_ratios) if len(self._randomized_mass_link_ids) > 0 else None,
                                        link_frictions =
                                            (self._randomized_friction_links_ids, self._current_episode_config.link_frictions_ratios) if len(self._randomized_friction_links_ids) > 0 else None,
                                        joint_armature_ratios =
                                            (self._randomized_dof_armature_joints_ids,     self._current_episode_config.joint_armatures_ratios) if len(self._randomized_dof_armature_joints_ids) > 0 else None,
                                        joint_damping_ratios =
                                            (self._randomized_dof_damping_joints_ids, self._current_episode_config.joint_dampings_ratios) if len(self._randomized_dof_damping_joints_ids) > 0 else None,
                                        joint_frictionloss_ratios = 
                                            (self._randomized_dof_frictionloss_joints_ids, self._current_episode_config.joint_frictionlosses_ratios) if len(self._randomized_dof_frictionloss_joints_ids) > 0 else None,
                                        com_position_diffs =
                                            (self._randomized_com_links_ids, self._current_episode_config.link_coms_diffs) if len(self._randomized_com_links_ids) > 0 else None,
                                        com_quatxyzw_diffs =
                                            None,
                                        vec_mask = reinit_vecs)
                record_time(f"RobotVecEnv._simulation_initialization: setted model randomization")
        self._adapter.set_reference_filter(self._current_episode_config.joint_reference_filter_freqs)
        record_time(f"RobotVecEnv._simulation_initialization: setted reference filter randomization")
        masked_assign(self._last_sent_v_j_pvesd, reinit_vecs, self._precomputed_sim_init.initial_cmd_vec_j_pvesd)
        record_time(f"RobotVecEnv._simulation_initialization: setted last sent j_pvesd")
        record_region_end("RobotVecEnv._simulation_initialization")




    @override
    def _build(self):
        envCtrlName = type(self._adapter).__name__
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            self._adapter.build_scenario(models = self._get_spawn_defs(),
                                         default_link_group_collisions = self._configuration.init_args.enable_link_collisions)
            self._arrow_base = ("arrow","arrow_link")
            self._arrow_yellow = ("arrow_yellow","arrow_link")
        elif isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
            subadapters = self._adapter.sub_adapters()
            if len(subadapters) != 1:
                raise RuntimeError(f"Expected exactly one subadapter in VecSimJointImpedanceAdapterWrapper, got {len(subadapters)}")
            subadapter = subadapters[0]
            if adarl.utils.utils.isinstance_noimport(subadapter, ("PyBulletJointImpedanceAdapter")):
                self._adapter.build_scenario(models = self._get_spawn_defs())
                self._arrow_base = ("arrow","world")
                self._arrow_yellow = ("arrow_yellow","world")
            elif adarl.utils.utils.isinstance_noimport(subadapter, ("RosXbotGazeboAdapter")):
                self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path( "adarl_envs",
                                                                                                            "gazebo/all_gazebo_xbot.launch"),
                                            launch_file_args={"gui":"false"})
                self._arrow_base = ("arrow","arrow_link")
                self._arrow_yellow = ("arrow_yellow","arrow_link")
            else:
                raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        elif isinstance_noimport(self._adapter, "VecRosXBotAdapterWrapper"):
            if adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("RosXbotAdapter")):
                if self._configuration.real:
                    raise NotImplementedError()
                else:
                    # self._adapter.build_scenario(   models = [],
                    #                                 launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path(  "adarl_envs",
                    #                                                                                                 "ros/all_kyon_mujoco.launch"),
                    #                                 launch_file_args={"gui":"false"})
                    self._arrow_base = ("arrow","arrow_link")
                    self._arrow_yellow = ("arrow_yellow","arrow_link")
        elif isinstance_noimport(self._adapter, "VecZmqXbotAdapter"):
            if adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("ZmqXbotAdapter")):
                if self._configuration.real:
                    raise NotImplementedError()
                else:
                    # self._adapter.build_scenario(   models = [],
                    #                                 launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path(  "adarl_envs",
                    #                                                                                                 "ros/all_kyon_mujoco.launch"),
                    #                                 launch_file_args={"gui":"false"})
                    self._arrow_base = ("arrow","arrow_link")
                    self._arrow_yellow = ("arrow_yellow","arrow_link")
            else:
                raise NotImplementedError("Unexpected sub adapter "+self._adapter.sub_adapter())
        elif isinstance_noimport(self._adapter, "MujocoAdapter"):
            self._adapter.build_scenario(models = self._get_spawn_defs())
            self._arrow_base = ("arrow","arrow_link")
            self._arrow_yellow = ("arrow_yellow","arrow_link")
        elif isinstance_noimport(self._adapter, "GenesisAdapter"):
            spawn_defs = self._get_spawn_defs()
            for sd in spawn_defs:
                # The visual marker models have no fixed joint in genesis (they are moved around with
                # setLinksStateDirect), so compensate gravity to keep them in place like mujoco mocap bodies.
                if sd.name in ("arrow", "arrow_yellow", "axes"):
                    sd.kwargs["genesis_gravity_compensation"] = 1.0
            # the ui camera is parsed automatically by the adapter from the camera mjcf model
            self._adapter.build_scenario(models = spawn_defs)
            self._arrow_base = ("arrow","arrow_link")
            self._arrow_yellow = ("arrow_yellow","arrow_link")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        


        if self._configuration.robot_is_floating:
                # Correct any offset on the floating joint, to make it as if it is just at the origin
                # TODO: actually perform some inverse kinematics here, this just work for simple cases
                self._pinocchio_corrected_homing_body_pose_xyzxyzw = self._configuration.homing_body_pose_xyz_xyzw.cpu().numpy()
                self._robot_model.set_joint_pose_by_names({self._configuration.robot_root_joint:np.array([0.,0.,0.,  0.,0.,0.,1.])})
                root_joint_offset = self._robot_model.get_frame_poses_xyzxyzw(frames=[self._configuration.main_body_link[1]])[self._configuration.main_body_link[1]]
                self._pinocchio_corrected_homing_body_pose_xyzxyzw = np.concatenate([self._pinocchio_corrected_homing_body_pose_xyzxyzw[:3]-root_joint_offset[:3],
                                                           quat_mul_xyzw_np(self._pinocchio_corrected_homing_body_pose_xyzxyzw[3:],
                                                                            quat_conj_xyzw_np(root_joint_offset[3:]).astype(np.float32))])
                # ggLog.info(f"joint_pose = {self._robot_model.get_joint_pose()}")                
                # ggLog.info(f"root_joint_offset = {root_joint_offset}")
                # ggLog.info(f"self._configuration.main_body_link[1] = {self._configuration.main_body_link[1]}")
                # ggLog.info(f"_corrected_homing_body_pose_xyzxyzw = {self._pinocchio_corrected_homing_body_pose_xyzxyzw}")
        else:
            self._pinocchio_corrected_homing_body_pose_xyzxyzw = self._configuration.homing_body_pose_xyz_xyzw.cpu().numpy()
        self._robot_model.disable_tree_self_collisions(root_frame=self._configuration.robot_root_link[1])
        self._ground_co_id = self._robot_model.add_collision_box(   pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
                                                                    collision_box_size_xyz=(100,100,1),
                                                                    collision_obj_id="ground_collision")
        ggLog.info(f"Detecting always present self collisions...")
        if self._initial_pose_randomization_enabled:
            self._excluded_collision_pairs : set[tuple[str,str]] = self._robot_model.detect_always_present_collisions(
                moving_joints=[jn[1] for jn in self._configuration.joints_agent_controlled],
                fixed_joints_pose={self._configuration.robot_root_joint : self._pinocchio_corrected_homing_body_pose_xyzxyzw}
                                                if self._configuration.robot_is_floating else {},
                samples=1000,
                threshold=1.0)
            self._excluded_collision_pairs.update(self._robot_model.get_adjacent_collision_pairs())
        else:
            self._excluded_collision_pairs = None # not used
        ggLog.info(f"Always present self collisions = {pprint.pformat(self._excluded_collision_pairs)}")
        self._adapter.set_monitored_joints(self._configuration.joints_all_env_controlled)
        self._monitored_joints = self._configuration.joints_all_env_controlled
        self._adapter.set_monitored_links([self._configuration.main_body_link])
        self._adapter.set_impedance_controlled_joints(self._configuration.joints_all_env_controlled)
        self._main_body_link_ids = self._adapter.get_monitored_links_ids([self._configuration.main_body_link])
        self._controlled_joints_ids = self._adapter.get_monitored_joints_ids(self._configuration.joints_agent_controlled)
        self._held_joints_ids = self._adapter.get_monitored_joints_ids(self._configuration.joints_env_held)
        self._all_controlled_joints_ids = self._adapter.get_monitored_joints_ids(self._configuration.joints_all_env_controlled)





    @override
    def close(self):
        self._adapter.destroy_scenario()

    def set_cam_pose(self, pose_dist_pitch_yaw : tuple[float,float,float] | th.Tensor):
        self._configuration.ui_rel_camera_pose_dist_pitch_yaw = self._thtens(pose_dist_pitch_yaw)

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
        cam_link_state = th.zeros((13,), device=self._configuration.init_args.th_device, dtype=th.float32)
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
                body_states13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_pose = False)[:,0,:]
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
        if self._configuration.init_args.enable_dbg_checks:
            if isinstance(self._adapter, BaseVecSimulationAdapter):
                dbg_check_finite(state, async_assert=True, assert_msg="Nonfinite state detected in RobotVecEnv")
                dbg_check_finite(self._last_obs, async_assert=True, assert_msg="Nonfinite observation detected in RobotVecEnv")
            else:
                dbg_check_finite(self._last_obs["base.vec"], async_assert=True, assert_msg="Nonfinite observation detected in RobotVecEnv")
        return self._last_obs


    @override
    def get_states(self) -> dict[Any, th.Tensor]:
        return self._current_state

    @th.compiler.disable
    def _apply_impulses(self, forcevector: th.Tensor, torque: th.Tensor, duration: th.Tensor, delays: th.Tensor, apply_impulse: th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._adapter.set_link_impulses(self._main_body_link_ids,
                                            force_torque_xyzxyz=th.cat([forcevector, torque], dim = 1).view((self.num_envs,1,6)),
                                            durations=duration.view((self.num_envs,1)),
                                            delays=delays.view((self.num_envs,1)),
                                            vec_mask=apply_impulse.view((self.num_envs,)))
    @override
    def pre_step(self):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            if self._impulse_disturbances_enabled:
                impulse_prob_per_env_dt = 1-th.pow(1-self._configuration.impulse_probability_per_sec, self._intendedStepLength_sec)
                apply_impulse = self._thrand((self.num_envs,1)) < impulse_prob_per_env_dt
                # impulse = self._thrandn_truncnorm((self.num_envs,1),
                #                                 mean = self._configuration.impulse_mean_std[0],
                #                                 std = self._configuration.impulse_mean_std[1],
                #                                 min_val = 0.0,
                #                                 max_val = self._configuration.impulse_mean_std[0]+self._configuration.impulse_mean_std[1]*5)
                impulse = th.tanh(self._thrandn((self.num_envs,1))/5)*5*self._configuration.init_args.impulse_mean_std[1] + self._configuration.init_args.impulse_mean_std[0]
                duration = unnormalize(self._thrand((self.num_envs,1)),self._configuration.init_args.impulse_duration_minmax[0], self._configuration.init_args.impulse_duration_minmax[1])
                # ggLog.info(f"impulse={impulse}\n"
                #            f"durations={duration}\n")
                force = impulse/duration
                dir_incl_azim = self._thrand((self.num_envs,2))*2*th.pi
                direction = th.stack([th.sin(dir_incl_azim[:,0])*th.cos(dir_incl_azim[:,1]),
                                        th.sin(dir_incl_azim[:,0])*th.sin(dir_incl_azim[:,1]),
                                        th.cos(dir_incl_azim[:,0])], dim=1)
                forcevector = direction*force
                th.where(apply_impulse, forcevector, th.zeros_like(forcevector[0,0]), out=forcevector)

                torque = th.zeros_like(forcevector)

                delays = self._thrand((self.num_envs,)) * self._intendedStepLength_sec

                # ggLog.info(f"Setting impulses:\n"
                #            f"impulse_prob_per_env_dt = {impulse_prob_per_env_dt}\n"
                #            f"impulse_mean = {self._configuration.impulse_mean_std[0]}\n"
                #            f"impulse_std = {self._configuration.impulse_mean_std[1]}\n"
                #            f"impulse = {impulse}\n"
                #            f"force_torque_xyzxyz={th.cat([forcevector, torque], dim = 1)}\n"
                #            f"durations={duration}\n"
                #            f"delays={delays}\n"
                #            f"vec_mask={apply_impulse}\n")
                self._apply_impulses(forcevector=forcevector, torque=torque, duration=duration, delays=delays, apply_impulse=apply_impulse)

    @override
    def post_step(self):
        record_region_start("RobotVecEnv.post_step")
        adapter_data = self._get_adapter_data()
        record_time("RobotVecEnv.post_step get_adapter_data done")
        self._post_step_optimized(adapter_data)
        self._current_state = {k:t.detach().clone() for k,t in self._current_state.items()} # just here out of caution (on 20/11/2025 it was still necessary, maybe...)
        record_region_end("RobotVecEnv.post_step")


    # @th.compile(mode="max-autotune", disable=disable_compile) # this seems to cause issues with the rolling buffer used in states with history
    def _post_step_optimized(self, adapter_data):
        record_region_start("RobotVecEnv._post_step_optimized")
        self._update_state(adapter_data)
        record_time("RobotVecEnv._post_step_optimized: update_state done")
        self._update_stats()
        record_region_end("RobotVecEnv._post_step_optimized")

    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune",fullgraph=True)
    def _compute_extr_from_bodystate(self, body_abs_linvel_xyz_vec, body_abs_angvel_xyz_vec, body_abs_quat_xyzw_vec):
        conj_body_abs_quat_xyzw_vec = th_quat_conj(body_abs_quat_xyzw_vec)
        vec_body_rel_gravity_dir = th_quat_rotate_py(self._abs_gravity_dir.expand_as(body_abs_linvel_xyz_vec), conj_body_abs_quat_xyzw_vec)
        vec_body_rel_linvel_xyz = th_quat_rotate_py(body_abs_linvel_xyz_vec, conj_body_abs_quat_xyzw_vec)
        vec_body_rel_angvel_xyz = th_quat_rotate_py(body_abs_angvel_xyz_vec, conj_body_abs_quat_xyzw_vec)
        return vec_body_rel_gravity_dir, vec_body_rel_linvel_xyz, vec_body_rel_angvel_xyz

    @th.compiler.disable
    def _get_adapter_data_no_compile(self):
        """get_adapter_data, explicitly not torch-compiled"""
        return self._get_adapter_data_raw()

    def _get_adapter_data_raw_mjx(self):
        """ get_adapter_data_raw, optimized for MJX """
        record_region_start("RobotVecEnv._get_adapter_data_raw_mjx")
        from adarl.adapters.MjxAdapter import MjxAdapter
        mjx_adapter : MjxAdapter = self._adapter # type: ignore
        states = mjx_adapter.get_sim_elements_state(
                                            joint_ids=self._all_controlled_joints_ids,
                                            link_ids=self._main_body_link_ids,
                                            )
        record_time("RobotVecEnv._get_adapter_data_raw_mjx: get_sim_elements_state done")
        vec_jstates_j_pveae =               states.joint_state_pveae[:,:len(self._configuration.joints_agent_controlled)]
        vec_held_jstates_j_pveae =          states.joint_state_pveae[:,len(self._configuration.joints_agent_controlled):len(self._configuration.joints_all_env_controlled)]
        vec_stats_minmaxavgstd_j_pvaeep =   states.joint_stats_pvaeep[:,:,:len(self._configuration.joints_agent_controlled)]
        vec_bodystates_13 = states.link_state[:,0,:]
        vec_body_rel_linacc_xyz = states.link_linacc[:,0,:]
        
        body_abs_quat_xyzw_vec  = vec_bodystates_13[:,3:7]
        vec_body_abs_linvel_xyz = vec_bodystates_13[:,7:10]
        vec_body_abs_angvel_xyz = vec_bodystates_13[:,10:13]
        vec_body_ground_dist    = vec_bodystates_13[:,2]

        vec_body_rel_gravity_dir, vec_body_rel_linvel_xyz, vec_body_rel_angvel_xyz = self._compute_extr_from_bodystate(body_abs_linvel_xyz_vec = vec_body_abs_linvel_xyz,
                                                                                                                        body_abs_angvel_xyz_vec = vec_body_abs_angvel_xyz,
                                                                                                                        body_abs_quat_xyzw_vec = body_abs_quat_xyzw_vec)

        vec_time_from_start = self._adapter.getEnvTimeFromStartup() - self._eps_start_stime

        record_region_end("RobotVecEnv._get_adapter_data_raw_mjx")
        return (vec_stats_minmaxavgstd_j_pvaeep,
                vec_jstates_j_pveae,
                vec_body_abs_linvel_xyz, # only used for visualization, can be wrong
                vec_body_abs_angvel_xyz,
                vec_body_ground_dist,
                vec_body_rel_gravity_dir,
                vec_body_rel_linvel_xyz,
                vec_body_rel_angvel_xyz,
                vec_body_rel_linacc_xyz,
                vec_time_from_start,
                vec_bodystates_13,
                vec_held_jstates_j_pveae)

    def _get_adapter_data_raw(self):
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            return self._get_adapter_data_raw_mjx()
        

        record_region_start("RobotVecEnv._get_adapter_data_raw")
        jstates_v_j_pve = self._adapter.getJointsState(requestedJoints=self._controlled_joints_ids)
        held_jstates_v_j_pve = self._adapter.getJointsState(requestedJoints=self._held_joints_ids)
        record_time("RobotVecEnv._get_adapter_data_raw: getJointsState done")
        vec_jstates_j_pveae = th.cat([jstates_v_j_pve, th.zeros_like(jstates_v_j_pve[:,:,:2])], dim = -1)
        vec_held_jstates_j_pveae = th.cat([held_jstates_v_j_pve, th.zeros_like(held_jstates_v_j_pve[:,:,:2])], dim = -1)
        # ggLog.info(f"jstates_v_j_pve = {jstates_v_j_pve}")
        # th.cuda.synchronize()
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            vec_bodystates_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_pose = False)[:,0,:]
            record_time("RobotVecEnv._get_adapter_data_raw: getLinksState done")
            body_abs_quat_xyzw_vec  = vec_bodystates_13[:,3:7]
            vec_body_abs_linvel_xyz = vec_bodystates_13[:,7:10]
            vec_body_abs_angvel_xyz = vec_bodystates_13[:,10:13]
            vec_linkstate_stats = self._adapter.get_links_state_step_stats()
            record_time("RobotVecEnv._get_adapter_data_raw: get_links_state_step_stats done")
            vec_body_abs_linvel_xyz = vec_linkstate_stats[:,2,0,0:3]
            vec_body_abs_angvel_xyz = vec_linkstate_stats[:,2,0,3:6]
            vec_body_rel_gravity_dir, vec_body_rel_linvel_xyz, vec_body_rel_angvel_xyz = self._compute_extr_from_bodystate(body_abs_linvel_xyz_vec = vec_body_abs_linvel_xyz,
                                                                                                                           body_abs_angvel_xyz_vec = vec_body_abs_angvel_xyz,
                                                                                                                           body_abs_quat_xyzw_vec = body_abs_quat_xyzw_vec)
            vec_body_ground_dist = vec_bodystates_13[:,2]              
            vec_body_rel_linacc_xyz = self._adapter.get_local_link_linear_acceleration(self._main_body_link_ids)[:,0,:]
            if self._configuration.init_args.enable_dbg_checks:
                dbg_check(lambda: th.all(th.isfinite(vec_bodystates_13)),
                        lambda: f"non finite values in body link state at {th.logical_not(th.isfinite(vec_bodystates_13)).nonzero()}: {vec_bodystates_13[th.logical_not(th.isfinite(vec_bodystates_13))]} : {vec_bodystates_13}",
                        just_warn=True,
                        async_assert=True,
                        assert_msg="non finite values in body link state")                
        else:
            vec_bodystates_13 = None
            vec_body_rel_gravity_dir = self._adapter.get_link_gravity_direction(self._main_body_link_ids)[:,0,:]
            vec_body_rel_angvel_xyz = self._adapter.get_link_relative_angular_velocity(self._main_body_link_ids)[:,0,:]
            example_vec_3d_tens = vec_jstates_j_pveae[:,0,:3]
            vec_body_abs_linvel_xyz = th.zeros_like(example_vec_3d_tens)
            vec_body_abs_angvel_xyz = th.zeros_like(example_vec_3d_tens)
            vec_body_rel_linvel_xyz = th.zeros_like(example_vec_3d_tens)
            vec_body_rel_linacc_xyz = th.zeros_like(example_vec_3d_tens)
            vec_body_ground_dist    = th.zeros_like(example_vec_3d_tens[:,0])
        # ggLog.info(f"axes pose = {self._adapter.getLinksState(requestedLinks = self._adapter.get_links_ids([('axes','root')]), use_com_pose = False)[:,0,:]}")
        # ggLog.info(f"bstates_v_13 = {bstates_v_13}")
        # th.cuda.synchronize()
        try:
            # The adapter returns stats for all monitored joints (agent-controlled first, then held);
            # keep only the agent-controlled ones, matching _get_adapter_data_raw_mjx.
            vec_stats_minmaxavgstd_j_pvaeep = self._adapter.get_joints_state_step_stats_extended()[:, :, :len(self._configuration.joints_agent_controlled)]
            record_time("RobotVecEnv._get_adapter_data_raw: get_joints_state_step_stats_extended done")
            if self._configuration.init_args.enable_dbg_checks:
                dbg_check(lambda: th.all(th.isfinite(vec_stats_minmaxavgstd_j_pvaeep)),
                        lambda: (f"non finite values in joint stats at indexes:\n{th.logical_not(th.isfinite(vec_stats_minmaxavgstd_j_pvaeep)).nonzero()}\n"
                                f"nonfinite values =\n{vec_stats_minmaxavgstd_j_pvaeep[th.logical_not(th.isfinite(vec_stats_minmaxavgstd_j_pvaeep))]}\n"
                                f"all values =\n{vec_stats_minmaxavgstd_j_pvaeep}"),
                        just_warn=True,
                        async_assert=True,
                        assert_msg="non finite values in joint stats")
                if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
                    def build_error_msg():
                        mjx_adapter : MjxAdapter = self._adapter # type: ignore
                        bad_sim_id = th.logical_not(th.isfinite(vec_bodystates_13)).nonzero()[0,0].item()
                        import jax.numpy as jnp
                        return (f"diverging sim {bad_sim_id}:\n"
                                f" model.geom_friction = {mjx_adapter._sim_state.mjx_model.geom_friction[bad_sim_id]} (avg = {jnp.mean(mjx_adapter._sim_state.mjx_model.geom_friction, axis=0)})\n"
                                f" model.body_mass = {mjx_adapter._sim_state.mjx_model.body_mass[bad_sim_id]} (avg = {jnp.mean(mjx_adapter._sim_state.mjx_model.body_mass, axis=0)})"
                                f" model.dof_frictionloss = {mjx_adapter._sim_state.mjx_model.dof_frictionloss[bad_sim_id]} (avg = {jnp.mean(mjx_adapter._sim_state.mjx_model.dof_frictionloss, axis=0)})"
                                f" model.dof_armature = {mjx_adapter._sim_state.mjx_model.dof_armature[bad_sim_id]} (avg = {jnp.mean(mjx_adapter._sim_state.mjx_model.dof_armature, axis=0)})")
                    dbg_check(lambda : th.logical_or(th.all(th.isfinite(vec_stats_minmaxavgstd_j_pvaeep)), th.all(th.isfinite(vec_bodystates_13))),
                            build_error_msg, just_warn = True, async_assert=True, assert_msg="diverging sim")
        except NotImplementedError:
            vec_stats_minmaxavgstd_j_pvaeep = self._thfull(float("nan"), (self.num_envs,4,len(self._configuration.joints_agent_controlled),6))
        vec_time_from_start = self._adapter.getEnvTimeFromStartup() - self._eps_start_stime

        record_region_end("RobotVecEnv._get_adapter_data_raw")
        # ggLog.info(f"getJoints={t1-t0:.6f} getlinks={t2-t1:.6f} getstats={t3-t2:.6f} others={t3_1-t3:.6f} tot = {t3_1-t0:.6f}s")
        return (vec_stats_minmaxavgstd_j_pvaeep,
                vec_jstates_j_pveae,
                vec_body_abs_linvel_xyz, # only used for visualization, can be wrong
                vec_body_abs_angvel_xyz,
                vec_body_ground_dist,
                vec_body_rel_gravity_dir,
                vec_body_rel_linvel_xyz,
                vec_body_rel_angvel_xyz,
                vec_body_rel_linacc_xyz,
                vec_time_from_start,
                vec_bodystates_13,
                vec_held_jstates_j_pveae)


    def _get_adapter_data(self):
        if th.compiler.is_compiling():
            r = self._get_adapter_data_no_compile()
        else:
            r = self._get_adapter_data_raw()
        self._obs2act_timings.mark_start()
        return r

    def _get_new_instantaneous_state(self, adapter_data) -> dict[str, dict[Any, th.Tensor] | th.Tensor]:        
        
        (   vec_stats_minmaxavgstd_j_pvaeep,
            vec_jstates_j_pveae,
            vec_body_abs_linvel_xyz, # only used for visualization, can be wrong
            vec_body_abs_angvel_xyz,
            vec_body_ground_dist,
            vec_body_rel_gravity_dir,
            vec_body_rel_linvel_xyz,
            vec_body_rel_angvel_xyz,
            vec_body_rel_linacc_xyz,
            vec_time_from_start,
            vec_bodystates_13,
            vec_held_jstates_j_pveae) = adapter_data
        prev_vec_joint_posrefs       = self._current_state[self.STATE_ROBOT][:,0,:,5]
        vec_step_count               = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT]
        prev_lims_safety_triggered   = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED].view((self.num_envs,)) > 0
        prev_posref_safety_triggered = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED].view((self.num_envs,)) > 0
        prev_posref_safety_violation_count = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_VIOLATION_COUNT].view((self.num_envs,))
        prev_vec_time_from_start     = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SIM_TIME]

        new_inst_state = self._build_new_instantaneous_state_vec(
                                    vec_step_count=vec_step_count,
                                    prev_vec_joint_posrefs = prev_vec_joint_posrefs,
                                    prev_lims_safety_triggered=prev_lims_safety_triggered,
                                    prev_posref_safety_triggered=prev_posref_safety_triggered,
                                    prev_posref_safety_violation_count=prev_posref_safety_violation_count,
                                    prev_vec_time_from_start=prev_vec_time_from_start,
                                    vec_stats_minmaxavgstd_j_pvaeep = vec_stats_minmaxavgstd_j_pvaeep,
                                    vec_jstates_j_pveae = vec_jstates_j_pveae,
                                    vec_held_jstates_j_pveae = vec_held_jstates_j_pveae,
                                    vec_last_sent_refs_j_pvesd = self._last_sent_v_j_pvesd,
                                    vec_last_sent_refs_held_j_pvesd = self._homing_held_joints_vec_pvesd,
                                    vec_body_abs_linvel_xyz = vec_body_abs_linvel_xyz, # only used for visualization, can be wrong
                                    vec_body_abs_angvel_xyz = vec_body_abs_angvel_xyz,
                                    vec_body_ground_dist = vec_body_ground_dist,
                                    vec_body_rel_gravity_dir = vec_body_rel_gravity_dir,
                                    vec_body_rel_linvel_xyz = vec_body_rel_linvel_xyz,
                                    vec_body_rel_angvel_xyz = vec_body_rel_angvel_xyz,
                                    vec_body_rel_linacc_xyz = vec_body_rel_linacc_xyz,
                                    vec_time_from_start=vec_time_from_start)
        # ggLog.info(f"insta_state sizes = "+str(map_tensor_tree(new_inst_state,lambda t: t.size())))
        # th.cuda.synchronize()
        # if not th.all(th.isfinite(new_inst_state[self.STATE_ROBOT_STATS])):
        #     ggLog.info(f"nonfinite vals in new_robot_stats_state = {new_inst_state[self.STATE_ROBOT_STATS]}")
        # dbg_check(lambda: th.all(new_inst_state[self.STATE_ROBOT][:,:,8:10]>=0), lambda: f"negative gains in new_robot_state") #type: ignore
        # th.cuda.synchronize()
        # ggLog.info(pprint.pformat(map_tensor_tree(new_inst_state, lambda t: t.size())))
        return new_inst_state

    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune",fullgraph=True)
    def _build_new_instantaneous_state_vec(self,    vec_step_count : th.Tensor,
                                                    prev_lims_safety_triggered : th.Tensor,
                                                    prev_posref_safety_triggered : th.Tensor,
                                                    prev_posref_safety_violation_count : th.Tensor,
                                                    prev_vec_joint_posrefs : th.Tensor,
                                                    prev_vec_time_from_start : th.Tensor,
                                                    vec_stats_minmaxavgstd_j_pvaeep : th.Tensor,
                                                    vec_jstates_j_pveae : th.Tensor,
                                                    vec_held_jstates_j_pveae : th.Tensor,
                                                    vec_last_sent_refs_j_pvesd : th.Tensor,
                                                    vec_last_sent_refs_held_j_pvesd : th.Tensor,
                                                    vec_body_abs_linvel_xyz : th.Tensor,
                                                    vec_body_abs_angvel_xyz : th.Tensor,
                                                    vec_body_ground_dist : th.Tensor,
                                                    vec_body_rel_gravity_dir : th.Tensor,
                                                    vec_body_rel_linvel_xyz : th.Tensor,
                                                    vec_body_rel_angvel_xyz : th.Tensor,
                                                    vec_body_rel_linacc_xyz : th.Tensor,
                                                    vec_time_from_start : th.Tensor) -> dict[str, dict[Any, th.Tensor] | th.Tensor]:
        vec_step_count = vec_step_count.view((self.num_envs,))        
        is_resetting = vec_step_count == -1
        has_settled = vec_step_count >= -10
        # prev_vec_time_from_start = prev_vec_internal_state[:,self.INTERNAL_FIELDS.SIM_TIME]
        # vec_prev_safety_triggered = vec_internal_state[:,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        # ggLog.info(f"stats_minmaxavgstd_j_pvae.device = {stats_minmaxavgstd_j_pvae.device}   self._safe_limits_minmax_j_pve[0].device = {self._safe_limits_minmax_j_pve[0].device}")
        pveidx = th.as_tensor([0,1,3]).to(device=vec_stats_minmaxavgstd_j_pvaeep.device, non_blocking=True)
        if self._configuration.init_args.enable_limits_safety:
            vec_triggered_limits = th.logical_or(   vec_stats_minmaxavgstd_j_pvaeep[:, 0, :, pveidx] < self._safe_limits_minmax_j_pve[0],
                                                    vec_stats_minmaxavgstd_j_pvaeep[:, 1, :, pveidx] > self._safe_limits_minmax_j_pve[1])
            vel_limits_safety_violation = th.any(vec_triggered_limits, dim=(1,2))
            triggered = th.logical_and(vel_limits_safety_violation, has_settled)
            vec_safety_lims_state = th.logical_or(prev_lims_safety_triggered,triggered)*1.0
        else:
            vec_safety_lims_state = self._thzeros((self.num_envs,))
        if self._configuration.init_args.enable_posref_safety:
            posref_diff = vec_last_sent_refs_j_pvesd[:,:,0] - prev_vec_joint_posrefs
            posref_safety_violation = th.logical_or(posref_diff < self._posref_safety_minmmax_diff[0],
                                                    posref_diff > self._posref_safety_minmmax_diff[1])
            posref_safety_violation = th.any(posref_safety_violation, dim=1)
            posref_triggered = th.logical_and(posref_safety_violation, has_settled)
            vec_safety_posref_state = (th.logical_or(prev_posref_safety_triggered,posref_triggered)*1.0).view(self.num_envs,)
        else:
            posref_safety_violation = self._thzeros((self.num_envs,), dtype=th.bool)
            vec_safety_posref_state = self._thzeros((self.num_envs,))
        last_step_dt = th.where(is_resetting,
                                self._configuration.init_args.stepLength_sec,
                                vec_time_from_start - prev_vec_time_from_start.view((self.num_envs,)))
        prev_posref_safety_violation_count = prev_posref_safety_violation_count.view(self.num_envs,)
        prev_posref_safety_violation_count = prev_posref_safety_violation_count * th.logical_not(is_resetting) # reset violation count if resetting
        posref_violation_count = prev_posref_safety_violation_count + posref_safety_violation
        # print(f"prev_posref_safety_violation_count = {prev_posref_safety_violation_count}")
        # print(f"posref_violation_count = {posref_violation_count}")
        # print(f"posref_safety_violation = {posref_safety_violation}")
        new_step_count = vec_step_count+1
        new_internal_state = {  self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED : vec_safety_lims_state.view(self.num_envs,1),
                                self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED : vec_safety_posref_state.view(self.num_envs,1),
                                self.INTERNAL_FIELDS.SAFETY_POSREF_VIOLATION_COUNT : posref_violation_count.view(self.num_envs,1),
                                self.INTERNAL_FIELDS.STEP_COUNT : new_step_count.view((self.num_envs,1)),
                                self.INTERNAL_FIELDS.SIM_TIME : vec_time_from_start.view(self.num_envs,1),
                                self.INTERNAL_FIELDS.LAST_STEP_DT : last_step_dt.view(self.num_envs,1)}
        new_jreferrs_vec_p_j = (vec_jstates_j_pveae[:,:,0] - vec_last_sent_refs_j_pvesd[:,:,0]).view(self.num_envs, 1, len(self._configuration.joints_agent_controlled))
        new_robot_state = th.cat([vec_jstates_j_pveae, vec_last_sent_refs_j_pvesd], dim = -1)
        new_held_joints_state = th.cat([vec_held_jstates_j_pveae, vec_last_sent_refs_held_j_pvesd], dim = -1)
        # build stats:
        # with permute the first dimension becomes the joint (ordered as in set_monitored_joints)
        # with flatten the second dimension becomes minp,minv,mina,mmine,maxp,maxv,...
        new_robot_stats_state_pvaeep = vec_stats_minmaxavgstd_j_pvaeep.permute(0,2,1,3).flatten(start_dim=2) # exchange minmaxavgstd and joint dim, then flatten minmaxavgstd into one dim
        # ggLog.info(f"new_robot_stats_state_pvaee = {new_robot_stats_state_pvaee.size()}, expected {(self.num_envs, len(self._configuration.controlled_joints), 4*5)}")
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : vec_body_rel_linvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : vec_body_rel_linvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : vec_body_rel_linvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : vec_body_rel_angvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : vec_body_rel_angvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : vec_body_rel_angvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINACC_X : vec_body_rel_linacc_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Y : vec_body_rel_linacc_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINACC_Z : vec_body_rel_linacc_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : vec_body_abs_linvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : vec_body_abs_linvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : vec_body_abs_linvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_X : vec_body_abs_angvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Y : vec_body_abs_angvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_ANGVEL_Z : vec_body_abs_angvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z    : vec_body_ground_dist.view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : vec_body_rel_gravity_dir[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : vec_body_rel_gravity_dir[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : vec_body_rel_gravity_dir[:,2].view(self.num_envs,1)}
        
        # step_avg_pos = vec_stats_minmaxavgstd_j_pvae[:,2,:,0]
        step_avg_pos = vec_jstates_j_pveae[:,:,0]
        prev_step_avg_pos = self._current_state[self.STATE_JOINT_LONGTERM_STATS][:,0,0]
        # print(f"vec_step_count={vec_step_count} step_avg_pos={step_avg_pos}")
        long_alpha = self._configuration.longterm_exp_smoothing_1s**self._configuration.init_args.stepLength_sec
        step_avg_pos = th.where(is_resetting.view((self.num_envs,1)),
                                step_avg_pos,
                                step_avg_pos*(1-long_alpha) + prev_step_avg_pos*long_alpha)
        new_longterm_stats_state = {self.JOINT_LONGTERM_STATS_FIELDS.AVG_POS : step_avg_pos}


        return {    self.STATE_EXTRINSIC    : new_extrinsic_state,
                    self.STATE_INTERNAL     : new_internal_state,
                    self.STATE_ROBOT        : new_robot_state,
                    self.STATE_POS_REF_ERR  : new_jreferrs_vec_p_j,
                    self.STATE_HELD_JOINTS   : new_held_joints_state,
                    self.STATE_JOINT_STEP_STATS  : new_robot_stats_state_pvaeep,
                    self.STATE_JOINT_LONGTERM_STATS : new_longterm_stats_state,
                    self.STATE_ACT_PREPROC : {self.ACT_FIELDS.ACTION : self._last_preprocessed_actions},
                    self.STATE_ACT_RAW_HIST : {self.ACT_FIELDS.ACTION : self._last_raw_actions},
                    self.STATE_LAST_ACT_RAW : {self.ACT_FIELDS.ACTION : self._last_raw_actions}}
        
    def _reinit_state(self, vec_mask : th.Tensor, adapter_data):
        """Reinitialize the state for the envs specified in vec_mask. This propagates the current instanateous state into the state history,
            it is used at environment resets."""
        record_region_start("RobotVecEnv._reinit_state")
        step_count = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT]
        masked_assign(step_count, vec_mask, -1)
        record_time("RobotVecEnv._reinit_state: masked_assign step_count done")
        instantaneous_state = self._get_new_instantaneous_state(adapter_data)
        record_time("RobotVecEnv._reinit_state: get_new_instantaneous_state done")
        # self._state_helper.check_size(instantaneous_state=instantaneous_state)
        instantaneous_state[self.STATE_INTERNAL][self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED].fill_(0.0)
        instantaneous_state[self.STATE_INTERNAL][self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED].fill_(0.0)
        record_time("RobotVecEnv._reinit_state: modified instantaneous_state done")
        self._current_state = self._state_helper.reset_state(instantaneous_state, vec_mask=vec_mask, old_state=self._current_state) # This repeats the instantaneous state across the history dimension
        record_region_end("RobotVecEnv._reinit_state")

    def _update_state(self, adapter_data):
        # th.cuda.synchronize()
        # t0 = time.monotonic()
        instantaneous_state = self._get_new_instantaneous_state(adapter_data)
        # th.cuda.synchronize()
        # t01 = time.monotonic()
        if not th.compiler.is_compiling():
            self._state_helper.check_size(instantaneous_state=instantaneous_state)
        # dbg_run(lambda: self._state_helper.check_size(instantaneous_state=instantaneous_state))
        # sizes = map_tensor_tree(flatten_tensor_tree(instantaneous_state), lambda t: t.size())
        # # {k:v.size() for k,v in instantaneous_state.items()}
        # n = "\n"
        # ggLog.info(f"Got instantaneous state with sizes: {n.join([str(kv) for kv in sizes.items()])}")
        # t1 = time.monotonic()
        self._state_helper.update(instantaneous_state, state=self._current_state) # rolls down the history and adds current state
        # ss = {k:t.size() for k,t in self._current_state.items()}
        # ggLog.info(f"state sizes = {ss}")
        dbg_check(lambda: th.all(self._current_state[self.STATE_INTERNAL][0,0,self.INTERNAL_FIELDS.STEP_COUNT] >= 0),
                  lambda: f"Negative step_counts {self._current_state[self.STATE_INTERNAL][0,0,self.INTERNAL_FIELDS.STEP_COUNT]}",
                  async_assert=True,
                  assert_msg="Negative step_counts detected in state")
        # map_tensor_tree(self._current_state, lambda t: t.detach().clone())
        # tf = time.monotonic()
        # print(f"newinst = {t01-t0}, check = {t1-t01}, map = {tf-t1}, tot = {tf-t0}")



    def _update_stats(self):
        # sub_rewards = {}
        # self.compute_rewards(self._current_state, 
        #                         sub_rewards_return=sub_rewards)
        # self._stats["rewards"] = sub_rewards
        pass
        
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        record_region_start("RobotVecEnv.get_infos")
        if self._configuration.init_args.no_infos:
            record_region_end("RobotVecEnv.get_infos")
            return {}
        sub_rews = {}
        reward = self.compute_rewards(state, sub_rews)
        state_internal = state[self.STATE_INTERNAL]
        step_count = state_internal[:,0,self.INTERNAL_FIELDS.STEP_COUNT]
        i : dict[str, th.Tensor] = {
            "ep_step_count" : step_count,
            "ep_count" : self._ep_counter,
            "rewards" : th.stack(list(sub_rews.values()), dim = 1) ,
            "tot_reward" : reward,
            "safety_limits_triggered" : state_internal[:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED],
            "safety_posref_triggered" : state_internal[:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED],
            "safety_posref_violation_rate" : state_internal[:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_VIOLATION_COUNT]/th.clamp(step_count+1, min=1.0)
        }
        if labels is not None:
            if not hasattr(self, "_sub_rew_names_th"): self._sub_rew_names_th = to_string_tensor(list(sub_rews.keys()))
            labels["rewards"] = self._sub_rew_names_th
        if self._configuration.init_args.minimal_infos:
            record_region_end("RobotVecEnv.get_infos")
            return i

        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        act_raw_state = state[self.STATE_ACT_RAW_HIST]   
        homing_ctrl_pos = self._configuration.homing_ctrl_joints_position
        normhoming = normalize(homing_ctrl_pos, lims[0,:,0], lims[1,:,0])
        normctrlhoming = normalize(homing_ctrl_pos, self._ctrl_limits[0,:,0], self._ctrl_limits[1,:,0])
        smoothed_joint_pose_norm = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)[:,0,0]
        actdiff             = th.flatten((act_raw_state[:,0] - act_raw_state[:,1])/2, start_dim=1)
        prev_actdiff        = th.flatten((act_raw_state[:,1] - act_raw_state[:,2])/2, start_dim=1)
        act_acc             = (actdiff - prev_actdiff)/2
        joint_pose = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)[:,0,:,0]
        joint_pose_ctrlnorm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], self._ctrl_limits, warn_limits_violation=False)[:,0,:,0]
        i.update({
            "tot_reward_sum" : reward.sum(dim=1) if reward.dim()>1 else reward,
            "joint_homing_dist" : state[self.STATE_JOINT_LONGTERM_STATS][:,0,0,:] - homing_ctrl_pos,
            "joint_pos_error" : th.mean(th.abs(smoothed_joint_pose_norm - normhoming), dim=1),
            "joint_pos_error_instant" : th.mean(th.abs(joint_pose - normhoming), dim=1),
            "joint_homing_dist_ctrlnorm" : joint_pose_ctrlnorm - normctrlhoming,
            "act_diff" : actdiff,
            "act_acc" : act_acc,
            "joint_avg_act_diff" : actdiff.abs().mean(dim=-1),
            "joint_avg_act_acc" : act_acc.abs().mean(dim=-1)
        })
        record_time("RobotVecEnv.get_infos: built dict")
        i.update({f"stats.{k}":v for k,v in self._stats.items()})
        record_time("RobotVecEnv.get_infos: added stats to dict")
        i.update({"ep_config."+k:v for k,v in dataclasses.asdict(self._current_episode_config).items()})
        record_time("RobotVecEnv.get_infos: added ep_config to dict")

        if labels is not None:
            if not hasattr(self, "_joint_names_th"): self._joint_names_th = to_string_tensor([jn[1] for jn in self._configuration.joints_agent_controlled])
            labels["joint_homing_dist"] = self._joint_names_th
            labels["act_diff"] = to_string_tensor(self._state_helper.sub_helpers[self.STATE_ACT_RAW_HIST].flat_state_names()[:12])
            labels["act_acc"] = labels["act_diff"]
        record_time("RobotVecEnv.get_infos: added labels to dict")

        if self._configuration.init_args.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            i.update({f"obs_{k}":o for k,o in self._last_obs.items()})
            if labels is not None:
                obs_names = self._state_helper.observation_names()
                ggLog.info(f"obs_names = "+str({k:v.shape for k,v in obs_names.items()}))
                all_obs_labels = {}
                for k in self._last_obs.keys():
                    subobs_names = obs_names[k]
                    if len(subobs_names.shape) == 1:
                        all_obs_labels[f"obs_{k}"] = to_string_tensor([n for n in subobs_names])
                    else:
                        ggLog.warn(f"Sub observation {k} has more than 1 dimension, cannot store labels")
                labels.update(all_obs_labels)
            # i["vec_obs"] = self._last_obs["base.vec"]
            # if not self._configuration.merge_privileged:
            #     i["vec_obs_privileged"] = self._last_obs["privileged.vec"]
            # if labels is not None:
            #     labels["vec_obs"] = to_string_tensor([n for n in self._state_helper.observation_names()["base.vec"]])
            #     if not self._configuration.merge_privileged:
            #         labels["vec_obs_privileged"] = to_string_tensor([n for n in self._state_helper.observation_names()["privileged.vec"]])
            posref_diff =      state[self.STATE_ROBOT][:,0,:,5] - state[self.STATE_ROBOT][:,1,:,5]
            prev_posref_diff = state[self.STATE_ROBOT][:,1,:,5] - state[self.STATE_ROBOT][:,2,:,5]
            posref_vel = posref_diff/self._configuration.init_args.stepLength_sec
            prev_posref_vel = prev_posref_diff/self._configuration.init_args.stepLength_sec
            posref_acc = (posref_vel - prev_posref_vel)/self._configuration.init_args.stepLength_sec
            i["posref_diff"] = posref_diff
            i["posref_vel"] = posref_vel
            i["posref_acc"] = posref_acc
            i["normposref_diff"] = statenorm[self.STATE_ROBOT][:,1,:,5] - statenorm[self.STATE_ROBOT][:,0,:,5]
            i["normposref_vel"] = i["normposref_diff"]/2/self._configuration.init_args.stepLength_sec
            for substate in self._state_helper.sub_helpers.keys():
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
        record_time("RobotVecEnv.get_infos: verbose part done")
        record_region_end("RobotVecEnv.get_infos")
        return i
    
    @override
    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune")
    def are_states_terminal(self, states) -> th.Tensor:
        if self._configuration.init_args.terminate_on_safety:
            safety_triggered = th.logical_or(   states[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED,0] > 0,
                                                states[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED,0] > 0)
            return (safety_triggered > 0).view((self.num_envs,))
        else:
            return th.zeros_like(self._no_envs)
        # if th.any(r):
        #     term_idxs = th.nonzero(r)
        #     ggLog.info(f"Env {term_idxs} terminated at step {self._ep_step_counter[term_idxs]}")
    
    @override
    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune")
    def are_states_timedout(self, states) -> th.Tensor:
        sinternal = states[self.STATE_INTERNAL]
        # r = sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT] >= self.get_max_episode_steps()
        # ggLog.info(f"sinternal.size() = {states[self.STATE_INTERNAL].size()}")
        # ggLog.info(f" sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT] = { sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT].size()}")
        # ggLog.info(f"self.get_max_episode_steps() = {self.get_max_episode_steps().size()}")
        # ggLog.info(f"r.size() = {r.size()}")
        step_counts = sinternal[:,0,self.INTERNAL_FIELDS.STEP_COUNT,0]
        max_steps = self.get_max_episode_steps()
        timedout = (step_counts >= max_steps)
        return timedout.view((self.num_envs,))



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
        dbg_check_bounded(robot_state_norm, -10,10, async_assert=False, just_warn=True)

    @override
    def compute_rewards(self,   state : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        # reward_health = th.ones((self.num_envs,), device=self._configuration.th_device, dtype=self._configuration.obs_dtype)
        # sub_rewards_return["health"] = reward_health

        max_rew = 100
        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_position, lims[0,:,0], lims[1,:,0])

        robot_state_norm = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)
        # dbg_run(lambda: self._warn_out_of_bounds(robot_state_norm))
        
        normposhomingdiff = robot_state_norm[:,0,:,0] - normhoming
        normvelocities =    robot_state_norm[:,0,:,1]
        normtorques =       robot_state_norm[:,0,:,2]
        normaccelerations = (robot_state_norm[:,0,:,1] - robot_state_norm[:,1,:,1])/self._configuration.init_args.stepLength_sec
        
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

    def _sample_distr(self, size, distribution : DistributionDefTh) -> th.Tensor:
        return sample_distr(size,
                            distribution,
                            device=self._th_device,
                            dtype=self._obs_dtype,
                            generator=self._rng)
    
    def _distr_to_tensor(self, distr : DistributionDef, size : tuple[int,...] | None = None) -> DistributionDefTh:
        return distr_to_tensor(distr, size, device=self._th_device, dtype=self._obs_dtype)