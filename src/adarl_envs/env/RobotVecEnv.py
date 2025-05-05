from __future__ import annotations
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
from adarl.envs.vec.BaseVecEnv import Observation
from adarl.utils.robot_helpers import Robot
from adarl.utils.utils import to_string_tensor, th_quat_rotate_py, th_quat_conj, ros_rpy_to_quaternion_xyzw_th, quat_mul_xyzw
from adarl.utils.vec_state_helper import    JointImpedanceActionHelper, ThBoxStateHelper,\
                                        RobotStateHelper, RobotStatsStateHelper,\
                                        StateNoiseGenerator, DictStateHelper, unnormalize, normalize
from adarl.utils.tensor_trees import map_tensor_tree, flatten_tensor_tree, map2_tensor_tree, space_from_tree
from adarl.utils.utils import build_pose, JointState, Pose, LinkState, isinstance_noimport, masked_assign, masked_assign_sc, quat_conj_xyzw_np, quat_mul_xyzw_np
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check, dbg_run
from dataclasses import dataclass
from gymnasium import Space
from enum import Enum, IntEnum
from typing import Sequence, Literal, TypedDict, Any, Callable
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
from adarl.utils.spaces import get_space_labels
import pprint


def hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))

JOINT_FILTERS = Enum("JOINT_FILTERS",["ALL_REVOLUTE",
                                         "ALL"])

LINK_FILTERS = Enum("LINK_FILTERS",["ALL","ALL_ROBOT"])

def find_pose(  root_joint : str,
                homing_body_pose_xyzxyzw : np.ndarray,
                controlled_joints : Sequence[tuple[str,str]],
                initial_pose_randomization : float,
                limits_minmax : th.Tensor,
                homing_pos : th.Tensor,
                noncontrolled_jointpos : dict[tuple[str,str], th.Tensor],
                robot_model : Robot,
                is_floating_base : bool,
                rng : th.Generator,
                always_present_collisions):
    found = False
    coll_counter = {}
    samples = 1000
    jp_dict = noncontrolled_jointpos
    dev = homing_pos.device
    for i in range(samples):
        normpos = (th.rand(size=(len(controlled_joints),), generator=rng, device = dev)*2-1)*initial_pose_randomization
        # initial_joint_pose = unnormalize(((npos)),limits_minmax[0],limits_minmax[1])                
        initial_joint_pose = ((normpos>=0)*((limits_minmax[1]-homing_pos)*normpos + homing_pos) + 
                                (normpos< 0)*((homing_pos-limits_minmax[0])*normpos + homing_pos))
        jp_dict.update({jn:initial_joint_pose[i] for i,jn in enumerate(controlled_joints)})
        robot_model.set_joint_pose_by_names({jn[1]:jp.cpu().numpy() for jn,jp in jp_dict.items()})
        if is_floating_base:
            robot_model.set_joint_pose_by_names({root_joint:homing_body_pose_xyzxyzw})
        collisions = robot_model.get_all_collisions()
        # all_link_poses = self._robot_model.get_frame_poses_xyzxyzw() #frames=self._robot_model.get_tree_frame_names_under_joint(self._configuration.robot_root_joint))
        # pprint.pprint(all_link_poses)
        # all_links_z = np.stack([pose[2] for pose in all_link_poses.values()])
        coll_counter.update({ln:coll_counter.get(ln,0)+1 for ln in collisions})                    
        if len(collisions) == 0: # and np.all(all_links_z>0):
            # ggLog.info(f"joint_pose = {self._robot_model.get_joint_pose()}")
            # ggLog.info(f"selected all_link_poses = {all_link_poses}")
            found = True
            initial_jpose = initial_joint_pose
            break
    if not found:
        initial_jpose = homing_pos
        coll_counter = {k:c/samples for k,c in coll_counter.items()}
        ggLog.warn(f"Failed to find initial joint configuration."
                    f" last collisions = {collisions}\n"
                    f" filtered collisions = {always_present_collisions}\n"
                    f" coll_ratio={coll_counter}")
    return initial_jpose
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
        fail_on_safety : bool
        frame_stack_length : int
        friction_randomized_links : tuple[tuple[str,str],...]
        friction_slide_spin_roll_rand_ratios : th.Tensor
        goal_err_exp_smoothing_1s : float
        ground_link : tuple[str,str]
        history_length : int
        homing_body_pose_xyz_xyzw : th.Tensor
        homing_ctrl_joints_pvesd : th.Tensor
        homing_nonctrl_joints_position : dict[tuple[str,str],th.Tensor]
        impulse_duration_minmax : th.Tensor
        impulse_mean_std : th.Tensor
        impulse_probability_per_sec : th.Tensor
        init_on_reset_ratio : float
        initial_pose_randomization : float
        joint_physical_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_damping : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_stiffness : dict[tuple[str,str],th.Tensor]
        longterm_stats_alpha : th.Tensor
        main_body_link : tuple[str,str]
        mass_randomization_ratios : th.Tensor
        mass_randomized_links : tuple[tuple[str,str],...]
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
        reward_penalties_max : th.Tensor
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
        stop_on_failure : bool
        th_device : th.device
        ui_camera_link : tuple[str,str]
        ui_camera_name : str
        ui_camera_resolution_hw : tuple[int,int]
        ui_rel_camera_pose_dist_pitch_yaw : th.Tensor
        vec_jimp_cmd_size : tuple[int,int,int]
        vec_size : int
        verbose_infos : bool
        enable_posref_safety : bool
        saturate_jimp_ref_limits : bool
        enable_limits_safety : bool
        posref_safety_period : float
        observe_full_robot_state : bool
        observe_body_vels_and_height : bool


    metadata = {'render.modes': ['rgb_array']}
    # STATE_BASE = "b" # component of the state that is a vector and is always the same regardless of the configuration
    STATE_ACT_PREPROC = "action"
    STATE_ACT_RAW = "action_raw"
    STATE_ROBOT = "robot"
    STATE_JOINT_STEP_STATS = "joint_step_stats"
    STATE_JOINT_LONGTERM_STATS = "joint_longterm_stats"
    STATE_EXTRINSIC = "extrinsic"
    STATE_INTERNAL = "internal"
    
    
    INTERNAL_FIELDS = IntEnum("INTERNAL_FIELDS", [  "SAFETY_TRIGGERED",
                                                    "STEP_COUNT",
                                                    "SIM_TIME"], start=0)

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

    JOINT_LONGTERM_STATS_FIELDS = IntEnum("LONGTERM_STATS_FIELDS", ["AVG_POS"])    
    
    joint_filters = {JOINT_FILTERS.ALL : lambda joint_name, robot_model: True,
                     JOINT_FILTERS.ALL_REVOLUTE : lambda joint_name, robot_model: robot_model.get_joint_properties([joint_name])[joint_name]["type"] == Robot.JOINT_TYPES.REVOLUTE}
    link_filters  = {LINK_FILTERS.ALL : lambda link_name, robot_model: True}

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
                        safety_limits_ratios_minmax_pve : float | tuple[float,float,float] | list[float] | th.Tensor | dict[tuple[str,str], th.Tensor | list[float] | tuple[float] | float], 
                        safe_limits_position_offset : dict[tuple[str,str], float],
                        seed : int,
                        stepLength_sec,
                        step_precision_tolerance : float,
                        stop_on_failure : bool,
                        fail_on_safety : bool,
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
                        ground_link : tuple[str,str],
                        ui_camera_resolution_hw : tuple[int,int] = (144,256),
                        enable_link_collisions : list[tuple[tuple[str,str],list[tuple[str,str]]]] | None = [],
                        mass_randomized_links : list[tuple[str,str]] = [],
                        mass_randomization_ratio : float = 0.1,
                        friction_randomized_links : list[tuple[str,str]] = [],
                        friction_slide_spin_roll_randomization_ratios : tuple[float, float, float] = (0.1,0.1,0.1),
                        longterm_states_decimation_time = 0.0001,
                        impulse_probability_per_sec : float = 0.0,
                        impulse_duration_minmax : tuple[float,float ]= (0.01, 5.0),
                        impulse_mean_std : tuple[float,float ]= (50.0, 50.0),
                        posref_safety_period = 0.001
                        ):
        self._main_seed = seed
        # self._rng_get_count = 0
        self._rng = th.Generator(device=th_device)
        self._rng.manual_seed(seed)
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._robot_model = Robot(model_urdf_string=robot_urdf_string)
        root_joint_name = self._robot_model.get_parent_joint(robot_root_link)
        is_floating = self._robot_model.get_joint_properties([root_joint_name])[root_joint_name]["type"] == Robot.JOINT_TYPES.FLOATING
        # self._build_new_instantaneous_state = th.vmap(self._build_new_instantaneous_state_single)
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

        self.link_filters[LINK_FILTERS.ALL_ROBOT] = lambda link_name, robot_model: link_name[0]==robot_name
        
        controlled_joints_rn : list[tuple[str,str]] = [(robot_name,jn) for jn in controlled_joints_str]
        phys_limits_minmax_pve = {(robot_name,k):self._thtens(l) 
                                    for k,l in self._robot_model.get_joint_limits(controlled_joints_str).items()}
        
        # if isinstance(safety_limits_ratios_minmax_pve,(float,int)):
        #     safety_limits_ratios_minmax_pve = [float(safety_limits_ratios_minmax_pve)]*3
        # if isinstance(safety_limits_ratios_minmax_pve, (list, tuple)):
        #     safety_limits_ratios_minmax_pve = self._thtens(safety_limits_ratios_minmax_pve)
        if isinstance(safety_limits_ratios_minmax_pve, dict):
            safety_limits_dict_ratios_minmax_pve = safety_limits_ratios_minmax_pve
        else:
            safety_limits_dict_ratios_minmax_pve = {k:safety_limits_ratios_minmax_pve for k in phys_limits_minmax_pve}
        safety_limits_ratios_minmax_pve_th = {k:self._thtens(v).expand((2,3,)) 
                                               for k,v in safety_limits_dict_ratios_minmax_pve.items()}
        # print(f"safety_limits_ratios_minmax_pve_th = {safety_limits_ratios_minmax_pve_th}")
        safe_limits_minmax_pve = {k: lim_minmax_pve*safety_limits_ratios_minmax_pve_th[k]
                                    for k,lim_minmax_pve in phys_limits_minmax_pve.items()}
        # print(f"safe_limits_minmax_pve = {safe_limits_minmax_pve}")
        safe_limits_minmax_pve = {jn:minmax_pve + self._thtens([safe_limits_position_offset[jn], 0, 0]).expand(2,3) for jn,minmax_pve in safe_limits_minmax_pve.items()}
        safe_limits_minmax_pve = {jn:lims.clamp(min=phys_limits_minmax_pve[jn][0].expand(2,-1), max=phys_limits_minmax_pve[jn][1].expand(2,-1)) 
                                  for jn,lims in safe_limits_minmax_pve.items()}
        # safe_limits_minmax_pve = {k:(lims_minmax-0.5*(lims_minmax[1]+lims_minmax[0]))*safety_limits_ratios_minmax_pve+(0.5+safety_limits_normalized_offset)*(lims_minmax[1]+lims_minmax[0])
        #                             for k,lims_minmax in phys_limits_minmax_pve.items()}
        # if safety_limits_normalized_offset + safety_limits_ratios_minmax_pve >= 1:
        #     raise RuntimeError(f"safety_limits_factor={safety_limits_ratios_minmax_pve} and safety_limits_normalized_offset={safety_limits_normalized_offset} exceed physical limits")

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
                                                    enable_dbg_checks = enable_dbg_checks,
                                                    fail_on_safety = fail_on_safety,
                                                    frame_stack_length = frame_stack_length,
                                                    friction_randomized_links=None, # Will fill up later
                                                    friction_slide_spin_roll_rand_ratios = None, # Will fill up later
                                                    goal_err_exp_smoothing_1s = goal_err_exp_smoothing_1s,
                                                    ground_link=ground_link,
                                                    history_length = max(2,frame_stack_length),
                                                    homing_body_pose_xyz_xyzw = self._thtens(homing_body_pose_xyz_xyzw),
                                                    homing_ctrl_joints_pvesd = homing_ctrl_joints_pvesd,
                                                    homing_nonctrl_joints_position = homing_nonctrl_joints_position,
                                                    impulse_duration_minmax = self._thtens(impulse_duration_minmax),
                                                    impulse_mean_std=self._thtens(impulse_mean_std),
                                                    impulse_probability_per_sec = self._thtens(impulse_probability_per_sec),
                                                    init_on_reset_ratio=init_on_reset_ratio,
                                                    initial_pose_randomization = initial_pose_randomization,
                                                    joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                    joint_safe_limits_minmax_damping = minmax_damping_thdict,
                                                    joint_safe_limits_minmax_pve = safe_limits_minmax_pve,
                                                    joint_safe_limits_minmax_stiffness = minmax_stiffness_thdict,
                                                    longterm_stats_alpha = self._thtens(0.1**(stepLength_sec/longterm_states_decimation_time)), # alpha so that the contribution of a sample longterm_states_decimation_time seconds ago is 0.1
                                                    main_body_link=(robot_name,robot_main_body_link),
                                                    mass_randomization_ratios = None, # Will fill up later
                                                    mass_randomized_links=None, # Will fill up later
                                                    model_urdf_string=robot_urdf_string,
                                                    noise_angvel_ep_mustdstd =  self._thtens(obs_noise_angvel_ep_mustd_step_std),
                                                    noise_gravity_ep_mustdstd = self._thtens(obs_noise_gravity_ep_mustd_step_std),
                                                    noise_joints_pve_mustdstd = self._thtens(obs_noise_joints_pve_ep_mustd_step_std),
                                                    noise_linvel_ep_mustdstd =  self._thtens(obs_noise_linvel_ep_mustd_step_std),
                                                    noise_posz_ep_mustdstd =    self._thtens(obs_noise_posz_ep_mustd_step_std),
                                                    obs_dtype = self._obs_dtype,
                                                    observe_body_state = observe_body_velocity,
                                                    original_max_epsteps = maxStepsPerEpisode,
                                                    quiet=quiet,
                                                    real = False,
                                                    reward_penalties_max = self._thtens(100.0),
                                                    robot_is_floating = is_floating,
                                                    robot_name = robot_name,
                                                    robot_root_joint = root_joint_name,
                                                    robot_root_link=(robot_name,robot_root_link),
                                                    safe_damping = safe_damping,
                                                    safe_stiffness = safe_stiffness,
                                                    seed = seed,
                                                    show_goal = True,
                                                    spawn_root_pose_xyz_xyzw = (0,0,0,0,0,0,1),
                                                    stepLength_sec = stepLength_sec,
                                                    stop_on_failure = stop_on_failure,
                                                    th_device = th_device,
                                                    ui_camera_link = ("simple_camera", "simple_camera_link"),
                                                    ui_camera_name="simple_camera",
                                                    ui_camera_resolution_hw = ui_camera_resolution_hw,
                                                    ui_rel_camera_pose_dist_pitch_yaw = self._thtens([2.5, 30/180*3.14159, -90/180*3.14159]),
                                                    vec_jimp_cmd_size=(adapter.vec_size(), len(controlled_joints_rn), 5),
                                                    vec_size=adapter.vec_size(),
                                                    verbose_infos = verbose_infos,
                                                    enable_posref_safety = True,
                                                    enable_limits_safety = True,
                                                    saturate_jimp_ref_limits = False,
                                                    posref_safety_period = posref_safety_period,
                                                    observe_full_robot_state = False,
                                                    observe_body_vels_and_height=False
                                                    )
        self._current_episode_config = RobotVecEnv.EpisodeConfiguration(
                                                    vec_initial_ctrl_joint_pose = self._configuration.homing_ctrl_joints_pvesd[:,0].expand(adapter.vec_size(), len(self._configuration.controlled_joints)).clone(),
                                                    vec_init_on_reset = th.ones(size=(adapter.vec_size(),), dtype=th.bool).to(device=th_device, non_blocking=th_device.type=="cuda"),
                                                    vec_max_ep_steps = th.full(fill_value=maxStepsPerEpisode, size=(adapter.vec_size(),), dtype=th.int64).to(device=th_device, non_blocking=th_device.type=="cuda"))
        
        self._last_sent_v_j_pvesd = homing_ctrl_joints_pvesd.repeat(adapter.vec_size(), 1, 1)
        self._always_present_collisions : set[tuple[str,str]] = set()
        self._safe_limits_minmax_j_pve = th.stack([safe_limits_minmax_pve[jn] for jn in controlled_joints_rn], dim=1)
        self._posref_safety_minmmax_diff = self._safe_limits_minmax_j_pve[:,:,1]*self._configuration.posref_safety_period
        self._posref_saturation_minmmax_diff = self._posref_safety_minmmax_diff*0.999
        self._impulse_disturbances_enabled = impulse_probability_per_sec > 0

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
        ggLog.info(f"self._state_helper.observation_names() = {self._state_helper.observation_names()}")
        self._current_state = self._state_helper.reset_state()
        self._last_obs = self._state_helper.observe(self._current_state)
        self._eps_start_stime = self._thzeros(size=(adapter.vec_size(),))
        self._safety_limits = self._state_helper.sub_helpers[self.STATE_ROBOT].build_robot_limits(
                                                    joint_limit_minmax_pve=self._configuration.joint_safe_limits_minmax_pve,
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
        # preallocate some things
        self._no_envs = th.zeros((self.num_envs,), dtype=th.bool, device=self._configuration.th_device)
        self._all_envs = th.zeros((self.num_envs,), dtype=th.bool, device=self._configuration.th_device)

        # Randomizations
        mass_randomized_links = self._expand_link_filters(mass_randomized_links)
        friction_randomized_links = self._expand_link_filters(friction_randomized_links)        
        self._configuration.mass_randomized_links = tuple(mass_randomized_links)
        self._configuration.mass_randomization_ratios = self._thtens(mass_randomization_ratio).expand((len(mass_randomized_links),))
        self._configuration.friction_randomized_links = tuple(friction_randomized_links)
        self._configuration.friction_slide_spin_roll_rand_ratios = self._thtens(friction_slide_spin_roll_randomization_ratios).expand((len(friction_randomized_links),3))
        self._mass_randomized_link_ids = self._adapter.get_links_ids(self._configuration.mass_randomized_links)
        self._friction_randomized_link_ids = self._adapter.get_links_ids(self._configuration.friction_randomized_links)
        self._abs_gravity_dir = self._thtens([0.0,0.0,-1.0])





        ggLog.info(f"enable_link_collisions = {enable_link_collisions}")
        if isinstance(self._adapter, BaseVecSimulationAdapter) and enable_link_collisions is not None:
            self._adapter.set_body_collisions(enable_link_collisions)
        ggLog.info(f"Built scenario")
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self._current_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__
        ggLog.info(f"Built info helper")
        # ggLog.info(f"get_space_labels(self.single_observation_space) = {get_space_labels(self.single_observation_space)}")

        self.set_seeds(th.as_tensor(seed))
        self._adapter.set_monitored_links([self._configuration.main_body_link])
        self._adapter.startup()
        self.initialize_episodes()
        


    def _expand_link_filters(self, links : Sequence[tuple[str,str]]) -> list[tuple[str,str]]:
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
                raise RuntimeError(f"Unexpected mass-randomized link {l} of type {type(l)} (self.link_filters = {self.link_filters})")
        return actual_links
    # @property
    # def _rng(self):
    #     import traceback
    #     ggLog.info(f"Getting rng {self._rng_get_count} {hash_tensor(self._rng_v.get_state())} at {''.join(traceback.format_list(traceback.extract_stack(limit=3)))}")
    #     self._rng_get_count += 1
    #     return self._rng_v

    def _build_stats(self):
        self._stats = {}

    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        if self._configuration.observe_full_robot_state:
            observable_robot_state = ["pos","vel","cmdeff","refpos","refvel","refeff","stiff","damp"] 
        else:
            observable_robot_state = ["pos","refpos","stiff","damp"] 
        robot_state_helper = RobotStateHelper(joint_limit_minmax_pveae=self._configuration.joint_physical_limits_minmax_pve,
                                              stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                              damping_minmax=self._configuration.joint_safe_limits_minmax_damping,
                                              obs_dtype=self._configuration.obs_dtype,
                                              th_device=self._configuration.th_device,
                                              history_length=self._configuration.history_length,
                                              obs_history_length = self._configuration.frame_stack_length,
                                              vec_size=adapter.vec_size(),
                                              observable_subfields = observable_robot_state)
        joint_step_stats_state_helper = RobotStatsStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                                        obs_dtype=self._configuration.obs_dtype,
                                                        th_device=self._configuration.th_device,
                                                        vec_size=adapter.vec_size())
        joint_longterm_stats_helper = ThBoxStateHelper( field_names=[e for e in self.JOINT_LONGTERM_STATS_FIELDS],
                                                        obs_dtype=self._obs_dtype,
                                                        th_device=self._th_device,
                                                        field_size=(len(self._configuration.joint_physical_limits_minmax_pve),),
                                                        fields_minmax={self.JOINT_LONGTERM_STATS_FIELDS.AVG_POS : 
                                                                       th.stack([minmax_pve[:,0]
                                                                                  for minmax_pve in self._configuration.joint_physical_limits_minmax_pve.values()],
                                                                                dim = 1)},
                                                        observable_fields=None,
                                                        vec_size=adapter.vec_size())
        internal_state_helper =   ThBoxStateHelper( field_names=[e for e in self.INTERNAL_FIELDS],
                                                    obs_dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(1,),
                                                    fields_minmax={   self.INTERNAL_FIELDS.SAFETY_TRIGGERED : [0,1000],
                                                                        self.INTERNAL_FIELDS.STEP_COUNT : [-1,1000_000],
                                                                        self.INTERNAL_FIELDS.SIM_TIME : [-1,1000_000]},
                                                    observable_fields=[self.INTERNAL_FIELDS.SAFETY_TRIGGERED],
                                                    vec_size=adapter.vec_size())
        if self._configuration.observe_body_vels_and_height:
            extrinsic_observable_fields = [
                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X,
                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y,
                                            self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z,
                                            self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X,
                                            self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y,
                                            self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z,
                                            self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z,
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z
                                            ]
        else:
            extrinsic_observable_fields = [
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X,
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y,
                                            self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z
                                            ]

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
                                                    observable_fields=extrinsic_observable_fields,
                                                    history_length=self._configuration.history_length,
                                                    obs_history_length = self._configuration.frame_stack_length,
                                                    vec_size=adapter.vec_size())
        act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                    obs_dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(self._action_helper.single_action_len(),),
                                                    fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                    history_length=3,
                                                    obs_history_length=3,
                                                    vec_size=adapter.vec_size(),
                                                    flatten_observation=True)
        raw_act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                                        obs_dtype=self._obs_dtype,
                                                        th_device=self._th_device,
                                                        field_size=(self._action_helper.single_action_len(),),
                                                        fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                                        history_length=3,
                                                        vec_size=adapter.vec_size(),
                                                        flatten_observation=True)
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
            observable_substates = [self.STATE_ROBOT,
                                    self.STATE_EXTRINSIC,
                                    self.STATE_INTERNAL,
                                    self.STATE_ACT_PREPROC,
                                    # self.STATE_ACT_RAW,
                                    self.STATE_JOINT_LONGTERM_STATS
                                    ]
        else:
            observable_substates = [self.STATE_ROBOT,
                                    self.STATE_INTERNAL,
                                    self.STATE_ACT_PREPROC,
                                    # self.STATE_ACT_RAW,
                                    self.STATE_JOINT_LONGTERM_STATS]
        statehelpers : dict[str,ThBoxStateHelper] = {self.STATE_ROBOT : robot_state_helper,
                        self.STATE_JOINT_STEP_STATS : joint_step_stats_state_helper,
                        self.STATE_EXTRINSIC : extrinsic_state_helper,
                        self.STATE_INTERNAL : internal_state_helper,
                        self.STATE_ACT_PREPROC: act_history_state_helper,
                        self.STATE_ACT_RAW : raw_act_history_state_helper,
                        self.STATE_JOINT_LONGTERM_STATS : joint_longterm_stats_helper}
        # ggLog.info("\n".join([f"{k} : state={s._state_space.shape}  obs ={s._obs_space.shape}" for k,s in statehelpers.items()]))

        self._state_helper = DictStateHelper(statehelpers,
                                              observable_fields=observable_substates,
                                              noise = {
                                                    self.STATE_ROBOT : robot_state_noise,
                                                    self.STATE_EXTRINSIC : extrinsic_state_noise},
                                              flatten_in_obs=[  self.STATE_ROBOT,
                                                                self.STATE_EXTRINSIC,
                                                                self.STATE_INTERNAL,
                                                                # self.STATE_ACT_RAW,
                                                                self.STATE_ACT_PREPROC,
                                                                self.STATE_JOINT_LONGTERM_STATS],
                                              flattened_part_name="vec")        

    
    # --------------------------------------------------------------------------------------------------------------------
    # Action
    # --------------------------------------------------------------------------------------------------------------------

    # @th.jit.script
    def _preproc_acts(self, actions : th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        dt = self._configuration.stepLength_sec
        alpha = self._configuration.action_exp_smoothing_1s**(dt/1)
        prev_actions = self._current_state[self.STATE_ACT_PREPROC][:,0,self.ACT_FIELDS.ACTION].detach().to(device=self._configuration.th_device)
        actions = actions*(1-alpha) + prev_actions*alpha
        actions = th.clamp(actions, min=-1, max=1)
        n = self._thrandn(size=(self._adapter.vec_size(),))
        action_delay = th.clamp(self._configuration.action_delay_mustd[0] + self._configuration.action_delay_mustd[1]*n, min = 0.0)
        return actions, action_delay

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        with th.no_grad():
            actions = self._thtens(actions).detach()
            dbg_check_size(actions, (self._adapter.vec_size(), self._action_helper.single_action_len()))
            self._last_raw_actions = actions
            actions, action_delay = self._preproc_acts(actions)
            self._last_preprocessed_actions = actions
            v_j_pvesd = self._action_helper.action_to_pvesd(actions)
            # do this better, avoid this if condition, put it in the helper
            if self._configuration.saturate_jimp_ref_limits:
                v_j_pvesd[:,:,:3] = th.clamp(v_j_pvesd[:,:,:3], min=self._safe_limits_minmax_j_pve[0], max=self._safe_limits_minmax_j_pve[1])
                posref_diff = v_j_pvesd[:,:,0] - self._last_sent_v_j_pvesd[:,:,0]
                posref_diff = th.clamp(posref_diff, min=self._posref_saturation_minmmax_diff[0], max=self._posref_saturation_minmmax_diff[1])
                v_j_pvesd[:,:,0] = self._last_sent_v_j_pvesd[:,:,0] + posref_diff
                # ggLog.info(f"prev_posref: {self._last_sent_v_j_pvesd[:,:,0]}")
                # ggLog.info(f"new_posref:  {v_j_pvesd[:,:,0]}")
                # ggLog.info(f"posref_diff: {posref_diff}")
            if self._configuration.control_mode in [JointImpedanceActionHelper.CONTROL_MODES.POSITION, JointImpedanceActionHelper.CONTROL_MODES.POSITION_AND_STIFFNESS, JointImpedanceActionHelper.CONTROL_MODES.POSITION_AND_TORQUES] :
                v_j_pvesd[:,:,1] = th.clamp((v_j_pvesd[:,:,0] - self._last_sent_v_j_pvesd[:,:,0])/self._intendedStepLength_sec, 
                                            min=self._safe_limits_minmax_j_pve[0,:,1], 
                                            max=self._safe_limits_minmax_j_pve[1,:,1]) # set velocity reference

            self._last_sent_v_j_pvesd = v_j_pvesd
            # ggLog.info(f"sending jimp: {self._last_sent_v_j_pvesd}")
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
            arrow_spawn_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/red_arrow.urdf.xacro")).read_text(),
                                            name="arrow",
                                            pose=arrow_pose,
                                            format="urdf.xacro",
                                            kwargs={"add_world_link":str(is_pybullet)})
            axes_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/axes.urdf.xacro")).read_text(),
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
            vec_mask = th.ones((self.num_envs,), dtype=th.bool).to(device=self._th_device, non_blocking=self._th_device.type=="cuda")
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            self._adapter.reset_model_alterations(vec_mask)
        # ggLog.info(f"initializing episodes {vec_mask}")
        resetted_state = self._state_helper.reset_state()
        # ggLog.info(f"resetted_state = {resetted_state}")
        map2_tensor_tree(self._current_state, resetted_state,
                        lambda l1, l2: masked_assign(l1, vec_mask, l2)) # should not be necessary, just for safety
        self._current_state[self.STATE_INTERNAL][vec_mask,0,self.INTERNAL_FIELDS.STEP_COUNT] = th.tensor(-1.) # all other fields will be overwritten accordingly in state_update
        masked_assign(self._eps_start_stime, vec_mask, self._adapter.getEnvTimeFromStartup())
        self._last_obs = self._state_helper.observe(self._current_state)
        self._set_current_ep_config(reset_options = options, vec_mask=vec_mask)
        
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._simulation_initialization(vec_mask=vec_mask)
        else:
            self._realworld_initialization(vec_mask=vec_mask)
        self._last_preprocessed_actions = th.clamp(self._action_helper.pvesd_to_action(self._last_sent_v_j_pvesd), min=-1, max=1)
        self._last_raw_actions = self._last_preprocessed_actions

        # ggLog.info(f"initial action {self._last_out_action}, pvesd = {self._last_sent_pvesd}")

        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            # ggLog.info(f"self._mass_randomized_link_ids = {self._mass_randomized_link_ids}")
            self._adapter.alter_model_rel(  link_masses = ( self._mass_randomized_link_ids,
                                                            (self._thrand(size=(self.num_envs, len(self._configuration.mass_randomization_ratios)))*2-1)*self._configuration.mass_randomization_ratios),
                                            link_frictions = (self._friction_randomized_link_ids,
                                                              (self._thrand(size=(self.num_envs,)+self._configuration.friction_slide_spin_roll_rand_ratios.size())*2-1)*self._configuration.friction_slide_spin_roll_rand_ratios))

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
        t0 = time.monotonic()
        if self._configuration.initial_pose_randomization > 0:
            initial_jposes_list : list[th.Tensor] = [None]*selected_vecs_num #type: ignore
            for v in range(selected_vecs_num): # TODO: this may be sloooooow, can I parallelize it?
                initial_jposes_list[v] = find_pose(  root_joint = self._configuration.robot_root_joint,
                                        homing_body_pose_xyzxyzw = self._pinocchio_corrected_homing_body_pose_xyzxyzw,
                                        controlled_joints = self._configuration.controlled_joints,
                                        initial_pose_randomization = self._configuration.initial_pose_randomization,
                                        limits_minmax = th.stack([self._configuration.joint_safe_limits_minmax_pve[jn][:,0] for jn in self._configuration.controlled_joints], dim = 1),
                                        homing_pos = self._configuration.homing_ctrl_joints_pvesd[:,0],
                                        noncontrolled_jointpos = {k:v for k,v in self._configuration.homing_nonctrl_joints_position.items()},
                                        robot_model = self._robot_model,
                                        is_floating_base = self._configuration.robot_is_floating,
                                        rng = self._rng,
                                        always_present_collisions = self._always_present_collisions)
            initial_jposes = th.stack(initial_jposes_list).to(device=self._configuration.th_device, non_blocking=True)
        else:
            initial_jposes = homing_pos.expand(selected_vecs_num, len(self._configuration.controlled_joints))
        if  self._configuration.init_on_reset_ratio<1.0 and self._init_counter_since_reset>1:
            vec_init_on_reset = self._thrand((selected_vecs_num,)) < self._configuration.init_on_reset_ratio
        else:
            vec_init_on_reset = th.ones((selected_vecs_num,), dtype=th.bool).to(device=self._th_device, non_blocking=self._th_device.type=="cuda")
        ggLog.info(f"pose randomization took {time.monotonic()-t0:.6f}s")
        # ggLog.info(f"initial_jpose = {initial_joint_pose}, homing = {homing}")
        self._robot_model.set_collision_pairs(original_collision_pairs)
        masked_assign(self._current_episode_config.vec_initial_ctrl_joint_pose, vec_mask, initial_jposes)
        masked_assign(self._current_episode_config.vec_init_on_reset, vec_mask, vec_init_on_reset)
        masked_assign(self._current_episode_config.vec_max_ep_steps, vec_mask, maxStepsPerEpisode)
        self.set_max_episode_steps(self._current_episode_config.vec_max_ep_steps)
        # ggLog.info(f"_current_episode_config = {self._current_episode_config}")


    def _realworld_robot_init_move(self, vec_mask : th.Tensor):
        if isinstance_noimport(self._adapter,"VecRosXBotAdapterWrapper"):
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
            ggLog.info(f"Moving robot...")
            try:
                self._adapter.moveToJointPoseSync(  joint_names = self._configuration.controlled_joints,
                                                    positions = initial_cmd_vec_j_pvesd[:,:,0],
                                                    velocity_scaling = 0.1,
                                                    acceleration_scaling = 0.1,
                                                    joint_position_tolerance = 0.01,
                                                    max_time_s = 60)
                ggLog.info(f"Moved robot.")
            except adarl.utils.utils.MoveFailError as e:
                ggLog.warn(f"Timed out reaching position: {adarl.utils.utils.exc_to_str(e)}")

            time.sleep(1)
            self._adapter.setJointsImpedanceCommand(initial_cmd_vec_j_pvesd, vec_mask=vec_mask)
            time.sleep(1)
            self._adapter.set_current_joint_impedance_command(initial_cmd_vec_j_pvesd, vec_mask=vec_mask)
            masked_assign(self._last_sent_v_j_pvesd, vec_mask, initial_cmd_vec_j_pvesd)
        else:
            raise NotImplementedError(f"Unsupported real-world initialization for adapter of type '{type(self._adapter)}'")

    def _realworld_initialization(self, vec_mask : th.Tensor):
        while True:
            print(f"Episode Initialization:\n"
                  f"Will move the robot joints into the homing pose and set the initial joint impedance command.")
            r = input("Enter 'move' to move the robot or 'skip' to skip the robot pose initialization > ")
            if r == "move":
                self._realworld_robot_init_move(vec_mask)
            elif r == "skip":
                masked_assign(self._last_sent_v_j_pvesd, vec_mask, self._adapter.get_current_joint_impedance_command())
                pass
            else:
                print(f"Invalid answer '{r}'")
                continue
            
            r = input("Please ensure the robot is in a suitable pose and type 'start' to start episode > ")
            if r == "start":
                return
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
            elif adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("RosXbotGazeboAdapter")):
                self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path( "adarl_envs",
                                                                                                            "gazebo/all_gazebo_xbot.launch"),
                                            launch_file_args={"gui":"false"})
                self._arrow_base = ("arrow","arrow_link")
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
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        
        self._main_body_link_ids = self._adapter.get_links_ids([self._configuration.main_body_link])
        self._controlled_joints_ids = self._adapter.get_joints_ids(self._configuration.controlled_joints)


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
                ggLog.info(f"_corrected_homing_body_pose_xyzxyzw = {self._pinocchio_corrected_homing_body_pose_xyzxyzw}")
        ggLog.info(f"Detecting always present self collisions...")
        self._robot_model.disable_tree_self_collisions(root_frame=self._configuration.robot_root_link[1])
        # self._robot_model.remove_collision_pairs([("rail_link_0","slider_link_0")])            
        self._ground_co_id = self._robot_model.add_collision_box(   pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
                                                                    collision_box_size_xyz=(100,100,1),
                                                                    collision_obj_id="ground_collision")
        self._always_present_collisions : set[tuple[str,str]] = self._robot_model.detect_always_present_collisions(
            moving_joints=[jn[1] for jn in self._configuration.controlled_joints],
            fixed_joints_pose={self._configuration.robot_root_joint : self._pinocchio_corrected_homing_body_pose_xyzxyzw}
                                            if self._configuration.robot_is_floating else {},
            samples=1000,
            threshold=1.0)
        ggLog.info(f"Always present self collisions = {pprint.pformat(self._always_present_collisions)}")
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
        if self._configuration.enable_dbg_checks:
            if isinstance(self._adapter, BaseVecSimulationAdapter) and not adarl.utils.tensor_trees.is_all_finite(state):
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
    def pre_step(self):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            if self._impulse_disturbances_enabled:
                impulse_prob_per_env_dt = 1-th.pow(1-self._configuration.impulse_probability_per_sec, self._intendedStepLength_sec)
                apply_impulse = self._thrand((self.num_envs,1)) < impulse_prob_per_env_dt
                impulse = self._thrand_truncnorm((self.num_envs,1),
                                                mean = self._configuration.impulse_mean_std[0].item(),
                                                std = self._configuration.impulse_mean_std[1].item(),
                                                min_val = 0.0,
                                                max_val = self._configuration.impulse_mean_std[0].item()+self._configuration.impulse_mean_std[1].item()*5)
                duration = self._thrand((self.num_envs,1)) * (self._configuration.impulse_duration_minmax[1] - self._configuration.impulse_duration_minmax[0]) + self._configuration.impulse_duration_minmax[0]

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
                #            f"impulse_mean = {self._configuration.impulse_mean_std[0].item()}\n"
                #            f"impulse_std = {self._configuration.impulse_mean_std[1].item()}\n"
                #            f"impulse = {impulse}\n"
                #            f"force_torque_xyzxyz={th.cat([forcevector, torque], dim = 1)}\n"
                #            f"durations={duration}\n"
                #            f"delays={delays}\n"
                #            f"vec_mask={apply_impulse}\n")
                self._adapter.set_link_impulses(self._main_body_link_ids,
                                                force_torque_xyzxyz=th.cat([forcevector, torque], dim = 1).view((self.num_envs,1,6)),
                                                durations=duration.view((self.num_envs,1)),
                                                delays=delays.view((self.num_envs,1)),
                                                vec_mask=apply_impulse.view((self.num_envs,)))

    @override
    def post_step(self):
        # t0 = time.monotonic()
        self._update_state()
        # t1 = time.monotonic()
        self._update_stats()
        # tf = time.monotonic()
        # ggLog.info(f"update_state: {t1-t0} update_stats: {tf-t1}")
        # ggLog.info(f"on_step(): {self._current_state[self.STATE_ROBOT][0,0]}")


    def _get_new_instantaneous_state(self):
        # ggLog.info(f"_stepCounter = {self._stepCounter}")
        # t0 = time.monotonic()
        if isinstance_noimport(self._adapter, "MjxAdapter"):
            jstates_v_j_pveae = self._adapter.getExtendedJointsState(requestedJoints=self._controlled_joints_ids)
        else:
            jstates_v_j_pve = self._adapter.getJointsState(requestedJoints=self._controlled_joints_ids)
            jstates_v_j_pveae = th.cat([jstates_v_j_pve, th.zeros(jstates_v_j_pve.shape[:2]+(2,), dtype=jstates_v_j_pve.dtype, device=jstates_v_j_pve.device)], dim = -1)
        # ggLog.info(f"jstates_v_j_pve = {jstates_v_j_pve}")
        # th.cuda.synchronize()
        # t1 = time.monotonic()
        bstates_v_13 = self._adapter.getLinksState(requestedLinks = self._main_body_link_ids, use_com_pose = False)[:,0,:]
        # ggLog.info(f"axes pose = {self._adapter.getLinksState(requestedLinks = self._adapter.get_links_ids([('axes','root')]), use_com_pose = False)[:,0,:]}")
        # ggLog.info(f"bstates_v_13 = {bstates_v_13}")
        # th.cuda.synchronize()
        # t2 = time.monotonic()
        internal_state = self._current_state[self.STATE_INTERNAL][:,0]
        try:
            vec_stats_minmaxavgstd_j_pvae = self._adapter.get_joints_state_step_stats()
        except NotImplementedError:
            vec_stats_minmaxavgstd_j_pvae = self._thzeros((self.num_envs,4,len(self._configuration.controlled_joints),4))*float("nan")
        # th.cuda.synchronize()
        # t3 = time.monotonic()
        # ggLog.info(f"vec_stats_minmaxavgstd_j_pvae = {vec_stats_minmaxavgstd_j_pvae}")
        # ggLog.info(f"bstates_v_13 = {bstates_v_13}")
        # ggLog.info(f"internal_states = {internal_states}")
        # ggLog.info(f"jstates_v_j_pve.device = {jstates_v_j_pve.device}")
        # ggLog.info(f"self._last_sent_v_j_pvesd.device = {self._last_sent_v_j_pvesd.device}")
        dbg_check(lambda: th.all(th.isfinite(vec_stats_minmaxavgstd_j_pvae)),
                  lambda: f"non finite values in joint stats at {th.logical_not(th.isfinite(vec_stats_minmaxavgstd_j_pvae)).nonzero()}: {vec_stats_minmaxavgstd_j_pvae[th.logical_not(th.isfinite(vec_stats_minmaxavgstd_j_pvae))]} : {vec_stats_minmaxavgstd_j_pvae}",
                  just_warn=True)
        dbg_check(lambda: th.all(th.isfinite(bstates_v_13)),
                  lambda: f"non finite values in body link state at {th.logical_not(th.isfinite(bstates_v_13)).nonzero()}: {bstates_v_13[th.logical_not(th.isfinite(bstates_v_13))]} : {bstates_v_13}",
                  just_warn=True)
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter") and not th.logical_or(th.all(th.isfinite(vec_stats_minmaxavgstd_j_pvae)), th.all(th.isfinite(bstates_v_13))):
            bad_sim_id = th.logical_not(th.isfinite(bstates_v_13)).nonzero()[0,0].item()
            import jax.numpy as jnp
            ggLog.info(f"diverging sim {bad_sim_id}:\n"
                       f" model.geom_friction = {self._adapter._sim_state.mjx_model.geom_friction[bad_sim_id]} (avg = {jnp.mean(self._adapter._sim_state.mjx_model.geom_friction, axis=0)})\n"
                       f" model.body_mass = {self._adapter._sim_state.mjx_model.body_mass[bad_sim_id]} (avg = {jnp.mean(self._adapter._sim_state.mjx_model.body_mass, axis=0)})")
        # bstates_v_13 = th.zeros(size=(1,13), dtype=th.float32, device=self._adapter._th_device)

        if isinstance(self._adapter, BaseVecSimulationAdapter):
            body_abs_linvel_xyz_vec = bstates_v_13[:,7:10]
            body_abs_angvel_xyz_vec = bstates_v_13[:,10:13]
            conj_body_abs_quat_xyzw_vec = th_quat_conj(bstates_v_13[:,3:7])
            vec_body_ground_dist = bstates_v_13[:,2]
            vec_body_rel_gravity_dir = th_quat_rotate_py(self._abs_gravity_dir.expand_as(body_abs_linvel_xyz_vec), conj_body_abs_quat_xyzw_vec)
            vec_body_rel_linvel_xyz = th_quat_rotate_py(body_abs_linvel_xyz_vec, conj_body_abs_quat_xyzw_vec)
            vec_body_rel_angvel_xyz = th_quat_rotate_py(body_abs_angvel_xyz_vec, conj_body_abs_quat_xyzw_vec)
        else:
            vec_body_rel_gravity_dir = self._adapter.get_link_gravity_direction(self._main_body_link_ids)[:,0,:]
            body_abs_linvel_xyz_vec = th.zeros_like(vec_body_rel_gravity_dir)
            vec_body_rel_linvel_xyz = th.zeros_like(vec_body_rel_gravity_dir)
            vec_body_rel_angvel_xyz = th.zeros_like(vec_body_rel_gravity_dir)
            vec_body_ground_dist = th.zeros_like(vec_body_rel_gravity_dir[:,0])
            # - vec_ground_dist can be obtained with some simple sensor, a depth camera, something like that
            # - gravity_dir with an accelerometer
            # - rel_linvel e rel_angvel, may be obtainable with some slam kind of thing, even something quite simple that tracks local features
            #     maybe something like these: 
            #         https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam
            #         https://wiki.ros.org/orb_slam2_ros
            # raise NotImplementedError("")
        
        robot_state = self._current_state[self.STATE_ROBOT]
        new_inst_state = self._build_new_instantaneous_state_vec(   
                                    internal_state,
                                    vec_stats_minmaxavgstd_j_pvae,
                                    jstates_v_j_pveae,
                                    self._last_sent_v_j_pvesd,
                                    body_abs_linvel_xyz_vec, # only used for visualization, can be wrong
                                    vec_body_ground_dist,
                                    vec_body_rel_gravity_dir,
                                    vec_body_rel_linvel_xyz,
                                    vec_body_rel_angvel_xyz,
                                    robot_state)
        # ggLog.info(f"insta_state sizes = "+str(map_tensor_tree(new_inst_state,lambda t: t.size())))
        new_inst_state[self.STATE_ACT_PREPROC] = {self.ACT_FIELDS.ACTION : self._last_preprocessed_actions}
        new_inst_state[self.STATE_ACT_RAW] = {self.ACT_FIELDS.ACTION : self._last_raw_actions}
        # th.cuda.synchronize()
        # t4 = time.monotonic()
        # if not th.all(th.isfinite(new_inst_state[self.STATE_ROBOT_STATS])):
        #     ggLog.info(f"nonfinite vals in new_robot_stats_state = {new_inst_state[self.STATE_ROBOT_STATS]}")
        dbg_check(lambda: th.all(new_inst_state[self.STATE_ROBOT][:,:,8:10]>=0), lambda: f"negative gains in new_robot_state") #type: ignore
        # th.cuda.synchronize()
        # t5 = time.monotonic()
        # ggLog.info(f"getJoints={t1-t0:.6f} getlinks={t2-t1:.6f} getstats={t3-t2:.6f} build={t4-t3:.6f} check={t5-t4:.6f} tot={t5-t0}")
        return new_inst_state


    def _build_new_instantaneous_state_vec(self,    vec_internal_state : th.Tensor,
                                                    vec_stats_minmaxavgstd_j_pvae : th.Tensor,
                                                    vec_jstates_j_pveae : th.Tensor,
                                                    vec_last_sent_j_pvesd : th.Tensor,
                                                    vec_body_abs_linvel_xyz : th.Tensor,
                                                    vec_body_ground_dist : th.Tensor,
                                                    vec_body_rel_gravity_dir : th.Tensor,
                                                    vec_body_rel_linvel_xyz : th.Tensor,
                                                    vec_body_rel_angvel_xyz : th.Tensor,
                                                    vec_robot_state : th.Tensor):


        vec_step_count = vec_internal_state[:,self.INTERNAL_FIELDS.STEP_COUNT]
        vec_safety_state = vec_internal_state[:,self.INTERNAL_FIELDS.SAFETY_TRIGGERED].view((self.num_envs,))
        # vec_prev_safety_triggered = vec_internal_state[:,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        # ggLog.info(f"stats_minmaxavgstd_j_pvae.device = {stats_minmaxavgstd_j_pvae.device}   self._safe_limits_minmax_j_pve[0].device = {self._safe_limits_minmax_j_pve[0].device}")
        pveidx = th.as_tensor([0,1,3]).to(device=vec_stats_minmaxavgstd_j_pvae.device, non_blocking=True)
        if self._configuration.enable_limits_safety:
            vec_triggered_limits = th.logical_or(   vec_stats_minmaxavgstd_j_pvae[:, 0, :, pveidx] < self._safe_limits_minmax_j_pve[0],
                                                    vec_stats_minmaxavgstd_j_pvae[:, 1, :, pveidx] > self._safe_limits_minmax_j_pve[1])
            vec_limits_safety_triggered = th.any(vec_triggered_limits, dim=(1,2))
            # vec_safety_triggered = th.logical_or(vec_limits_safety_triggered, vec_prev_safety_triggered)
            newly_triggered = th.logical_and(vec_limits_safety_triggered,
                                             th.logical_not((vec_safety_state>0)))
            vec_safety_state = th.where(newly_triggered,
                                        10.0, # 10 means safety triggered by limits
                                        vec_safety_state)
        if self._configuration.enable_posref_safety:
            posref_diff = vec_last_sent_j_pvesd[:,:,0] - vec_robot_state[:,0,:,5]
            posref_safety_triggered = th.logical_or(posref_diff < self._posref_safety_minmmax_diff[0],
                                                    posref_diff > self._posref_safety_minmmax_diff[1])
            posref_safety_triggered = th.any(posref_safety_triggered, dim=1)
            newly_triggered = th.logical_and(posref_safety_triggered,
                                             th.logical_not((vec_safety_state>0)))
            vec_safety_state = th.where(newly_triggered,
                                        100.0, # 100 means safety triggered by posref
                                        vec_safety_state)
        vec_safety_state = th.logical_and(vec_safety_state, vec_step_count.view((self.num_envs,))>=1)

        new_internal_state = {  self.INTERNAL_FIELDS.SAFETY_TRIGGERED : vec_safety_state.view(self.num_envs,1),
                                self.INTERNAL_FIELDS.STEP_COUNT : (vec_step_count+1).view(self.num_envs,1),
                                self.INTERNAL_FIELDS.SIM_TIME : (self._adapter.getEnvTimeFromStartup() - self._eps_start_stime).view(self.num_envs,1)}
        new_robot_state = th.cat([vec_jstates_j_pveae, vec_last_sent_j_pvesd], dim = -1)
        # build stats:
        # with permute the first dimension becomes the joint (ordered as in set_monitored_joints)
        # with flatten the second dimension becomes minp,minv,mina,mmine,maxp,maxv,...
        new_robot_stats_state = vec_stats_minmaxavgstd_j_pvae.permute(0,2,1,3).flatten(start_dim=2)
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : vec_body_rel_linvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : vec_body_rel_linvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : vec_body_rel_linvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : vec_body_rel_angvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : vec_body_rel_angvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : vec_body_rel_angvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_X : vec_body_abs_linvel_xyz[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Y : vec_body_abs_linvel_xyz[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_LINVEL_Z : vec_body_abs_linvel_xyz[:,2].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z    : vec_body_ground_dist.view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : vec_body_rel_gravity_dir[:,0].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : vec_body_rel_gravity_dir[:,1].view(self.num_envs,1),
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : vec_body_rel_gravity_dir[:,2].view(self.num_envs,1)}
        
        # step_avg_pos = vec_stats_minmaxavgstd_j_pvae[:,2,:,0]
        step_avg_pos = vec_jstates_j_pveae[:,:,0]
        # print(f"vec_step_count={vec_step_count} step_avg_pos={step_avg_pos}")
        step_avg_pos = th.where(vec_step_count < 1,
                                step_avg_pos,
                                step_avg_pos*(1-self._configuration.longterm_stats_alpha) + self._current_state[self.STATE_JOINT_LONGTERM_STATS][0,0]*self._configuration.longterm_stats_alpha)
        new_longterm_stats_state = {self.JOINT_LONGTERM_STATS_FIELDS.AVG_POS : step_avg_pos}

        return {    self.STATE_EXTRINSIC    : new_extrinsic_state,
                    self.STATE_INTERNAL     : new_internal_state,
                    self.STATE_ROBOT        : new_robot_state,
                    self.STATE_JOINT_STEP_STATS  : new_robot_stats_state,
                    self.STATE_JOINT_LONGTERM_STATS : new_longterm_stats_state}
        


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
        i["joint_homing_dist"] = state[self.STATE_JOINT_LONGTERM_STATS][:,0,0,:] - self._configuration.homing_ctrl_joints_pvesd[:,0]
        if labels is not None:
            if not hasattr(self, "_joint_names_th"):
                self._joint_names_th = to_string_tensor([jn[1] for jn in self._configuration.controlled_joints])
            labels["joint_homing_dist"] = self._joint_names_th
        lims = self._state_helper.sub_helpers[self.STATE_ROBOT].get_limits()
        normhoming = normalize(self._configuration.homing_ctrl_joints_pvesd[:,0], lims[0,:,0], lims[1,:,0])
        smoothed_joint_pose_norm     = self._state_helper.sub_helpers[self.STATE_JOINT_LONGTERM_STATS].normalize(state[self.STATE_JOINT_LONGTERM_STATS],
                                                                                                      warn_limits_violation=False)[:,0,0]
        joint_pose     = self._state_helper.sub_helpers[self.STATE_ROBOT].normalize(state[self.STATE_ROBOT], warn_limits_violation=False)[:,0,:,0]
        i["joint_pos_error"] = th.mean(th.abs(smoothed_joint_pose_norm - normhoming), dim=1)
        i["joint_pos_error_instant"] = th.mean(th.abs(joint_pose - normhoming), dim=1)

        if self._configuration.verbose_infos:
            act_raw_state = state[self.STATE_ACT_RAW]   
            statenorm = self._state_helper.normalize(state)
            for substate in [   self.STATE_ROBOT,
                                self.STATE_EXTRINSIC,
                                self.STATE_INTERNAL,
                                self.STATE_ACT_PREPROC,
                                self.STATE_JOINT_STEP_STATS,
                                self.STATE_ACT_RAW,
                                self.STATE_JOINT_LONGTERM_STATS]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["vec_obs"] = to_string_tensor([n for n in self._state_helper.observation_names()["vec"]])
                i["posref_diff"] = state[self.STATE_ROBOT][:,1,:,5] - state[self.STATE_ROBOT][:,0,:,5]
                i["posref_vel"] = i["posref_diff"]/self._configuration.stepLength_sec
            i["vec_obs"] = self._last_obs["vec"]
            if labels is not None:
                labels["vec_obs"] = to_string_tensor([n for n in self._state_helper.observation_names()["vec"]])
            actdiff             = th.flatten((act_raw_state[:,0] - act_raw_state[:,1])/2, start_dim=1)
            prev_actdiff        = th.flatten((act_raw_state[:,1] - act_raw_state[:,2])/2, start_dim=1)
            act_acc             = actdiff - prev_actdiff
            i["act_diff"] = actdiff
            i["act_acc"] = act_acc
            if labels is not None:
                labels["act_diff"] = to_string_tensor(self._state_helper.sub_helpers[self.STATE_ACT_RAW].flat_state_names()[:12])
                labels["act_acc"] = labels["act_diff"]
        sub_rews = {}
        reward = self.compute_rewards(state, sub_rews)
        i["rewards"] = th.stack(list(sub_rews.values()), dim = 1) 
        # ggLog.info(f"i['rewards'] = {i['rewards'].size()}")
        if labels is not None:
            if not hasattr(self, "_sub_rew_names_th"):
                self._sub_rew_names_th = to_string_tensor(list(sub_rews.keys()))
            labels["rewards"] = self._sub_rew_names_th
        i.update({f"tot_reward":reward})
            
        i.update({"ep_config."+k:v for k,v in dataclasses.asdict(self._current_episode_config).items()})
        i["safety_triggered"] = (state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0) * 1.0
        
        return i
    
    @override
    @th.compile(mode="max-autotune-no-cudagraphs")
    def are_states_terminal(self, states) -> th.Tensor:
        if not self._configuration.stop_on_failure:
            return th.zeros_like(self._no_envs)
        # if th.any(r):
        #     term_idxs = th.nonzero(r)
        #     ggLog.info(f"Env {term_idxs} terminated at step {self._ep_step_counter[term_idxs]}")
        return (states[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_TRIGGERED,0] > 0).view((self.num_envs,))
    
    @override
    @th.compile(mode="max-autotune-no-cudagraphs")
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
