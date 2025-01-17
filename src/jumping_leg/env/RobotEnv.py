from __future__ import annotations
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.envs.ControlledEnv import ControlledEnv
from adarl.utils.robot_helpers import Robot
from adarl.utils.utils import to_string_tensor, th_quat_rotate, th_quat_conj, ros_rpy_to_quaternion_xyzw
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
import dataclasses
import numpy as np
import torch as th
import time
from adarl.utils.utils import isinstance_noimport
from typing_extensions import deprecated

@deprecated("Use RobotVecEnv") 
class RobotEnv(ControlledEnv[BaseJointImpedanceAdapter]):

    @dataclass
    class Configuration:
        action_delay_mustd : th.Tensor
        action_exp_smoothing_1s : float
        action_noise_mustd : th.Tensor
        control_mode : JointImpedanceActionHelper.CONTROL_MODES
        control_limits_minmax_pve : dict[tuple[str,str], th.Tensor]
        controlled_joints : Sequence[tuple[str,str]]
        frame_stack_length : int
        goal_err_exp_smoothing_1s : float
        history_length : int
        homing_body_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float]
        spawn_root_pose_xyz_xyzw : tuple[float,float,float,float,float,float,float]
        homing_ctrl_joints_pvesd : th.Tensor
        joint_physical_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_damping : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_pve : dict[tuple[str,str],th.Tensor]
        joint_safe_limits_minmax_stiffness : dict[tuple[str,str],th.Tensor]
        main_body_link : tuple[str,str]
        model_urdf_string : str
        obs_dtype : th.dtype
        observe_body_state : bool
        original_max_epsteps : int
        initial_pose_randomization : float
        real : bool
        robot_name : str
        robot_root_link : tuple[str,str]
        robot_root_joint : str
        robot_is_floating : bool
        safe_damping : float
        safe_stiffness : float
        seed : int
        show_goal : bool
        stepLength_sec : float
        stop_on_safety : bool
        th_device : th.device
        ui_camera_name : str
        ui_camera_link : tuple[str,str]
        verbose_infos : bool
        quiet : bool
        init_on_reset_ratio : float
        noise_joints_pve_mustdstd : th.Tensor
        noise_linvel_ep_mustdstd : th.Tensor
        noise_angvel_ep_mustdstd : th.Tensor
        noise_posz_ep_mustdstd : th.Tensor
        noise_gravity_ep_mustdstd : th.Tensor
        ui_rel_camera_pose_dist_pitch_yaw : th.Tensor
        ui_camera_resolution_hw : tuple[int,int]


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
                                                   "BODY_ABS_POS_Z",
                                                   "BODY_REL_GRAVITY_X",
                                                   "BODY_REL_GRAVITY_Y",
                                                   "BODY_REL_GRAVITY_Z"], start=0)
    ACT_FIELDS = IntEnum("ACT_FIELDS", ["ACTION"], start=0)
    
    JOINT_FILTERS = Enum("JointFilters",["ALL_REVOLUTE",
                                         "ALL"])
    
    joint_filters = {JOINT_FILTERS.ALL : lambda joint_name, robot_model: True,
                     JOINT_FILTERS.ALL_REVOLUTE : lambda joint_name, robot_model: robot_model.get_joint_properties([joint_name])[joint_name]["type"] == Robot.JOINT_TYPES.REVOLUTE}

    @dataclass
    class EpisodeConfiguration:
        initial_ctrl_joint_pose : th.Tensor
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
                        robot_main_body_link : str,
                        robot_root_link : str,
                        robot_name : str,
                        robot_urdf_string : str,
                        safe_damping : float,
                        safe_stiffness : float,
                        safety_limits_factor : float,
                        seed,
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
                        ui_camera_resolution_hw : tuple[int,int] = (144,256)
                        ):
        
        self._rng = th.Generator(device=th_device)
        self._spawned = False
        self._robot_model = Robot(adarl.utils.utils.compile_xacro_string(  model_definition_string=robot_urdf_string))
        self._current_state = {}
        self._enable_dbg_checks = enable_dbg_checks
        root_joint_name = self._robot_model.get_parent_joint(robot_root_link)
        is_floating = self._robot_model.get_joint_properties([root_joint_name])[root_joint_name]["type"] == Robot.JOINT_TYPES.FLOATING


        # ggLog.info("Properties:"+("\n".join([str(jp) for jp in self._robot_model.get_joint_properties(self._robot_model.get_joint_names()).items()])))
        # exit()
        controlled_joints_str = []
        for j in controlled_joints:
            if isinstance(j, str):
                controlled_joints_str.append(j)
            elif isinstance(j, self.JOINT_FILTERS):
                for jn in self._robot_model.get_joint_names():
                    if self.joint_filters[j](jn,self._robot_model):
                        controlled_joints_str.append(jn)

        controlled_joints_rn : list[tuple[str,str]] = [(robot_name,jn) for jn in controlled_joints_str]
        phys_limits_minmax_pve = {(robot_name,k):th.as_tensor(l,device=th_device) 
                                    for k,l in self._robot_model.get_joint_limits(controlled_joints_str).items()}
        safe_limits_minmax_pve = {k:(lims_minmax-0.5*(lims_minmax[1]+lims_minmax[0]))*safety_limits_factor+0.5*(lims_minmax[1]+lims_minmax[0])
                                    for k,lims_minmax in phys_limits_minmax_pve.items()}

        for jn in safe_limits_minmax_pve.keys():
            if jn not in control_limits_minmax_pve:
                control_limits_minmax_pve[jn] = safe_limits_minmax_pve[jn]

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
            ggLog.info(f"homing_joint_pose = "+"\n".join([f"{jn}:{p}" for jn,p in homing_joint_pose.items()]))

        obs_dtype = th.float32
        homing_ctrl_joints_pvesd = th.as_tensor([(homing_joint_pose[jn], 0, 0, safe_stiffness, safe_damping)
                                    for jn in controlled_joints_rn],
                                    device=th_device,
                                    dtype=obs_dtype)
        self._configuration = self.Configuration(  action_delay_mustd = th.as_tensor(action_delay_mustd, device=th_device),
                                                            action_exp_smoothing_1s = action_exp_smoothing_1s,
                                                            action_noise_mustd = th.as_tensor(action_noise_mustd, device=th_device),
                                                            control_limits_minmax_pve = control_limits_minmax_pve,
                                                            control_mode = JointImpedanceActionHelper.CONTROL_MODES[control_mode.upper()],
                                                            controlled_joints = controlled_joints_rn,
                                                            frame_stack_length = frame_stack_length,
                                                            goal_err_exp_smoothing_1s = goal_err_exp_smoothing_1s,
                                                            history_length = max(2,frame_stack_length),
                                                            homing_body_pose_xyz_xyzw = homing_body_pose_xyz_xyzw,
                                                            homing_ctrl_joints_pvesd = homing_ctrl_joints_pvesd,
                                                            joint_physical_limits_minmax_pve = phys_limits_minmax_pve,
                                                            joint_safe_limits_minmax_damping = minmax_damping_thdict,
                                                            joint_safe_limits_minmax_pve = safe_limits_minmax_pve,
                                                            joint_safe_limits_minmax_stiffness = minmax_stiffness_thdict,
                                                            main_body_link=(robot_name,robot_main_body_link),
                                                            robot_root_link=(robot_name,robot_root_link),
                                                            model_urdf_string=robot_urdf_string,
                                                            obs_dtype = obs_dtype,
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
                                                            noise_joints_pve_mustdstd = th.as_tensor(obs_noise_joints_pve_ep_mustd_step_std, device=th_device),
                                                            noise_linvel_ep_mustdstd =  th.as_tensor(obs_noise_linvel_ep_mustd_step_std, device=th_device),
                                                            noise_angvel_ep_mustdstd =  th.as_tensor(obs_noise_angvel_ep_mustd_step_std, device=th_device),
                                                            noise_posz_ep_mustdstd =    th.as_tensor(obs_noise_posz_ep_mustd_step_std, device=th_device),
                                                            noise_gravity_ep_mustdstd = th.as_tensor(obs_noise_gravity_ep_mustd_step_std, device=th_device),
                                                            ui_rel_camera_pose_dist_pitch_yaw = th.as_tensor([2.5, 30/180*3.14159, -90/180*3.14159], device=th_device),
                                                            ui_camera_resolution_hw = ui_camera_resolution_hw
                                                            )

        self._always_present_collisions : set[tuple[str,str]] = self._robot_model.detect_always_present_collisions(
            moving_joints=[jn[1] for jn in self._configuration.controlled_joints],
            fixed_joints_pose={self._configuration.robot_root_joint : np.array(self._configuration.homing_body_pose_xyz_xyzw)}
                                            if self._configuration.robot_is_floating else {})

        self._safe_limits_minmax_j_pve = th.stack([safe_limits_minmax_pve[jn] for jn in controlled_joints_rn], dim=1)
        self._action_helper= JointImpedanceActionHelper(control_mode=self._configuration.control_mode,
                                joints=controlled_joints_rn,
                                joints_minmax_pvesd={jn:th.cat([control_limits_minmax_pve[jn],
                                                                minmax_stiffness_thdict[jn].unsqueeze(1),
                                                                minmax_damping_thdict[jn].unsqueeze(1)], dim=1) 
                                                        for jn in controlled_joints_rn},
                                safe_stiffness=th.as_tensor([self._configuration.safe_stiffness]).repeat(len(controlled_joints_rn)),
                                safe_damping=th.as_tensor([self._configuration.safe_damping]).repeat(len(controlled_joints_rn)),
                                th_device=self._configuration.th_device)

        robot_state_helper = RobotStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                              stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                              damping_minmax=self._configuration.joint_safe_limits_minmax_damping,
                                              obs_dtype=self._configuration.obs_dtype,
                                              th_device=self._configuration.th_device,
                                              history_length=self._configuration.history_length,
                                              obs_history_length = self._configuration.frame_stack_length)
        robot_stats_state_helper = RobotStatsStateHelper(joint_limit_minmax_pve=self._configuration.joint_physical_limits_minmax_pve,
                                                        obs_dtype=self._configuration.obs_dtype,
                                                        th_device=self._configuration.th_device)
        internal_state_helper =   ThBoxStateHelper(field_names=[e for e in self.INTERNAL_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.INTERNAL_FIELDS.SAFETY_TRIGGERED : [0,1],
                                                                self.INTERNAL_FIELDS.STEP_COUNT : [-1,1000_000_000]},
                                                observable_fields=[])
        extrinsic_state_helper =  ThBoxStateHelper(field_names=[e for e in self.EXTRINSIC_FIELDS],
                                              obs_dtype=th.float32,
                                              th_device=th_device,
                                              field_size=(1,),
                                              fields_minmax={   self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : [-10,10],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : [-10,10],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : [-10,10],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : [-100,100],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : [-100,100],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : [-100,100],
                                                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : [-1,1],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : [-1,1],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : [-1,1],
                                                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : [-1,1]},
                                               history_length=self._configuration.history_length,
                                               obs_history_length = self._configuration.frame_stack_length)
        act_history_state_helper = ThBoxStateHelper(field_names=[a for a in self.ACT_FIELDS],
                                               obs_dtype=th.float32,
                                               th_device=th_device,
                                               field_size=(self._action_helper.action_len(),),
                                               fields_minmax = {self.ACT_FIELDS.ACTION : [-1.0,1.0]},
                                               history_length=2)
        robot_state_noise =  StateNoiseGenerator(robot_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = self._configuration.noise_joints_pve_mustdstd[:2],
                                            step_std = self._configuration.noise_joints_pve_mustdstd[2])
        ggLog.info(f"Built robot noise")
        extrinsic_state_noise =  StateNoiseGenerator(extrinsic_state_helper,
                                            self._rng, dtype=self._configuration.obs_dtype, device=self._configuration.th_device,
                                            episode_mu_std = th.cat([   self._configuration.noise_linvel_ep_mustdstd[:2].expand(3,2),
                                                                        self._configuration.noise_angvel_ep_mustdstd[:2].expand(3,2),
                                                                        self._configuration.noise_posz_ep_mustdstd[:2].expand(1,2),
                                                                        self._configuration.noise_gravity_ep_mustdstd[:2].expand(3,2)]).permute(1,0).unsqueeze(-1),
                                            step_std = th.cat([ self._configuration.noise_linvel_ep_mustdstd[2].expand(3),
                                                                self._configuration.noise_angvel_ep_mustdstd[2].expand(3),
                                                                self._configuration.noise_posz_ep_mustdstd[2].expand(1),
                                                                self._configuration.noise_gravity_ep_mustdstd[2].expand(3)]).unsqueeze(-1))
        ggLog.info(f"Built extrinsic noise")
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

        ggLog.info(f"Built substate helpers")
        self._state_helper = DictStateHelper(statehelpers,
                                              observable_fields=observable_fields,
                                              noise = {
                                                    self.STATE_ROBOT : robot_state_noise,
                                                    self.STATE_EXTRINSIC : extrinsic_state_noise},
                                              flatten_in_obs=[   self.STATE_ROBOT,
                                                                self.STATE_EXTRINSIC,
                                                                self.STATE_INTERNAL],
                                              flattened_part_name="vec")
        ggLog.info(f"Built state helper")
        self._safety_limits = robot_state_helper.build_robot_limits(joint_limit_minmax_pve=self._configuration.joint_safe_limits_minmax_pve,
                                                                    stiffness_minmax=self._configuration.joint_safe_limits_minmax_stiffness,
                                                                    damping_minmax=self._configuration.joint_safe_limits_minmax_damping)
        ggLog.info(f"Built safety limits")
        
        state_space = self._state_helper.get_space()
        observation_space = self._state_helper.get_obs_space()
        action_space = self._action_helper.action_space(seed=seed)
        ggLog.info(f"Built state/obs/action helpers")

        super().__init__(maxStepsPerEpisode,
                         stepLength_sec,
                         adapter,
                         action_space,
                         observation_space,
                         state_space,
                         startSimulation = True,
                         step_precision_tolerance = step_precision_tolerance)
        
        self._adapter.set_monitored_links([self._configuration.main_body_link])
        self._adapter.startup()


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
            self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_pvesd,
                                                                delay_sec=action_delay.item())
            





    





    # --------------------------------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------
    @override
    def initializeEpisode(self, options = {}) -> None:

        self._current_state = self._state_helper.reset_state()
        self._current_state[self.STATE_INTERNAL][0,self.INTERNAL_FIELDS.STEP_COUNT] = th.tensor(-1.)
        self._last_obs = None

        if not self._spawned and isinstance(self._adapter, BaseSimulationAdapter):            
            robot_pose = build_pose(*self._configuration.spawn_root_pose_xyz_xyzw)
            camera_pose = build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707)
            arrow_pose = robot_pose
            self._spawned = True
            camera_file = adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.sdf.xacro")
            if isinstance_noimport(self._adapter, "PyBulletAdapter"):
                self._adapter.spawn_model(model_definition_string=self._configuration.model_urdf_string,
                                            model_name=self._configuration.robot_name,
                                            pose=robot_pose,
                                            model_format="urdf")
            self._adapter.spawn_model(model_file=camera_file,
                                        model_name="simple_camera",
                                        pose=camera_pose,
                                        model_format="sdf.xacro",
                                        model_kwargs={"camera_width":self._configuration.ui_camera_resolution_hw[1],
                                                      "camera_height":self._configuration.ui_camera_resolution_hw[0],
                                                      "frame_rate":1/self._intendedStepLength_sec})
            if self._configuration.show_goal:
                self._adapter.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/red_arrow.urdf.xacro"),
                                            model_name="arrow",
                                            pose=arrow_pose,
                                            model_format="urdf.xacro",
                                            model_kwargs={"add_world_link":str(isinstance_noimport(self._adapter, "PyBulletAdapter"))})
                self._adapter.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("jumping_leg","models/axes.urdf.xacro"),
                                            model_name="axes",
                                            pose=build_pose(0,0,0,0,0,0,1),
                                            model_format="urdf.xacro",
                                            model_kwargs={"add_world_link":str(isinstance_noimport(self._adapter, "PyBulletAdapter"))})
            self._robot_model.disable_tree_self_collisions(root_frame=self._configuration.robot_root_link[1])
            # self._robot_model.remove_collision_pairs([("rail_link_0","slider_link_0")])            
            self._ground_co_id = self._robot_model.add_collision_box(   pose_xyz_xyzw=np.array([0.,0.,-0.5,0.,0.,0.,1.]),
                                                                        collision_box_size_xyz=(100,100,1),
                                                                        collision_obj_id="ground_collision")
            self._adapter.set_monitored_joints(self._configuration.controlled_joints)
            self._adapter.set_impedance_controlled_joints(self._configuration.controlled_joints)

        
        self._set_current_ep_config(reset_options = options)
        
        if isinstance(self._adapter, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            self._realworld_initialization()
        self._last_out_action = th.clamp(self._action_helper._pvesd_to_action(self._last_sent_pvesd), min=-1, max=1)
        # ggLog.info(f"initial action {self._last_out_action}, pvesd = {self._last_sent_pvesd}")

        self._update_state()
        self._update_stats()


    def _set_current_ep_config(self, reset_options : dict = {}):
        maxStepsPerEpisode = reset_options.get("max_ep_steps", self._configuration.original_max_epsteps)           
        found_good_configuration = False
        original_collision_pairs = self._robot_model.get_enabled_collision_pairs()
        self._robot_model.set_collision_pairs("all")
        self._robot_model.remove_collision_pairs(self._always_present_collisions)
        homing_pos = self._configuration.homing_ctrl_joints_pvesd[:,0]
        if self._configuration.initial_pose_randomization > 0:
            for i in range(1000):
                npos = (th.rand(size=(len(self._configuration.controlled_joints),), dtype=th.float32, device=self._configuration.th_device)*2-1)*self._configuration.initial_pose_randomization
                limits_minmax = th.stack([self._configuration.joint_safe_limits_minmax_pve[jn][:,0] for jn in self._configuration.controlled_joints], dim = 1)
                # initial_joint_pose = unnormalize(((npos)),limits_minmax[0],limits_minmax[1])                
                initial_joint_pose = ((npos>=0)*((limits_minmax[1]-homing_pos)*npos + homing_pos) + (npos<0)*((homing_pos-limits_minmax[0])*npos + homing_pos))
                self._robot_model.set_joint_pose_by_names({jn[1]:initial_joint_pose[i].cpu().numpy() for i,jn in enumerate(self._configuration.controlled_joints)})
                if self._configuration.robot_is_floating:
                    self._robot_model.set_joint_pose_by_names({self._configuration.robot_root_joint:np.array([self._configuration.homing_body_pose_xyz_xyzw])})
                collisions = self._robot_model.get_all_collisions()
                if len(collisions) == 0:
                    found_good_configuration = True
                    break
            if not found_good_configuration:
                ggLog.warn(f"Failed to find initial joint configuration. Last collisions = {collisions}, always present collisions = {self._always_present_collisions}")
        if not found_good_configuration:
            initial_joint_pose = homing_pos
        # ggLog.info(f"initial_jpose = {initial_joint_pose}, homing = {homing}")
        self._robot_model.set_collision_pairs(original_collision_pairs)
        self._current_episode_config = RobotEnv.EpisodeConfiguration(   initial_ctrl_joint_pose = initial_joint_pose,
                                                                        max_ep_steps = maxStepsPerEpisode)
        self.set_max_episode_steps(self._current_episode_config.max_ep_steps)

    def _realworld_initialization(self):
        raise NotImplementedError()
    
    def _simulation_initialization(self):
        if not isinstance(self._adapter, BaseSimulationAdapter):
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
        
        if self._configuration.init_on_reset_ratio > 0 and th.rand((1,), generator=self._rng) >= self._configuration.init_on_reset_ratio and self._resetCounter > 1:
            return
        
        if self._configuration.homing_body_pose_xyz_xyzw is not None:
            self._adapter.setLinksStateDirect({self._configuration.main_body_link :
                                                            LinkState( position_xyz = th.tensor(self._configuration.homing_body_pose_xyz_xyzw[:3], device=self._configuration.th_device),
                                                                        orientation_xyzw = th.tensor(self._configuration.homing_body_pose_xyz_xyzw[3:7], device=self._configuration.th_device),
                                                                        pos_com_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                        ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
        jpose = self._current_episode_config.initial_ctrl_joint_pose
        self._adapter.setJointsStateDirect({jn:JointState(  position=jpose[i],
                                                            rate = 0,
                                                            effort = 0) for i,jn in enumerate(self._configuration.controlled_joints)})
        initial_jimp_cmd = th.stack([jpose,
                                    th.zeros_like(jpose),
                                    th.zeros_like(jpose),
                                    th.full_like(jpose, self._configuration.safe_stiffness),
                                    th.full_like(jpose, self._configuration.safe_damping)], dim=1)
        self._adapter.setJointsImpedanceCommand(initial_jimp_cmd)
        self._adapter.apply_joint_impedances(initial_jimp_cmd)
        self._last_sent_pvesd = {jn:initial_jimp_cmd[i] for i,jn in enumerate(self._configuration.controlled_joints)}

    @override
    def build(self):
        ggLog.info(f"RobotEnv.build()...")
        envCtrlName = type(self._adapter).__name__
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._adapter.build_scenario()
            self._arrow_base = ("arrow","world")
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._configuration.real:
                raise NotImplementedError()
            else:
                self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path("jumping_leg",
                                                                                                                          "gazebo/all_gazebo_xbot.launch"),
                                                           launch_file_args={"gui":"false"})
                self._arrow_base = ("arrow","arrow_link")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")

    @override
    def _destroy(self):
        self._adapter.destroy_scenario()


    def set_cam_pose(self, pose_dist_pitch_roll : tuple[float,float,float] | th.Tensor):
        self._configuration.ui_rel_camera_pose_dist_pitch_yaw = th.as_tensor(pose_dist_pitch_roll, device=self._configuration.th_device)


    def get_cam_pose(self):
        return self._configuration.ui_rel_camera_pose_dist_pitch_yaw    
    
    def _get_cam_pose_xyz_xyzw(self):
        cam_rel_pos_dist_pitch_yaw = self._configuration.ui_rel_camera_pose_dist_pitch_yaw
        cam_rel_pos  = th.as_tensor([-cam_rel_pos_dist_pitch_yaw[0], 0.0, 0.0], device=self._configuration.th_device)
        cam_rel_quat = th.as_tensor(ros_rpy_to_quaternion_xyzw([0.0, cam_rel_pos_dist_pitch_yaw[1], cam_rel_pos_dist_pitch_yaw[2]]),
                                   device=self._configuration.th_device)
        # ggLog.info(f"cam pos0 = {cam_rel_pos}")
        return th.cat([th_quat_rotate(cam_rel_pos, cam_rel_quat), cam_rel_quat])

    # --------------------------------------------------------------------------------------------------------------------
    # State & Observation
    # --------------------------------------------------------------------------------------------------------------------
    @override
    def getUiRendering(self) -> tuple[np.ndarray | th.Tensor | None, float]:
        # camera by default looks down the x axis
        rel_cam_pose_xyz_xyzw = self._get_cam_pose_xyz_xyzw()
        # ggLog.info(f"cam pos = {cam_rel_pos}")
        # ggLog.info(f"cam quat = {cam_rel_quat}")
        try:
            if isinstance(self._adapter, BaseSimulationAdapter):
                body_state : LinkState = self._adapter.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
                self._adapter.setLinksStateDirect({self._configuration.ui_camera_link :
                                                                LinkState(  position_xyz = body_state.pose.position + rel_cam_pose_xyz_xyzw[0:3],
                                                                            orientation_xyzw = rel_cam_pose_xyz_xyzw[3:7],
                                                                            pos_com_velocity_xyz = th.tensor((0.,0.,0), device=self._configuration.th_device),
                                                                            ang_velocity_xyz = th.tensor((0.,0.,0.), device=self._configuration.th_device))})
            img, time = self._adapter.getRenderings([self._configuration.ui_camera_name])[self._configuration.ui_camera_name]
            if img is None:
                time = -1
            return img, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return None, -1
    
    @override
    def getObservation(self, state) -> dict[Any, th.Tensor]:
        self._last_obs = self._state_helper.observe(state)
        if self._enable_dbg_checks:
            if not adarl.utils.tensor_trees.is_all_finite(state):
                ggLog.warn(f"Non-finite values in state {state}")
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
        self._last_step_simtime = self._adapter.getEnvTimeFromReset()


    def _get_new_instantaneous_state(self):
        # ggLog.info(f"_stepCounter = {self._stepCounter}")
        
        jstates = self._adapter.getJointsState(requestedJoints=self._configuration.controlled_joints)
        body_state : LinkState = self._adapter.getLinksState(requestedLinks = [self._configuration.main_body_link], use_com_frame = True)[self._configuration.main_body_link]
        body_linvel_xyz = body_state.pos_velocity_xyz
        body_angvel_xyz = body_state.ang_velocity_xyz
        body_position_xyz = body_state.pose.position
        gravity_vec         = th_quat_rotate(th.tensor([0.0,0,-1]), th_quat_conj(body_state.pose.orientation_xyzw))
        body_rel_linvel_xyz = th_quat_rotate(body_linvel_xyz,     th_quat_conj(body_state.pose.orientation_xyzw))
        body_rel_angvel_xyz = th_quat_rotate(body_angvel_xyz,     th_quat_conj(body_state.pose.orientation_xyzw))


        internal_state = self._current_state[self.STATE_INTERNAL][0]
        step_count = internal_state[self.INTERNAL_FIELDS.STEP_COUNT]
        safety_triggered = step_count!=-1 and internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED] > 0
        
        if not isinstance_noimport(self._adapter, "PyBulletAdapter"): # for now the only adapter that really supports joint stats
            stats_minmaxavgstd_j_pvae = self._adapter.get_joints_state_step_stats()
            if not th.all(th.isfinite(stats_minmaxavgstd_j_pvae)):
                raise RuntimeError(f"non finite values in joint stats: stats_minmaxavgstd_hipknee_pve = {stats_minmaxavgstd_j_pvae}")
            if step_count>=1: # stats are not valid at step 0
                triggered_limits = th.logical_or(stats_minmaxavgstd_j_pvae[0, :, [0,1,3]] < self._safe_limits_minmax_j_pve[0],
                                                stats_minmaxavgstd_j_pvae[1, :, [0,1,3]] > self._safe_limits_minmax_j_pve[1])
                safety_triggered = th.any(triggered_limits)
                if safety_triggered:       
                    elements = np.array([[f"{jn[1]}_pos",f"{jn[1]}_vel",f"{jn[1]}_eff"] for jn in self._configuration.controlled_joints], dtype=object) #type: ignore
                    triggered = []
                    for i in np.ndindex(elements.shape):
                        if triggered_limits[i]:
                            triggered.append(elements[i])
                    if not self._configuration.quiet:
                        ggLog.info( f"SAFETY TRIGGERED (step {step_count.item()}):"
                                    f"\n    triggered ({len(triggered)}) = {triggered}"
                                    # f"\n    joints_minmax = \n{stats_minmaxavgstd_j_pve[:2]}"
                                    # f"\n    j_safety_lims  = \n{self._safe_limits_minmax_j_pve} "
                                    )
            else:
                safety_triggered = False
        elif isinstance_noimport(self._adapter, "RosXbotAdapter"):
            safety_triggered = self._adapter.safety_triggered()
            stats_minmaxavgstd_j_pvae = th.zeros((4,len(self._adapter.get_monitored_joints()), 4), device=self._configuration.th_device) # TODO, put here some more proper values
        else:
            raise NotImplementedError()

        new_internal_state = {  self.INTERNAL_FIELDS.SAFETY_TRIGGERED : 1 if safety_triggered else 0,
                                self.INTERNAL_FIELDS.STEP_COUNT : step_count+1}
        new_robot_state = {jn : th.concat([ jstates[jn].position[[0]],
                                            jstates[jn].rate[[0]],
                                            jstates[jn].effort[[0]],
                                            th.as_tensor(self._last_sent_pvesd[jn])])
                                for jn in self._configuration.controlled_joints}
        new_robot_stats_state = {jname : stats_minmaxavgstd_j_pvae[:,i,:].flatten()
                                 for i,jname in enumerate(self._adapter.get_monitored_joints())}
        new_extrinsic_state = { self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_X : body_rel_linvel_xyz[0],
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Y : body_rel_linvel_xyz[1],
                                self.EXTRINSIC_FIELDS.BODY_REL_LINVEL_Z : body_rel_linvel_xyz[2],
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_X : body_rel_angvel_xyz[0],
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Y : body_rel_angvel_xyz[1],
                                self.EXTRINSIC_FIELDS.BODY_REL_ANGVEL_Z : body_rel_angvel_xyz[2],
                                self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z : body_position_xyz[2],
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X : gravity_vec[0],
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y : gravity_vec[1],
                                self.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Z : gravity_vec[2]}
        new_act_state = {self.ACT_FIELDS.ACTION : self._last_out_action}
        instantaneous_state = { self.STATE_EXTRINSIC    : new_extrinsic_state,
                                self.STATE_ACT          : new_act_state,
                                self.STATE_INTERNAL     : new_internal_state,
                                self.STATE_ROBOT        : new_robot_state,
                                self.STATE_ROBOT_STATS  : new_robot_stats_state}              
        if th.any(th.concat([new_robot_state[jn][6:] for jn in self._configuration.controlled_joints])<0):
            ggLog.warn(f"negative gains in new_robot_state = {new_robot_state}")
        return instantaneous_state
        


    def _update_state(self):
        # t0 = time.monotonic()
        instantaneous_state = self._get_new_instantaneous_state()
        # t1 = time.monotonic()
        step_count = self._current_state[self.STATE_INTERNAL][0][self.INTERNAL_FIELDS.STEP_COUNT]
        if step_count <= 0:
            self._current_state = self._state_helper.reset_state(instantaneous_state)
        else:
            self._state_helper.update(instantaneous_state, state=self._current_state)
        # map_tensor_tree(self._current_state, lambda t: t.detach().clone())
        # tf = time.monotonic()
        # print(f"newinst = {t1-t0}, map = {tf-t1}, tot = {tf-t0}")



    def _update_stats(self):
        rew_dbg_info = {}
        self.computeReward( {},
                            self._current_state, 
                            th.tensor([]), 
                            env_conf=self.get_configuration(),
                            dbg_info=rew_dbg_info)
        if self._current_state[self.STATE_INTERNAL][0][self.INTERNAL_FIELDS.STEP_COUNT]<=0:
            self._stats = {}
        self._stats["rewards"] = rew_dbg_info
        
    @override
    def getInfo(self,state) -> dict[Any,Any]:
        i = super().getInfo(state=state)
        internal_state = state[self.STATE_INTERNAL][0]
        i.update(self._stats)
        i["step_count"] = self._stepCounter

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_ROBOT, self.STATE_EXTRINSIC, self.STATE_INTERNAL, self.STATE_ACT, self.STATE_ROBOT_STATS]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["state_"+substate+"_labels"] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                i["statenorm_"+substate+"_labels"] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                i["vec_obs"] = self._last_obs["vec"]
                i["vec_obs_labels"] = to_string_tensor([n for n in self._state_helper.observation_names()["vec"]])
            
        i.update(self._stats["rewards"])
        i["ep_config"] = dataclasses.asdict(self._current_episode_config)
        i["safety_triggered"] = internal_state[self.INTERNAL_FIELDS.SAFETY_TRIGGERED]
        
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
