from __future__ import annotations

from adarl_envs.env.env_utils import joint_penalty_reward, norm_penalty, smoothclip_flattener
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.utils.utils import (LinkState, to_string_tensor, th_quat_rotate, th_quat_conj, vector_projection, isinstance_noimport, 
                               quat_xyzw_between_vecs_py, masked_assign, quat_mul_xyzw, quat_angle_xyzw, ros_rpy_to_quaternion_xyzw_th)
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
from adarl_envs.env.RobotVecEnv import RobotVecEnv, JOINT_FILTERS, DistributionDef, RobotVecEnvInitArgs
from adarl.utils.tensor_trees import map_tensor_tree, space_from_tree
import adarl.utils.tensor_trees
import traceback
import pprint
import dataclasses
from pathlib import Path
from adarl_envs.env.env_utils import flattened_joint_penalty_reward
from adarl.utils.dbg.dbg_checks import dbg_check_finite

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
    gripper_links : list[tuple[str,str]]
    manipulator_links : list[tuple[str,str]]
    reward_health_weight : float
    reward_joint_actacc_weight : float
    reward_joint_actdiff_weight : float
    reward_joint_position_limit_weight : float
    reward_joint_power_weight : float
    reward_joint_torque_weight : float
    reward_safety_weight : float
    reward_scale : float
    reward_object_pose_weight : float
    reward_gripper_pose_weight : float
    target_object_link : tuple[str,str]
    observe_object_pose : bool = False

class GraspVecEnv(RobotVecEnv):
    STATE_GRASPING = "grasp"
    STATE_CAMERA = "camera"

    @dataclass
    class GraspingConfiguration:
        reward_scale : th.Tensor
        target_object_link : tuple[str,str]
        gripper_links : list[tuple[str,str]]
        observe_object_pose : bool
        camera_resolution_hw : tuple[int,int]
        init_obj_area_minmax_xyz : th.Tensor
        goal_obj_area_minmax_xyz : th.Tensor
        table_link : tuple[str,str]
        manipulator_links : list[tuple[str,str]]
        use_head_cam_as_ui_camera : bool
        split_rewards : bool

    @dataclass
    class SubRewards:
        health : th.Tensor
        joint_actacc : th.Tensor
        joint_actdiff : th.Tensor
        joint_power : th.Tensor
        joint_torque : th.Tensor
        joint_position_limit : th.Tensor
        safety_triggered : th.Tensor
        reward_object_pose : th.Tensor
        reward_gripper_pose : th.Tensor


    @dataclass
    class EpisodeGraspingConfiguration:
        initial_object_pose : th.Tensor
        goal_object_pose : th.Tensor

    GRASPING_FIELDS = IntEnum("GRASPING_FIELDS",   ["GOAL_POSE",
                                                    "OBJECT_POSE",
                                                    "GRIPPER_POSE"], start=0)

    CAMERA_FIELDS = IntEnum("CAMERA_FIELDS",   ["IMAGE"], start=0)
    

    def __init__(self,  grasp_init_args : GrapVecEnvInitArgs):
        robot_init_args = grasp_init_args.robot_init_args
        adapter = robot_init_args.adapter
        th_device = robot_init_args.th_device
        self._th_device = th_device
        self._obs_dtype = th.float32
        self._all_vecs = th.ones((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._no_vecs = th.zeros((adapter.vec_size(),), device=th_device, dtype=th.bool)
        self._unit_3d_vector = self._thtens([1.0, 0.0, 0.0])
        self._unit_quaternion = self._thtens([0.0, 0.0, 0.0, 1.0])
        self._zero = self._thtens([0.0])
        self._head_camera_name = "head_camera"
        self._table_height = 0.8
        cube_spawn_height = self._table_height + 0.031
        spawn_area_minmax_xyz = [[ 0.50,  0.0, cube_spawn_height],
                                        [ 0.60,  0.1, cube_spawn_height]]
        
        manipulation_area_minmax_xyz = [[ 0.50,  0.0, cube_spawn_height],
                                        [ 0.60,  0.1, self._table_height + 0.1]]
        self._grasping_conf = GraspVecEnv.GraspingConfiguration(
                        reward_scale = self._thtens(grasp_init_args.reward_scale),
                        target_object_link=grasp_init_args.target_object_link,
                        gripper_links=grasp_init_args.gripper_links,
                        observe_object_pose=grasp_init_args.observe_object_pose,
                        camera_resolution_hw = (256,256),
                        init_obj_area_minmax_xyz = th.as_tensor(spawn_area_minmax_xyz, device=th_device),
                        goal_obj_area_minmax_xyz = th.as_tensor(manipulation_area_minmax_xyz, device=th_device),
                        table_link = ("table","cube"),
                        manipulator_links = grasp_init_args.manipulator_links,
                        use_head_cam_as_ui_camera = True,
                        split_rewards = False
                        )

        self._sub_rewards_weights2 = GraspVecEnv.SubRewards(
                health = self._thtens(grasp_init_args.reward_health_weight),
                joint_actacc = self._thtens(grasp_init_args.reward_joint_actacc_weight),
                joint_actdiff = self._thtens(grasp_init_args.reward_joint_actdiff_weight),
                joint_power = self._thtens(grasp_init_args.reward_joint_power_weight),
                joint_torque = self._thtens(grasp_init_args.reward_joint_torque_weight),
                joint_position_limit = self._thtens(grasp_init_args.reward_joint_position_limit_weight),
                safety_triggered = self._thtens(grasp_init_args.reward_safety_weight),
                reward_object_pose = self._thtens(grasp_init_args.reward_object_pose_weight),
                reward_gripper_pose = self._thtens(grasp_init_args.reward_gripper_pose_weight)
        )
        self._sub_rewards_enabled = {k:v for k,v in dataclasses.asdict(self._sub_rewards_weights2).items() if v!=0.0}
        self._sub_rewards_enabled_weights_th = self._thtens([v for v in self._sub_rewards_enabled.values()])
        
        self._grasping_episode_config = GraspVecEnv.EpisodeGraspingConfiguration(initial_object_pose = self._thzeros((adapter.vec_size(), 7)),
                                                                                   goal_object_pose = self._thzeros((adapter.vec_size(), 7)))
        if robot_init_args.enable_link_collisions is None:
            robot_init_args.enable_link_collisions = []
        cube_colliding_links = [self._grasping_conf.table_link]
        # cube_colliding_links += self._grasping_conf.manipulator_links
        robot_init_args.enable_link_collisions.append((self._grasping_conf.target_object_link, cube_colliding_links))
        robot_init_args.initial_height_randomization_range_meters=0.0
        robot_init_args.obs_abs_noise_linacc_ep_mustd_step_std=(0.0,0.0,0.0)
        super().__init__(robot_init_args)

    @override
    def _build_stats(self):
        self._stats = {}
        self._buff_sizes = int(self._configuration.original_max_epsteps/10)
        self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
        self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))
        self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))

    @override
    def _build(self):
        super()._build()
        self._adapter.set_monitored_links(self._adapter.get_monitored_links() + [self._grasping_conf.target_object_link] + self._grasping_conf.gripper_links)
        self._object_link_id = self._adapter.get_monitored_links_ids([self._grasping_conf.target_object_link])
        self._gripper_link_ids = self._adapter.get_monitored_links_ids(self._grasping_conf.gripper_links)
        self._obj_and_gripper_link_ids = self._adapter.get_monitored_links_ids([self._grasping_conf.target_object_link]+self._grasping_conf.gripper_links)



    def _build_state_helper(self, adapter : BaseVecJointImpedanceAdapter):
        super()._build_state_helper(adapter)
        if self._grasping_conf.observe_object_pose:
            observable_fields=[ self.GRASPING_FIELDS.GOAL_POSE,
                                self.GRASPING_FIELDS.OBJECT_POSE
                                ]
        else:
            observable_fields=[ self.GRASPING_FIELDS.GOAL_POSE]
        grasping_state_helper = ThBoxStateHelper( field_names=[e for e in self.GRASPING_FIELDS],
                                                    dtype=self._obs_dtype,
                                                    th_device=self._th_device,
                                                    field_size=(7,),
                                                    fields_minmax={ self.GRASPING_FIELDS.GOAL_POSE :    [-10, 10],
                                                                    self.GRASPING_FIELDS.OBJECT_POSE :  [-10, 10],
                                                                    self.GRASPING_FIELDS.GRIPPER_POSE : [-10, 10]},
                                                    vec_size=adapter.vec_size(),
                                                    history_length=2,
                                                    observation_definitions={"base":
                                                                             ThBoxStateHelper.SimpleObsDef( obs_history_length=1,
                                                                                                            observable_fields=observable_fields,
                                                                                                            observable_subfields=None)})
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_GRASPING,
                                                            grasping_state_helper,
                                                        obs_defs={"base":{"observable":True,"concatenate":True,"noise":None}})
        camera_state_helper = ThBoxStateHelper( field_names=[e for e in self.CAMERA_FIELDS],
                                                dtype=self._obs_dtype,
                                                th_device=self._th_device,
                                                field_size=self._grasping_conf.camera_resolution_hw,
                                                fields_minmax={ self.CAMERA_FIELDS.IMAGE : [-1,1]},
                                                vec_size=adapter.vec_size(),
                                                observation_definitions={"base":
                                                                         ThBoxStateHelper.SimpleObsDef( obs_history_length=1,
                                                                                                        observable_fields=None,
                                                                                                        observable_subfields=None)})
        # if not self._grasping_conf.observe_object_pose:
        self._state_helper = self._state_helper.add_substate(GraspVecEnv.STATE_CAMERA,
                                                            camera_state_helper,
                                                            obs_defs={"base":{"observable":not self._grasping_conf.observe_object_pose,
                                                                              "concatenate":False,
                                                                              "noise":None}})
        ggLog.info(f"Built state/obs/action helpers")
        

    def _get_adapter_data_raw(self):
        super_adapter_data =  super()._get_adapter_data_raw()
        poses = self._adapter.getLinksState(self._obj_and_gripper_link_ids, use_com_pose=False)[:,:,:7]
        current_object_pose = poses[:,0]
        current_gripper_poses = poses[:,1:]
        grasp_adapter_data = current_object_pose, current_gripper_poses
        return super_adapter_data, grasp_adapter_data

    @override
    def _get_new_instantaneous_state(self, adapter_data):
        super_adapter_data, (current_object_pose, current_gripper_poses) = adapter_data
        new_inst_state = super()._get_new_instantaneous_state(super_adapter_data)

        current_gripper_pose = current_gripper_poses.mean(dim=1)
        # ggLog.info(f"current_object_pose = {current_object_pose}")
        # ggLog.info(f"current_gripper_pose = {current_gripper_pose}")
        new_grasping_state = {self.GRASPING_FIELDS.GOAL_POSE   : self._grasping_episode_config.goal_object_pose.expand(self.num_envs,7),
                              self.GRASPING_FIELDS.OBJECT_POSE : current_object_pose.expand(self.num_envs,7),
                              self.GRASPING_FIELDS.GRIPPER_POSE : current_gripper_pose.expand(self.num_envs,7)}
        new_inst_state[self.STATE_GRASPING] = new_grasping_state
        
        # if not self._grasping_conf.observe_object_pose:
        new_camera_state = self._thzeros((self.num_envs,
                                        1,
                                        self._grasping_conf.camera_resolution_hw[1],
                                        self._grasping_conf.camera_resolution_hw[0]))
        new_inst_state[self.STATE_CAMERA] = new_camera_state
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
   

    # @adarl.utils.utils.th_compile_ext(copy_outs=True, mode="max-autotune",
    #                                 #   skip_eval_unsafe_warmup=100, skip_eval_unsafe_manual_arg_guard=0,
    #                                   disable=disable_compile)
    def compute_rewards(self,   state : dict[str,th.Tensor], 
                                sub_rewards_return: dict[str,th.Tensor] = {}) -> th.Tensor:
        max_rew = self._configuration.reward_penalties_max
        # curr_state_extr_vec = state[self.STATE_EXTRINSIC][:, 0,:,0]
        current_state_internal = state[self.STATE_INTERNAL][:, 0,:,0]
        state_action_raw_vec = state[self.STATE_ACT_RAW_HIST]
        state_stats_v_h_j_minmaxavgstd_pvaeep = state[self.STATE_JOINT_STEP_STATS].view(self.num_envs, 1, -1, 4, 6)
        last_step_dt = current_state_internal[:,self.INTERNAL_FIELDS.LAST_STEP_DT].view((self.num_envs,))
        
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
        position_safenorm   = state_robot_safenorm[:,0,:,0]


        # ---------------- JOINT-LEVEL PENALTIES ----------------

        reward_position         = flattened_joint_penalty_reward(norm_posstathomingdiff,max_rew=max_rew, exponent=2.0, flattening_scale=0.02)
        reward_actdiff          = joint_penalty_reward(actdiff,max_rew=1, exponent=2, presquash_factor=10)
        reward_actacc           = joint_penalty_reward(act_acc,max_rew=1, exponent=2, presquash_factor=100)
        reward_position_limit   = joint_penalty_reward(position_safenorm,max_rew=1,exponent=50)

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

        # ---------------- SAFETY TRIGGERED REWARD ----------------
        # This is a penalty for triggering safety mechanisms
        safety_triggered = th.logical_or(state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED,0],
                                         state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED,0])
        reward_safety_triggered = -1*safety_triggered

        # # FAILURE SCALING
        # failed = (curr_state_extr_vec[:,self.EXTRINSIC_FIELDS.BODY_ABS_POS_Z] < 0)
        # if self._configuration.fail_on_safety:
        #     failed = th.logical_or(failed, safety_triggered)

        # ----------------- GRASPING REWARDS ----------------

        obj_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.OBJECT_POSE,:3]
        goal_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GOAL_POSE,:3]
        gripper_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GRIPPER_POSE,:3]
        obj2goal_dist = th.linalg.norm(obj_position - goal_position, dim = -1)
        obj2hand_dist = th.linalg.norm(obj_position - gripper_position, dim = -1)

        reward_object_pose = 1-th.tanh(obj2goal_dist/0.5)
        reward_gripper_pose = 1-th.tanh(obj2hand_dist/0.5)


        raw_rewards = GraspVecEnv.SubRewards(
            health = th.ones_like(reward_position),
            joint_actacc = reward_actacc,
            joint_actdiff = reward_actdiff,
            joint_power = reward_power,
            joint_torque = reward_cmdtorque,
            safety_triggered = reward_safety_triggered,
            reward_object_pose = reward_object_pose,
            reward_gripper_pose = reward_gripper_pose,
            joint_position_limit = reward_position_limit
        )
        sub_rew_unscaled = th.stack([dataclasses.asdict(raw_rewards)[k] for k in self._sub_rewards_enabled], dim=1)
        sub_rew_scaled = sub_rew_unscaled*self._sub_rewards_enabled_weights_th.unsqueeze(0)*self._grasping_conf.reward_scale

        sub_rewards_return.update({k:sub_rew_scaled[:,i] for i,k in enumerate(self._sub_rewards_enabled.keys())})
        if self._grasping_conf.split_rewards:
            reward = sub_rew_scaled
            dbg_check_size(reward, (self._adapter.vec_size(),len(sub_rewards_return)), f"Unexpected reward size")
        else:
            reward = th.sum(sub_rew_scaled, dim =1, keepdim=True)
            dbg_check_size(reward, (self._adapter.vec_size(),1), f"Unexpected reward size")
        reward = th.clamp(reward, -self._configuration.reward_clamp, self._configuration.reward_clamp)
        
        dbg_check_finite(sub_rewards_return, async_assert=True, assert_msg="Nonfinite sub rewards detected")
        
        return reward









    # def _update_stats(self):
    #     super()._update_stats()

    #     self._stats["obj2hand_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
    #     self._stats["obj2goal_dist"] = self._thzeros((self._configuration.vec_size, self._buff_sizes))
    #     self._stats["ep_obj2hand_dist"] = self._thzeros((self._configuration.vec_size,))
    #     self._stats["ep_obj2goal_dist"] = self._thzeros((self._configuration.vec_size,))
    #     self._stats["ep_obj_travel"] = self._thzeros((self._configuration.vec_size,))

    #     obj_pose  = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.OBJECT_POSE]
    #     goal_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GOAL_POSE]
    #     gripper_pose = self._current_state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GRIPPER_POSE]
    #     obj2goal_dist = th.linalg.norm(obj_pose[:,:3]-goal_pose[:,:3], dim = -1)
    #     obj2hand_dist = th.linalg.norm(obj_pose[:,:3]-gripper_pose[:,:3], dim = -1)
    #     prev_obj_pose  = self._current_state[self.STATE_GRASPING][:,1,self.GRASPING_FIELDS.OBJECT_POSE]
    #     obj_travel = th.linalg.norm(obj_pose[:,:3]-prev_obj_pose[:,:3], dim = -1)

    #     step_counts = self._current_state[self.STATE_INTERNAL][:,0,self.INTERNAL_FIELDS.STEP_COUNT,0].to(th.long)
    #     dbg_check_size(step_counts, (self._adapter.vec_size(),))
        
    #     # Update episode averages
    #     self._stats["ep_obj2hand_dist"]          = (self._stats["ep_obj2hand_dist"]*(step_counts-1) + obj2hand_dist)/step_counts # Elements with step_count == 0 will be inf
    #     self._stats["ep_obj2goal_dist"]          = (self._stats["ep_obj2goal_dist"]*(step_counts-1) + obj2goal_dist)/step_counts # Elements with step_count == 0 will be inf
    #     self._stats["ep_obj_travel"]             = (self._stats["ep_obj_travel"] + obj_travel) # Elements with step_count == 0 will be inf
    #     # Correct the episode averages for episodes that have just started
    #     starting_eps = step_counts==0
    #     masked_assign(self._stats["ep_obj2hand_dist"],      starting_eps, obj2hand_dist)
    #     masked_assign(self._stats["ep_obj2goal_dist"],      starting_eps, obj2goal_dist)
    #     masked_assign(self._stats["ep_obj_travel"],         starting_eps, obj_travel)
    #     # Fill the buffers for episodes that have just staretd
    #     masked_assign(self._stats["obj2hand_dist"],     step_counts==0, obj2hand_dist.unsqueeze(1).expand(-1, self._buff_sizes))
    #     masked_assign(self._stats["obj2goal_dist"],     step_counts==0, obj2goal_dist.unsqueeze(1).expand(-1, self._buff_sizes))
        
    #     # Update the buffers
    #     # idxs = step_counts%self._buff_sizes
    #     idxs = step_counts%self._stats["obj2hand_dist"].size()[1]
    #     # print(f"torch.is_grad_enabled()) = {th.is_grad_enabled()}")
    #     # print(f"idx.size() = {idxs.size()}, idx = {idxs}")
    #     # print(f"vel_error_vec.size() = {vel_error_vec.size()}, {vel_error_vec}")
    #     self._stats["obj2hand_dist"][:,idxs] = obj2hand_dist
    #     self._stats["obj2goal_dist"][:,idxs] = obj2goal_dist
   
    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[Any,Any]:
        i = super().get_infos(state=state, labels=labels)
        
        obj_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.OBJECT_POSE,:3]
        goal_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GOAL_POSE,:3]
        gripper_position = state[self.STATE_GRASPING][:,0,self.GRASPING_FIELDS.GRIPPER_POSE,:3]
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

        if self._configuration.verbose_infos:
            statenorm = self._state_helper.normalize(state)
            for substate in [self.STATE_GRASPING]:
                i["state_"+substate] = self._state_helper.sub_helpers[substate].flatten(state[substate])
                i["statenorm_"+substate] = self._state_helper.sub_helpers[substate].flatten(statenorm[substate])
                # Would make sense to put the labels in the info_space definition, maybe make an info_helper?
                if labels is not None:
                    labels["state_"+substate] =  to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())
                    labels["statenorm_"+substate] = to_string_tensor(self._state_helper.sub_helpers[substate].flat_state_names())

        return i
    
    def _set_current_ep_config(self, vec_mask : th.Tensor, reset_options : dict = {}):
        if "goal_pose" in reset_options:
            sampled_goal_obj_pose_v_xyzxyzw = th.as_tensor(reset_options["goal_pose"],device=self._configuration.th_device)
        else:
            sampled_goal_obj_pose_v_xyzxyzw = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                                self._grasping_conf.goal_obj_area_minmax_xyz[0],
                                                self._grasping_conf.goal_obj_area_minmax_xyz[1])
            goal_obj_quat = ros_rpy_to_quaternion_xyzw_th(self._thrand((self.num_envs,3))*math.pi*2)
            sampled_goal_obj_pose_v_xyzxyzw = th.cat([sampled_goal_obj_pose_v_xyzxyzw, goal_obj_quat], dim=1).view((self.num_envs,7))
            
        sampled_initial_object_position = unnormalize(self._thrand((self.num_envs,3))*2-1,
                                              self._grasping_conf.init_obj_area_minmax_xyz[0],
                                              self._grasping_conf.init_obj_area_minmax_xyz[1])
        sampled_initial_object_yaw = self._thrand((self.num_envs,))*math.pi*2
        sampled_initial_obj_quat = ros_rpy_to_quaternion_xyzw_th(th.cat([self._thzeros((self.num_envs,2)),sampled_initial_object_yaw.unsqueeze(1)], dim=1))
        sampled_initial_obj_pose = th.cat([sampled_initial_object_position, 
                                   sampled_initial_obj_quat], dim=1).view((self.num_envs,7))
        super()._set_current_ep_config(vec_mask=vec_mask, reset_options=reset_options)
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

    def _set_goal_marker_pose(self, vec_mask : th.Tensor):
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._adapter.setLinksStateDirect(  link_names=[self._goal_marker_base_link],
                                                link_states_pose_vel=th.cat([self._grasping_episode_config.goal_object_pose,
                                                                             self._thzeros((self.num_envs,6,))], dim = 1).unsqueeze(1),
                                                vec_mask=vec_mask)

    @override
    def _simulation_initialization(self, vec_mask : th.Tensor):
        super()._simulation_initialization(vec_mask = vec_mask)
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            obj_state = self._thzeros((self.num_envs,1,13))
            obj_state[:,0,:7] = self._grasping_episode_config.initial_object_pose[:]
            self._adapter.setLinksStateDirect(link_names=[self._grasping_conf.target_object_link],
                                              link_states_pose_vel=obj_state,
                                              vec_mask=vec_mask)
        else:
            raise RuntimeError(f"Cannot run simulation initialization on non-simulated adapter")
            
    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        self.set_cam_pose((2.5, 30/180*3.14159, -200/180*3.14159))
        if self._grasping_conf.use_head_cam_as_ui_camera:
            if th.any(vec_mask[1:]):
                raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
            try:
                head_imgs, head_times = self._adapter.getRenderings([self._head_camera_name], vec_mask=vec_mask)
                return head_imgs, head_times
                external_imgs, external_times = super().get_ui_renderings(vec_mask=vec_mask)
                imgs = head_imgs + external_imgs
                times = th.cat([head_times,external_times], dim = -1)
                return imgs, times
            except Exception as e:
                ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
                return [], th.empty((0,))
        else:
            if isinstance(self._adapter, BaseVecSimulationAdapter):
                self._set_goal_marker_pose(vec_mask=self._all_vecs)        
            return super().get_ui_renderings(vec_mask=vec_mask)
    
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
                                            kwargs={"add_world_link":str(is_pybullet)})
        spawn_defs.append(self._cube_spawn_def)
        if not hasattr(self,"_axes_spawn_def"):
            self._axes_spawn_def = ModelSpawnDef(   definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl_envs","models/axes.urdf.xacro")).read_text(),
                                                    name="goal_axes",
                                                    pose=None,
                                                    format="urdf.xacro",
                                                    kwargs={"add_world_link":str(is_pybullet),
                                                            "size" : 0.2})
            if not is_pybullet:
                self._goal_marker_base_link = ("goal_axes","root")
            else:
                self._goal_marker_base_link = ("goal_axes","world")
        spawn_defs.append(self._axes_spawn_def)
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
                                                    "fixed_joint_xyz" : f"0.8 0.0 {self._table_height-0.5}",
                                                    "fixed_joint_rpy" : "0 0 0"},)
        spawn_defs.append(self._table_spawn_def)
        if adarl.utils.utils.isinstance_noimport(self._adapter, ("MjxAdapter", "MujocoAdapter")):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:            
            cam_file = "models/simple_camera.sdf.xacro"
        if not hasattr(self,"_head_camera"):
            if self._configuration.robot_name == "centauro":
                self._cam_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                                name=self._head_camera_name,
                                                pose=None,
                                                format="sdf.xacro",
                                                kwargs={"camera_width":self._grasping_conf.camera_resolution_hw[0],
                                                        "camera_height":self._grasping_conf.camera_resolution_hw[1],
                                                        "frame_rate":1/self._intendedStepLength_sec,
                                                        "camera_name": self._head_camera_name},
                                                attachment_link=("centauro","D435_head_camera_link"))
            else:
                self._cam_spawn_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                                name=self._head_camera_name,
                                                pose=None,
                                                format="sdf.xacro",
                                                kwargs={"camera_width":self._grasping_conf.camera_resolution_hw[0],
                                                        "camera_height":self._grasping_conf.camera_resolution_hw[1],
                                                        "frame_rate":1/self._intendedStepLength_sec,
                                                        "camera_name": self._head_camera_name,
                                                        "position_xyz": "1.0 0 1.5",
                                                        "orientation_wxyz": "0.707 0 0.707 0"},)
        spawn_defs.append(self._cam_spawn_def)
        return spawn_defs