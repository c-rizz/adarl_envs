#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, SupportsFloat
import gymnasium as gym
from typing import Any, SupportsFloat, Mapping
from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter
from adarl_ros.adapters.RosXbotGazeboAdapter import RosXbotGazeboAdapter
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
import torch as th
import numpy as np
import time
from adarl.utils.utils import LinkState, pkgutil_get_path, build_pose, JointState
import adarl.utils.dbg.ggLog as ggLog


class LegReachMinimal(gym.Env):
    def __init__(self, show_gui : bool= False, mode : str ="gazebo-xbot"):
        self.observation_space = gym.spaces.Box(low = -1.0, high = 1.0, shape=(19,))
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (10,))
        self._stepLength_sec = 0.02
        self._show_gui = show_gui

        if mode == "gazebo-xbot":
            self._adapter : BaseJointImpedanceAdapter = RosXbotGazeboAdapter(model_name = "leg",
                                                                            stepLength_sec = self._stepLength_sec,
                                                                            forced_ros_master_uri = None,
                                                                            maxObsDelay = float("+inf"),
                                                                            blocking_observation = False,
                                                                            is_floating_base = True,
                                                                            reference_frame = "base_link",
                                                                            torch_device = th.device("cpu"),
                                                                            fallback_cmd_stiffness = 200.0,
                                                                            fallback_cmd_damping = 100.0,
                                                                            allow_fallback = True,
                                                                            jpos_cmd_max_vel = {},
                                                                            jpos_cmd_max_vel_default = 10.0,
                                                                            jpos_cmd_max_acc = {},
                                                                            jpos_cmd_max_acc_default = 10.0)
        elif mode == "pybullet":
            self._adapter = PyBulletJointImpedanceAdapter(  stepLength_sec=self._stepLength_sec,
                                                            restore_on_reset=False,
                                                            debug_gui=show_gui,
                                                            simulation_step=1/1024,
                                                            enable_redering=False,
                                                            global_max_torque_position_control = 100)
        else:
            raise NotImplementedError()
        self._knee_joint = ("leg","knee_joint_1")
        self._hip_joint = ("leg","hip_joint_1")
        self._thigh_base_link = ("leg", "thigh_link1")
        self._shin_base_link = ("leg", "shin_link1")
        self._thigh_com_link = ("leg", "thigh_link1_com")
        self._shin_com_link = ("leg", "shin_link1_com")
        self._rail_joint = ("leg","rail_joint")
        self._camera_name = "simple_camera"
        self._last_sent_pvesd = {self._hip_joint  : [0,0,0,0,0],
                                 self._knee_joint : [0,0,0,0,0]}
        self._hip_goal_z = 0.6
        self._step_count = 0
        self._action_max_pvesd = np.array([ 2.4, 20, 100,1000, 400, 2.4, 20, 100,1000, 400])
        self._action_min_pvesd = np.array([-2.4,-20,-100,  10,  10,-2.4,-20,-100,  10,  10])
        
        self._obs_max = np.array([  2.5,
                                    30,
                                    150,
                                    2.5,
                                    30,
                                    150,
                                    3,
                                    50,
                                    3,
                                    2.5,
                                    30,
                                    150,
                                    1000,
                                    400,
                                    2.5,
                                    30,
                                    150,
                                    1000,
                                    400])
        self._obs_min = -self._obs_max
        self._obs_min[[12,13,17,18]] = 10,10,10,10 # stiffness and damping for hip and knee
        self._real = False # some day
        self._spawned = False
        self._build_scenario()
        self._adapter.set_monitored_links([self._shin_com_link,self._shin_base_link,self._thigh_com_link,self._thigh_base_link])
        # self._adapter.set_monitored_cameras([self._camera_name])
        self._adapter.startup()
        super().__init__()

    def _get_obs_rew(self):
        jstates = self._adapter.getJointsState()
        # jstates = self._adapter.getJointsState(requestedJoints=[self._knee_joint, self._hip_joint])
        lstates = self._adapter.getLinksState(requestedLinks = [self._thigh_com_link,
                                                                self._shin_com_link,
                                                                self._thigh_base_link])
        hip_height = lstates[self._thigh_base_link].pose.position[2]
        hip_vel_z = lstates[self._thigh_base_link].pos_velocity_xyz[2]
        print(f"Position: knee = {jstates[0,0]:.3f} hip = {jstates[1,0]:.3f}")
        print(f"Effort:   knee = {jstates[0,2]:.3f} hip = {jstates[1,2]:.3f}")
        obs = np.concatenate([jstates.detach().flatten().cpu().numpy(),
                              np.array([hip_height,
                                        hip_vel_z,
                                        self._hip_goal_z,
                                        self._last_sent_pvesd[self._hip_joint][0],
                                        self._last_sent_pvesd[self._hip_joint][1],
                                        self._last_sent_pvesd[self._hip_joint][2],
                                        self._last_sent_pvesd[self._hip_joint][3],
                                        self._last_sent_pvesd[self._hip_joint][4],
                                        self._last_sent_pvesd[self._knee_joint][0],
                                        self._last_sent_pvesd[self._knee_joint][1],
                                        self._last_sent_pvesd[self._knee_joint][2],
                                        self._last_sent_pvesd[self._knee_joint][3],
                                        self._last_sent_pvesd[self._knee_joint][4]])])
        # np.array([jstates[self._hip_joint].position[0],
        #                 jstates[self._hip_joint].rate[0],
        #                 jstates[self._hip_joint].effort[0],
        #                 jstates[self._knee_joint].position[0],
        #                 jstates[self._knee_joint].rate[0],
        #                 jstates[self._knee_joint].effort[0],
        #                 hip_height,
        #                 hip_vel_z,
        #                 self._hip_goal_z,
        #                 self._last_sent_pvesd[self._hip_joint][0],
        #                 self._last_sent_pvesd[self._hip_joint][1],
        #                 self._last_sent_pvesd[self._hip_joint][2],
        #                 self._last_sent_pvesd[self._hip_joint][3],
        #                 self._last_sent_pvesd[self._hip_joint][4],
        #                 self._last_sent_pvesd[self._knee_joint][0],
        #                 self._last_sent_pvesd[self._knee_joint][1],
        #                 self._last_sent_pvesd[self._knee_joint][2],
        #                 self._last_sent_pvesd[self._knee_joint][3],
        #                 self._last_sent_pvesd[self._knee_joint][4]])
        obs = (obs-self._obs_min)/(self._obs_max-self._obs_min)*2-1
        reward = 1-hip_height-self._hip_goal_z
        return obs, reward

    def _action_to_jimp(self, action) -> Mapping[tuple[str,str], tuple[float,float,float,float,float]]:
        action = (action+1)/2*(self._action_max_pvesd-self._action_min_pvesd)+self._action_min_pvesd
        return {self._knee_joint : action[0:5],
                self._hip_joint  : action[5:10]}
    
    def _jimp_to_action(self, jimp : Mapping[tuple[str,str], tuple[float,float,float,float,float]]) -> np.ndarray:
        action = np.concatenate([jimp[self._knee_joint], jimp[self._hip_joint]])
        action = (action - self._action_min_pvesd)/(self._action_max_pvesd-self._action_min_pvesd)*2-1
        return action

    def step(self, action: th.Tensor) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
            # ggLog.info(f"Step {self._step_count}")
            self._last_sent_pvesd = self._action_to_jimp(action)
            # ggLog.info(f"Setting jimp cmd {self._last_sent_pvesd}")
            use_th_interface = True
            if use_th_interface:
                action_unnorm = (action+1)/2*(self._action_max_pvesd-self._action_min_pvesd)+self._action_min_pvesd
                self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = action_unnorm.reshape(2,5),
                                                        delay_sec = 0.0)
            else:
                self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = self._last_sent_pvesd,
                                                        delay_sec = 0.0)
            
            
            # ggLog.info(f"Stepping")
            dt = self._adapter.step()
            # ggLog.info(f"Stepped of {dt}s")

            th.set_printoptions(sci_mode=False)
            ggLog.info(f"Joint stats:\n{self._adapter.get_joints_state_step_stats()}")
            th.set_printoptions()


            obs, reward = self._get_obs_rew()
            truncated = self._step_count > 500
            terminated = False
            info = {}
            self._step_count += 1
            return obs, reward, terminated, truncated, info
            
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if not self._spawned and isinstance(self._adapter, BaseSimulationAdapter):
            leg_model_name = "leg"
            cam_model_name = "camera"
            leg_pose = build_pose(0,0,0,0,0,0,1)
            self._spawned = True
            if isinstance(self._adapter, PyBulletAdapter):
                leg_file = pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")
                # import rospkg
                # leg_file = rospkg.RosPack().get_path("protoleg")+"/description/urdf/protoleg_test_rig.urdf.xacro"
                self._adapter.spawn_model(  model_file=leg_file,
                                            model_name=leg_model_name,
                                            pose=leg_pose,
                                            model_format="urdf.xacro")
            # self._adapter.spawn_model(  model_file=pkgutil_get_path("adarl","models/simple_camera.sdf.xacro"),
            #                             model_name=cam_model_name,
            #                             pose=build_pose(0,2.5,0.7, 0.0,0.0,-0.707,0.707),
            #                             model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._stepLength_sec},
            #                             model_format="sdf.xacro")
            
        self._adapter.set_monitored_joints([self._knee_joint,self._hip_joint])
        self._adapter.set_impedance_controlled_joints([self._knee_joint,self._hip_joint])

        if isinstance(self._adapter, BaseSimulationAdapter):
            self._simulation_initialization()
        else:
            raise NotImplementedError()
        obs = self._get_obs_rew()[0]
        info = {}
        self._step_count = 0
        return obs, info
    

    def _simulation_initialization(self):
        if isinstance(self._adapter, BaseSimulationAdapter):
            ggLog.info(f"Moving to initial pose...")
            rpos, hpos, kpos = 0.8, 3.14159/4,  3.14159/2
            self._adapter.setJointsStateDirect({self._rail_joint: JointState(position = rpos, rate=0, effort=0),
                                                self._hip_joint:  JointState(position = hpos, rate=0, effort=0),
                                                self._knee_joint: JointState(position = kpos, rate=0, effort=0)})
            ggLog.info(f"Moved to initial pose.")
            # start_action = np.zeros(shape=(10,),dtype = np.float32)
            # start_jimp = self._action_to_jimp(start_action)         
            start_jimp : dict[tuple[str,str], tuple[float,float,float,float,float]] = { self._knee_joint : (kpos,0.,0.,200.,50.),
                                                                                        self._hip_joint  : (hpos,0.,0.,200.,50.)}
            start_action = self._jimp_to_action(start_jimp)
            ggLog.info(f"Setting initial jimp command")
            self._adapter.setJointsImpedanceCommand(start_jimp)
            ggLog.info(f"Set jimp cmd")
            self._adapter.apply_joint_impedances(start_jimp)
            ggLog.info(f"Applied jimp cmd")
            ggLog.info(f"Settling...")
            self._adapter.run(3.0) # let the leg fall
            ggLog.info(f"Sim init done.")
            self._last_sent_pvesd = start_jimp
            self._last_out_action = start_action
        else:
            raise RuntimeError(f"called simulation initialization with non-simulated adapter")
    
    def _build_scenario(self):
        # ggLog.info("Building env")
        envCtrlName = type(self._adapter).__name__
        if envCtrlName == "PyBulletJointImpedanceAdapter":
            self._adapter.build_scenario()
        elif envCtrlName in ["RosXbotAdapter", "RosXbotGazeboAdapter"]:
            if self._real:
                raise NotImplementedError()
            else:
                self._adapter.build_scenario(launch_file_pkg_and_path = pkgutil_get_path("jumping_leg",
                                                                                         "gazebo/all_gazebo_xbot.launch"),
                                                launch_file_args={"gui":self._show_gui})
        else:
            raise RuntimeError(f"unexpected adapter type {type(self._adapter)}")
        
    def close(self):
        self._adapter.destroy_scenario()
        return super().close()
    

if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.set_defaults(feature=True)
    ap.add_argument("--mode", default="pybullet", type=str, help="Adapter to use ('pybullet','gazebo-xbot')")
    args = vars(ap.parse_args())

    env = LegReachMinimal(show_gui = True, mode=args["mode"])

    def zero(obs):
        return th.zeros(size=(10,)), None
    
    policy_time = 0
    def oscillate_policy(obs):
        import math
        hz = 50
        rate = 0.5
        hip_range_rad = 1.1
        knee_range_rad = 2.2
        global policy_time
        hip_pos_rad = obs[0]*2.4
        start_hip_pos = 3.14159/4
        #time offset to make the starting pose correspond with time 0
        time_offset = (math.asin(start_hip_pos/hip_range_rad)+2*3.14159)/(rate*2*3.14159)
        
        t = policy_time + time_offset
        policy_time+=1/hz

        href_rad = hip_range_rad*math.sin(t*(2*3.14159)*rate)
        kref_rad = knee_range_rad*math.sin(t*(2*3.14159)*rate)

        print(f"t = {t} href = {href_rad:.3f}={href_rad/2.4:.3f} kref {kref_rad:.3f}={kref_rad/2.4:.3f}")
        return th.tensor([kref_rad/2.4, 0, 0, 0.0, -0.5,
                          href_rad/2.4, 0, 0, 0.0, -0.5]), None
    def ep_done_cb(episodeReward, steps, episode):
        global policy_time
        policy_time = 0 # reset time
        import adarl.utils.session
        # adarl.utils.session.default_session.run_info["collected_episodes"] += 1
        # adarl.utils.session.default_session.run_info["collected_steps"] += steps
        # print(f"Completed episode {adarl.utils.session.default_session.run_info['collected_episodes']}")
    from adarl.utils.utils import evaluatePolicy
    res = evaluatePolicy(env = env, model = None, episodes = 5, predict_func=oscillate_policy,
                                                images_return = None, obs_return=None,
                                                on_ep_done_callback=ep_done_cb)
    print(f"evaluation returned {res}")

    env.close()