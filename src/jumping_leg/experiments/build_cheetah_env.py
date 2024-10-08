
from __future__ import annotations
import adarl.utils.utils
from jumping_leg.experiments.build_locomotion_env import locomotion_env_builder

def env_builder(seed,
                    log_folder,
                    is_eval,
                    env_builder_args : dict,
                    no_dict = False):
    model_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/cheetah.urdf.xacro")
    homing_joint_pose={}
    disallowed_contact_links = [("quad","thigh_link_back_left"),
                                                        ("quad","shin_link_back_left"),
                                                        ("quad","thigh_link_back_right"),
                                                        ("quad","shin_link_back_right"),
                                                        ("quad","thigh_link_front_left"),
                                                        ("quad","shin_link_front_left"),
                                                        ("quad","thigh_link_front_right"),
                                                        ("quad","shin_link_front_right"),
                                                        ("quad","body_link")]
    terminating_contact_pairs=[(("cheetah","body_link"),("ground_plane","planeLink"))]
    robot_name="cheetah"
    robot_main_body_link="body_link"
    homing_body_pose_xyz_xyzw=(0.,0.,0.5,0.,0.,0.,1.)

    return locomotion_env_builder(seed = seed,
                                    log_folder = log_folder,
                                    is_eval = is_eval,
                                    env_builder_args = env_builder_args,
                                    model_file = model_file,
                                    no_dict = no_dict,
                                    homing_body_pose_xyz_xyzw=homing_body_pose_xyz_xyzw,
                                    homing_joint_pose=homing_joint_pose,
                                    disallowed_contact_links=disallowed_contact_links,
                                    terminating_contact_pairs=terminating_contact_pairs,
                                    robot_name=robot_name,
                                    robot_main_body_link=robot_main_body_link)