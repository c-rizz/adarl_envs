

def quad_env_builder(seed,
                    log_folder,
                    is_eval,
                    env_builder_args : dict,
                    no_dict = False):
    import adarl.utils.utils
    from adarl_envs.experiments.build_locomotion_env import locomotion_env_builder
    from adarl_envs.env.LocomotionEnv import LocomotionEnv    
    model_file = adarl.utils.utils.pkgutil_get_path("adarl_envs","models/quad_simple.urdf.xacro")
    homing_joint_pose={ ("quad","hip_joint_x_back_left") : -3.14159*0.4,
                        ("quad","hip_joint_x_back_right") : -3.14159*0.4,
                        ("quad","hip_joint_x_front_left") : -3.14159*0.4,
                        ("quad","hip_joint_x_front_right") : -3.14159*0.4,
                        ("quad","hip_joint_y_back_left") : 0.75,
                        ("quad","hip_joint_y_back_right") : 0.75,
                        ("quad","hip_joint_y_front_left") : 0.75,
                        ("quad","hip_joint_y_front_right") : 0.75,
                        ("quad","knee_joint_back_left") : 1.8,
                        ("quad","knee_joint_back_right") : 1.8,
                        ("quad","knee_joint_front_left") : 1.8,
                        ("quad","knee_joint_front_right") : 1.8}
    disallowed_contact_links = [("quad","thigh_link_back_left"),
                                ("quad","shin_link_back_left"),
                                ("quad","thigh_link_back_right"),
                                ("quad","shin_link_back_right"),
                                ("quad","thigh_link_front_left"),
                                ("quad","shin_link_front_left"),
                                ("quad","thigh_link_front_right"),
                                ("quad","shin_link_front_right"),
                                ("quad","body_link")]
    terminating_contact_pairs=[(("quad","body_link"),("ground_plane","planeLink"))]
    robot_name="quad"
    robot_main_body_link="body_link"
    robot_root_link="body_link"
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
                                    robot_main_body_link=robot_main_body_link,
                                    robot_root_link=robot_root_link,
                                    controlled_joints=[LocomotionEnv.JOINT_FILTERS.ALL_REVOLUTE])
