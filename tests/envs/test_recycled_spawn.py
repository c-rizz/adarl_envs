"""Initial poses with randomization_recycle_init_pose, as loco_kyon_ppo sets it.

The poses are sampled once and reused at each reset. For months the reuse went to the homing pose
instead, so after the first sample every robot started at the origin in the same joint configuration,
with nothing to show for it but the spawn box having no effect.
"""
import importlib.util

import pytest
import torch as th

# checked without importing it: adarl picks the MuJoCo GL backend before mujoco's first import
if importlib.util.find_spec("mujoco") is None:
    pytest.skip("mujoco is not installed", allow_module_level=True)
from _env_builders import build_loco_env

pytestmark = [pytest.mark.mujoco, pytest.mark.slow]


@pytest.fixture(scope="module")
def loco(tmp_path_factory):
    return build_loco_env(str(tmp_path_factory.mktemp("recycled_spawn")), {"init_on_reset_ratio": 1.0})


def _body_position(env):
    return env._adapter.getLinksState(requestedLinks=env._feet_and_body_link_ids, use_com_pose=False)[0, -1, :3].clone()


def _joint_positions(env):
    return env.get_states()[env.STATE_ROBOT][0, 0, :, 0].clone()


def test_resets_go_back_to_the_sampled_pose_and_not_to_homing(loco):
    env, runner = loco.env, loco.runner
    assert env._configuration.init_args.randomization_recycle_init_pose
    runner.reset()
    spawn_xy, spawn_joints = _body_position(env)[:2], _joint_positions(env)

    spawn_box = env._configuration.homing_body_pose_minmax_xyz[:, :2]
    assert th.all(spawn_xy >= spawn_box[0]) and th.all(spawn_xy <= spawn_box[1])
    assert not th.allclose(spawn_xy, env._configuration.homing_body_pose_xyz_xyzw[:2], atol=1e-3), "spawned at the homing pose"
    assert not th.allclose(spawn_joints, env._configuration.homing_ctrl_joints_position, atol=1e-3), "joints at the homing pose"

    for _ in range(3):
        for _ in range(5):
            runner.step(th.zeros((env.num_envs,)+tuple(env.single_action_space.shape)), autoreset=False)
        runner.reset()
        th.testing.assert_close(_body_position(env)[:2], spawn_xy, atol=1e-3, rtol=0.0)
        th.testing.assert_close(_joint_positions(env), spawn_joints, atol=1e-3, rtol=0.0)
