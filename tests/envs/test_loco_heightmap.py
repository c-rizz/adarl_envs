"""The terrain map of LocomotionVecEnv, on the kyon env of loco_kyon_ppo in the pyramids world.

The pyramids are axis-aligned boxes, so Robot.get_ground_z gives the terrain height at any point
independently of the render path the map comes from.
"""
import importlib.util

import numpy as np
import pytest
import torch as th

# checked without importing it: adarl picks the MuJoCo GL backend before mujoco's first import
if importlib.util.find_spec("mujoco") is None:
    pytest.skip("mujoco is not installed", allow_module_level=True)
from _env_builders import build_loco_env

pytestmark = [pytest.mark.slow]
requires_gpu = pytest.mark.skipif(not th.cuda.is_available(), reason="needs a GPU")

RESOLUTION_XY = (20, 12)
SIZE_XY = (2.0, 1.2)
NUM_ENVS = 8
ENV_ARGS = {"heightmap_resolution_xy": RESOLUTION_XY, "heightmap_size_xy": SIZE_XY, "observe_heightmap": True}


@pytest.fixture(scope="module")
def loco(tmp_path_factory):
    return build_loco_env(str(tmp_path_factory.mktemp("loco_heightmap")), ENV_ARGS,
                          mode="mjx", world="pyramids", num_envs=NUM_ENVS)


def _expected_heightmap(env):
    """Heights of the terrain around each body from its pose, with no use of the sensor."""
    from adarl.utils.utils import th_quat_rotate
    body = env._adapter.getLinksState(requestedLinks=env._feet_and_body_link_ids, use_com_pose=False)[:, -1]
    orient = env.to_gait_frame_orientation(body[:, 3:7])
    forward = th_quat_rotate(th.tensor([1.0, 0.0, 0.0], device=orient.device).expand(NUM_ENVS, 3), orient)
    yaw = th.atan2(forward[:, 1], forward[:, 0]).cpu().numpy()
    body = body.cpu().numpy()
    local_x = (np.arange(RESOLUTION_XY[0])+0.5)/RESOLUTION_XY[0]*SIZE_XY[0] - SIZE_XY[0]/2
    local_y = (np.arange(RESOLUTION_XY[1])+0.5)/RESOLUTION_XY[1]*SIZE_XY[1] - SIZE_XY[1]/2
    x, y = np.meshgrid(local_x, local_y, indexing="ij")
    expected = []
    for e in range(NUM_ENVS):
        points = np.stack([body[e, 0] + np.cos(yaw[e])*x - np.sin(yaw[e])*y,
                           body[e, 1] + np.sin(yaw[e])*x + np.cos(yaw[e])*y], axis=-1)
        ground = env._robot_model.get_ground_z(points.reshape(-1, 2)).reshape(RESOLUTION_XY)
        expected.append(ground - body[e, 2])
    return np.clip(np.stack(expected), *env.HEIGHTMAP_MINMAX)


def _state_heightmap(env):
    return env.get_states()[env.STATE_HEIGHTMAP][:, 0, 0].cpu().numpy()


def _assert_matches_terrain(env):
    heightmap, expected = _state_heightmap(env), _expected_heightmap(env)
    assert heightmap.shape == expected.shape
    # without steps in the maps the comparison would prove little: all the robots could be on flat ground
    assert (heightmap.max(axis=(1, 2)) - heightmap.min(axis=(1, 2)) > 0.03).any(), "no map contains a step"
    # a point right on a step edge can fall on either side, depending on rounding
    wrong = np.abs(heightmap - expected) > 2e-3
    assert wrong.mean() < 0.01, f"{wrong.mean():.1%} of the cells are wrong, worst error {np.abs(heightmap - expected).max()}"


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_map_has_its_own_flat_privileged_observation(loco):
    env = loco.env
    space = env._state_helper.get_vec_obs_space()
    assert space["privileged.heightmap"].shape == (NUM_ENVS, RESOLUTION_XY[0]*RESOLUTION_XY[1])
    others = {k: v for k, v in env._state_helper.observation_names().items() if k != "privileged.heightmap"}
    in_others = [k for k, names in others.items() if any("heightmap" in str(n) for n in np.asarray(names).reshape(-1))]
    assert in_others == [] and "heightmap" not in str(list(others.keys()))

    obs = env.get_observations(env.get_states())
    normalized = env._state_helper.sub_helpers[env.STATE_HEIGHTMAP].normalize(env.get_states()[env.STATE_HEIGHTMAP])
    th.testing.assert_close(obs["privileged.heightmap"], normalized.flatten(start_dim=1))


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_map_is_the_terrain_around_each_body_after_a_reset(loco):
    loco.runner.reset()
    _assert_matches_terrain(loco.env)


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_map_follows_the_body_while_stepping(loco):
    env = loco.env
    loco.runner.reset()
    generator = th.Generator().manual_seed(0)
    for _ in range(20):
        actions = (th.rand((NUM_ENVS,)+tuple(env.single_action_space.shape), generator=generator)*2 - 1)*0.3
        loco.runner.step(actions.to(env._th_device), autoreset=False)
    _assert_matches_terrain(env)


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_feet_contacts_are_detected_on_the_floor_and_on_the_pyramids(loco):
    """Regression: only the empty ground link of the ground model was monitored, while the terrain of
    a world is in the world model, so on the pyramids no foot ever touched the ground, and all the
    rewards based on contacts were silently off."""
    env = loco.env
    loco.runner.reset()
    for _ in range(40):
        loco.runner.step(th.zeros((NUM_ENVS,)+tuple(env.single_action_space.shape), device=env._th_device), autoreset=False)
    feet = env.get_states()[env.STATE_FEET][:, 0].cpu().numpy()
    feet_z = feet[:, env.FEET_FIELDS.FEET_ABS_POS_Z]
    contact = feet[:, env.FEET_FIELDS.FEET_CONTACT] > 0.5
    assert contact[feet_z < 0.05].any(), "no contact with the floor"
    assert contact[feet_z >= 0.05].any(), "no contact with the pyramids"
    assert contact.mean() > 0.5, f"only {contact.mean():.0%} of the feet are in contact, with the robots standing still"


def _body_z_and_height(env):
    body_z = env._adapter.getLinksState(requestedLinks=env._feet_and_body_link_ids, use_com_pose=False)[:, -1, 2].cpu().numpy()
    height = env.get_states()[env.STATE_EXTRINSIC][:, 0, env.EXTRINSIC_FIELDS.BODY_HEIGHT_FROM_GROUND, 0].cpu().numpy()
    return body_z, height


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_body_height_is_measured_from_the_ground_under_the_body(loco):
    env = loco.env
    loco.runner.reset()
    body_z, height = _body_z_and_height(env)
    cells_x, cells_y = env._ground_patch_cells_xy
    ground_z = (_expected_heightmap(env) + body_z[:, None, None])[:, cells_x, cells_y].mean(axis=(1, 2))
    assert (ground_z > 0.03).any(), "no robot is on raised terrain"
    np.testing.assert_allclose(height, body_z - ground_z, atol=0.01)


@pytest.mark.mjx
@pytest.mark.gpu
@requires_gpu
def test_terrain_is_measured_even_if_the_map_is_not_observed(tmp_path):
    built = build_loco_env(str(tmp_path), {}, mode="mjx", world="pyramids", num_envs=NUM_ENVS)
    env = built.env
    assert "privileged.heightmap" not in env._state_helper.observation_names()
    assert env.STATE_HEIGHTMAP not in env.get_states()
    body_z, height = _body_z_and_height(env)
    assert (body_z - height > 0.03).any(), "the height is still the world z"


@pytest.mark.mujoco
def test_an_adapter_with_no_sensor_says_so(tmp_path):
    with pytest.raises(NotImplementedError):
        build_loco_env(str(tmp_path), ENV_ARGS)


@pytest.mark.mujoco
@pytest.mark.parametrize("world", ["flat_ground", "pyramids"])
def test_without_a_sensor_the_height_is_the_world_z(tmp_path, world):
    built = build_loco_env(str(tmp_path), {"init_on_reset_ratio": 1.0}, world=world) # the env lives as long as this
    env = built.env
    assert not env._heightmap_sensor_enabled
    body_z = env._adapter.getLinksState(requestedLinks=env._feet_and_body_link_ids, use_com_pose=False)[0, -1, 2]
    height = env.get_states()[env.STATE_EXTRINSIC][0, 0, env.EXTRINSIC_FIELDS.BODY_HEIGHT_FROM_GROUND, 0]
    assert height == pytest.approx(float(body_z))
