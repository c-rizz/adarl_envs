"""Reward terms of GraspVecEnv, on the kyon env of solve_grasp_env.

Each test takes the state right after a reset, overwrites only the fields a term reads, and checks
how the term responds: when it is zero, its sign, and how it orders states. Exact values are left
out, so tuning the reward shapes and weights does not break these tests. Every term is enabled, so
the ones solve_grasp_env leaves at zero weight are checked too.
"""
import dataclasses
import importlib.util
import math

import pytest
import torch as th

# checked without importing it: adarl picks the MuJoCo GL backend before mujoco's first import
if importlib.util.find_spec("mujoco") is None:
    pytest.skip("mujoco is not installed", allow_module_level=True)
from _env_builders import build_grasp_env, field_index, sub_rewards

pytestmark = [pytest.mark.mujoco, pytest.mark.slow]

# disabled in solve_grasp_env, enabled here so the terms show up in the sub-rewards
REWARD_WEIGHTS = {
    "reward_health_weight" :                1.0,
    "reward_joint_power_weight" :           1.0,
    "reward_joint_actacc_weight" :          1.0,
    "reward_joint_position_limit_weight" :  1.0,
    "reward_gripper_grasp_weight" :         1.0,
    "reward_velocity_tracking_weight" :     1.0,
    "reward_yaw_vel_track_weight" :         1.0,
    "reward_feet_linvel_weight" :           1.0,
    "reward_safety_weight" :                1.0,
}


@pytest.fixture(scope="module")
def grasp(tmp_path_factory):
    return build_grasp_env(str(tmp_path_factory.mktemp("grasp")), REWARD_WEIGHTS)


def _rew(grasp, state, name : str) -> float:
    return sub_rewards(grasp.env, state)[name]


def _set_extrinsic(grasp, state, field, value : float):
    env = grasp.env
    state[env.STATE_EXTRINSIC][:, 0, field_index(env, env.STATE_EXTRINSIC, field), 0] = value


def test_rewards_on_the_reset_state_are_finite_and_complete(grasp):
    env = grasp.env
    assert env._has_object_state, "the grasping terms are all zero without the object state"
    rews = sub_rewards(env, grasp.fresh_state())
    assert set(rews) == {f.name for f in dataclasses.fields(env.SubRewards)}
    assert all(math.isfinite(v) for v in rews.values()), rews


class TestGraspingPoses:

    def _gripper_pose(self, grasp, distance : float) -> float:
        env = grasp.env
        state = grasp.fresh_state()
        poses = state[env.STATE_GRASPING]
        obj = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.OBJECT_POSE)
        gripper = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.GRIPPER_POSE)
        poses[:, 0, gripper, :3] = poses[:, 0, obj, :3] + th.tensor([distance, 0.0, 0.0])
        return _rew(grasp, state, "gripper_pose")

    def _object_pose(self, grasp, distance : float) -> float:
        env = grasp.env
        state = grasp.fresh_state()
        poses = state[env.STATE_GRASPING]
        obj = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.OBJECT_POSE)
        goal = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.GOAL_POSE)
        poses[:, 0, obj, :3] = poses[:, 0, goal, :3] + th.tensor([0.0, distance, 0.0])
        return _rew(grasp, state, "object_pose")

    def test_gripper_pose_grows_as_the_gripper_nears_the_object(self, grasp):
        assert self._gripper_pose(grasp, 0.0) > self._gripper_pose(grasp, 0.1) > self._gripper_pose(grasp, 0.5) > 0.0

    def test_object_pose_grows_as_the_object_nears_the_goal(self, grasp):
        assert self._object_pose(grasp, 0.0) > self._object_pose(grasp, 0.1) > self._object_pose(grasp, 0.5) > 0.0


class TestGripperClosing:
    """gripper_close pays for closing near the object, gripper_grasp for closing while touching it."""

    def _state(self, grasp, closed : bool, distance : float, contact : float):
        env = grasp.env
        if len(env._gripper_joint_idxs) == 0:
            pytest.skip("this robot config has no gripper joints")
        state = grasp.fresh_state()
        closes_at_lower = env._grasping_conf.grasping_init_args.gripper_closes_at_lower_limit
        closed_end = 0 if closes_at_lower else 1
        end = closed_end if closed else 1-closed_end
        for j in env._gripper_joint_idxs:
            state[env.STATE_ROBOT][:, 0, j, 0] = env._ctrl_limits[end, j, 0]
        poses = state[env.STATE_GRASPING]
        obj = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.OBJECT_POSE)
        gripper = field_index(env, env.STATE_GRASPING, env.GRASPING_POSES.GRIPPER_POSE)
        poses[:, 0, gripper, :3] = poses[:, 0, obj, :3] + th.tensor([distance, 0.0, 0.0])
        state[env.STATE_GRIPPER][:, 0, field_index(env, env.STATE_GRIPPER, env.GRIPPER.OBJECT_CONTACT)] = contact
        return state

    def test_closing_is_rewarded_near_the_object(self, grasp):
        close_distance = grasp.env._grasping_conf.grasping_init_args.gripper_close_distance
        closed_near = _rew(grasp, self._state(grasp, closed=True, distance=0.0, contact=0.0), "gripper_close")
        closed_far = _rew(grasp, self._state(grasp, closed=True, distance=2*close_distance, contact=0.0), "gripper_close")
        open_near = _rew(grasp, self._state(grasp, closed=False, distance=0.0, contact=0.0), "gripper_close")
        assert closed_near > closed_far > 0.0
        assert closed_near > open_near
        assert open_near == pytest.approx(0.0, abs=1e-6)

    def test_grasp_is_rewarded_only_when_closed_on_the_object(self, grasp):
        closed_touching = _rew(grasp, self._state(grasp, closed=True, distance=0.0, contact=1.0), "gripper_grasp")
        closed_not_touching = _rew(grasp, self._state(grasp, closed=True, distance=0.0, contact=0.0), "gripper_grasp")
        open_touching = _rew(grasp, self._state(grasp, closed=False, distance=0.0, contact=1.0), "gripper_grasp")
        assert closed_touching > 0.0
        assert closed_not_touching == 0.0
        assert open_touching == pytest.approx(0.0, abs=1e-6)


class TestBodyPosture:

    def test_height_is_best_at_the_neutral_height_and_symmetric(self, grasp):
        env = grasp.env
        neutral = env._grasping_conf.grasping_init_args.neutral_body_height
        def height(h):
            state = grasp.fresh_state()
            _set_extrinsic(grasp, state, env.EXTRINSIC_FIELDS.BODY_HEIGHT_FROM_GROUND, h)
            return _rew(grasp, state, "height_position")
        assert height(neutral) == 0.0
        assert height(neutral+0.05) < 0.0
        assert height(neutral+0.05) == pytest.approx(height(neutral-0.05), rel=1e-4) # float32 heights
        assert height(neutral+0.1) < height(neutral+0.05)

    def test_pitchnroll_is_best_level(self, grasp):
        env = grasp.env
        def pitchnroll(gx, gy):
            state = grasp.fresh_state()
            _set_extrinsic(grasp, state, env.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_X, gx)
            _set_extrinsic(grasp, state, env.EXTRINSIC_FIELDS.BODY_REL_GRAVITY_Y, gy)
            return _rew(grasp, state, "pitchnroll")
        assert pitchnroll(0.0, 0.0) == 0.0
        assert pitchnroll(0.2, 0.0) < 0.0
        assert pitchnroll(0.2, 0.0) == pytest.approx(pitchnroll(0.0, 0.2))

    def test_body_velocities_are_penalized(self, grasp):
        env = grasp.env
        E = env.EXTRINSIC_FIELDS
        def velocities(vx, vy, vz, yaw):
            state = grasp.fresh_state()
            for field, value in ((E.BODY_ABS_LINVEL_X, vx), (E.BODY_ABS_LINVEL_Y, vy), (E.BODY_ABS_LINVEL_Z, vz), (E.BODY_ABS_ANGVEL_Z, yaw)):
                _set_extrinsic(grasp, state, field, value)
            rews = sub_rewards(env, state)
            return rews["velocity_tracking"], rews["yaw_vel_track"]
        still_lin, still_yaw = velocities(0.0, 0.0, 0.0, 0.0)
        assert still_lin == 0.0 and still_yaw == 0.0
        for v in ((0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.3)):
            assert velocities(*v, 0.0)[0] < 0.0
        assert velocities(0.0, 0.0, 0.0, 0.5)[1] < 0.0


class TestFeetSlip:
    """Penalizes the horizontal speed of the feet that are on the ground."""

    def _feet_linvel(self, grasp, vel_x : float, vel_y : float, on_ground : float) -> float:
        env = grasp.env
        state = grasp.fresh_state()
        feet = state[env.STATE_FEET]
        nfeet = feet.shape[-1]
        feet[:, 0] = 0.0
        feet[:, 0, field_index(env, env.STATE_FEET, env.FEET.FEET_ON_GROUND), :] = 1.0
        feet[:, 0, field_index(env, env.STATE_FEET, env.FEET.FEET_ON_GROUND), 0] = on_ground
        feet[:, 0, field_index(env, env.STATE_FEET, env.FEET.FEET_VEL_X), 0] = vel_x
        feet[:, 0, field_index(env, env.STATE_FEET, env.FEET.FEET_VEL_Y), 0] = vel_y
        assert nfeet > 0
        return _rew(grasp, state, "feet_linvel")

    def test_sliding_foot_on_the_ground_is_penalized_along_x_and_y(self, grasp):
        still = self._feet_linvel(grasp, 0.0, 0.0, 1.0)
        assert self._feet_linvel(grasp, 0.5, 0.0, 1.0) < still
        assert self._feet_linvel(grasp, 0.0, 0.5, 1.0) < still

    def test_moving_foot_in_the_air_is_not_penalized(self, grasp):
        assert self._feet_linvel(grasp, 0.5, 0.5, 0.0) == self._feet_linvel(grasp, 0.0, 0.0, 1.0)


class TestJointPenalties:

    def test_joint_position_is_best_at_homing(self, grasp):
        env = grasp.env
        homing = env._configuration.homing_ctrl_joints_position
        def position(offset):
            state = grasp.fresh_state()
            state[env.STATE_JOINT_LONGTERM_STATS][:, 0, 0, :] = homing + offset
            return _rew(grasp, state, "joint_position")
        assert position(0.0) > position(0.2) > position(0.4)

    def test_actdiff_penalizes_changing_actions(self, grasp):
        env = grasp.env
        def actdiff(change):
            state = grasp.fresh_state()
            actions = state[env.STATE_ACT_RAW_HIST]
            actions[:, 1:] = 0.0
            actions[:, 0] = change
            return _rew(grasp, state, "joint_actdiff")
        assert actdiff(0.0) > actdiff(0.1) > actdiff(0.5)


class TestSafetyTriggered:

    def _safety(self, grasp, posref : float, limits : float) -> float:
        env = grasp.env
        state = grasp.fresh_state()
        internal = state[env.STATE_INTERNAL]
        internal[:, 0, field_index(env, env.STATE_INTERNAL, env.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED), 0] = posref
        internal[:, 0, field_index(env, env.STATE_INTERNAL, env.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED), 0] = limits
        return _rew(grasp, state, "safety_triggered")

    def test_zero_when_nothing_triggered_and_negative_when_triggered(self, grasp):
        assert self._safety(grasp, 0.0, 0.0) == 0.0
        assert self._safety(grasp, 1.0, 0.0) < 0.0
        assert self._safety(grasp, 0.0, 1.0) < 0.0

    @pytest.mark.xfail(strict=True, reason="GraspVecEnv binarizes the safety states (logical_or), so the "
                                           "penalty stays full for the whole safety cooldown decay, "
                                           "unlike LocomotionVecEnv where it follows the decaying state")
    def test_penalty_follows_the_decaying_safety_state(self, grasp):
        assert self._safety(grasp, 0.5, 0.0) == pytest.approx(0.5*self._safety(grasp, 1.0, 0.0))
