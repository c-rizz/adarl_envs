"""Reward terms of LocomotionVecEnv, and the feet state they read, on the kyon env of loco_kyon_ppo.

Each reward test takes the state right after a reset, overwrites only the fields a term reads, and
checks how the term responds: when it is zero, its sign, and how it orders states. Exact values are
left out, so tuning the reward shapes and weights does not break these tests. Fields are addressed
through the state helper's index rather than the enum value, so a term that reads the wrong field
fails here (the slip term once read a feet duration as a velocity).
"""
import importlib.util
import math

import pytest
import torch as th

# checked without importing it: adarl picks the MuJoCo GL backend before mujoco's first import
if importlib.util.find_spec("mujoco") is None:
    pytest.skip("mujoco is not installed", allow_module_level=True)
from _env_builders import build_loco_env, field_index, sub_rewards

pytestmark = [pytest.mark.mujoco, pytest.mark.slow]

# disabled in loco_kyon_ppo, enabled here so the term shows up in the sub-rewards
REWARD_WEIGHTS = {"reward_feet_ground_time_weight": 1.0}

# the clamps LocomotionVecEnv applies to the feet durations
MAX_AIR_TIME_SEC = 2.0
MAX_GROUND_TIME_SEC = 0.5


@pytest.fixture(scope="module")
def loco(tmp_path_factory):
    return build_loco_env(str(tmp_path_factory.mktemp("loco")), REWARD_WEIGHTS)


def _nfeet(loco) -> int:
    return loco.reset_state[loco.env.STATE_FEET].shape[-1]


def _first_foot(loco, value : float, others : float = 0.0) -> list[float]:
    return [value] + [others]*(_nfeet(loco)-1)


def _set_feet(loco, state, field, values : list[float]):
    env = loco.env
    feet = state[env.STATE_FEET]
    feet[:, 0, field_index(env, env.STATE_FEET, field), :] = th.as_tensor(values, dtype=feet.dtype)


def _stance_state(loco, moving : bool):
    """All feet still on the ground, no step just ended, and a goal speed that does or does not ask to move."""
    env = loco.env
    state = loco.fresh_state()
    state[env.STATE_FEET][:, 0] = 0.0
    _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, [1.0]*_nfeet(loco))
    loco_state = state[env.STATE_LOCOMOTION]
    loco_state[:, 0, field_index(env, env.STATE_LOCOMOTION, env.LOCOMOTION_FIELDS.GOAL_LINVEL_SPEED), 0] = 0.5 if moving else 0.0
    loco_state[:, 0, field_index(env, env.STATE_LOCOMOTION, env.LOCOMOTION_FIELDS.GOAL_YAW_VEL), 0] = 0.0
    return state


def _durations(loco):
    a = loco.env._loco_conf.init_args
    return a.min_good_step_air_duration, a.max_good_step_air_duration, a.step_min_good_ground_duration, a.step_max_good_ground_duration


def test_rewards_on_the_reset_state_are_finite_and_complete(loco):
    rews = sub_rewards(loco.env, loco.fresh_state())
    assert set(rews) == set(loco.env._loco_conf.enabled_rewards)
    assert all(math.isfinite(v) for v in rews.values()), rews


class TestFeetAirTime:
    """Paid once per step, at touchdown (on the ground with an air duration), from the swing just completed."""

    def _air_time(self, loco, duration : float, touchdown : bool = True, moving : bool = True) -> float:
        env = loco.env
        state = _stance_state(loco, moving)
        _set_feet(loco, state, env.FEET_FIELDS.FEET_AIR_DURATIONS, _first_foot(loco, duration))
        # without a touchdown the foot is still in the air, in the middle of its swing
        _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, _first_foot(loco, 1.0 if touchdown else 0.0, others=1.0))
        return sub_rewards(env, state)["feet_air_time"]

    def test_zero_during_the_swing(self, loco):
        min_air, max_air, _, _ = _durations(loco)
        assert self._air_time(loco, (min_air+max_air)/2, touchdown=False) == 0.0
        assert self._air_time(loco, min_air/4, touchdown=False) == 0.0

    def test_good_step_is_rewarded_and_a_longer_one_more(self, loco):
        min_air, max_air, _, _ = _durations(loco)
        shorter = self._air_time(loco, min_air + 0.25*(max_air-min_air))
        longer = self._air_time(loco, min_air + 0.75*(max_air-min_air))
        assert 0.0 < shorter < longer

    def test_short_step_is_penalized_even_when_standing_still(self, loco):
        min_air, _, _, _ = _durations(loco)
        assert self._air_time(loco, min_air/4, moving=True) < 0.0
        assert self._air_time(loco, min_air/4, moving=False) < 0.0

    def test_good_step_is_not_rewarded_when_standing_still(self, loco):
        min_air, max_air, _, _ = _durations(loco)
        assert self._air_time(loco, (min_air+max_air)/2, moving=False) == 0.0


class TestFeetGroundTime:
    """Paid once per stance, at liftoff (in the air with a ground duration), from the stance just completed."""

    def _ground_time(self, loco, duration : float, liftoff : bool = True, moving : bool = True) -> float:
        env = loco.env
        state = _stance_state(loco, moving)
        _set_feet(loco, state, env.FEET_FIELDS.FEET_GROUND_DURATIONS, _first_foot(loco, duration))
        # without a liftoff the foot is still on the ground, in the middle of its stance
        _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, _first_foot(loco, 0.0 if liftoff else 1.0, others=1.0))
        return sub_rewards(env, state)["feet_ground_time"]

    def test_zero_during_the_stance(self, loco):
        _, _, min_ground, max_ground = _durations(loco)
        assert self._ground_time(loco, (min_ground+max_ground)/2, liftoff=False) == 0.0

    def test_good_stance_is_rewarded_and_a_longer_one_more(self, loco):
        _, _, min_ground, max_ground = _durations(loco)
        shorter = self._ground_time(loco, min_ground + 0.25*(max_ground-min_ground))
        longer = self._ground_time(loco, min_ground + 0.75*(max_ground-min_ground))
        assert 0.0 < shorter < longer

    def test_short_stance_is_penalized_even_when_standing_still(self, loco):
        _, _, min_ground, _ = _durations(loco)
        assert self._ground_time(loco, min_ground/4, moving=True) < 0.0
        assert self._ground_time(loco, min_ground/4, moving=False) < 0.0

    def test_good_stance_is_not_rewarded_when_standing_still(self, loco):
        _, _, min_ground, max_ground = _durations(loco)
        assert self._ground_time(loco, (min_ground+max_ground)/2, moving=False) == 0.0


class TestSlip:
    """Penalizes the horizontal speed of the feet that are on the ground."""

    def _slip(self, loco, vel_x : float = 0.0, vel_y : float = 0.0, contact : float = 1.0, durations : float = 0.0) -> float:
        env = loco.env
        state = _stance_state(loco, moving=True)
        _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, _first_foot(loco, contact, others=1.0))
        _set_feet(loco, state, env.FEET_FIELDS.FEET_VEL_X, _first_foot(loco, vel_x))
        _set_feet(loco, state, env.FEET_FIELDS.FEET_VEL_Y, _first_foot(loco, vel_y))
        _set_feet(loco, state, env.FEET_FIELDS.FEET_AIR_DURATIONS, [durations]*_nfeet(loco))
        _set_feet(loco, state, env.FEET_FIELDS.FEET_GROUND_DURATIONS, [durations]*_nfeet(loco))
        return sub_rewards(env, state)["slip"]

    def test_sliding_foot_on_the_ground_is_penalized_along_x_and_y(self, loco):
        still = self._slip(loco)
        assert self._slip(loco, vel_x=0.5) < still
        assert self._slip(loco, vel_y=0.5) < still
        assert self._slip(loco, vel_x=0.5) == pytest.approx(self._slip(loco, vel_y=0.5))

    def test_moving_foot_in_the_air_is_not_penalized(self, loco):
        assert self._slip(loco, vel_x=0.5, vel_y=0.5, contact=0.0) == self._slip(loco)

    def test_feet_durations_do_not_count_as_velocity(self, loco):
        assert self._slip(loco, durations=0.4) == self._slip(loco)


class TestFeetOnGround:
    """Rewards feet on the ground, only while the goal is to stand still."""

    def _on_ground(self, loco, contacts : int, moving : bool) -> float:
        env = loco.env
        state = _stance_state(loco, moving)
        _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, [1.0]*contacts + [0.0]*(_nfeet(loco)-contacts))
        return sub_rewards(env, state)["feet_on_ground"]

    def test_grows_with_the_feet_on_the_ground_when_standing_still(self, loco):
        n = _nfeet(loco)
        assert self._on_ground(loco, n, moving=False) > self._on_ground(loco, n//2, moving=False) > self._on_ground(loco, 0, moving=False)
        assert self._on_ground(loco, 0, moving=False) == 0.0

    def test_zero_when_moving(self, loco):
        assert self._on_ground(loco, _nfeet(loco), moving=True) == 0.0


class TestFeetStepHeight:
    """Rewards good steps whose swing peaked at the desired foot clearance."""

    def _step_height(self, loco, peak_offset : float, touchdown : bool = True, duration : float | None = None, moving : bool = True) -> float:
        env = loco.env
        min_air, max_air, _, _ = _durations(loco)
        duration = (min_air+max_air)/2 if duration is None else duration
        state = _stance_state(loco, moving)
        _set_feet(loco, state, env.FEET_FIELDS.FEET_AIR_DURATIONS, _first_foot(loco, duration))
        _set_feet(loco, state, env.FEET_FIELDS.FEET_CONTACT, _first_foot(loco, 1.0 if touchdown else 0.0, others=1.0))
        _set_feet(loco, state, env.FEET_FIELDS.PEAK_POS_Z, _first_foot(loco, env._loco_conf.init_args.desired_foot_clearance + peak_offset))
        return sub_rewards(env, state)["feet_step_height"]

    def test_highest_at_the_desired_clearance(self, loco):
        assert self._step_height(loco, 0.0) > self._step_height(loco, 0.03) > self._step_height(loco, 0.3)
        assert self._step_height(loco, 0.03) == pytest.approx(self._step_height(loco, -0.03))

    def test_only_good_steps_at_touchdown_while_moving_count(self, loco):
        min_air, _, _, _ = _durations(loco)
        assert self._step_height(loco, 0.0, touchdown=False) == 0.0
        assert self._step_height(loco, 0.0, duration=min_air/4) == 0.0
        assert self._step_height(loco, 0.0, moving=False) == 0.0


class TestSafetyTriggered:
    """Penalizes the decaying safety states, averaged over the posref and the limits safety."""

    def _safety(self, loco, posref : float, limits : float) -> float:
        env = loco.env
        state = loco.fresh_state()
        internal = state[env.STATE_INTERNAL]
        internal[:, 0, field_index(env, env.STATE_INTERNAL, env.INTERNAL_FIELDS.SAFETY_POSREF_TRIGGERED), 0] = posref
        internal[:, 0, field_index(env, env.STATE_INTERNAL, env.INTERNAL_FIELDS.SAFETY_LIMITS_TRIGGERED), 0] = limits
        return sub_rewards(env, state)["safety_triggered"]

    def test_zero_when_nothing_triggered(self, loco):
        assert self._safety(loco, 0.0, 0.0) == 0.0

    def test_penalty_follows_the_safety_states(self, loco):
        posref = self._safety(loco, 1.0, 0.0)
        assert posref < 0.0
        assert self._safety(loco, 0.0, 1.0) == pytest.approx(posref)
        assert self._safety(loco, 1.0, 1.0) == pytest.approx(2*posref)
        assert self._safety(loco, 0.5, 0.0) == pytest.approx(0.5*posref)


def test_feet_state_follows_the_contacts_over_a_rollout(loco):
    """Steps the sim with random actions and checks every feet state against the contacts it saw.

    The contact flag is the contact. Each duration counts up during its own phase (from zero after
    the phase starts), keeps its final value for the one step after the phase ends, and is zero for
    the rest of the other phase, so a nonzero duration of the other phase marks exactly the edges.
    """
    env, runner = loco.env, loco.runner
    F = env.FEET_FIELDS
    fi = lambda f: field_index(env, env.STATE_FEET, f)
    dt_idx = field_index(env, env.STATE_INTERNAL, env.INTERNAL_FIELDS.LAST_STEP_DT)

    runner.reset()
    prev = env.get_states()[env.STATE_FEET][0, 0].clone()
    # a reset is neither a touchdown nor a liftoff
    reset_on_ground = prev[fi(F.FEET_CONTACT)] > 0.5
    assert th.all(prev[fi(F.FEET_AIR_DURATIONS)][reset_on_ground] == 0.0)
    assert th.all(prev[fi(F.FEET_GROUND_DURATIONS)][~reset_on_ground] == 0.0)

    generator = th.Generator().manual_seed(0)
    action_shape = (env.num_envs,) + tuple(env.single_action_space.shape)
    touchdowns = liftoffs = 0
    for _ in range(200):
        actions = (th.rand(action_shape, generator=generator)*2 - 1)*0.5
        _, _, _, terminated, truncated, _, _, _ = runner.step(actions, autoreset=False)
        state = env.get_states()
        cur = state[env.STATE_FEET][0, 0].clone()
        dt = float(state[env.STATE_INTERNAL][0, 0, dt_idx, 0])

        contact = cur[fi(F.FEET_CONTACT)]
        assert th.all((contact == 0.0) | (contact == 1.0)), contact
        on_ground, was_on_ground = contact > 0.5, prev[fi(F.FEET_CONTACT)] > 0.5
        touchdown, liftoff = on_ground & ~was_on_ground, ~on_ground & was_on_ground

        prev_air, prev_ground = prev[fi(F.FEET_AIR_DURATIONS)], prev[fi(F.FEET_GROUND_DURATIONS)]
        zero = th.zeros_like(prev_air)
        expected_air = th.where(~on_ground,
                                th.where(liftoff, zero, prev_air) + dt,
                                th.where(touchdown, prev_air, zero)).clamp(max=MAX_AIR_TIME_SEC)
        expected_ground = th.where(on_ground,
                                   th.where(touchdown, zero, prev_ground) + dt,
                                   th.where(liftoff, prev_ground, zero)).clamp(max=MAX_GROUND_TIME_SEC)
        air, ground = cur[fi(F.FEET_AIR_DURATIONS)], cur[fi(F.FEET_GROUND_DURATIONS)]
        th.testing.assert_close(air, expected_air)
        th.testing.assert_close(ground, expected_ground)
        # what the rewards read the edges from
        assert th.equal(on_ground & (air > 0), touchdown)
        assert th.equal(~on_ground & (ground > 0), liftoff)

        # the peak of a swing is measured from where the foot lifted off
        feet_z, liftoff_z = cur[fi(F.FEET_ABS_POS_Z)], cur[fi(F.FEET_LIFTOFF_POS_Z)]
        th.testing.assert_close(liftoff_z[liftoff], feet_z[liftoff])
        assert th.all(cur[fi(F.PEAK_POS_Z)][~on_ground] >= (feet_z - liftoff_z)[~on_ground] - 1e-5)

        touchdowns += int(touchdown.sum())
        liftoffs += int(liftoff.sum())
        prev = cur
        if bool(terminated.any() or truncated.any()):
            break
    # without any edge the checks above would hold trivially
    assert touchdowns > 0 and liftoffs > 0, f"touchdowns={touchdowns}, liftoffs={liftoffs}"
