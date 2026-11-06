"""The goal arrows only belong to the locomotion env."""
from _env_builders import build_grasp_env, build_loco_env


def _models(env) -> set[str]:
    return {link[0] for link in env._adapter.get_detected_links()}


def test_loco_env_spawns_and_moves_the_arrows(tmp_path):
    built = build_loco_env(str(tmp_path), {}) # keeps the runner alive, it closes the env when collected
    env = built.env
    assert {"arrow", "arrow_yellow"} <= _models(env)
    env._set_arrow_pose(vec_mask=env._all_vecs)


def test_grasp_env_has_no_arrows(tmp_path):
    built = build_grasp_env(str(tmp_path), {})
    assert not {"arrow", "arrow_yellow"} & _models(built.env)
