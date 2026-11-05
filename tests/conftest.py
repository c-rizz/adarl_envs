"""Root pytest configuration for the adarl_envs test suite.

Layout
------
    tests/envs/                     reward tests on the real envs, built on the MuJoCo (CPU) adapter
    tests/envs/_env_builders.py     builds the envs from the training scripts' configs, state helpers

Running, from the dir with pyproject.toml::

    python -m pytest
"""
import os

# The env modules read this at import time, so it must be set before any test imports them.
# Compiling the env functions on the CPU would take far longer than the tests themselves.
os.environ.setdefault("DISABLE_ENV_TH_COMPILE", "1")
