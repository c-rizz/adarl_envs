#!/usr/bin/env python3
"""Filesystem locations owned by adarl_envs."""
from __future__ import annotations

import os
from pathlib import Path

import adarl_envs


def _is_editable_install() -> bool:
    """Whether adarl_envs is imported from a source tree rather than from an installed copy."""
    package_dir = Path(adarl_envs.__path__[0]).resolve()
    return not any(part in ("site-packages", "dist-packages") for part in package_dir.parts)


def descriptions_dir(create : bool = True) -> Path:
    """Directory holding the robot description repositories the environments load their models from.

    These checkouts are data of this package, so where they live follows the package:

    - **editable install** -- the package is in a workspace, so they go inside it
      (``<adarl_envs>/descriptions``), next to the code that uses them and easy to edit.
    - **normal install** -- the package directory belongs to the installer and may be read-only, so a
      per-user location is used instead (``$XDG_DATA_HOME/adarl_envs/descriptions``).

    ``ADARL_ENVS_DESCRIPTIONS_DIR`` overrides both, which is what to use when the source tree is
    read-only or several checkouts should share one directory.
    """
    override = os.environ.get("ADARL_ENVS_DESCRIPTIONS_DIR")
    if override:
        root = Path(override).expanduser()
    elif _is_editable_install():
        root = Path(adarl_envs.__path__[0])/"descriptions"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home()/".local"/"share"
        root = base/"adarl_envs"/"descriptions"

    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
