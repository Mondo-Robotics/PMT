"""Public-safe path defaults for standalone config construction.

Builder-composed tasks receive concrete data paths from ``configs/paths.yaml``.
These helpers only keep direct config instantiation importable without embedding
machine-specific absolute paths.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def repo_root() -> Path:
    return _env_path("PMT_REPO_ROOT", Path(__file__).resolve().parents[1])


def data_root() -> Path:
    # Mirrors configs/paths.yaml local DATA_ROOT default ($HOME/whole_body_tracking).
    return _env_path("PMT_DATA_ROOT", Path.home() / "whole_body_tracking")


def motion_root() -> Path:
    return _env_path("PMT_MOTION_ROOT", data_root() / "motions")


def dataset_root() -> Path:
    # Mirrors configs/paths.yaml local DATASET_ROOT default
    # ($HOME/whole_body_tracking_motions/motions).
    return _env_path(
        "PMT_DATASET_ROOT",
        Path.home() / "whole_body_tracking_motions" / "motions",
    )


def terrain_root() -> Path:
    # configs/paths.yaml derives TERRAIN_ROOT == DATA_ROOT (meshes live under DATA_ROOT);
    # there is no separate PMT_TERRAIN_ROOT env var.
    return data_root()


def terrain_motion_root() -> Path:
    return _env_path("PMT_TERRAIN_MOTION_ROOT", dataset_root() / "terrain")


def sonic_root() -> Path:
    return _env_path("PMT_SONIC_ROOT", dataset_root() / "sonic")


def repo_path(*parts: str) -> str:
    return str(repo_root().joinpath(*parts))


def motion_path(*parts: str) -> str:
    return str(motion_root().joinpath(*parts))


def terrain_asset_path(*parts: str) -> str:
    return str(terrain_root().joinpath(*parts))


def terrain_motion_path(*parts: str) -> str:
    return str(terrain_motion_root().joinpath(*parts))


def sonic_path(*parts: str) -> str:
    return str(sonic_root().joinpath(*parts))
