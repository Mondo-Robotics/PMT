from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# PMT port: the ~181 MB robot-description files (URDF/meshes) are NOT copied
# into PMT. This resolver locates them via (1) PMT/legacy env vars, (2) a
# PMT-local assets dir if one is ever populated, and (3) a fallback to the
# original whole_body_tracking source repo where the binaries already live.
_ASSET_ENV_VARS = (
    "PMT_ASSET_DIR",
    "WHOLE_BODY_TRACKING_ASSET_DIR",
    "WHOLE_BODY_TRACKING_REMOTE_ASSET_DIR",
)
_DEFAULT_RELATIVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
# Fallback: assets dir inside the original source repo (no binaries duplicated).
_PMT_REPO_ROOT = Path(os.environ.get("PMT_REPO_ROOT", Path(__file__).resolve().parents[1])).expanduser()
_SOURCE_REPO_ASSET_DIR = str(
    _PMT_REPO_ROOT.parent
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
)
_REQUIRED_SUBDIR = "unitree_description"
_REQUIRED_ROBOT_FILE = os.path.join(_REQUIRED_SUBDIR, "urdf", "g1", "main.urdf")


def _split_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(os.pathsep) if p.strip()]


def _normalize(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def asset_dir_candidates(manual_override: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    candidates.extend(_split_paths(manual_override))
    for env_var in _ASSET_ENV_VARS:
        candidates.extend(_split_paths(os.environ.get(env_var)))
    candidates.append(_DEFAULT_RELATIVE_DIR)
    candidates.append(_SOURCE_REPO_ASSET_DIR)

    normalized: List[str] = []
    seen = set()
    for path in candidates:
        norm = _normalize(path)
        if norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized


def _normalize_candidate(directory: str) -> Optional[str]:
    """Return a directory whose child contains the required asset folder."""
    if not directory or not os.path.isdir(directory):
        return None

    candidate = directory
    # If the user pointed directly to .../unitree_description, use its parent.
    if os.path.basename(os.path.normpath(directory)) == _REQUIRED_SUBDIR:
        candidate = os.path.dirname(os.path.normpath(directory))

    required_path = os.path.join(candidate, _REQUIRED_ROBOT_FILE)
    if os.path.isfile(required_path):
        return candidate
    return None


def resolve_asset_dir(manual_override: Optional[str] = None) -> str:
    searched = asset_dir_candidates(manual_override)
    for directory in searched:
        normalized = _normalize_candidate(directory)
        if normalized:
            return normalized
    raise FileNotFoundError(
        "Unable to locate robot asset directory.\n"
        f"Set PMT_ASSET_DIR, WHOLE_BODY_TRACKING_ASSET_DIR, or WHOLE_BODY_TRACKING_REMOTE_ASSET_DIR to the\n"
        f"directory that contains '{_REQUIRED_ROBOT_FILE}'. Checked paths:\n  - "
        + "\n  - ".join(searched)
        + "\nInstall the required Unitree assets from the PMT repository root with:\n"
        "  python scripts/download_robot_assets.py"
    )


ASSET_DIR = resolve_asset_dir()

__all__ = ["ASSET_DIR", "resolve_asset_dir", "asset_dir_candidates"]
