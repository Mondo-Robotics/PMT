from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "pmt_tasks" / "asset_config.py"


def _load_asset_config(monkeypatch, asset_dir: Path):
    for name in (
        "WHOLE_BODY_TRACKING_ASSET_DIR",
        "WHOLE_BODY_TRACKING_REMOTE_ASSET_DIR",
        "PMT_REPO_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PMT_ASSET_DIR", str(asset_dir))
    spec = importlib.util.spec_from_file_location("pmt_asset_config_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_dir_requires_the_g1_urdf(monkeypatch, tmp_path):
    asset_dir = tmp_path / "assets"
    urdf = asset_dir / "unitree_description/urdf/g1/main.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot/>")
    asset_config = _load_asset_config(monkeypatch, asset_dir)

    assert asset_config.ASSET_DIR == str(asset_dir)

    incomplete = tmp_path / "incomplete"
    (incomplete / "unitree_description").mkdir(parents=True)
    assert asset_config._normalize_candidate(str(incomplete)) is None


def test_missing_asset_error_points_to_downloader(monkeypatch, tmp_path):
    valid = tmp_path / "valid"
    urdf = valid / "unitree_description/urdf/g1/main.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot/>")
    asset_config = _load_asset_config(monkeypatch, valid)

    monkeypatch.delenv("PMT_ASSET_DIR")
    monkeypatch.setattr(asset_config, "asset_dir_candidates", lambda manual_override=None: [])
    with pytest.raises(FileNotFoundError, match="python scripts/download_robot_assets.py"):
        asset_config.resolve_asset_dir()
