from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_robot_assets.py"
SPEC = importlib.util.spec_from_file_location("download_robot_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download_robot_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_robot_assets)


def _archive(
    tmp_path: Path,
    *,
    unsafe: bool = False,
    symlink: bool = False,
    omit_mesh: bool = False,
) -> tuple[Path, str]:
    archive = tmp_path / "unitree_description.tar.gz"
    files = {
        "unitree_description/package.xml": b"<package/>",
        "unitree_description/urdf/g1/main.urdf": b"<robot/>",
        "unitree_description/meshes/g1/pelvis.STL": b"solid pelvis\nendsolid pelvis\n",
    }
    if unsafe:
        files["unitree_description/../../escaped"] = b"unsafe"
    if omit_mesh:
        del files["unitree_description/meshes/g1/pelvis.STL"]
    with tarfile.open(archive, "w:gz") as output:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
        if symlink:
            member = tarfile.TarInfo("unitree_description/unsafe-link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escaped"
            output.addfile(member)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def test_install_from_verified_archive(capsys, tmp_path):
    archive, digest = _archive(tmp_path)
    asset_dir = tmp_path / "assets"

    installed = download_robot_assets.install(
        asset_dir, archive_path=archive, expected_sha256=digest
    )

    assert installed == asset_dir / "unitree_description"
    assert (installed / "urdf/g1/main.urdf").is_file()
    assert (installed / "meshes/g1/pelvis.STL").is_file()
    assert download_robot_assets.install(asset_dir) == installed
    assert f"export PMT_ASSET_DIR={asset_dir}" in capsys.readouterr().out


def test_checksum_mismatch_does_not_install(tmp_path):
    archive, _ = _archive(tmp_path)
    asset_dir = tmp_path / "assets"

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download_robot_assets.install(
            asset_dir, archive_path=archive, expected_sha256="0" * 64
        )

    assert not (asset_dir / "unitree_description").exists()


def test_archive_path_traversal_is_rejected(tmp_path):
    archive, digest = _archive(tmp_path, unsafe=True)
    asset_dir = tmp_path / "assets"

    with pytest.raises(RuntimeError, match="Unsafe path"):
        download_robot_assets.install(
            asset_dir, archive_path=archive, expected_sha256=digest
        )

    assert not (asset_dir / "unitree_description").exists()
    assert not (tmp_path / "escaped").exists()


def test_archive_links_are_rejected(tmp_path):
    archive, digest = _archive(tmp_path, symlink=True)

    with pytest.raises(RuntimeError, match="Unsupported entry"):
        download_robot_assets.install(
            tmp_path / "assets", archive_path=archive, expected_sha256=digest
        )


def test_extracted_size_limit_is_enforced(monkeypatch, tmp_path):
    archive, digest = _archive(tmp_path)
    monkeypatch.setattr(download_robot_assets, "MAX_EXTRACTED_BYTES", 1)

    with pytest.raises(RuntimeError, match="safety limit"):
        download_robot_assets.install(
            tmp_path / "assets", archive_path=archive, expected_sha256=digest
        )


def test_incomplete_archive_does_not_install(tmp_path):
    archive, digest = _archive(tmp_path, omit_mesh=True)
    asset_dir = tmp_path / "assets"

    with pytest.raises(RuntimeError, match="archive is incomplete"):
        download_robot_assets.install(
            asset_dir, archive_path=archive, expected_sha256=digest
        )

    assert not (asset_dir / "unitree_description").exists()


def test_download_size_limit_removes_partial_file(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(download_robot_assets, "MAX_DOWNLOAD_BYTES", 3)
    monkeypatch.setattr(download_robot_assets, "urlopen", lambda *args, **kwargs: Response(b"data"))
    destination = tmp_path / "download.tar.gz"

    with pytest.raises(RuntimeError, match="download reports"):
        download_robot_assets._download("https://example.invalid/assets", destination)

    assert not destination.exists()


def test_malformed_content_length_is_reported(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {"Content-Length": "not-a-number"}

    monkeypatch.setattr(download_robot_assets, "urlopen", lambda *args, **kwargs: Response(b""))

    with pytest.raises(RuntimeError, match="invalid Content-Length"):
        download_robot_assets._download(
            "https://example.invalid/assets", tmp_path / "download.tar.gz"
        )
