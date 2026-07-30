#!/usr/bin/env python3
"""Download and install the Unitree robot descriptions required by PMT."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROBOT_ASSET_URL = (
    "https://storage.googleapis.com/qiayuanl_robot_descriptions/"
    "unitree_description.tar.gz"
)
ROBOT_ASSET_SHA256 = "b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674"
ARCHIVE_ROOT = "unitree_description"
MAX_EXTRACTED_BYTES = 1_000_000_000
MAX_DOWNLOAD_BYTES = 100_000_000
REQUIRED_FILES = (
    Path("unitree_description/package.xml"),
    Path("unitree_description/urdf/g1/main.urdf"),
    Path("unitree_description/meshes/g1/pelvis.STL"),
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = REPO_ROOT / "pmt_tasks" / "assets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}. "
            "The upstream object may have changed; do not extract it until PMT maintainers "
            "verify the new artifact and update the pinned checksum."
        )


def _download(url: str, destination: Path, *, attempts: int = 3) -> None:
    request = Request(url, headers={"User-Agent": "PMT-robot-asset-installer/1"})
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading Unitree robot assets (attempt {attempt}/{attempts})...", flush=True)
            with urlopen(request, timeout=60) as response, destination.open("wb") as output:
                content_length = response.headers.get("Content-Length", "0")
                try:
                    total = int(content_length)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Robot asset server returned invalid Content-Length: {content_length!r}"
                    ) from exc
                if total > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        f"Robot asset download reports {total} bytes, above the "
                        f"{MAX_DOWNLOAD_BYTES}-byte safety limit"
                    )
                downloaded = 0
                next_report = 5 * 1024 * 1024
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(
                            f"Robot asset download exceeded the {MAX_DOWNLOAD_BYTES}-byte "
                            "safety limit"
                        )
                    if downloaded >= next_report:
                        suffix = f" / {total // (1024 * 1024)} MiB" if total else ""
                        print(f"  {downloaded // (1024 * 1024)} MiB{suffix}", flush=True)
                        next_report += 5 * 1024 * 1024
            return
        except HTTPError as exc:
            destination.unlink(missing_ok=True)
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise RuntimeError(
                    f"Robot asset URL returned permanent HTTP error {exc.code}: {url}"
                ) from exc
            if attempt == attempts:
                raise RuntimeError(f"Unable to download robot assets from {url}: {exc}") from exc
            time.sleep(attempt)
        except RuntimeError:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, TimeoutError, URLError) as exc:
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Unable to download robot assets from {url}: {exc}") from exc
            time.sleep(attempt)


def _validated_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    total_size = 0
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in robot asset archive: {member.name!r}")
        if not path.parts or path.parts[0] != ARCHIVE_ROOT:
            raise RuntimeError(f"Unexpected top-level path in robot asset archive: {member.name!r}")
        if not (member.isdir() or member.isfile()):
            raise RuntimeError(f"Unsupported entry in robot asset archive: {member.name!r}")
        if member.isfile():
            total_size += member.size
    if total_size > MAX_EXTRACTED_BYTES:
        raise RuntimeError(
            f"Robot asset archive expands to {total_size} bytes, above the "
            f"{MAX_EXTRACTED_BYTES}-byte safety limit"
        )
    return members


def _extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validated_members(archive)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(destination, members=members, **kwargs)


def _missing_required_files(asset_dir: Path) -> list[Path]:
    return [relative for relative in REQUIRED_FILES if not (asset_dir / relative).is_file()]


def _print_install_location(asset_dir: Path, install_dir: Path, *, already_installed: bool) -> None:
    state = "are already installed" if already_installed else "were installed"
    print(f"Unitree robot assets {state} at {install_dir}")
    if asset_dir != DEFAULT_ASSET_DIR.resolve():
        print("Use this custom location in future shells with:")
        print(f"  export PMT_ASSET_DIR={asset_dir}")


def install(
    asset_dir: Path,
    *,
    archive_path: Path | None = None,
    url: str = ROBOT_ASSET_URL,
    expected_sha256: str = ROBOT_ASSET_SHA256,
) -> Path:
    """Install into ``asset_dir/unitree_description`` and return that directory."""
    asset_dir = asset_dir.expanduser().resolve()
    install_dir = asset_dir / ARCHIVE_ROOT
    if install_dir.exists():
        missing = _missing_required_files(asset_dir)
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise RuntimeError(
                f"Existing robot asset directory is incomplete: {install_dir}. "
                f"Missing: {formatted}. Move it aside and run this command again."
            )
        _print_install_location(asset_dir, install_dir, already_installed=True)
        return install_dir

    asset_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unitree-assets-", dir=asset_dir) as temporary:
        temporary_dir = Path(temporary)
        downloaded_archive = temporary_dir / "unitree_description.tar.gz"
        if archive_path is None:
            _download(url, downloaded_archive)
            source_archive = downloaded_archive
        else:
            source_archive = archive_path.expanduser().resolve()
            if not source_archive.is_file():
                raise FileNotFoundError(f"Robot asset archive does not exist: {source_archive}")

        print("Verifying SHA-256 checksum...", flush=True)
        _verify_checksum(source_archive, expected_sha256)
        extracted_dir = temporary_dir / "extracted"
        extracted_dir.mkdir()
        print("Extracting verified robot assets...", flush=True)
        _extract(source_archive, extracted_dir)

        missing = _missing_required_files(extracted_dir)
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise RuntimeError(f"Robot asset archive is incomplete; missing: {formatted}")
        os.replace(extracted_dir / ARCHIVE_ROOT, install_dir)

    _print_install_location(asset_dir, install_dir, already_installed=False)
    return install_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the checksum-verified Unitree G1 URDF and meshes required by PMT."
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=(
            "Parent directory for unitree_description "
            f"(default: {DEFAULT_ASSET_DIR})"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Install from an already-downloaded archive instead of using the network.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        install(args.asset_dir, archive_path=args.archive)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
