from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass(frozen=True)
class MotionDiscoveryResult:
    """Result of motion file discovery."""
    files: List[str]
    searched_paths: List[str]


@dataclass(frozen=True)
class PairedMotionDiscoveryResult:
    """Result of paired optimized/raw motion discovery."""

    optimized_files: List[str]
    raw_files: List[str]
    optimized_searched_paths: List[str]
    raw_searched_paths: List[str]


def _normalize(path: str) -> str:
    """Normalize a path to absolute form."""
    return os.path.abspath(os.path.expanduser(path))


def _canonical_motion_key(path: str) -> str:
    """Normalize paired-motion keys by removing raw/optimized filename suffixes."""
    stem, ext = os.path.splitext(path)
    for suffix in ("_optimized", "_raw"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + ext


def _find_npz_files_recursive(directory: str, robot_only: bool = True) -> List[str]:
    """Recursively find all .npz files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search
        robot_only: If True, only return files in robot_* folders (exclude human_* folders)
    """
    pattern = os.path.join(directory, "**", "*.npz")
    all_files = sorted(glob.glob(pattern, recursive=True))
    
    if robot_only:
        # Filter to only include files in robot_* folders, excluding human_* folders
        filtered = []
        for f in all_files:
            # Check if any parent folder starts with "human"
            path_parts = f.split(os.sep)
            is_human = any(part.lower().startswith("human") for part in path_parts)
            if not is_human:
                filtered.append(f)
        return filtered
    
    return all_files


def find_motion_files(
    motion_paths: Optional[Union[str, List[str]]] = None,
    strict: bool = True,
    robot_only: bool = True,
) -> MotionDiscoveryResult:
    """Discover motion files from given paths.

    Args:
        motion_paths: A single path or list of paths. Each path can be:
            - A .npz file: added directly to the result
            - A directory: recursively searched for all .npz files
        strict: When True, raise an error if no motion files are discovered.
        robot_only: When True, filter out files in human_* folders (default True).

    Returns:
        MotionDiscoveryResult containing all discovered .npz files.
    """
    # Handle None or empty input
    if motion_paths is None:
        motion_paths = []
    elif isinstance(motion_paths, str):
        motion_paths = [motion_paths]
    
    searched_paths: List[str] = []
    all_files: List[str] = []
    seen_files = set()
    
    for path in motion_paths:
        normalized_path = _normalize(path)
        searched_paths.append(normalized_path)
        
        if os.path.isfile(normalized_path):
            # Single file - add if it's a .npz file
            if normalized_path.endswith(".npz") and normalized_path not in seen_files:
                all_files.append(normalized_path)
                seen_files.add(normalized_path)
        elif os.path.isdir(normalized_path):
            # Directory - recursively find all .npz files
            found_files = _find_npz_files_recursive(normalized_path, robot_only=robot_only)
            for f in found_files:
                if f not in seen_files:
                    all_files.append(f)
                    seen_files.add(f)
        else:
            # Path doesn't exist - warn but continue
            print(f"[MotionPaths] Warning: Path does not exist: {normalized_path}")
    
    # Sort final list
    all_files = sorted(all_files)
    
    if strict and not all_files:
        raise ValueError(
            "No motion files were found. Searched paths:\n  - "
            + "\n  - ".join(searched_paths) if searched_paths else "No paths provided"
        )
    
    return MotionDiscoveryResult(files=all_files, searched_paths=searched_paths)


def pair_motion_files(
    optimized_motion_paths: Optional[Union[str, List[str]]],
    raw_motion_paths: Optional[Union[str, List[str]]],
    strict: bool = True,
    robot_only: bool = True,
) -> PairedMotionDiscoveryResult:
    """Discover and pair optimized/raw motion files by relative path or basename.

    This is intended for datasets where raw and optimized motion trees contain
    the same clips. Any mismatch raises immediately instead of silently
    continuing with incomplete pairs.
    """

    optimized = find_motion_files(optimized_motion_paths, strict=strict, robot_only=robot_only)
    raw = find_motion_files(raw_motion_paths, strict=strict, robot_only=robot_only)

    def _build_lookup(files: List[str], roots: List[str]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        basename_counts: dict[str, int] = {}
        normalized_roots = [_normalize(root) for root in roots]

        for path in files:
            key: str | None = None
            for root in normalized_roots:
                try:
                    rel = os.path.relpath(path, root)
                except ValueError:
                    continue
                if not rel.startswith(".."):
                    key = _canonical_motion_key(rel)
                    break
            if key is None:
                key = _canonical_motion_key(os.path.basename(path))

            if key in lookup:
                raise ValueError(f"Duplicate motion key '{key}' discovered for path: {path}")
            lookup[key] = path

            basename = _canonical_motion_key(os.path.basename(path))
            basename_counts[basename] = basename_counts.get(basename, 0) + 1

        duplicate_basenames = sorted(name for name, count in basename_counts.items() if count > 1)
        if duplicate_basenames:
            raise ValueError(
                "Motion pairing requires unique basenames or matching relative paths. "
                f"Found duplicate basenames: {duplicate_basenames}"
            )
        return lookup

    optimized_lookup = _build_lookup(optimized.files, optimized.searched_paths)
    raw_lookup = _build_lookup(raw.files, raw.searched_paths)

    optimized_keys = set(optimized_lookup)
    raw_keys = set(raw_lookup)
    missing_in_raw = sorted(optimized_keys - raw_keys)
    missing_in_optimized = sorted(raw_keys - optimized_keys)
    if missing_in_raw or missing_in_optimized:
        messages = []
        if missing_in_raw:
            messages.append(f"missing in raw: {missing_in_raw[:10]}")
        if missing_in_optimized:
            messages.append(f"missing in optimized: {missing_in_optimized[:10]}")
        raise ValueError("Optimized/raw motion pairing mismatch: " + "; ".join(messages))

    paired_keys = sorted(optimized_keys)
    optimized_files = [optimized_lookup[key] for key in paired_keys]
    raw_files = [raw_lookup[key] for key in paired_keys]

    return PairedMotionDiscoveryResult(
        optimized_files=optimized_files,
        raw_files=raw_files,
        optimized_searched_paths=optimized.searched_paths,
        raw_searched_paths=raw.searched_paths,
    )


__all__ = [
    "MotionDiscoveryResult",
    "PairedMotionDiscoveryResult",
    "find_motion_files",
    "pair_motion_files",
]

