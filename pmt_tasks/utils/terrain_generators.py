from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def positive_stepping_stones_terrain(
    difficulty: float, cfg: terrain_gen.HfSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate stepping stones as positive cuboid obstacles on flat ground."""
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_distance = max(0, int(stone_distance / cfg.horizontal_scale))
    stone_width = max(1, int(stone_width / cfg.horizontal_scale))
    stone_height_max = max(1, int(cfg.stone_height_max / cfg.vertical_scale))
    platform_width = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    stone_height_range = np.arange(1, stone_height_max + 1, step=1)
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    start_x, start_y = 0, 0
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width)
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            start_y += stone_width + stone_distance
    else:
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width)
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            start_x += stone_width + stone_distance

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@configclass
class PositiveSteppingStonesTerrainCfg(terrain_gen.HfSteppingStonesTerrainCfg):
    """Stepping-stones terrain with only non-negative obstacle heights."""

    function = positive_stepping_stones_terrain
    holes_depth: float = 0.0


@configclass
class RandomRepeatedBoxesTerrainCfg(terrain_gen.MeshRepeatedBoxesTerrainCfg):
    """Repeated-box terrain with an explicit platform height field."""

    platform_height: float = -1.0
