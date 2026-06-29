"""
Terrain height query for G1 stair scenes.

Two implementations:
  - StairTerrain: analytical model parsed from XML box geoms (fast, box-only)
  - RaycastTerrain: MuJoCo mj_ray queries (works with any geometry)

Both expose the same API: height_at(), height_batch(), steps, half_width_y,
print_info(), so they are drop-in replacements for each other.

Usage:
    python -m stair_mppi.terrain
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mujoco
import numpy as np


@dataclass
class StairStep:
    """One stair step's geometry."""
    x_lo: float    # left edge
    x_hi: float    # right edge
    top_z: float   # top surface height
    name: str = ""


@dataclass
class StairTerrain:
    """
    Analytical terrain height for stair scenes.

    Supports ascending stairs, plateau, descending stairs.
    Matches MuJoCo box geometry exactly.
    """
    steps: List[StairStep] = field(default_factory=list)
    half_width_y: float = 1.0  # stairs extend y in [-half_width, +half_width]

    def height_at(self, x: float, y: float = 0.0) -> float:
        """Single-point height query."""
        if abs(y) > self.half_width_y:
            return 0.0
        z = 0.0
        for step in self.steps:
            if step.x_lo <= x < step.x_hi and step.top_z > z:
                z = step.top_z
        return z

    def height_batch(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized height query for arbitrary-shape arrays."""
        z = np.zeros_like(x, dtype=np.float64)

        if y is not None:
            in_y = np.abs(y) <= self.half_width_y
        else:
            in_y = np.ones_like(x, dtype=bool)

        for step in self.steps:
            mask = in_y & (x >= step.x_lo) & (x < step.x_hi)
            z = np.where(mask & (step.top_z > z), step.top_z, z)

        return z

    def local_heightmap(
        self,
        x_robot: float,
        y_robot: float = 0.0,
        forward: float = 0.8,
        backward: float = 0.2,
        lateral: float = 0.3,
        resolution: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a local heightmap centered on the robot.

        Args:
            x_robot: Robot x position (world frame)
            y_robot: Robot y position (world frame)
            forward: Distance ahead to query (m)
            backward: Distance behind to query (m)
            lateral: Half-width of lateral extent (m)
            resolution: Grid resolution (m)

        Returns:
            heightmap: (ny, nx) height array
            x_coords: (nx,) world x coordinates
            y_coords: (ny,) world y coordinates
        """
        x_coords = np.arange(x_robot - backward, x_robot + forward, resolution)
        y_coords = np.arange(y_robot - lateral, y_robot + lateral, resolution)

        xx, yy = np.meshgrid(x_coords, y_coords)
        heightmap = self.height_batch(xx, yy)

        return heightmap, x_coords, y_coords

    @staticmethod
    def from_params(
        num_steps_up: int = 5,
        step_height: float = 0.10,
        step_depth: float = 0.20,
        stair_start_x: float = 0.50,
        plateau_depth: float = 0.40,
        num_steps_down: int = 5,
        half_width_y: float = 1.0,
    ) -> 'StairTerrain':
        """Create terrain from stair parameters."""
        steps = []

        # Ascending stairs
        for i in range(num_steps_up):
            x_lo = stair_start_x + i * step_depth
            x_hi = stair_start_x + (i + 1) * step_depth
            top_z = (i + 1) * step_height
            steps.append(StairStep(x_lo=x_lo, x_hi=x_hi, top_z=top_z, name=f"step_up_{i}"))

        # Plateau
        max_z = num_steps_up * step_height
        plateau_x_start = stair_start_x + num_steps_up * step_depth
        plateau_x_end = plateau_x_start + plateau_depth
        steps.append(StairStep(x_lo=plateau_x_start, x_hi=plateau_x_end, top_z=max_z, name="plateau"))

        # Descending stairs
        down_start_x = plateau_x_end
        for i in range(num_steps_down):
            x_lo = down_start_x + i * step_depth
            x_hi = down_start_x + (i + 1) * step_depth
            top_z = max_z - (i + 1) * step_height
            if top_z > 0:
                steps.append(StairStep(x_lo=x_lo, x_hi=x_hi, top_z=top_z, name=f"step_down_{i}"))

        return StairTerrain(steps=steps, half_width_y=half_width_y)

    @staticmethod
    def from_scene_xml(xml_path: str) -> 'StairTerrain':
        """
        Parse a MuJoCo scene XML to extract stair geometry.

        Reads all box geoms named 'step_*' or 'platform_*' and computes
        their x-range and top-z from pos and size attributes.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        steps = []
        half_width_y = 1.0

        # Search all geom elements in worldbody
        for geom in root.iter('geom'):
            name = geom.get('name', '')
            if not (name.startswith('step_') or name.startswith('platform')):
                continue

            gtype = geom.get('type', '')
            if gtype != 'box':
                continue

            pos = np.array([float(x) for x in geom.get('pos', '0 0 0').split()])
            size = np.array([float(x) for x in geom.get('size', '0 0 0').split()])

            # Box: pos is center, size is half-extents [half_x, half_y, half_z]
            x_lo = pos[0] - size[0]
            x_hi = pos[0] + size[0]
            top_z = pos[2] + size[2]
            half_width_y = max(half_width_y, size[1])

            steps.append(StairStep(x_lo=x_lo, x_hi=x_hi, top_z=top_z, name=name))

        # Sort by x_lo for deterministic ordering
        steps.sort(key=lambda s: s.x_lo)

        return StairTerrain(steps=steps, half_width_y=half_width_y)

    def print_info(self):
        """Print terrain summary."""
        print(f"StairTerrain: {len(self.steps)} segments, y_width=±{self.half_width_y}m")
        for s in self.steps:
            print(f"  {s.name:20s}: x=[{s.x_lo:.3f}, {s.x_hi:.3f}), top_z={s.top_z:.3f}m")


class RaycastTerrain:
    """
    Terrain height query using MuJoCo mj_ray.

    Works with any geometry type (box, mesh, heightfield, etc.).
    Drop-in replacement for StairTerrain — exposes the same API:
      height_at(), height_batch(), steps, half_width_y, print_info().

    The ``steps`` list is built by scanning the x-axis at construction time
    so that FootstepResolver edge-clamping logic works unchanged.
    """

    RAY_ORIGIN_Z = 20.0  # start height for downward ray

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        half_width_y: float = 10.0,
        scan_x_range: Tuple[float, float] = (-5.0, 5.0),
        scan_resolution: float = 0.005,
        step_merge_z_tol: float = 0.005,
        step_min_width: float = 0.02,
    ):
        self._model = model
        self._data = data
        self.half_width_y = float(half_width_y)

        # All geom groups enabled for raycasting.
        self._geomgroup = np.ones(6, dtype=np.uint8)

        # Build a set of geom IDs that belong to the worldbody (body 0).
        # Only hits on these geoms count as terrain.
        self._terrain_geom_ids = set()
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == 0:
                self._terrain_geom_ids.add(gid)

        # Pre-scan the x-axis to build a ``steps`` list for edge-clamping.
        self.steps: List[StairStep] = self._scan_steps(
            scan_x_range, scan_resolution, step_merge_z_tol, step_min_width,
        )

    # ------------------------------------------------------------------
    # Core raycast
    # ------------------------------------------------------------------

    def _ray_z(self, x: float, y: float) -> float:
        """Single downward raycast → terrain z at (x, y), or 0.0 if no hit.

        Skips non-terrain geoms (robot body parts) by re-casting from
        just below each non-terrain hit until a terrain geom or miss.
        """
        pnt = np.array([x, y, self.RAY_ORIGIN_Z], dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        max_attempts = 10
        for _ in range(max_attempts):
            dist = mujoco.mj_ray(
                self._model, self._data, pnt, vec,
                self._geomgroup, 1, -1, geomid,
            )
            if dist < 0:
                return 0.0
            hit_z = pnt[2] - dist
            if int(geomid[0]) in self._terrain_geom_ids:
                return hit_z
            # Hit a non-terrain geom (robot); restart just below it
            pnt[2] = hit_z - 1e-4
        return 0.0

    # ------------------------------------------------------------------
    # Public API  (same as StairTerrain)
    # ------------------------------------------------------------------

    def height_at(self, x: float, y: float = 0.0) -> float:
        """Single-point height query."""
        return self._ray_z(float(x), float(y))

    def height_batch(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized height query.  Loops internally (mj_ray is single-point)."""
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        x_flat = x.ravel()
        if y is not None:
            y_flat = np.asarray(y, dtype=np.float64).ravel()
        else:
            y_flat = np.zeros_like(x_flat)
        z_flat = np.empty_like(x_flat)
        for i in range(x_flat.shape[0]):
            z_flat[i] = self._ray_z(x_flat[i], y_flat[i])
        return z_flat.reshape(shape)

    def side_clearance(self, x: float, y: float, z: float,
                        direction: np.ndarray) -> float:
        """Horizontal ray from (x, y, z) along direction → distance to terrain side.

        Returns distance to nearest terrain surface in the xy plane at height z.
        If no hit within 2m, returns 2.0.
        """
        dir_xy = np.array([float(direction[0]), float(direction[1]), 0.0], dtype=np.float64)
        norm = np.linalg.norm(dir_xy)
        if norm < 1e-8:
            return 2.0
        dir_xy /= norm
        pnt = np.array([float(x), float(y), float(z)], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self._model, self._data, pnt, dir_xy,
            self._geomgroup, 1, -1, geomid,
        )
        if dist < 0 or int(geomid[0]) not in self._terrain_geom_ids:
            return 2.0
        return float(dist)

    def foot_side_penetration(self, ankle_x: float, ankle_y: float, ankle_z: float,
                              fwd_x: float, fwd_y: float,
                              foot_half_width: float = 0.05,
                              foot_half_length: float = 0.12) -> tuple:
        """Check if foot rectangle penetrates terrain sidewalls.

        Shoots horizontal rays from ankle position at foot-sole height in 4 directions.

        Returns:
            (push_x, push_y): correction to move foot away from walls. (0,0) if clear.
        """
        check_z = ankle_z - 0.03  # check near sole level
        if check_z < 0.005:
            check_z = 0.005

        lat_x = -fwd_y
        lat_y = fwd_x
        push_x, push_y = 0.0, 0.0

        checks = [
            (fwd_x, fwd_y, foot_half_length),
            (-fwd_x, -fwd_y, foot_half_length),
            (lat_x, lat_y, foot_half_width),
            (-lat_x, -lat_y, foot_half_width),
        ]
        for dx, dy, half_size in checks:
            direction = np.array([dx, dy, 0.0])
            dist = self.side_clearance(ankle_x, ankle_y, check_z, direction)
            if dist < half_size:
                penetration = half_size - dist
                push_x -= penetration * dx
                push_y -= penetration * dy

        return push_x, push_y

    def local_heightmap(
        self,
        x_robot: float,
        y_robot: float = 0.0,
        forward: float = 0.8,
        backward: float = 0.2,
        lateral: float = 0.3,
        resolution: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local heightmap grid (same signature as StairTerrain)."""
        x_coords = np.arange(x_robot - backward, x_robot + forward, resolution)
        y_coords = np.arange(y_robot - lateral, y_robot + lateral, resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)
        heightmap = self.height_batch(xx, yy)
        return heightmap, x_coords, y_coords

    def print_info(self):
        """Print terrain summary (matches StairTerrain format)."""
        print(f"RaycastTerrain: {len(self.steps)} detected steps, y_width=±{self.half_width_y}m")
        for s in self.steps:
            print(f"  {s.name:20s}: x=[{s.x_lo:.3f}, {s.x_hi:.3f}), top_z={s.top_z:.3f}m")

    # ------------------------------------------------------------------
    # Step detection via x-axis scan  (builds self.steps)
    # ------------------------------------------------------------------

    def _scan_steps(
        self,
        x_range: Tuple[float, float],
        resolution: float,
        z_tol: float,
        min_width: float,
    ) -> List[StairStep]:
        """
        Scan along y=0 to detect flat segments with z > 0 (steps).

        Groups consecutive samples with the same z (within z_tol) into
        StairStep objects with x_lo / x_hi / top_z.
        """
        xs = np.arange(x_range[0], x_range[1], resolution)
        zs = np.array([self._ray_z(float(x), 0.0) for x in xs])

        steps: List[StairStep] = []
        i = 0
        n = len(xs)
        while i < n:
            z_cur = zs[i]
            if z_cur < 1e-6:
                # ground level, skip
                i += 1
                continue
            # find contiguous run with same z
            j = i + 1
            while j < n and abs(zs[j] - z_cur) < z_tol:
                j += 1
            x_lo = float(xs[i])
            x_hi = float(xs[j - 1]) + resolution
            width = x_hi - x_lo
            if width >= min_width:
                avg_z = float(np.mean(zs[i:j]))
                steps.append(StairStep(
                    x_lo=x_lo, x_hi=x_hi, top_z=avg_z,
                    name=f"scan_{len(steps)}",
                ))
            i = j

        steps.sort(key=lambda s: s.x_lo)
        return steps

    @staticmethod
    def from_xml_path(
        xml_path: str,
        half_width_y: float = 2.0,
        scan_x_range: Tuple[float, float] = (-1.0, 5.0),
        scan_resolution: float = 0.005,
    ) -> "RaycastTerrain":
        """Convenience: load model from XML, build terrain."""
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return RaycastTerrain(
            model, data,
            half_width_y=half_width_y,
            scan_x_range=scan_x_range,
            scan_resolution=scan_resolution,
        )


def main():
    """Validation: load G1 scene XML, query heights, compare with known values."""
    import os
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'g1', 'g1_29dof_scene_stairs_ud.xml')
    xml_path = os.path.abspath(xml_path)

    print(f"Loading terrain from: {xml_path}")
    terrain = StairTerrain.from_scene_xml(xml_path)
    terrain.print_info()

    # Known ground truth from the XML
    expected = {
        (0.0, 0.0): 0.0,       # before stairs
        (0.55, 0.0): 0.10,     # step_up_0
        (0.75, 0.0): 0.20,     # step_up_1
        (0.95, 0.0): 0.30,     # step_up_2
        (1.15, 0.0): 0.40,     # step_up_3
        (1.35, 0.0): 0.50,     # step_up_4
        (1.70, 0.0): 0.50,     # plateau
        (2.05, 0.0): 0.50,     # step_down_0 (same height as plateau)
        (2.25, 0.0): 0.40,     # step_down_1
        (2.45, 0.0): 0.30,     # step_down_2
        (2.65, 0.0): 0.20,     # step_down_3
        (2.85, 0.0): 0.10,     # step_down_4
        (3.00, 0.0): 0.0,      # past stairs
    }

    print(f"\n{'x':>6} {'y':>4} {'expected':>10} {'query':>10} {'match':>6}")
    all_pass = True
    for (x, y), exp_z in expected.items():
        q_z = terrain.height_at(x, y)
        match = abs(q_z - exp_z) < 0.001
        all_pass &= match
        print(f"{x:6.2f} {y:4.1f} {exp_z:10.3f} {q_z:10.3f} {'OK' if match else 'FAIL':>6}")

    # Test batch query
    xs = np.array([x for x, y in expected.keys()])
    ys = np.array([y for x, y in expected.keys()])
    batch_z = terrain.height_batch(xs, ys)
    exp_zs = np.array(list(expected.values()))
    batch_match = np.allclose(batch_z, exp_zs, atol=0.001)

    print(f"\nBatch query: {'PASS' if batch_match else 'FAIL'}")

    # Test local heightmap
    hmap, x_coords, y_coords = terrain.local_heightmap(x_robot=1.0, y_robot=0.0)
    print(f"Local heightmap: shape={hmap.shape}, x=[{x_coords[0]:.2f}, {x_coords[-1]:.2f}], "
          f"y=[{y_coords[0]:.2f}, {y_coords[-1]:.2f}]")
    print(f"Height range: [{hmap.min():.3f}, {hmap.max():.3f}]")

    print(f"\nOverall: {'ALL PASS' if all_pass and batch_match else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
