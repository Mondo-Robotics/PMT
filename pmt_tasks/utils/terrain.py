# Copyright (c) 2024, The Isaac Lab Project Developers (whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Derived from https://github.com/HybridRobotics/whole_body_tracking (MIT).

"""Terrain utilities for height querying from trimesh terrains.

This module provides utilities to precompute and query terrain heights from
a terrain mesh. The key use case is during robot reset in the command system,
where we need terrain height BEFORE the robot is positioned (so we can't use
a sensor attached to the robot).

Usage:
    1. After terrain is generated, create a HeightMapTerrain from the mesh
    2. Use get_height(x, y) to query terrain height at any (x, y) position
    3. Heights are returned as tensors for efficient batched operations
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporter


class HeightMapTerrain:
    """Precomputed height map for efficient terrain height queries.
    
    This class takes a terrain mesh and provides efficient height queries
    using either:
    1. Grid-based precomputation (fast lookups, fixed resolution)
    2. Ray-mesh intersection (accurate, slower for many queries)
    
    The grid-based approach is recommended for RL training where speed matters.
    
    Attributes:
        mesh: The original trimesh mesh
        resolution: Grid resolution in meters (e.g., 0.1 = 10cm cells)
        height_grid: Precomputed height values [num_rows, num_cols]
        x_range: (x_min, x_max) bounds of the mesh
        y_range: (y_min, y_max) bounds of the mesh
    """
    
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        resolution: float = 0.1,
        device: str | torch.device = "cuda",
        use_grid: bool = True,
    ):
        """Initialize the height map from a terrain mesh.
        
        Args:
            mesh: The terrain trimesh mesh
            resolution: Grid cell size in meters for precomputation
            device: Device to store the height grid tensor
            use_grid: If True, precompute a grid for fast lookups.
                     If False, use ray casting for each query (slower but exact).
        """
        self.mesh = mesh
        self.resolution = resolution
        self.device = torch.device(device)
        self._use_grid = use_grid
        
        # Get mesh bounds
        bounds = mesh.bounds  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        self.x_min, self.y_min, self.z_min = bounds[0]
        self.x_max, self.y_max, self.z_max = bounds[1]
        
        # Store ranges for convenience
        self.x_range = (self.x_min, self.x_max)
        self.y_range = (self.y_min, self.y_max)
        self.z_range = (self.z_min, self.z_max)
        
        if use_grid:
            self._precompute_height_grid()
        else:
            # Create ray intersection accelerator for on-demand queries
            self._ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        # Cache for smoothed grids keyed by radius in cells
        self._smoothed_grids: dict[int, torch.Tensor] = {}
    
    def _precompute_height_grid(self):
        """Precompute a 2D grid of height values using ray casting."""
        # Calculate grid dimensions
        self.num_cols = int(np.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.num_rows = int(np.ceil((self.y_max - self.y_min) / self.resolution)) + 1
        
        print(f"[HeightMapTerrain] Precomputing height grid: {self.num_rows}x{self.num_cols} "
              f"({self.num_rows * self.num_cols} points)")
        
        # Create grid of (x, y) query points
        x_coords = np.linspace(self.x_min, self.x_max, self.num_cols)
        y_coords = np.linspace(self.y_min, self.y_max, self.num_rows)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Create ray origins above the mesh (high z value)
        z_start = self.z_max + 10.0
        num_points = self.num_rows * self.num_cols
        
        ray_origins = np.stack([
            xx.flatten(),
            yy.flatten(),
            np.full(num_points, z_start),
        ], axis=1)
        
        # Ray directions pointing downward
        ray_directions = np.zeros((num_points, 3))
        ray_directions[:, 2] = -1.0
        
        # Perform ray-mesh intersection
        ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        locations, index_ray, index_tri = ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        
        # Initialize height grid with z_min (default for rays that miss)
        height_grid = np.full((self.num_rows, self.num_cols), self.z_min, dtype=np.float32)
        
        # Fill in heights from ray hits
        if len(locations) > 0:
            # For each ray, take the first (highest) hit point
            # Group by ray index and take max z
            for i, hit_z in zip(index_ray, locations[:, 2]):
                row = i // self.num_cols
                col = i % self.num_cols
                # Take the highest z value for this cell (in case of multiple hits)
                height_grid[row, col] = max(height_grid[row, col], hit_z)
        
        # Store as torch tensor
        self.height_grid = torch.tensor(height_grid, dtype=torch.float32, device=self.device)
        
        # Store coordinate arrays for interpolation
        self._x_coords = torch.tensor(x_coords, dtype=torch.float32, device=self.device)
        self._y_coords = torch.tensor(y_coords, dtype=torch.float32, device=self.device)
        
        print(f"[HeightMapTerrain] Height grid precomputed. "
              f"Height range: [{height_grid.min():.3f}, {height_grid.max():.3f}]")
    
    def get_height(
        self,
        x: torch.Tensor | float | np.ndarray,
        y: torch.Tensor | float | np.ndarray,
        interpolate: bool = True,
    ) -> torch.Tensor:
        """Get terrain height at (x, y) positions.
        
        Args:
            x: X coordinates (can be scalar, 1D tensor, or batched)
            y: Y coordinates (must match x shape)
            interpolate: If True, use bilinear interpolation for sub-grid accuracy.
                        If False, use nearest neighbor (faster).
        
        Returns:
            Tensor of heights with same shape as input x/y
        """
        # Convert inputs to tensors
        scalar_input = False
        if isinstance(x, (float, int)):
            x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor([y], dtype=torch.float32, device=self.device)
            scalar_input = True
        elif isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y).to(device=self.device, dtype=torch.float32)
        else:
            # x and y are torch.Tensor (guaranteed by type check above)
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            x_tensor = x.to(device=self.device, dtype=torch.float32)
            y_tensor = y.to(device=self.device, dtype=torch.float32)
        
        original_shape = x_tensor.shape
        x_flat = x_tensor.flatten()
        y_flat = y_tensor.flatten()
        
        if self._use_grid:
            heights = self._grid_lookup(x_flat, y_flat, interpolate)
        else:
            heights = self._ray_lookup(x_flat, y_flat)
        
        heights = heights.view(original_shape)
        
        if scalar_input:
            return heights.squeeze()
        return heights
    
    def _grid_lookup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        interpolate: bool,
        grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up heights from precomputed grid."""
        height_grid = self.height_grid if grid is None else grid
        # Normalize coordinates to grid indices
        # x -> col, y -> row
        col_float = (x - self.x_min) / self.resolution
        row_float = (y - self.y_min) / self.resolution
        
        if interpolate:
            # Bilinear interpolation
            col_lo = col_float.floor().long()
            row_lo = row_float.floor().long()
            col_hi = col_lo + 1
            row_hi = row_lo + 1
            
            # Clamp to valid range
            col_lo = col_lo.clamp(0, self.num_cols - 1)
            col_hi = col_hi.clamp(0, self.num_cols - 1)
            row_lo = row_lo.clamp(0, self.num_rows - 1)
            row_hi = row_hi.clamp(0, self.num_rows - 1)
            
            # Interpolation weights
            col_frac = (col_float - col_lo.float()).clamp(0, 1)
            row_frac = (row_float - row_lo.float()).clamp(0, 1)
            
            # Get corner values
            h00 = height_grid[row_lo, col_lo]
            h01 = height_grid[row_lo, col_hi]
            h10 = height_grid[row_hi, col_lo]
            h11 = height_grid[row_hi, col_hi]
            
            # Bilinear interpolation
            h0 = h00 * (1 - col_frac) + h01 * col_frac
            h1 = h10 * (1 - col_frac) + h11 * col_frac
            heights = h0 * (1 - row_frac) + h1 * row_frac
        else:
            # Nearest neighbor
            col = col_float.round().long().clamp(0, self.num_cols - 1)
            row = row_float.round().long().clamp(0, self.num_rows - 1)
            heights = height_grid[row, col]
        
        return heights

    def _get_smoothed_grid(self, radius_m: float) -> torch.Tensor:
        """Return a cached box-filtered height grid for a given radius (meters)."""
        if radius_m <= 0.0:
            return self.height_grid

        radius_cells = max(1, int(np.ceil(radius_m / self.resolution)))
        cached = self._smoothed_grids.get(radius_cells)
        if cached is not None:
            return cached

        kernel_size = 2 * radius_cells + 1
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size),
            dtype=self.height_grid.dtype,
            device=self.height_grid.device,
        ) / float(kernel_size * kernel_size)
        grid = self.height_grid.unsqueeze(0).unsqueeze(0)
        smoothed = torch.nn.functional.conv2d(grid, kernel, padding=radius_cells)
        smoothed = smoothed.squeeze(0).squeeze(0)
        self._smoothed_grids[radius_cells] = smoothed
        return smoothed

    def get_flat_height(
        self,
        x: torch.Tensor | float | np.ndarray,
        y: torch.Tensor | float | np.ndarray,
        radius: float = 0.2,
        interpolate: bool = True,
    ) -> torch.Tensor:
        """Get a smoothed terrain height by averaging a local neighborhood.

        Args:
            x: X coordinates
            y: Y coordinates
            radius: Neighborhood radius (meters) used for box filtering
            interpolate: If True, use bilinear interpolation on the smoothed grid

        Returns:
            Tensor of heights with same shape as input x/y
        """
        if not self._use_grid:
            return self.get_height(x, y, interpolate=interpolate)

        scalar_input = False
        if isinstance(x, (float, int)):
            x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor([y], dtype=torch.float32, device=self.device)
            scalar_input = True
        elif isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y).to(device=self.device, dtype=torch.float32)
        else:
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            x_tensor = x.to(device=self.device, dtype=torch.float32)
            y_tensor = y.to(device=self.device, dtype=torch.float32)

        original_shape = x_tensor.shape
        x_flat = x_tensor.flatten()
        y_flat = y_tensor.flatten()

        smoothed = self._get_smoothed_grid(radius)
        heights = self._grid_lookup(x_flat, y_flat, interpolate, grid=smoothed)
        heights = heights.view(original_shape)

        if scalar_input:
            return heights.squeeze()
        return heights
    
    def _ray_lookup(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Look up heights using ray casting (slower but exact)."""
        n = x.shape[0]
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Create ray origins above the mesh
        z_start = self.z_max + 10.0
        ray_origins = np.stack([x_np, y_np, np.full(n, z_start)], axis=1)
        ray_directions = np.zeros((n, 3))
        ray_directions[:, 2] = -1.0
        
        # Perform ray-mesh intersection
        locations, index_ray, index_tri = self._ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        
        # Default to z_min for rays that miss
        heights_np = np.full(n, self.z_min, dtype=np.float32)
        
        if len(locations) > 0:
            # For each ray, take the first hit
            for i, hit_z in zip(index_ray, locations[:, 2]):
                heights_np[i] = max(heights_np[i], hit_z)
        
        return torch.tensor(heights_np, dtype=torch.float32, device=self.device)
    
    def get_height_at_positions(
        self,
        positions: torch.Tensor,
        interpolate: bool = True,
    ) -> torch.Tensor:
        """Get terrain heights for a batch of 2D or 3D positions.
        
        Args:
            positions: Tensor of shape (..., 2) or (..., 3) containing (x, y) or (x, y, z)
            interpolate: Use bilinear interpolation if True
        
        Returns:
            Tensor of heights with shape positions.shape[:-1]
        """
        x = positions[..., 0]
        y = positions[..., 1]
        return self.get_height(x, y, interpolate)
    
    @classmethod
    def from_terrain_importer(
        cls,
        terrain_importer: "TerrainImporter",
        resolution: float = 0.1,
        device: str | torch.device = "cuda",
    ) -> "HeightMapTerrain":
        """Create HeightMapTerrain from an IsaacLab TerrainImporter.
        
        Note: This requires accessing the terrain mesh from the TerrainGenerator
        that was used during terrain import. The mesh is stored in the generator.
        
        Args:
            terrain_importer: IsaacLab TerrainImporter instance
            resolution: Grid cell size in meters
            device: Device for the height tensor
        
        Returns:
            HeightMapTerrain instance
        
        Raises:
            ValueError: If terrain mesh cannot be accessed
        """
        # TerrainImporter doesn't directly store the mesh anymore,
        # but we can access it if terrain_type was "generator"
        # The mesh was passed to create_prim_from_mesh during import
        
        # For now, we need to recreate the terrain generator to get the mesh
        # This is a workaround since the mesh is not stored in TerrainImporter
        if terrain_importer.cfg.terrain_type == "generator" and terrain_importer.cfg.terrain_generator is not None:
            generator = terrain_importer.cfg.terrain_generator.class_type(
                cfg=terrain_importer.cfg.terrain_generator,
                device=device,
            )
            return cls(generator.terrain_mesh, resolution, device)
        else:
            raise ValueError(
                "HeightMapTerrain can only be created from generated terrains. "
                f"Got terrain_type='{terrain_importer.cfg.terrain_type}'"
            )


# Global registry for height map (singleton pattern for cross-component access)
_HEIGHT_MAP_REGISTRY: dict[str, HeightMapTerrain] = {}


def register_height_map(name: str, height_map: HeightMapTerrain):
    """Register a height map in the global registry.
    
    Args:
        name: Unique name for this height map (e.g., "main_terrain")
        height_map: HeightMapTerrain instance
    """
    _HEIGHT_MAP_REGISTRY[name] = height_map
    print(f"[HeightMapTerrain] Registered height map '{name}'")


def get_height_map(name: str = "main_terrain") -> HeightMapTerrain | None:
    """Get a height map from the global registry.
    
    Args:
        name: Name of the height map to retrieve
    
    Returns:
        HeightMapTerrain instance, or None if not found
    """
    return _HEIGHT_MAP_REGISTRY.get(name)


def clear_height_map_registry():
    """Clear all registered height maps."""
    _HEIGHT_MAP_REGISTRY.clear()


def create_height_map_from_terrain_cfg(
    terrain_generator_cfg,
    device: str | torch.device = "cuda",
    resolution: float = 0.1,
    register_name: str | None = "main_terrain",
) -> HeightMapTerrain:
    """Create and optionally register a height map from a terrain generator config.
    
    This is the recommended way to create a height map during environment setup.
    Call this after the terrain is generated (e.g., in prestartup event).
    
    Args:
        terrain_generator_cfg: TerrainGeneratorCfg instance
        device: Device for the height tensor
        resolution: Grid cell size in meters
        register_name: If provided, register the height map with this name
    
    Returns:
        HeightMapTerrain instance
    """
    # Create generator
    generator = terrain_generator_cfg.class_type(cfg=terrain_generator_cfg, device=device)
    
    # Create height map from mesh
    height_map = HeightMapTerrain(
        mesh=generator.terrain_mesh,
        resolution=resolution,
        device=device,
        use_grid=True,
    )
    
    # Optionally register
    if register_name:
        register_height_map(register_name, height_map)
    
    return height_map
