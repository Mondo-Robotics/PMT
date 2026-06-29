from __future__ import annotations

from typing import Literal

import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass


class MeshCompatibleTerrainImporter(TerrainImporter):
    """Terrain importer compatibility layer for Isaac Lab versions without mesh terrain cfg support."""
    
    def __init__(self, cfg: "MeshCompatibleTerrainImporterCfg"):
        if cfg.terrain_type != "mesh":
            super().__init__(cfg)
            return

        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore[assignment]

        self.terrain_prim_paths = []
        self._meshes = None
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        if self.cfg.mesh_path is None:
            raise ValueError("Input terrain type is 'mesh' but no value provided for 'mesh_path'.")

        mesh = trimesh.load(self.cfg.mesh_path)
        mesh = self._coerce_to_trimesh(mesh)
        self._meshes = mesh
        self.import_mesh("terrain", mesh)
        self.configure_env_origins()
        self.set_debug_vis(self.cfg.debug_vis)

    @staticmethod
    def _coerce_to_trimesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
        """Convert a loaded trimesh asset into a single Trimesh."""
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        if isinstance(mesh, trimesh.Scene):
            geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
            if not geometries:
                raise ValueError("Loaded mesh scene does not contain any trimesh geometries.")
            if len(geometries) == 1:
                return geometries[0].copy()
            return trimesh.util.concatenate(geometries)
        raise ValueError(f"Unsupported mesh asset type: {type(mesh)!r}")


@configclass
class MeshCompatibleTerrainImporterCfg(TerrainImporterCfg):
    """Backward-compatible terrain importer config with mesh terrain support."""

    class_type: type = MeshCompatibleTerrainImporter
    terrain_type: Literal["generator", "plane", "usd", "mesh"] | str = "generator"
    mesh_path: str | None = None
