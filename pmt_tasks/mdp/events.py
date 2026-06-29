from __future__ import annotations

import glob
import numpy as np
import os
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import omni.usd
from pxr import Sdf, Gf, UsdGeom, Vt
import isaaclab.sim as sim_utils
import trimesh
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.terrains.trimesh.utils import make_plane

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def initialize_terrain_height_map(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mesh_prim_path: list[str]  = "/World/ground",
):
    """Initialize the precomputed terrain height map from the terrain mesh.
    
    This function should be called at STARTUP (not prestartup) because:
    1. The terrain mesh is created during scene initialization
    2. At startup, the scene is fully built and physics view is available
    3. The command term may need the height map for robot positioning
    
    This function reads the terrain generator config from the scene and creates
    a HeightMapTerrain that precomputes heights via ray casting. The height map
    is registered globally so commands can access it.
    
    Args:
        env: The environment instance.
        env_ids: Not used, but required by event manager signature.
    """
    from pmt_tasks.utils.terrain import (
        HeightMapTerrain,
        register_height_map,
        create_height_map_from_terrain_cfg,
    )
    
    # Check if scene has terrain configuration with a generator
    scene_cfg = env.scene.cfg
    terrain_cfg = getattr(scene_cfg, 'terrain', None)
    if terrain_cfg is None:
        print("[initialize_terrain_height_map] No terrain found in scene config. Skipping height map creation.")
        return
    
    if not hasattr(terrain_cfg, 'terrain_generator') or terrain_cfg.terrain_generator is None:
        print("[initialize_terrain_height_map] No terrain generator found. Skipping height map creation.")
        return
    
    if terrain_cfg.terrain_type != "generator":
        print(f"[initialize_terrain_height_map] Terrain type is '{terrain_cfg.terrain_type}', not 'generator'. Skipping.")
        return
    
    print("[initialize_terrain_height_map] Creating precomputed height map from terrain mesh...")
    
    try:
        # Create the terrain generator to get the mesh
        # Note: The generator was already created during TerrainImporter init,
        # but we need to recreate it to access the mesh (it's not stored)
        # mesh_prim = sim_utils.get_first_matching_child_prim(
        #     mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        # )
        # for mesh_prim_path in self.cfg.mesh_prim_paths:
        #     # check if the prim is a plane - handle PhysX plane as a special case
        #     # if a plane exists then we need to create an infinite mesh that is a plane
        #     mesh_prim = sim_utils.get_first_matching_child_prim(
        #         mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        #     )
        #     # if we did not find a plane then we need to read the mesh
        if mesh_prim_path is not None:
            # obtain the mesh prim
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
            )
            # check if valid
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            # cast into UsdGeomMesh
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            # read the vertices and faces
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = np.matmul(points, transform_matrix[:3, :3].T)
            points += transform_matrix[:3, 3]
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            _terrain_mesh = trimesh.Trimesh(vertices=points, faces=indices.reshape(-1, 3))
            # print info
            omni.log.info(
                f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
            )
        else:
            _terrain_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            # print info
            omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
        #     self.meshes[mesh_prim_path] = wp_mesh   
        # Create height map from mesh
        height_map = HeightMapTerrain(
            mesh=_terrain_mesh,#env.scene._terrain._terrain_mesh,
            resolution=0.1,  # 10cm grid cells
            device=str(env.device),
            use_grid=True,
        )
        
        # Register globally so commands can access it
        register_height_map("main_terrain", height_map)
        
        print(f"[initialize_terrain_height_map] Height map created successfully!")
        print(f"  - Grid size: {height_map.num_rows}x{height_map.num_cols}")
        print(f"  - X range: [{height_map.x_min:.2f}, {height_map.x_max:.2f}]")
        print(f"  - Y range: [{height_map.y_min:.2f}, {height_map.y_max:.2f}]")
        print(f"  - Z range: [{height_map.z_min:.2f}, {height_map.z_max:.2f}]")
        
    except Exception as e:
        print(f"[initialize_terrain_height_map] Failed to create height map: {e}")
        import traceback
        traceback.print_exc()


