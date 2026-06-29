# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-augmented SONIC Actor-Critic with Modulated Residual Architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.actor_critic import SonicActorCritic, FSQ


class MapCNN(nn.Module):
    """Hybrid CNN for terrain feature extraction with minimal spatial context.

    Architecture (Option B - Hybrid):
    1. Conv2d(3x3) for local spatial context (slopes, edges)
    2. Conv2d(1x1) for per-pixel feature refinement without blurring

    Final output channels = dim_out - 3 (leaving room for raw coordinate skip connection).
    """
    def __init__(self, dim_out: int = 64, in_channels: int = 1):
        super().__init__()
        # Output dim should leave 3 channels for (x, y, z) skip connection
        cnn_out_dim = dim_out - 3
        assert cnn_out_dim > 0, f"dim_out must be > 3 to accommodate skip connection, got {dim_out}"

        # Hybrid: 3x3 for spatial context + 1x1 for feature refinement
        intermediate_dim = max(16, cnn_out_dim // 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(intermediate_dim, cnn_out_dim, kernel_size=1),
            # No final ReLU - preserve feature distribution for attention
        )
        self.out_dim = cnn_out_dim

    def forward(self, height_only: torch.Tensor) -> torch.Tensor:
        """
        Args:
            height_only: [B, 1, H, W] - just the z/height channel
        Returns:
            features: [B, cnn_out_dim, H, W]
        """
        return self.cnn(height_only)


# NOTE: This MapTransformer intentionally DIVERGES from the canonical one in
# networks/layers/map_transformer.py (its MapCNN uses dim_out=/in_channels=1 and
# the ctor takes map_height/map_width). State_dict bound to this variant's
# checkpoints, so it is NOT merged into the canonical class.
class MapTransformer(nn.Module):
    """Hardened Transformer-based terrain perception module with normalization.

    Architecture:
    1. CNN extracts features from height map z-channel
    2. Raw (x, y, z) coordinates are concatenated (skip connection)
    3. LayerNorm on query components (proprio, z_fsq) for fairness
    4. Cross-Attention: Query = (proprio_norm + z_fsq_norm), Keys/Values = map features
    5. Output: terrain encoding that can be fused with policy

    This allows the policy to attend to relevant terrain features based on
    current proprioception state and motion intent (FSQ token).
    """
    def __init__(
        self,
        dim_proprio: int,
        dim_intent: int,  # FSQ code dimension
        dim_map_embed: int = 64,
        num_heads: int = 4,
        map_height: int = 16,
        map_width: int = 10,
    ):
        super().__init__()
        self.dim_map_embed = dim_map_embed
        self.map_height = map_height
        self.map_width = map_width

        # CNN for height feature extraction (outputs dim_map_embed - 3 channels)
        self.map_cnn = MapCNN(dim_out=dim_map_embed, in_channels=1)

        # Normalization layers for query components (ensures fairness)
        self.norm_proprio = nn.LayerNorm(dim_proprio)
        self.norm_intent = nn.LayerNorm(dim_intent)

        # Query projection: proprio + intent → embedding dim
        self.query_proj = nn.Linear(dim_proprio + dim_intent, dim_map_embed)

        # Normalization for KV sequences (after reshape)
        self.norm_kv = nn.LayerNorm(dim_map_embed)

        # Cross-attention: Query looks into map features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_map_embed,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Linear(dim_map_embed, dim_map_embed)

        # Layer norm for stability
        self.norm_out = nn.LayerNorm(dim_map_embed)
    
    def forward(
        self,
        map_3d: torch.Tensor,
        proprio: torch.Tensor,
        z_fsq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            map_3d: [B, 3, H, W] - (x, y, z) coordinates per cell
            proprio: [B, dim_proprio] - proprioception
            z_fsq: [B, dim_intent] - FSQ motion token

        Returns:
            z_map: [B, dim_map_embed] - terrain encoding
        """
        B = map_3d.shape[0]

        # 1. Extract height channel and run through CNN
        height_only = map_3d[:, 2:3, :, :]  # [B, 1, H, W]
        cnn_feat = self.map_cnn(height_only)  # [B, d-3, H, W]

        # 2. Skip connection: concat raw coordinates
        combined_feat = torch.cat([cnn_feat, map_3d], dim=1)  # [B, d, H, W]

        # 3. Reshape to sequence: [B, H*W, d]
        kv_seq = combined_feat.view(B, self.dim_map_embed, -1).permute(0, 2, 1)

        # 4. Normalize KV sequence (after reshape)
        kv_seq = self.norm_kv(kv_seq)

        # 5. Build query from normalized proprio + motion intent
        p_norm = self.norm_proprio(proprio)
        z_norm = self.norm_intent(z_fsq)
        query_input = torch.cat([p_norm, z_norm], dim=-1)
        query = self.query_proj(query_input).unsqueeze(1)  # [B, 1, d]

        # 6. Cross-attention
        attn_out, _ = self.cross_attn(query, kv_seq, kv_seq)  # [B, 1, d]

        # 7. Output projection with normalization
        out = self.out_proj(attn_out.squeeze(1))  # [B, d]
        out = self.norm_out(out)

        return out


class ResidualVisionSonicActorCritic(SonicActorCritic):
    """Vision-augmented SONIC Actor-Critic with Modulated Residual Architecture.

    Extends SonicActorCritic with Transformer-based terrain perception and
    modulated residual policy structure.

    Architecture:
    1. Mimic Stream: robot_encoder → FSQ → z_fsq (motion intent)
    2. Vision Stream: height_map → CNN → Transformer → z_map
    3. Modulation: z_map → MLP → delta_z; z_modulated = z_fsq + delta_z
    4. Base Policy: MLP(proprio, z_modulated) → action_base
    5. Residual Policy: MLP(proprio, z_map) → action_residual
    6. Final Action: action_base + action_residual * residual_scale

    Key features:
    1. Intent modulation: Vision adaptively modifies motion intent
    2. Residual corrections: Vision provides direct stability adjustments
    3. Zero initialization: Both modulation and residual start at identity
    4. Frozen encoder support: Preserve learned motion priors
    5. Critic uses privileged observations (proprio + z_fsq + z_map)
    """
    is_recurrent: bool = False
    
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # SONIC configs (inherited)
        robot_motion_dim: int = 580,
        human_motion_dim: int = 660,
        proprio_dim: int = 0,
        latent_dim: int = 256,
        fsq_levels: list[int] = [8, 8, 8, 5],
        actor_hidden_dims: list[int] = [2048, 2048, 1024, 1024, 512, 512],
        encoder_hidden_dims: list[int] = [2048, 1024, 512, 512],
        motion_decoder_hidden_dims: list[int] = [2048, 1024, 512, 512],
        init_noise_std: float = 1.0,
        # Vision-specific configs
        map_height: int = 17,
        map_width: int = 11,
        map_resolution: float = 0.1,
        dim_map_embed: int = 64,
        num_attn_heads: int = 4,
        vision_dropout: float = 0.0,  # Probability of dropping vision features
        freeze_encoder: bool = False,  # Freeze SONIC encoder weights
        # Modulated Residual configs
        modulation_hidden_dims: list[int] = [256, 128],  # MLP: z_map → delta_z
        residual_hidden_dims: list[int] = [256, 128],    # MLP: (proprio, z_map) → delta_action
        residual_scale: float = 1.0,  # Scaling factor for residual actions
        **kwargs: dict[str, Any],
    ) -> None:
        # Initialize parent (SONIC components)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            robot_motion_dim=robot_motion_dim,
            human_motion_dim=human_motion_dim,
            proprio_dim=proprio_dim,
            latent_dim=latent_dim,
            fsq_levels=fsq_levels,
            actor_hidden_dims=actor_hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            motion_decoder_hidden_dims=motion_decoder_hidden_dims,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        # Store vision and residual config
        self.map_height = map_height
        self.map_width = map_width
        self.map_resolution = map_resolution
        self.dim_map_embed = dim_map_embed
        self.vision_dropout = vision_dropout
        self.freeze_encoder = freeze_encoder
        self.residual_scale = residual_scale

        # Create pre-computed coordinate grids (registered as buffers)
        self._register_coordinate_grids(map_height, map_width, map_resolution)

        # Vision Transformer module
        self.map_transformer = MapTransformer(
            dim_proprio=self.proprio_dim,
            dim_intent=self.latent_dim,  # FSQ output dimension
            dim_map_embed=dim_map_embed,
            num_heads=num_attn_heads,
            map_height=map_height,
            map_width=map_width,
        )

        # Modulation Network: z_map → delta_z (modifies motion intent)
        # Output dimension must match latent_dim to be added to z_fsq
        self.modulation_mlp = MLP(
            dim_map_embed,
            latent_dim,
            modulation_hidden_dims,
            "elu"
        )
        # Zero initialization: Start as identity (no modulation)
        self._zero_init_network(self.modulation_mlp)

        # Base Policy (Blind to raw map): proprio + z_modulated → action_base
        # This replaces the original control_decoder with REDUCED input dim
        base_policy_input_dim = self.proprio_dim + self.latent_dim
        self.control_decoder = MLP(
            base_policy_input_dim,
            num_actions,
            actor_hidden_dims,
            "elu"
        )

        # Residual Policy: (proprio, z_map) → delta_action (stability correction)
        residual_input_dim = self.proprio_dim + dim_map_embed
        self.residual_mlp = MLP(
            residual_input_dim,
            num_actions,
            residual_hidden_dims,
            "elu"
        )
        # Zero initialization: Start with no residual corrections
        self._zero_init_network(self.residual_mlp)

        print(f"[VisionSonicActorCritic - Modulated Residual] Initialized with:")
        print(f"  - Proprio dim: {self.proprio_dim}")
        print(f"  - FSQ dim (latent): {self.latent_dim}")
        print(f"  - Map embed dim: {dim_map_embed}")
        print(f"  - Map size: {map_height}x{map_width} @ {map_resolution}m")
        print(f"  - Vision dropout: {vision_dropout}")
        print(f"  - Freeze encoder: {freeze_encoder}")
        print(f"  - Modulation hidden dims: {modulation_hidden_dims}")
        print(f"  - Residual hidden dims: {residual_hidden_dims}")
        print(f"  - Residual scale: {residual_scale}")

        # Optionally freeze encoder
        if freeze_encoder:
            self._freeze_encoder()
    
    def _register_coordinate_grids(
        self,
        height: int,
        width: int,
        resolution: float,
    ) -> None:
        """Pre-compute and register (x, y) coordinate grids as buffers.
        
        Uses grid indices (0 to m, 0 to n) instead of physical coordinates.
        This is simpler and resolution-independent - the CNN can learn
        spatial patterns regardless of physical scale.
        
        The grid indices are normalized to [0, 1] range for better network behavior.
        """
        # Create coordinate grids using grid indices
        # xs: row indices from 0 to height-1, normalized to [0, 1]
        # ys: col indices from 0 to width-1, normalized to [0, 1]
        xs = torch.linspace(0, 1, height)  # [0, 1] normalized
        ys = torch.linspace(0, 1, width)   # [0, 1] normalized
        
        # Create meshgrid: [H, W] each
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        
        # Register as buffers (will move with model, not trainable)
        self.register_buffer("_grid_x", grid_x)  # [H, W]
        self.register_buffer("_grid_y", grid_y)  # [H, W]
    
    def _freeze_encoder(self) -> None:
        """Freeze SONIC encoder and FSQ weights."""
        for param in self.robot_encoder.parameters():
            param.requires_grad = False
        for param in self.human_encoder.parameters():
            param.requires_grad = False
        if self.hybrid_encoder is not None:
            for param in self.hybrid_encoder.parameters():
                param.requires_grad = False
        for param in self.robot_encoder_proj.parameters():
            param.requires_grad = False
        for param in self.human_encoder_proj.parameters():
            param.requires_grad = False
        if self.hybrid_encoder_proj is not None:
            for param in self.hybrid_encoder_proj.parameters():
                param.requires_grad = False
        # Note: FSQ has no learnable parameters (just operations on buffers)
        print("[VisionSonicActorCritic] Frozen encoder weights")

    def _zero_init_network(self, network: nn.Module) -> None:
        """Initialize the last layer of an MLP to near-zero output.

        This ensures the network starts as an identity/zero transform:
        - Modulation MLP: delta_z ≈ 0 → z_modulated ≈ z_fsq
        - Residual MLP: delta_action ≈ 0 → action ≈ action_base

        Strategy:
        - Hidden layers: Keep default initialization (Xavier/Kaiming)
        - Last layer: Scale weights to 0.01 and zero bias
        """
        # MLP inherits from nn.Sequential, so iterate through modules
        last_linear = None
        for module in network.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is not None:
            # Scale weights to near-zero
            nn.init.xavier_uniform_(last_linear.weight)
            last_linear.weight.data *= 0.01
            # Zero bias
            if last_linear.bias is not None:
                last_linear.bias.data.zero_()
            print(f"[VisionSonicActorCritic] Zero-initialized last linear layer")
        else:
            print(f"[Warning] Could not find linear layer in {network.__class__.__name__}")
    
    def process_height_scan(self, height_scan: torch.Tensor) -> torch.Tensor:
        """Convert flat height scan to 3D map tensor.
        
        Args:
            height_scan: [B, H*W] - flat height values from sensor
            
        Returns:
            map_3d: [B, 3, H, W] - (x, y, z) channels
        """
        B = height_scan.shape[0]
        
        # Reshape height to [B, 1, H, W]
        map_z = height_scan.view(B, 1, self.map_height, self.map_width)
        
        # Expand coordinate grids to batch: [B, 1, H, W]
        map_x = self._grid_x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        map_y = self._grid_y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Concatenate: [B, 3, H, W] where channels are (x, y, z)
        map_3d = torch.cat([map_x, map_y, map_z], dim=1)
        
        return map_3d
    
    def _get_vision_encoding(
        self,
        obs: TensorDict,
        proprio: torch.Tensor,
        z_fsq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vision encoding from height scan.
        
        Args:
            obs: Observation dict (must contain 'vision' group)
            proprio: Proprioception tensor
            z_fsq: FSQ motion token
            
        Returns:
            z_map: [B, dim_map_embed] vision encoding
        """
        # Get height scan from vision observation group
        if "vision" in obs:
            height_scan = obs["vision"]
        else:
            # Fallback: try to get from a specific key
            raise KeyError("Vision observation group 'vision' not found in obs. "
                          "Make sure to add 'vision' to obs_groups config.")
        
        # Convert to 3D map
        map_3d = self.process_height_scan(height_scan)
        
        # Run through transformer
        z_map = self.map_transformer(map_3d, proprio, z_fsq)
        
        # Apply vision dropout during training
        if self.training and self.vision_dropout > 0:
            mask = torch.rand(z_map.shape[0], 1, device=z_map.device) > self.vision_dropout
            z_map = z_map * mask.float()
        
        return z_map
    
    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Sample action with modulated residual policy.

        Flow:
        1. Get proprio from policy obs
        2. Encode motion → z_fsq (motion intent)
        3. Encode vision → z_map (terrain features)
        4. Modulate intent: z_modulated = z_fsq + modulation_mlp(z_map)
        5. Base policy: action_base = control_decoder(proprio, z_modulated)
        6. Residual policy: action_residual = residual_mlp(proprio, z_map)
        7. Final action: action_base + action_residual * residual_scale
        """
        # Get proprioception
        obs_list = [obs[k] for k in self.obs_groups["policy"]]
        proprio = torch.cat(obs_list, dim=-1)

        # Get FSQ token (motion intent)
        if self.freeze_encoder:
            with torch.no_grad():
                z_fsq = self._select_action_token(obs)
        else:
            z_fsq = self._select_action_token(obs)

        # Get vision encoding (with attention)
        z_map = self._get_vision_encoding(obs, proprio, z_fsq)

        # Modulation: Vision adaptively modifies motion intent
        delta_z = self.modulation_mlp(z_map)
        z_modulated = z_fsq + delta_z

        # Base Policy: Uses modulated intent (blind to raw map)
        base_input = torch.cat([proprio, z_modulated], dim=-1)
        action_base = self.control_decoder(base_input)

        # Residual Policy: Uses raw vision for stability correction
        residual_input = torch.cat([proprio, z_map], dim=-1)
        action_residual = self.residual_mlp(residual_input)

        # Apply tanh + scaling to residual for stability
        action_residual = torch.tanh(action_residual) * self.residual_scale

        # Final action
        mean = action_base + action_residual

        # Distribution
        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)

        return self.distribution.sample()
    
    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action for evaluation (modulated residual)."""
        # Get proprioception
        obs_list = [obs[k] for k in self.obs_groups["policy"]]
        proprio = torch.cat(obs_list, dim=-1)

        # Get FSQ token (always no grad at inference)
        with torch.no_grad():
            z_fsq = self._select_action_token(obs)

        # Get vision encoding (no dropout at inference)
        z_map = self._get_vision_encoding(obs, proprio, z_fsq)

        # Modulation: Vision adaptively modifies motion intent
        delta_z = self.modulation_mlp(z_map)
        z_modulated = z_fsq + delta_z

        # Base Policy: Uses modulated intent (blind to raw map)
        base_input = torch.cat([proprio, z_modulated], dim=-1)
        action_base = self.control_decoder(base_input)

        # Residual Policy: Uses raw vision for stability correction
        residual_input = torch.cat([proprio, z_map], dim=-1)
        action_residual = self.residual_mlp(residual_input)

        # Apply tanh + scaling to residual for stability
        action_residual = torch.tanh(action_residual) * self.residual_scale

        # Final action
        mean = action_base + action_residual

        # Populate distribution for potential log_prob calls
        std = self._get_action_std(mean)
        self.distribution = Normal(mean, std)

        return mean
    
    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get parameters that should be trained (modulated residual version).

        Useful for creating optimizer with only unfrozen parameters.
        """
        trainable = []

        # Always train vision and modulated residual components
        trainable.extend(self.map_transformer.parameters())
        trainable.extend(self.modulation_mlp.parameters())
        trainable.extend(self.control_decoder.parameters())
        trainable.extend(self.residual_mlp.parameters())

        # Conditionally add encoder params
        if not self.freeze_encoder:
            trainable.extend(self.robot_encoder.parameters())
            trainable.extend(self.human_encoder.parameters())
            if self.hybrid_encoder is not None:
                trainable.extend(self.hybrid_encoder.parameters())
            trainable.extend(self.robot_encoder_proj.parameters())
            trainable.extend(self.human_encoder_proj.parameters())
            if self.hybrid_encoder_proj is not None:
                trainable.extend(self.hybrid_encoder_proj.parameters())

        # Always train critic and std
        trainable.extend(self.critic.parameters())
        trainable.append(self.std)

        return trainable
