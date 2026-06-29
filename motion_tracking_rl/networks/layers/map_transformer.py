# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical cross-attention terrain-perception block (``MapTransformer``).

PMT plan §2 / Phase 0 step 3 asked to de-duplicate the ``MapTransformer`` class
that appeared in four files. A diff of the four definitions (see the module-level
note below) found they are **materially different** — different constructor
signatures, different submodule names, and therefore **different state_dict
keys**. Merging all four would break checkpoint loading.

Resolution (conservative): this module hosts the **canonical** variant — the one
used by the registered slice network ``VisionTransformerActorCritic`` (it imports
``MapTransformer`` from ``residual_vision_action``). ``residual_vision_action``
now imports the class from here so there is a single source of truth for that
variant. The three divergent variants (in ``vision_sonic``,
``residual_vision_sonic``, ``deploy_residual_vision_sonic``) intentionally keep
their own local definitions, because their state_dict layouts differ and are
bound to their own checkpoints.

Divergence summary (do NOT blindly merge):

* ``vision_sonic.MapTransformer`` — original/no-norm: submodule ``norm`` (single
  output LayerNorm), no ``norm_proprio``/``norm_intent``/``norm_kv``; ``MapCNN``
  uses ``dim_out=`` and hardcodes ``in_channels=1``; ctor takes
  ``map_height``/``map_width``.
* ``residual_vision_sonic.MapTransformer`` — "hardened": adds
  ``norm_proprio``/``norm_intent``/``norm_kv``/``norm_out`` (renames ``norm`` →
  ``norm_out``); still uses ``MapCNN(dim_out=..., in_channels=1)``.
* ``deploy_residual_vision_sonic.MapTransformer`` — deploy-trimmed: like the
  hardened variant's submodules but with a slimmer ctor and ``z_intent`` arg.
* ``residual_vision_action.MapTransformer`` (THIS canonical one) — most general:
  ``terrain_input_channels``/``map_coord_channels`` ctor knobs, ``MapCNN`` with
  ``dim_map_embed=``/``map_coord_channels=``, shape validation in ``forward``.

All four use **distinct submodule names**, so their checkpoints are NOT
interchangeable — that is why only the canonical one is extracted here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MapCNN(nn.Module):
    """Hybrid CNN: 3x3 conv (local context) + 1x1 conv (refine).

    Output channels = ``dim_map_embed - map_coord_channels`` (reserving room for
    the raw coordinate channels concatenated back in by ``MapTransformer``).
    """

    def __init__(self, dim_map_embed: int = 64, in_channels: int = 1, map_coord_channels: int = 3):
        super().__init__()
        self.map_coord_channels = int(map_coord_channels)
        out_ch = dim_map_embed - self.map_coord_channels
        if out_ch <= 0:
            raise ValueError(
                f"dim_map_embed must be greater than map_coord_channels={self.map_coord_channels}, "
                f"got {dim_map_embed}"
            )

        mid_ch = max(16, out_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

    def forward(self, height_only: torch.Tensor) -> torch.Tensor:
        # height_only: [B,1,H,W]
        return self.net(height_only)  # [B, dim_map_embed-3, H, W]


class MapTransformer(nn.Module):
    """Cross-attention from a (proprio + intent) query into map KV features.

    - KV: CNN(z or [z,mask]) concatenated with raw (x,y,z[,mask]) => dim_map_embed
    - Query: [proprio, z_fsq] (proprio optional if dim_proprio==0)
    """

    def __init__(
        self,
        dim_proprio: int,
        dim_intent: int,
        dim_map_embed: int = 64,
        num_heads: int = 4,
        terrain_input_channels: int = 1,
        map_coord_channels: int = 3,
    ):
        super().__init__()
        self.dim_map_embed = dim_map_embed
        self.dim_proprio = dim_proprio
        self.terrain_input_channels = int(terrain_input_channels)
        self.map_coord_channels = int(map_coord_channels)

        # Safe norms
        self.norm_proprio = nn.LayerNorm(dim_proprio) if dim_proprio > 0 else nn.Identity()
        self.norm_intent = nn.LayerNorm(dim_intent)

        query_dim = dim_intent + dim_proprio
        self.query_proj = nn.Linear(query_dim, dim_map_embed)

        self.map_cnn = MapCNN(
            dim_map_embed=dim_map_embed,
            in_channels=self.terrain_input_channels,
            map_coord_channels=self.map_coord_channels,
        )

        self.norm_kv = nn.LayerNorm(dim_map_embed)
        self.cross_attn = nn.MultiheadAttention(dim_map_embed, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim_map_embed, dim_map_embed)
        self.norm_out = nn.LayerNorm(dim_map_embed)

    def forward(
        self,
        map_3d: torch.Tensor,   # [B,3,H,W] or [B,4,H,W] : (x,y,z[,mask]) metric coords + terrain
        proprio: torch.Tensor,  # [B,dim_proprio] or [B,0]
        z_fsq: torch.Tensor,    # [B,dim_intent]
    ) -> torch.Tensor:
        B = map_3d.shape[0]
        if map_3d.ndim != 4:
            raise ValueError(f"map_3d must be 4D [B,C,H,W], got shape {tuple(map_3d.shape)}")
        if map_3d.shape[1] != self.map_coord_channels:
            raise ValueError(
                f"map_3d channel mismatch: expected {self.map_coord_channels}, got {map_3d.shape[1]}"
            )

        # CNN on terrain channels: z only, or [z, validity_mask].
        terrain_only = map_3d[:, 2:, :, :]
        cnn_feat = self.map_cnn(terrain_only)           # [B, d-map_coord_channels, H, W]

        # concat raw coords (x,y,z)
        combined = torch.cat([cnn_feat, map_3d], dim=1)  # [B, d, H, W]
        if combined.shape[1] != self.dim_map_embed:
            raise RuntimeError(
                f"MapTransformer combined feature dim mismatch: expected {self.dim_map_embed}, "
                f"got {combined.shape[1]}"
            )

        # KV sequence
        kv = combined.reshape(B, self.dim_map_embed, -1).permute(0, 2, 1)  # [B, HW, d]
        kv = self.norm_kv(kv)

        # Query
        z_norm = self.norm_intent(z_fsq)
        if self.dim_proprio > 0:
            p_norm = self.norm_proprio(proprio)
            q_in = torch.cat([p_norm, z_norm], dim=-1)
        else:
            q_in = z_norm
        q = self.query_proj(q_in).unsqueeze(1)  # [B,1,d]

        attn_out, _ = self.cross_attn(q, kv, kv)        # [B,1,d]
        out = self.out_proj(attn_out.squeeze(1))        # [B,d]
        return self.norm_out(out)                       # [B,d]


__all__ = ["MapCNN", "MapTransformer"]
