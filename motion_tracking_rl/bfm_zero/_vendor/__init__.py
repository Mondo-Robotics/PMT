"""Vendored subset of the BFM-Zero ``humanoidverse`` package (FB-CPR-Aux algorithm core).

This package is a *verbatim* copy of the minimal import-closure needed to build and train the
BFM-Zero ``FBcprAuxAgent`` inside PMT, so PMT no longer requires the sibling ``BFM-Zero`` repo on
``sys.path``. Only pure-PyTorch algorithm / network / buffer code is vendored — none of the Isaac
Sim, humenv, or training-harness modules are included.

Provenance
----------
- Source repo : https://github.com/LeCAR-Lab/BFM-Zero  (humanoidverse package)
- Source commit: b87916f52d3d9e6eeba484f5e80851a235191837
- Original path mapping (relative imports preserved unchanged):
    humanoidverse.agents.*           -> motion_tracking_rl.bfm_zero._vendor.agents.*
    humanoidverse.utils.torch_utils  -> motion_tracking_rl.bfm_zero._vendor.torch_utils

License
-------
The vendored files are licensed under **CC BY-NC 4.0** (Creative Commons
Attribution-NonCommercial 4.0 International), Copyright (c) Meta Platforms, Inc. and affiliates.
See the upstream ``LICENSE`` in the BFM-Zero repository. This subset is included for
non-commercial research reproduction only; retain attribution if redistributing.

Files vendored (import closure of the FB-CPR-Aux agent):
    agents/{base, base_model, nn_models, nn_filters, nn_filter_models, normalizers, pytree_utils}.py
    agents/fb/{agent, model}.py
    agents/fb_cpr/{agent, model}.py
    agents/fb_cpr_aux/{agent, model}.py
    agents/misc/zbuffer.py
    agents/buffers/transition.py
    agents/envs/utils/gym_spaces.py
    torch_utils.py
"""
