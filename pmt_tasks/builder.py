"""Task config builder: SELECT -> DERIVE -> VALIDATE -> emit (PMT plan §3a/§3b/§10).

Wave 3 scope (no isaaclab/omni): this builder composes the chosen config-group
YAMLs with a small manual OmegaConf defaults-list composer (more robust than a
standalone Hydra compose(); see report), resolves ${paths.*}/${checkpoints.*}
against the active profile, runs the §3a derivation rules, validates the combo
against compat.py (§3b), and RETURNS the resolved OmegaConf config. The eventual
@configclass population + gym.register are Phase 1 (plan §10), deliberately not
wired here.

Layering (plan §10): this is the "build time, PMT-owned, before gym.register"
pass. It is strictly upstream of Isaac Lab's @hydra_task_config.
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from motion_tracking_rl import compat, registry
from pmt_tasks import derive as _derive

# configs/ lives at repo root, one level up from pmt_tasks/
CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"

# Axis order for composing the defaults list deterministically.
_AXES = (
    "robot", "terrain", "motion", "scene", "sensor", "obs",
    "reward", "network", "algorithm", "stage", "runner",
)


# --- paths -----------------------------------------------------------------

def load_paths(profile: str | None = None) -> DictConfig:
    """Load configs/paths.yaml and return the ACTIVE profile block as a flat
    namespace (DATA_ROOT/MOTION_ROOT/TERRAIN_ROOT/CKPT_ROOT), fully resolved.

    Profile precedence: explicit `profile` arg > $PMT_PROFILE > "local".
    Plan §5: switching the profile must select different roots, and the ${paths.*}
    interpolations must RESOLVE to concrete strings.
    """
    raw = OmegaConf.load(CONFIGS_ROOT / "paths.yaml")
    if profile is None:
        profile = os.environ.get("PMT_PROFILE", "local")
    if profile not in raw:
        raise ValueError(
            f"unknown PMT profile '{profile}'; available: "
            f"{[k for k in raw.keys() if k != 'profile']}"
        )
    block = raw[profile]
    # resolve the within-block ${.DATA_ROOT} relative interps to concrete strings
    resolved = OmegaConf.create(OmegaConf.to_container(block, resolve=True))
    return resolved


def load_checkpoints(paths: DictConfig) -> DictConfig:
    """Load configs/checkpoints/*.yaml into a `checkpoints` namespace, resolving
    each entry's run_dir against the active `paths`. Wave 3 resolves the STRING
    only (fail-loud filesystem glob-latest is a Phase-1 deliverable, plan §5)."""
    ckpt_dir = CONFIGS_ROOT / "checkpoints"
    out: dict[str, str] = {}
    for f in sorted(ckpt_dir.glob("*.yaml")):
        node = OmegaConf.load(f)
        merged = OmegaConf.merge(OmegaConf.create({"paths": paths}), node)
        merged = OmegaConf.create(OmegaConf.to_container(merged, resolve=True))
        run_dir = merged["run_dir"]
        ckpt = merged.get("checkpoint", "latest")
        # Wave 3: represent "latest" as a string token under run_dir; Phase 1
        # will glob-latest the actual model_*.pt fail-loud.
        out[f.stem] = run_dir if ckpt == "latest" else os.path.join(run_dir, ckpt)
    return OmegaConf.create(out)


# --- compose (manual defaults-list composer) -------------------------------

def _load_task_yaml(task_name_or_dict) -> DictConfig:
    if isinstance(task_name_or_dict, (dict, DictConfig)):
        return OmegaConf.create(task_name_or_dict)
    name = str(task_name_or_dict)
    path = CONFIGS_ROOT / "task" / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"task yaml not found: {path}")
    return OmegaConf.load(path)


def _compose_defaults(task_cfg: DictConfig) -> DictConfig:
    """Manual OmegaConf composer over a Hydra-style `defaults:` list.

    Reads `defaults: [{axis: choice}, ...]`, loads each configs/<axis>/<choice>.yaml
    under an `<axis>:` namespace, merges them in order, then merges the task's own
    top-level overrides (everything except `defaults`). Returns the merged config.
    """
    merged = OmegaConf.create({})
    defaults = task_cfg.get("defaults", [])
    for entry in defaults:
        # each entry is a single-key mapping {axis: choice}
        items = OmegaConf.to_container(entry) if isinstance(entry, DictConfig) else dict(entry)
        for axis, choice in items.items():
            group_path = CONFIGS_ROOT / axis / f"{choice}.yaml"
            if not group_path.exists():
                raise FileNotFoundError(
                    f"group yaml not found for axis '{axis}' choice '{choice}': {group_path}"
                )
            group_cfg = OmegaConf.load(group_path)
            merged = OmegaConf.merge(merged, OmegaConf.create({axis: group_cfg}))

    # task-level overrides (drop the defaults key)
    overrides = OmegaConf.create(
        {k: v for k, v in OmegaConf.to_container(task_cfg).items() if k != "defaults"}
    )
    merged = OmegaConf.merge(merged, overrides)
    return merged


# --- compat-name resolution (single source of truth = registry, plan §3b/§4) ---

def _resolve_compat_name(*, kind, cfg_block, registry_table):
    """Resolve a compat axis name from the YAML `name` via the registry table.

    Single source of truth: the registry decorator (`@register_*(name, compat_name=...)`)
    binds a class `name` to its compat axis. So given the YAML's `name`, the compat
    axis is looked up in `registry_table` (ALGORITHM_COMPAT / NETWORK_COMPAT).

    Precedence (fail-loud):
      1. registry_table[name]  -- the unified mapping; YAML needs only `name`.
      2. explicit YAML `compat_name` override -- for classes NOT yet @register_*'d
         (KNOWN_PENDING: e.g. diffusion, vision_student_latent_anchor). If both the
         registry AND an explicit override exist, they MUST agree (drift guard).
      3. neither -> raise a clear build-time error.
    """
    name = cfg_block.get("name")
    explicit = cfg_block.get("compat_name")
    mapped = registry_table.get(name) if name is not None else None

    if mapped is not None:
        if explicit is not None and explicit != mapped:
            raise ValueError(
                f"{kind} '{name}': YAML compat_name '{explicit}' disagrees with the "
                f"registry mapping '{mapped}'. Remove the redundant YAML override or fix it."
            )
        return mapped

    if explicit is not None:
        # name not in registry table (unregistered/pending class) -> trust the override
        return explicit

    raise ValueError(
        f"{kind} '{name}' has no compat axis: it is not in the registry "
        f"{kind.upper()}_COMPAT table and the YAML provides no `compat_name` override. "
        f"Either decorate the class with @register_{kind}(\"{name}\", compat_name=...) "
        f"or add an explicit `compat_name:` to its config/{kind}/*.yaml. "
        f"(registry-mapped {kind} names: {sorted(registry_table)})"
    )


# --- public API ------------------------------------------------------------

def build_task_config(task_name_or_dict, profile: str | None = None, overrides=None) -> DictConfig:
    """SELECT -> DERIVE -> VALIDATE -> emit a resolved OmegaConf config (plan §3a/§3b).

    Args:
      task_name_or_dict: a configs/task/<name>(.yaml) stem, or an inline dict with
        a `defaults:` list (used by the Phase-0.5 compat matrix loop).
      profile: paths profile override (else $PMT_PROFILE, else "local").
      overrides: optional list of dotlist override strings (e.g.
        ["network.use_identity_gate=false"]) merged after composition.

    Returns the resolved DictConfig with a derived `obs_groups`, `reward_weights`,
    and `runner` field set from compat.validate.
    """
    registry.autoload()  # ensure registries are populated (idempotent)
    registry.assert_compat_consistency()  # fail loud if compat tables drifted (§3b/§4)

    task_cfg = _load_task_yaml(task_name_or_dict)

    # (1) SELECT: compose the defaults list
    cfg = _compose_defaults(task_cfg)

    # inject paths + checkpoints namespaces so ${paths.*}/${checkpoints.*} resolve
    paths = load_paths(profile)
    cfg = OmegaConf.merge(OmegaConf.create({"paths": paths}), cfg)
    cfg = OmegaConf.merge(OmegaConf.create({"checkpoints": load_checkpoints(paths)}), cfg)

    # apply dotlist overrides (ablation sweeps etc.) before resolution
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # resolve all ${...} interpolations to concrete values (fail loud on dangling)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # (3) VALIDATE first to learn the runner (needed for obs-group derivation)
    alg_cfg = cfg["algorithm"]
    net_cfg = cfg["network"]
    alg_compat = _resolve_compat_name(
        kind="algorithm", cfg_block=alg_cfg,
        registry_table=registry.ALGORITHM_COMPAT,
    )
    net_compat = _resolve_compat_name(
        kind="network", cfg_block=net_cfg,
        registry_table=registry.NETWORK_COMPAT,
    )
    rnd = bool(alg_cfg.get("rnd", False))
    symmetry = bool(alg_cfg.get("symmetry", False))
    recurrent = bool(alg_cfg.get("recurrent", False))

    # derive the required obs sets up front so validate can check them
    spec = compat.SPECS.get(alg_compat)
    if spec is None:
        # let validate raise the canonical clear error
        compat.validate(alg_compat, net_compat, rnd=rnd, symmetry=symmetry, recurrent=recurrent)

    obs_groups = _derive.derive_obs_groups_with_features(
        net_cfg, cfg.get("stage"), spec, cfg["obs"], rnd=rnd
    )

    runner_name = compat.validate(
        alg_compat, net_compat,
        rnd=rnd, symmetry=symmetry, recurrent=recurrent,
        obs_sets=set(obs_groups.keys()),
    )

    # if a runner axis was explicitly selected, assert it matches the derived one
    explicit_runner = cfg.get("runner", None)
    if explicit_runner is not None and explicit_runner.get("name") not in (None, runner_name):
        raise ValueError(
            f"explicit runner '{explicit_runner.get('name')}' != derived runner "
            f"'{runner_name}' for algorithm '{alg_compat}'"
        )

    # (2) DERIVE the rest (reward weights as data)
    reward_weights = _derive.derive_reward_weights(cfg["reward"], cfg["obs"])

    # (4) emit: attach derived fields
    cfg["runner"] = OmegaConf.create({"name": runner_name})
    cfg["obs_groups"] = OmegaConf.create(obs_groups)
    cfg["reward_weights"] = OmegaConf.create(reward_weights)
    cfg["_derived"] = OmegaConf.create({
        "runner": runner_name,
        "anchor_in_actor_obs": _derive.anchor_in_actor_obs(cfg["obs"]),
    })

    return cfg


# --- Phase 1: emit concrete @configclass instances (plan §10 PART A/D) -------
#
# These functions import isaaclab-dependent modules lazily so the pure builder
# (build_task_config + tests above) stays importable in the wbt env. They are the
# "build time, PMT-owned, before gym.register" pass: compose+derive+validate, then
# populate a concrete @configclass instance. Both return a FRESH instance per call
# (no module-level singleton) per §10/D.


# Per-task env-cfg dispatch. Each task stem maps to a builder helper that knows how to
# inject the config-driven values into its concrete @configclass. Adding a Phase-1 task
# = one helper here (no class-tree edits). Helpers import isaaclab-dependent modules
# lazily so the pure builder stays importable in the wbt env.

def _resolve_task_stem(task_name_or_dict) -> str:
    if isinstance(task_name_or_dict, str):
        s = task_name_or_dict
        return s[:-5] if s.endswith(".yaml") else s
    return ""  # inline dict (compat-matrix loop): no env builder


def _build_stepping_stone_env(cfg):
    from pmt_tasks.env_cfgs.pmt.stepping_stone import PMTSteppingStoneEnvCfg

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTSteppingStoneEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_history_length = int(cfg["obs"].get("history_length", 10))
    reward_weights = cfg.get("reward_weights")
    if reward_weights:
        env_cfg.pmt_reward_weights = dict(reward_weights)
    env_cfg.__post_init__()
    return env_cfg


def _build_perceptive_motion_token_env(cfg):
    """PMT token-tracker PPO-pretrain env: stepping-stone base + height-scan
    ``vision`` group, so the env exposes both future_motion_window and height_scan
    obs groups. Same data-driven values as the stepping-stone env (plan §6 Phase 2.5)."""
    from pmt_tasks.env_cfgs.pmt_token.perceptive_motion_token import PMTPerceptiveMotionTokenTrackerEnvCfg

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTPerceptiveMotionTokenTrackerEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_history_length = int(cfg["obs"].get("history_length", 10))
    reward_weights = cfg.get("reward_weights")
    if reward_weights:
        env_cfg.pmt_reward_weights = dict(reward_weights)
    env_cfg.__post_init__()
    return env_cfg


def _build_add_env(cfg):
    from pmt_tasks.env_cfgs.add.add_multimotion import PMTADDMultiMotionEnvCfg

    motion = cfg["motion"]
    env_cfg = PMTADDMultiMotionEnvCfg()
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.__post_init__()
    return env_cfg


def _build_backflip_env(cfg):
    from pmt_tasks.env_cfgs.pmt.backflip import PMTBackFlipEnvCfg

    motion = cfg["motion"]
    terrain = cfg["terrain"]
    env_cfg = PMTBackFlipEnvCfg()
    # back_flip_merged clips are terrain-anchored -> run on the big_map mesh (§5 paths).
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    # §3a dt-from-motion: decimation/sim.dt come from the motion config (NOT defaults).
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.__post_init__()
    return env_cfg


def _build_multimotion_flat_env(cfg):
    """One env builder for the whole MultiMotion/Flat family (base/uniform/adaptive/
    streaming/bpo). Sampler + storage mode are config flags from motion/*.yaml — no
    env-class proliferation (plan §3/§9b)."""
    from pmt_tasks.env_cfgs.multi_motion_flat import PMTMultiMotionFlatEnvCfg

    motion = cfg["motion"]
    env_cfg = PMTMultiMotionFlatEnvCfg()
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_sampler_type = str(motion.get("sampler", "bin_adaptive"))
    env_cfg.pmt_storage_mode = str(motion.get("storage_mode", "eager"))
    # streaming-only knobs (ignored when storage_mode == "eager").
    env_cfg.pmt_max_working_set = int(motion.get("max_working_set", 0))
    env_cfg.pmt_num_load_workers = int(motion.get("num_load_workers", 16))
    env_cfg.pmt_use_process_pool = bool(motion.get("use_process_pool", False))
    env_cfg.__post_init__()
    return env_cfg


def _build_pcrbt_env(cfg):
    """P-CaRBT FSQ-behavior-tokenizer FLAT env (pmt_pcrbt task).

    Flat-plane base (PMTPCaRBTFlatEnvCfg) exposing the token/window obs groups but NO
    vision/height-scan group (reduced OBS; pmt_only_mode skips the terrain adapter).
    Same data-driven motion knobs as the multimotion-flat family."""
    from pmt_tasks.env_cfgs.pmt_pcrbt import PMTPCaRBTFlatEnvCfg

    motion = cfg["motion"]
    env_cfg = PMTPCaRBTFlatEnvCfg()
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_sampler_type = str(motion.get("sampler", "bin_adaptive"))
    env_cfg.pmt_storage_mode = str(motion.get("storage_mode", "eager"))
    env_cfg.pmt_max_working_set = int(motion.get("max_working_set", 0))
    env_cfg.pmt_num_load_workers = int(motion.get("num_load_workers", 16))
    env_cfg.pmt_use_process_pool = bool(motion.get("use_process_pool", False))
    env_cfg.__post_init__()
    return env_cfg


def _build_rgmt_env(cfg):
    """Paper-faithful RGMT deploy env for the `rgmt` task only."""
    from pmt_tasks.env_cfgs.rgmt.rgmt import RGMTG1EnvCfg

    motion = cfg["motion"]
    env_cfg = RGMTG1EnvCfg()
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_sampler_type = str(motion.get("sampler", "bin_adaptive"))
    env_cfg.pmt_storage_mode = str(motion.get("storage_mode", "eager"))
    env_cfg.pmt_max_working_set = int(motion.get("max_working_set", 0))
    env_cfg.pmt_num_load_workers = int(motion.get("num_load_workers", 16))
    env_cfg.pmt_use_process_pool = bool(motion.get("use_process_pool", False))
    env_cfg.pmt_history_length = int(cfg["obs"].get("history_length", 10))
    env_cfg.__post_init__()
    return env_cfg


def _build_pmt_adaptive_sampling_env(cfg):
    """Standalone PMT adaptive-sampling env with the hybrid streaming command."""
    from pmt_tasks.env_cfgs.pmt.adaptive_sampling import PMTAdaptiveSamplingEnvCfg

    motion = cfg["motion"]
    env_cfg = PMTAdaptiveSamplingEnvCfg()
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_sampler_type = str(motion.get("sampler", "hybrid"))
    env_cfg.pmt_storage_mode = str(motion.get("storage_mode", "hybrid"))
    env_cfg.pmt_max_working_set = int(motion.get("max_working_set", 0))
    env_cfg.pmt_num_load_workers = int(motion.get("num_load_workers", 16))
    env_cfg.pmt_use_process_pool = bool(motion.get("use_process_pool", False))
    env_cfg.pmt_history_length = int(cfg["obs"].get("history_length", 10))

    def _mset(attr, key, cast):
        if key in motion:
            setattr(env_cfg, attr, cast(motion.get(key)))

    _mset("pmt_offline_prior_path", "offline_prior_path", str)
    _mset("pmt_offline_prior_strength", "offline_prior_strength", float)
    _mset("pmt_hybrid_error_weight", "hybrid_error_weight", float)
    _mset("pmt_hybrid_failure_weight", "hybrid_failure_weight", float)
    _mset("pmt_hybrid_error_good", "hybrid_error_good", float)
    _mset("pmt_hybrid_error_bad", "hybrid_error_bad", float)
    _mset("pmt_hybrid_retention_ratio", "hybrid_retention_ratio", float)
    _mset("pmt_hybrid_topk_motion", "hybrid_topk_motion", int)
    _mset("pmt_hybrid_topk_motion_weight", "hybrid_topk_motion_weight", float)
    _mset("pmt_hybrid_retention_success_thresh", "hybrid_retention_success_thresh", float)
    _mset("pmt_global_age_ratio", "global_age_ratio", float)
    _mset("pmt_global_age_tau", "global_age_tau", float)
    _mset("pmt_hybrid_uncertainty_weight", "hybrid_uncertainty_weight", float)
    _mset("pmt_hybrid_uncertainty_gate_lo", "hybrid_uncertainty_gate_lo", float)
    _mset("pmt_hybrid_uncertainty_gate_hi", "hybrid_uncertainty_gate_hi", float)
    _mset("pmt_hybrid_uncertainty_norm", "hybrid_uncertainty_norm", float)
    _mset("pmt_hybrid_hard_buffer_ratio", "hybrid_hard_buffer_ratio", float)
    _mset("pmt_hybrid_hard_buffer_k", "hybrid_hard_buffer_k", int)
    env_cfg.__post_init__()
    return env_cfg


def _build_sonic_multimotion_flat_env(cfg):
    """SONIC flat multi-motion env selected by task-level ``sonic_mode``."""
    from pmt_tasks.env_cfgs.sonic.sonic_multimotion import PMTSonicMultiMotionFlatEnvCfg

    motion = cfg["motion"]
    env_cfg = PMTSonicMultiMotionFlatEnvCfg()
    env_cfg.pmt_sonic_mode = str(cfg.get("sonic_mode", "scratch"))
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_sampler_type = str(motion.get("sampler", "uniform"))
    env_cfg.pmt_storage_mode = str(motion.get("storage_mode", "eager"))
    env_cfg.__post_init__()
    return env_cfg


def _build_terrain_flat_mix_env(cfg):
    """Mixed terrain+flat transformer env via the FLEXIBLE UnifiedMotionCommandV2 (plan §9b).

    ONE store, ONE sampler, per-clip {origin offset + noise} — NOT the hard env partition
    (GroupedMultiMotionCommandV2). The two clip lists (terrain subset + flat subset) and
    the flat_origin come from motion/grouped_terrain_flat.yaml; decimation/dt §3a.
    """
    from pmt_tasks.env_cfgs.pmt.terrain_flat_mix import PMTTerrainFlatMixEnvCfg

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTTerrainFlatMixEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_terrain_motion_paths = _as_path_list(motion.get("terrain_motion_files"))
    env_cfg.pmt_flat_motion_paths = _as_path_list(motion.get("flat_motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    flat_origin = motion.get("flat_origin")
    if flat_origin is not None:
        env_cfg.pmt_flat_origin = list(flat_origin)
    env_cfg.__post_init__()
    return env_cfg


def _build_distill_stepping_stone_env(cfg):
    """Teacher/student distillation env on the stepping-stone terrain (paired clips)."""
    from pmt_tasks.env_cfgs.pmt.distill_stepping_stone import PMTSteppingStoneDistillEnvCfg

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTSteppingStoneDistillEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_optimized_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_raw_motion_paths = _as_path_list(motion.get("raw_motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.__post_init__()
    return env_cfg


def _build_distill_latent_anchor_env(cfg):
    """vision-transformer student / blind-transformer teacher latent-anchor distillation env.

    Same paired optimized/raw stepping-stone clips + terrain mesh as the simpler MLP
    distill env, but swaps in the vision-transformer teacher/student obs groups and the
    latent-anchor student-actor contract (PMTSteppingStoneVisionLatentAnchorDistillEnvCfg).
    """
    from pmt_tasks.env_cfgs.pmt.distill_stepping_stone import (
        PMTSteppingStoneVisionLatentAnchorDistillEnvCfg,
    )

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTSteppingStoneVisionLatentAnchorDistillEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_optimized_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_raw_motion_paths = _as_path_list(motion.get("raw_motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.__post_init__()
    return env_cfg


def _build_finetune_vision_teacher_env(cfg):
    """PPO-finetune env for the distilled vision-transformer student (P3).

    Single ``motion`` command (NOT paired) + height-scan vision group on the big_map
    mesh. Mirrors _build_distill_latent_anchor_env's data-driven injection but uses the
    SINGLE optimized clip dir (motion.motion_files) — there is no raw/student pairing in
    finetune.
    """
    from pmt_tasks.env_cfgs.pmt.finetune_stepping_stone import (
        PMTSteppingStoneVisionTeacherFinetuneEnvCfg,
    )

    terrain = cfg["terrain"]
    motion = cfg["motion"]
    env_cfg = PMTSteppingStoneVisionTeacherFinetuneEnvCfg()
    env_cfg.pmt_mesh_path = str(terrain["mesh_path"])
    env_cfg.pmt_motion_paths = _as_path_list(motion.get("motion_files"))
    env_cfg.pmt_decimation = int(motion.get("decimation", 4))
    env_cfg.pmt_sim_dt = float(motion.get("sim_dt", 0.005))
    env_cfg.pmt_history_length = int(cfg["obs"].get("history_length", 10))
    env_cfg.__post_init__()
    return env_cfg


_ENV_BUILDERS = {
    "pmt_stepping_stone": _build_stepping_stone_env,
    "walk_dance_bigmap": _build_stepping_stone_env,  # same transformer stack; big_map mesh + walk_dance clips
    "cartwheel_bigmap": _build_stepping_stone_env,  # same transformer stack; big_map mesh + cartwheel clips
    "perceptive_motion_token_tracker": _build_perceptive_motion_token_env,
    "pmt_pcrbt": _build_pcrbt_env,
    "pmt_pcrbt_100style": _build_pcrbt_env,  # same env; motion axis = 100style (1620 clips)
    "add_multimotion_flat": _build_add_env,
    "backflip": _build_backflip_env,
    "multimotionv2_flat": _build_multimotion_flat_env,
    "multimotionv2_uniform_flat": _build_multimotion_flat_env,
    "multimotionv2_adaptive_flat": _build_multimotion_flat_env,
    "multimotionv2_streaming_flat": _build_multimotion_flat_env,
    "multimotionv2_streaming_100style": _build_multimotion_flat_env,  # streaming swap on 1620 clips
    "multimotionv2_100style_flat": _build_multimotion_flat_env,  # same plane MLP path; 100style data
    "bpo_multimotionv2_flat": _build_multimotion_flat_env,
    "fpo_plus_flat": _build_multimotion_flat_env,
    "rgmt": _build_rgmt_env,
    "pmt_adaptive_sampling": _build_pmt_adaptive_sampling_env,
    "pmt_adaptive_sampling_baseline": _build_pmt_adaptive_sampling_env,  # A/B baseline: same env, sampling knobs zeroed
    "sonic_multimotion_flat": _build_sonic_multimotion_flat_env,
    "distill_stepping_stone": _build_distill_stepping_stone_env,
    "distill_stepping_stone_latent_anchor": _build_distill_latent_anchor_env,
    "ppofinetune_vision_teacher_stepping_stone_latent_anchor": _build_finetune_vision_teacher_env,
    "terrain_flat_mix": _build_terrain_flat_mix_env,
}


_NOT_DIRECT_TRAIN_REASONS = {
    "vision_ablation_base": (
        "it is a composition/ablation demonstration and pure-test target; runtime "
        "sweeps need network overrides plus a vision-teacher checkpoint and "
        "height-scan data."
    ),
}


def _unwired_task_message(stem: str, cfg_kind: str) -> str:
    wired = ", ".join(sorted(_ENV_BUILDERS))
    reason = _NOT_DIRECT_TRAIN_REASONS.get(stem)
    if reason is not None:
        return (
            f"task '{stem}' has no {cfg_kind} builder because it is not a "
            f"direct-train task: {reason} Launch one of the wired task stems "
            f"instead: {wired}."
        )
    return f"no {cfg_kind} builder wired for task '{stem}'. Wired task stems: {wired}."


def _as_path_list(motion_files):
    if motion_files is None:
        raise ValueError("motion config has no 'motion_files'")
    # OmegaConf ListConfig is not a (list, tuple) subclass; normalize it to a plain
    # list of strings so a list-valued path field (e.g. terrain/flat clip dirs) is not
    # stringified whole into a single bogus path.
    from omegaconf import ListConfig

    if isinstance(motion_files, (list, tuple, ListConfig)):
        return [str(p) for p in motion_files]
    return [str(motion_files)]


def _resolve_backend(cfg, backend: str | None) -> str:
    """Backend precedence: explicit arg > $PMT_BACKEND > cfg['backend']['name'] > 'isaaclab'.

    (MJLAB_BACKEND_PLAN.md Phase C — default isaaclab keeps every existing task unchanged.)
    """
    if backend is not None:
        return backend
    env = os.environ.get("PMT_BACKEND")
    if env in ("isaaclab", "mjlab"):
        return env
    node = cfg.get("backend") if hasattr(cfg, "get") else None
    if node is not None:
        name = node.get("name") if hasattr(node, "get") else node
        if name in ("isaaclab", "mjlab"):
            return name
    return "isaaclab"


def build_env_cfg(
    task_name_or_dict, profile: str | None = None, overrides=None, backend: str | None = None
):
    """Emit a concrete env cfg instance for the task (plan §10 PART A).

    Backend dispatch (Phase C): with backend='isaaclab' (default) this emits a PMT
    ``@configclass`` TrackingEnvCfg via ``_ENV_BUILDERS`` exactly as before. With
    backend='mjlab' it emits an mjlab ``ManagerBasedRlEnvCfg`` via
    ``pmt_tasks.backends.mjlab._MJLAB_ENV_BUILDERS`` (populating mjlab's tracking template
    from the same resolved config). Backend is chosen by the explicit arg, then
    ``$PMT_BACKEND``, then a ``backend:`` config axis, else isaaclab.

    Strategy (isaaclab path, hybrid-reuse-mdp-cfgs): the env STRUCTURE reuses the ported
    manager cfgs in ``pmt_tasks.env_cfgs.*``; the VALUES that vary — terrain mesh path,
    motion clip dir, decimation/sim.dt, reward weights, obs history_length — are injected
    from the resolved OmegaConf config.

    Returns a fresh cfg instance per call (§10/D).
    """
    cfg = build_task_config(task_name_or_dict, profile=profile, overrides=overrides)
    stem = _resolve_task_stem(task_name_or_dict)
    resolved_backend = _resolve_backend(cfg, backend)

    if resolved_backend == "mjlab":
        from pmt_tasks.backends.mjlab import _MJLAB_ENV_BUILDERS

        builder = _MJLAB_ENV_BUILDERS.get(stem)
        if builder is None:
            raise NotImplementedError(
                f"task '{stem}' has no mjlab env builder (wired: "
                f"{sorted(_MJLAB_ENV_BUILDERS)}). isaaclab remains the default backend."
            )
        return builder(cfg)

    builder = _ENV_BUILDERS.get(stem)
    if builder is None:
        raise NotImplementedError(_unwired_task_message(stem, "env"))
    return builder(cfg)


def build_agent_cfg(task_name_or_dict, profile: str | None = None, overrides=None):
    """Emit a concrete RslRlOnPolicyRunnerCfg instance for the task (plan §10 PART D).

    Fresh per call. The runner/policy/algorithm class_name fields are the
    registry/eval names the runner constructs from (TransformerActorCritic / PPO /
    OnPolicyRunner). obs_groups + experiment_name + hyperparams come from the ported
    agent cfg; the derived runner name and obs_groups from the resolved config are
    asserted to agree (fail-loud) so the agent cfg and the composition stay in sync.
    """
    cfg = build_task_config(task_name_or_dict, profile=profile, overrides=overrides)
    stem = _resolve_task_stem(task_name_or_dict)

    # Per-task agent-cfg dispatch (mirrors _ENV_BUILDERS). Each Phase-1 task pairs an
    # env cfg with a runner cfg. The derived runner axis is asserted to match the
    # runner class the cfg uses (fail-loud, so cfg + composition stay in sync).
    if stem == "rgmt":
        from pmt_tasks.agent_cfgs.rgmt import G1RGMTPPORunnerCfg

        agent_cfg = G1RGMTPPORunnerCfg()
        expected_runner = "on_policy"
    elif stem in ("pmt_adaptive_sampling", "pmt_adaptive_sampling_baseline"):
        from pmt_tasks.agent_cfgs.pmt_transformer import G1PMTAdaptiveSamplingPPORunnerCfg

        agent_cfg = G1PMTAdaptiveSamplingPPORunnerCfg()
        # Baseline keeps an identical agent/runner but a distinct experiment_name so its
        # logs/checkpoints land in a separate dir for the A/B comparison.
        if stem == "pmt_adaptive_sampling_baseline":
            agent_cfg.experiment_name = "pmt_adaptive_sampling_baseline"
        try:
            freq = int(cfg.get("motion_resample_frequency", 0))
        except Exception:
            freq = 0
        agent_cfg.motion_resample_frequency = freq
        expected_runner = "on_policy"
    elif stem in ("backflip", "pmt_stepping_stone", "walk_dance_bigmap", "cartwheel_bigmap", "terrain_flat_mix"):
        from pmt_tasks.agent_cfgs.transformer import G1SteppingStonePPORunnerCfg
        agent_cfg = G1SteppingStonePPORunnerCfg()
        expected_runner = "on_policy"
    elif stem == "perceptive_motion_token_tracker":
        from pmt_tasks.agent_cfgs.perceptive_motion_token import (
            G1PerceptiveMotionTokenTrackerPMTPretrainRunnerCfg,
        )
        agent_cfg = G1PerceptiveMotionTokenTrackerPMTPretrainRunnerCfg()
        expected_runner = "on_policy"
    elif stem in ("pmt_pcrbt", "pmt_pcrbt_100style"):
        from pmt_tasks.agent_cfgs.perceptive_residual_behavior_token import (
            G1PerceptiveResidualBehaviorTokenTrackerRunnerCfg,
        )
        agent_cfg = G1PerceptiveResidualBehaviorTokenTrackerRunnerCfg()
        if stem == "pmt_pcrbt_100style":
            agent_cfg.experiment_name = "g1_pcrbt_flat_100style"
        expected_runner = "on_policy"
    elif stem == "add_multimotion_flat":
        from pmt_tasks.agent_cfgs.add_ppo import G1AddFlatRunnerCfg
        agent_cfg = G1AddFlatRunnerCfg()
        expected_runner = "on_policy"
    elif stem in (
        "multimotionv2_flat",
        "multimotionv2_uniform_flat",
        "multimotionv2_adaptive_flat",
        "multimotionv2_streaming_flat",
        "multimotionv2_streaming_100style",
        "multimotionv2_100style_flat",
    ):
        from pmt_tasks.agent_cfgs.multi_motion_ppo import G1FlatMultiMotionPPORunnerCfg
        agent_cfg = G1FlatMultiMotionPPORunnerCfg()
        expected_runner = "on_policy"
    elif stem == "bpo_multimotionv2_flat":
        from pmt_tasks.agent_cfgs.multi_motion_ppo import G1FlatMultiMotionBPORunnerCfg
        agent_cfg = G1FlatMultiMotionBPORunnerCfg()
        expected_runner = "on_policy"
    elif stem == "fpo_plus_flat":
        from pmt_tasks.agent_cfgs.multi_motion_ppo import G1FlatSingleClipFpoPlusRunnerCfg
        agent_cfg = G1FlatSingleClipFpoPlusRunnerCfg()
        expected_runner = "on_policy"
    elif stem == "sonic_multimotion_flat":
        from pmt_tasks.agent_cfgs.sonic_ppo import make_g1_sonic_runner_cfg
        agent_cfg = make_g1_sonic_runner_cfg(str(cfg.get("sonic_mode", "scratch")))
        expected_runner = "on_policy"
    elif stem == "distill_stepping_stone":
        from pmt_tasks.agent_cfgs.distillation import G1SteppingStoneDistillRunnerCfg
        agent_cfg = G1SteppingStoneDistillRunnerCfg()
        expected_runner = "distillation"
    elif stem == "distill_stepping_stone_latent_anchor":
        from pmt_tasks.agent_cfgs.distillation import (
            G1SteppingStoneVisionLatentAnchorDistillRunnerCfg,
        )
        agent_cfg = G1SteppingStoneVisionLatentAnchorDistillRunnerCfg()
        expected_runner = "distillation"
        # Resolve the named teacher ckpt (task yaml: network.teacher_ckpt =
        # ${checkpoints.ss_teacher}) to the concrete model_*.pt path and inject it
        # into BOTH the frozen distillation target (policy.teacher_ckpt_path) and the
        # student warm-start backbone (student_cfg.base_policy_ckpt). The blind
        # TransformerActorCritic teacher is loaded from this file by the
        # VisionStudentTeacher wrapper.
        teacher_ckpt = cfg["network"].get("teacher_ckpt")
        if teacher_ckpt:
            agent_cfg.policy.teacher_ckpt_path = str(teacher_ckpt)
            # student_cfg is a per-instance dict (built in VisionStudentTeacherCfg
            # __post_init__); set the warm-start backbone ckpt too.
            agent_cfg.policy.student_cfg = dict(agent_cfg.policy.student_cfg)
            agent_cfg.policy.student_cfg["base_policy_ckpt"] = str(teacher_ckpt)
    elif stem == "ppofinetune_vision_teacher_stepping_stone_latent_anchor":
        from pmt_tasks.agent_cfgs.finetune import (
            G1SteppingStoneVisionTeacherFinetuneRunnerCfg,
        )

        agent_cfg = G1SteppingStoneVisionTeacherFinetuneRunnerCfg()
        expected_runner = "on_policy"
        # Resolve the distilled-student ckpt (task yaml: network.base_policy_ckpt =
        # ${checkpoints.distilled_student}) to the concrete model_*.pt path and inject it
        # into the standalone VisionTransformerActorCritic policy. The network's
        # _smart_load_checkpoint strips the "student." prefix from the distill
        # model_state_dict and loads the overlapping tensors (full transfer: the distilled
        # student already carries the vision encoder).
        base_ckpt = cfg["network"].get("base_policy_ckpt")
        if base_ckpt:
            agent_cfg.policy.base_policy_ckpt = str(base_ckpt)
    else:
        raise NotImplementedError(_unwired_task_message(stem, "agent"))

    derived_runner = cfg["_derived"]["runner"]
    if derived_runner != expected_runner:
        raise ValueError(
            f"build_agent_cfg for '{stem}' expected derived runner '{expected_runner}', "
            f"got '{derived_runner}'"
        )

    # apply optional CLI/max_iterations overrides surfaced by the task config.
    if "experiment_name" in cfg:
        agent_cfg.experiment_name = str(cfg["experiment_name"])
    return agent_cfg
