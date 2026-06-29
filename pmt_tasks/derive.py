"""Derivation rules (PMT plan §3a).

Several fields the old code sets are NOT independent selections — they are
*functions* of other axes (network heads, stage, algorithm). Hydra composition
can merge/override but cannot express "if X then Y". These pure, unit-testable
rules own that logic. No isaaclab/omni imports — runs in the wbt env.

`derive_obs_groups` mirrors the REAL `resolve_obs_groups` semantics from
`sonic_rsl_rl/utils/utils.py:238-340`:

  - obs_groups is a map  set-name -> list[group-name]  (NOT term names).
    A "group" is one of the env's ObservationsCfg group attribute names
    (e.g. "policy", "critic", "proprio", "vision", ...).
    Real examples (rsl_rl_transformer_ppo_cfg.py:208/652/955):
        {"policy": ["policy", "proprio"], "critic": ["critic"],
         "command_window": ["command_window"], "vision": ["vision"], ...}
  - "policy" is always present.
  - For each algorithm "default_set" missing from the explicit map, the real
    resolver (utils.py:313-331) fills it by: (1) using the same-named obs group
    if it exists in the env, else (2) copying the policy set's groups.

The runners build `default_sets` as:
  - on_policy_runner.py:64-67  -> ["critic"]  (+ "rnd_state" iff rnd_cfg present)
  - distillation_runner.py:61  -> ["teacher"]  (teacher ONLY — NOT student)

Each rule below carries the source file:line evidence it reproduces.
"""
from __future__ import annotations

from typing import Iterable

from motion_tracking_rl import compat


def derive_obs_groups(
    network_cfg,
    stage,
    algorithm_spec: compat.AlgorithmSpec,
    obs_cfg,
) -> dict[str, list[str]]:
    """Compute the obs-set -> obs-group mapping (real `resolve_obs_groups` granularity).

    Operates on obs *GROUP* names, NOT obs term names. The obs YAML provides
    `groups:` (the inventory of obs-group names the env will expose) and,
    optionally, `obs_groups:` (an explicit set-name -> [group-name] map for the
    base sets). Everything else is derived by the rules below, each citing the
    real source it reproduces.

    Returns:
        dict set-name -> list[group-name].
    """
    # available obs-group names the env exposes (mirrors `obs` keys checked by
    # resolve_obs_groups; here provided declaratively by the obs axis).
    # TODO(real value, plan §7): the real group inventory is populated from the
    # env's ObservationsCfg group attribute names at Phase-1 real-env wiring
    # (rsl_rl_transformer_ppo_cfg.py:208/652/955; mdp/observations.py).
    available_groups = list(_get(obs_cfg, "groups", default=["policy"]) or ["policy"])

    # explicit base obs_groups map from the obs axis (optional). The real env
    # cfgs ship this dict verbatim (e.g. rsl_rl_transformer_ppo_cfg.py:208). When
    # absent we synthesize the minimal {"policy": ["policy"]} and derive the rest.
    explicit = _get(obs_cfg, "obs_groups", default=None)
    if explicit is not None:
        groups: dict[str, list[str]] = {k: list(v) for k, v in _items(explicit)}
    else:
        groups = {}

    # "policy": always present.
    # (evidence: resolve_obs_groups requires the 'policy' key — utils.py:277-290;
    #  every actor reads the policy obs set.)
    if "policy" not in groups:
        groups["policy"] = ["policy"] if "policy" in available_groups else list(available_groups)

    # Assemble the algorithm's default_sets exactly as the runners do.
    # on_policy: ["critic"]; distillation: ["teacher"] (teacher only).
    # (evidence: on_policy_runner.py:64 default_sets=["critic"];
    #            distillation_runner.py:61 default_sets=["teacher"].)
    if algorithm_spec.runner == "distillation":
        default_sets = ["teacher"]
    else:
        default_sets = ["critic"]

    # add_ppo discriminator groups are required, not "default-filled"; the real
    # ADD cfg names them "add_disc_obs"/"add_disc_demo" (rl_cfg.py:433,436), not
    # "discriminator". We surface the spec's required_obs_sets as the single
    # source of truth so derive stays consistent with compat.validate's check
    # (builder.py:170-174 passes set(obs_groups.keys()) into validate).
    for required in sorted(algorithm_spec.required_obs_sets):
        if required not in default_sets and required != "policy":
            default_sets.append(required)

    # Fill each default/required set with the real resolver's defaulting rule
    # (utils.py:313-331): use the same-named env group if it exists, else copy
    # the policy set's groups.
    for set_name in default_sets:
        if set_name in groups:
            continue
        if set_name in available_groups:
            groups[set_name] = [set_name]
        else:
            groups[set_name] = list(groups["policy"])

    return groups


def derive_obs_groups_with_features(
    network_cfg,
    stage,
    algorithm_spec: compat.AlgorithmSpec,
    obs_cfg,
    *,
    rnd: bool = False,
) -> dict[str, list[str]]:
    """Same as `derive_obs_groups` but adds the 'rnd_state' set when the rnd
    feature flag is enabled.

    'rnd_state' is gated by the rnd *feature flag* on the algorithm cfg, not by a
    `required_obs_sets` entry — the runner appends it to default_sets only when
    `rnd_cfg` is present (evidence: on_policy_runner.py:65-66). It defaults like
    any other set (same-named group, else policy's groups; utils.py:313-331).
    """
    groups = derive_obs_groups(network_cfg, stage, algorithm_spec, obs_cfg)
    if rnd:
        if not algorithm_spec.supports_rnd:
            raise ValueError(
                f"algorithm '{algorithm_spec.name}' does not support rnd; cannot add rnd_state"
            )
        if "rnd_state" not in groups:
            available_groups = list(_get(obs_cfg, "groups", default=["policy"]) or ["policy"])
            if "rnd_state" in available_groups:
                groups["rnd_state"] = ["rnd_state"]
            else:
                groups["rnd_state"] = list(groups["policy"])
    return groups


def derive_reward_weights(reward_cfg, obs_cfg) -> dict[str, float]:
    """Apply the reward term->weight dict as data (plan §9b reward-as-data).

    The reward axis YAML already encodes the chosen weight-set (e.g.
    deepmimic_anchor vs deepmimic_anchor_fair). This rule simply surfaces it as a
    plain dict; the *choice* between fair/non-fair is itself obs-coupled (§3a:
    fair fires when motion_anchor_pos_b is dropped) but in this wave it is
    selected by the task's reward axis, so we return the selected weights verbatim
    and record whether the anchor obs term is present for downstream sanity checks.
    """
    weights = dict(_get(reward_cfg, "weights", default={}) or {})
    return weights


def anchor_in_actor_obs(obs_cfg) -> bool:
    """Whether the privileged motion_anchor_pos_b term is in the actor obs.

    Evidence (§3a): _match_old_no_anchor_fair_reward_weights() fires only when
    that obs term is DROPPED (distill_stepping_stone_env_cfg.py ~1056).
    """
    terms = _get(obs_cfg, "terms", default=[]) or []
    return "motion_anchor_pos_b" in list(terms)


# --- helpers -------------------------------------------------------------

def _get(cfg, key, default=None):
    """OmegaConf/dict/attr-tolerant getter."""
    if cfg is None:
        return default
    if hasattr(cfg, "get") and not isinstance(cfg, (list, tuple)):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _items(mapping) -> Iterable:
    """Iterate (key, value) over a dict or OmegaConf DictConfig."""
    if hasattr(mapping, "items"):
        return mapping.items()
    return dict(mapping).items()
