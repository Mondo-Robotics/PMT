"""Phase 0.5 — COMPATIBILITY-MATRIX GATE (PMT plan §3b, §6).

Proves the algorithm × network × feature × obs_sets coupling is fully handled by
`motion_tracking_rl.compat.validate` BEFORE any task is ported. Every invalid combo
must raise a clear build-time ValueError; every valid combo must return the spec's
runner. This protects the clean-break decision.

This phase exercises `compat.validate` ONLY — no Isaac Lab, no network instantiation.

Design rule (data-driven, no hand-typed per-cell expectations): for each
(algorithm, network, feature, obs_sets) cell the EXPECTED outcome is computed purely
from `compat.SPECS`, so this test cannot silently drift from the matrix it guards.
"""
from __future__ import annotations

import itertools

import pytest

from motion_tracking_rl import compat, registry


# ---------------------------------------------------------------------------
# Matrix axes (compat AXIS-name space, not registry class-name space).
# ---------------------------------------------------------------------------
ALGORITHMS = ["ppo", "bpo", "add_ppo", "fpo_plus", "distillation"]

# networks in compat-axis-name space
NETWORKS = [
    "mlp",
    "transformer",
    "vision_transformer",
    "sonic",
    "student_teacher",
    "vision_student_latent_anchor",
    "diffusion",
]

# each feature variation is a single-flag dict (plus the empty no-feature case)
FEATURES = [{}, {"rnd": True}, {"symmetry": True}, {"recurrent": True}]

# obs_sets variations:
#   - None  -> skip the required_obs_sets subset check entirely
#   - a representative provided set {"policy","critic"} that DOES satisfy ppo/bpo/fpo_plus
#     but is INSUFFICIENT for add_ppo (needs "discriminator") and distillation
#     (needs "teacher") -> exercises the required_obs_sets failure path.
OBS_SETS = [None, {"policy", "critic"}]


# ---------------------------------------------------------------------------
# compat-axis-name (network) -> registry NETWORKS class-name mapping.
# The compat.SPECS speak axis names ("transformer"); the registry speaks class
# names ("TransformerActorCritic"). configs/network/*.yaml carry the seam via
# (name == class-name, compat_name == axis-name). We reproduce the full seam here
# so STEP 3 can check each compatible_network has (or lacks) a backing class.
# ---------------------------------------------------------------------------
NETWORK_COMPAT_TO_CLASS = {
    "mlp": "ActorCritic",
    "transformer": "TransformerActorCritic",
    "vision_transformer": "VisionTransformerActorCritic",
    "sonic": "SonicActorCritic",
    "student_teacher": "StudentTeacher",
    "vision_student_latent_anchor": "VisionStudentTeacher",  # registered (Phase 2.2)
    # PerceptiveMotion family + vision ablation (registered Phase 2.5):
    "perceptive_motion_adapter": "PerceptiveMotionAdapter",
    "perceptive_motion_adapter_tracker": "PerceptiveMotionAdapterTracker",
    "perceptive_motion_token_tracker": "PerceptiveMotionTokenTracker",
    "vision_ablation": "VisionAblationActorCritic",
    # not yet @register_network'd:
    "diffusion": "DiffusionActorCritic",
}

# Networks referenced by SPECS whose backing class is a real module but is NOT yet
# decorated with @register_network (so it won't appear in registry.NETWORKS after
# autoload). Phase 2.2 cleared VisionStudentTeacher; diffusion is now registered.
# Asserted as a documented WARNING set, not a hard failure — but anything OUTSIDE
# (registered ∪ KNOWN_PENDING) IS a failure.
KNOWN_PENDING_NETWORKS = set()

# Runner axis names that are valid even if no class is registered under that exact
# name (the runner is a derived axis; classes register BOTH axis+class names).
KNOWN_RUNNER_AXES = {"on_policy", "distillation"}


@pytest.fixture(scope="module", autouse=True)
def _autoload():
    """Populate registries so STEP-3 self-consistency checks see real classes."""
    registry.autoload()


# ---------------------------------------------------------------------------
# Expectation derived PURELY from compat.SPECS (the single source of truth).
# ---------------------------------------------------------------------------
def _expected(spec: compat.AlgorithmSpec, network: str, feature: dict, obs_sets):
    """Return (should_raise: bool, expected_runner: str | None) for one cell,
    computed only from the spec. No per-cell hand-typing."""
    # 1. network must be in the spec's compatible set
    if network not in spec.compatible_networks:
        return True, None
    # 2. a set feature flag must be supported by the spec
    if feature.get("rnd") and not spec.supports_rnd:
        return True, None
    if feature.get("symmetry") and not spec.supports_symmetry:
        return True, None
    if feature.get("recurrent") and not spec.supports_recurrent:
        return True, None
    # 3. if obs_sets provided, required_obs_sets must be a subset
    if obs_sets is not None and not set(spec.required_obs_sets) <= set(obs_sets):
        return True, None
    return False, spec.runner


def _matrix_cells():
    for alg, net, feat, obs in itertools.product(
        ALGORITHMS, NETWORKS, FEATURES, OBS_SETS
    ):
        yield alg, net, feat, obs


@pytest.mark.parametrize(
    "algorithm,network,feature,obs_sets",
    list(_matrix_cells()),
    ids=[
        f"{a}-{n}-{('+'.join(f) or 'nofeat')}-{('obs' if o else 'noobs')}"
        for a, n, f, o in _matrix_cells()
    ],
)
def test_compat_matrix_cell(algorithm, network, feature, obs_sets):
    """Every (algorithm, network, feature, obs_sets) cell matches the SPEC-derived
    expectation: invalid combos raise ValueError; valid combos return spec.runner."""
    spec = compat.SPECS[algorithm]
    should_raise, expected_runner = _expected(spec, network, feature, obs_sets)

    if should_raise:
        with pytest.raises(ValueError):
            compat.validate(
                algorithm, network,
                rnd=feature.get("rnd", False),
                symmetry=feature.get("symmetry", False),
                recurrent=feature.get("recurrent", False),
                obs_sets=obs_sets,
            )
    else:
        runner = compat.validate(
            algorithm, network,
            rnd=feature.get("rnd", False),
            symmetry=feature.get("symmetry", False),
            recurrent=feature.get("recurrent", False),
            obs_sets=obs_sets,
        )
        assert runner == expected_runner == spec.runner


def test_matrix_covers_every_algorithm_in_specs():
    """The enumerated ALGORITHMS list must equal compat.SPECS keys (no drift)."""
    assert set(ALGORITHMS) == set(compat.SPECS), (
        f"matrix algorithms {sorted(ALGORITHMS)} != SPECS {sorted(compat.SPECS)}"
    )


def test_network_compat_map_covers_every_compatible_network():
    """Every network named in any spec.compatible_networks must have an entry in the
    compat-name->class map (otherwise STEP-3's backing-class check is incomplete)."""
    referenced = set().union(*(s.compatible_networks for s in compat.SPECS.values()))
    missing = referenced - set(NETWORK_COMPAT_TO_CLASS)
    assert not missing, f"compatible_networks with no compat->class mapping: {sorted(missing)}"


# ---------------------------------------------------------------------------
# STEP 3 — SELF-CONSISTENCY of the matrix itself (catch SPEC mistakes).
# ---------------------------------------------------------------------------
def test_every_spec_runner_is_registered_or_known_axis():
    """Each spec.runner must be a registered runner (registry.RUNNERS, post-autoload)
    OR a known runner axis name {"on_policy","distillation"}."""
    for alg, spec in compat.SPECS.items():
        assert spec.runner in registry.RUNNERS or spec.runner in KNOWN_RUNNER_AXES, (
            f"spec '{alg}'.runner='{spec.runner}' is neither a registered runner "
            f"{sorted(registry.RUNNERS)} nor a known axis {sorted(KNOWN_RUNNER_AXES)}"
        )


def test_compatible_networks_have_backing_class_or_known_pending(capsys):
    """Every network in any spec.compatible_networks must be either a registered
    network (via the compat-name->class seam) OR in KNOWN_PENDING_NETWORKS.

    Prints a WARNING list of compatible_networks that have NO backing registered
    class yet — direct input for Phase 1/2 registration work. Anything referenced
    that is neither registered nor known-pending is a hard failure.
    """
    referenced = set().union(*(s.compatible_networks for s in compat.SPECS.values()))
    pending_actual = set()
    unbacked_unknown = set()
    for net in sorted(referenced):
        cls_name = NETWORK_COMPAT_TO_CLASS.get(net)
        if cls_name in registry.NETWORKS:
            continue  # has a backing registered class
        if net in KNOWN_PENDING_NETWORKS:
            pending_actual.add(net)
        else:
            unbacked_unknown.add(net)

    # WARNING artifact for Phase 1/2 (printed, not a failure)
    print("\n[compat-matrix] compatible_networks with NO registered backing class yet "
          f"(KNOWN_PENDING, register in Phase 1/2): {sorted(pending_actual)}")

    assert not unbacked_unknown, (
        "compatible_networks referenced that are neither registered nor known-pending: "
        f"{sorted(unbacked_unknown)} (registered={sorted(registry.NETWORKS)})"
    )


def test_fpo_plus_rejects_rnd_symmetry_recurrent():
    """fpo_plus must reject rnd, symmetry, and recurrent.

    fpo_plus extends an internal FPO-style base, so it inherits these constraints.
    """
    spec = compat.SPECS["fpo_plus"]
    assert spec.supports_rnd is False
    assert spec.supports_symmetry is False
    assert spec.supports_recurrent is False
    # and validate() enforces it (diffusion is the only compatible network)
    for feat in ("rnd", "symmetry", "recurrent"):
        with pytest.raises(ValueError):
            compat.validate("fpo_plus", "diffusion", **{feat: True})


def test_add_ppo_rejects_recurrent():
    """add_ppo must reject recurrent policies.

    Grounded against the real source:
      sonic_rsl_rl/algorithms/add_ppo.py:97-98 raises
      NotImplementedError("Recurrent policies are not supported in ADDPPO v1.")
    (Phase 0.5 gate caught a spec that wrongly declared supports_recurrent=True.)
    """
    spec = compat.SPECS["add_ppo"]
    assert spec.supports_recurrent is False
    with pytest.raises(ValueError):
        compat.validate(
            "add_ppo", "transformer", recurrent=True,
            obs_sets={"policy", "critic", "discriminator"},
        )


def test_distillation_supports_recurrent():
    """distillation DOES support recurrent: distillation.py accepts StudentTeacherRecurrent
    (algorithms/distillation.py:24,28 policy union) with no recurrent rejection. This grounds
    the supports_recurrent=True flag flagged at the Phase 0.5 gate."""
    spec = compat.SPECS["distillation"]
    assert spec.supports_recurrent is True
    assert compat.validate(
        "distillation", "student_teacher", recurrent=True,
        obs_sets={"policy", "teacher"},
    ) == "distillation"


def test_distillation_runner_and_paired_command():
    """distillation must route to the distillation runner and require a paired command."""
    spec = compat.SPECS["distillation"]
    assert spec.runner == "distillation"
    assert spec.requires_paired_command is True
    # the other algorithms must NOT require a paired command
    for alg, s in compat.SPECS.items():
        if alg != "distillation":
            assert s.requires_paired_command is False, f"'{alg}' unexpectedly paired"


def test_unknown_algorithm_raises():
    """An algorithm not in SPECS raises a clear ValueError."""
    with pytest.raises(ValueError):
        compat.validate("nonexistent_alg", "mlp")
