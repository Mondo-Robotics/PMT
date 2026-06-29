"""Generate the human-readable compatibility-matrix artifact (PMT plan §6).

Writes docs/compat_matrix.md: an algorithm-rows × network-cols table where each cell
is the runner name (valid combo) or "X" (invalid). Validity is computed purely from
compat.SPECS via compat.validate (no feature flags, obs_sets=None — i.e. the bare
algorithm×network coupling). A footnote records the feature/obs-set and backing-class
constraints that the matrix cells alone do not show.

Run: PYTHONPATH=<repo> python tests/gen_compat_matrix_doc.py
"""
from __future__ import annotations

from pathlib import Path

from motion_tracking_rl import compat, registry

ALGORITHMS = ["ppo", "bpo", "add_ppo", "fpo_plus", "distillation"]
NETWORKS = [
    "mlp",
    "transformer",
    "vision_transformer",
    "sonic",
    "student_teacher",
    "vision_student_latent_anchor",
    "diffusion",
]
NETWORK_COMPAT_TO_CLASS = {
    "mlp": "ActorCritic",
    "transformer": "TransformerActorCritic",
    "vision_transformer": "VisionTransformerActorCritic",
    "sonic": "SonicActorCritic",
    "student_teacher": "StudentTeacher",
    "vision_student_latent_anchor": "VisionStudentTeacher",
    "diffusion": "DiffusionActorCritic",
}
KNOWN_PENDING_NETWORKS = {"diffusion"}  # VisionStudentTeacher registered (Phase 2.2)


def _cell(alg: str, net: str) -> str:
    try:
        return compat.validate(alg, net)  # bare coupling; runner name on success
    except ValueError:
        return "X"


def build_markdown() -> str:
    registry.autoload()

    # header
    cols = " | ".join(NETWORKS)
    sep = " | ".join(["---"] * (len(NETWORKS) + 1))
    lines = [
        "# PMT Compatibility Matrix (Phase 0.5 artifact)",
        "",
        "Generated from `motion_tracking_rl/compat.py` SPECS via `compat.validate`.",
        "Rows = algorithm (compat axis name). Cols = network (compat axis name).",
        "Cell = derived **runner** name for the valid `(algorithm, network)` coupling, "
        "or **X** if the network is not in that algorithm's `compatible_networks`.",
        "",
        f"| algorithm \\ network | {cols} |",
        f"| {sep} |",
    ]
    for alg in ALGORITHMS:
        row = " | ".join(_cell(alg, net) for net in NETWORKS)
        lines.append(f"| **{alg}** | {row} |")

    # feature-support sub-table (cells alone don't encode this)
    lines += [
        "",
        "## Feature support (per algorithm)",
        "",
        "| algorithm | runner | rnd | symmetry | recurrent | requires_paired_command | required_obs_sets |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for alg in ALGORITHMS:
        s = compat.SPECS[alg]
        lines.append(
            f"| {alg} | {s.runner} | {s.supports_rnd} | {s.supports_symmetry} | "
            f"{s.supports_recurrent} | {s.requires_paired_command} | "
            f"{{{', '.join(sorted(s.required_obs_sets))}}} |"
        )

    # backing-class status (Phase 1/2 input)
    referenced = sorted(set().union(*(s.compatible_networks for s in compat.SPECS.values())))
    lines += [
        "",
        "## Network backing-class status",
        "",
        "| network (compat) | class name | registered? |",
        "| --- | --- | --- |",
    ]
    # Derive the compat-name -> backing class(es) map from the LIVE registry so the
    # doc can never drift from what is actually @register_network'd (Phase 3 fix:
    # the hand-kept NETWORK_COMPAT_TO_CLASS missed the perceptive_motion/ablation
    # classes registered in Phase 2.5).
    compat_to_classes: dict[str, list[str]] = {}
    for cls_name, compat_name in registry.NETWORK_COMPAT.items():
        compat_to_classes.setdefault(compat_name, []).append(cls_name)
    for net in referenced:
        registered_classes = sorted(compat_to_classes.get(net, []))
        if registered_classes:
            cls = ", ".join(registered_classes)
            status = "yes"
        else:
            cls = NETWORK_COMPAT_TO_CLASS.get(net, "?")
            status = "pending (Phase 1/2)" if net in KNOWN_PENDING_NETWORKS else "MISSING"
        lines.append(f"| {net} | {cls} | {status} |")

    lines += [
        "",
        "## Notes",
        "",
        "- A valid `(algorithm, network)` coupling can still fail to build if a "
        "feature flag (rnd/symmetry/recurrent) is set that the algorithm does not "
        "support, or if the provided `obs_sets` omit a `required_obs_sets` member.",
        "- `fpo_plus` rejects rnd/symmetry/recurrent "
        "(source: `motion_tracking_rl/algorithms/fpo.py`).",
        "- `distillation` routes to the `distillation` runner and requires a paired "
        "(teacher/student) command.",
        "- `pending` networks have a real class but are not yet `@register_network`'d.",
    ]
    return "\n".join(lines) + "\n"


def main():
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    out = docs_dir / "compat_matrix.md"
    md = build_markdown()
    out.write_text(md)
    print(f"wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
