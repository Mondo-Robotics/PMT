"""Unit tests for derive_obs_groups (PMT plan §3a).

These assertions validate the GROUNDED semantics of the real `resolve_obs_groups`
(`sonic_rsl_rl/utils/utils.py:238-340`), NOT derive.py's earlier invented
term-granularity behavior:

  * obs_groups is  set-name -> list[GROUP-name]  (group = an env ObservationsCfg
    group attr name like "policy"/"critic"/"vision"), not a flat list of obs terms.
  * "policy" is always present (utils.py:277-290).
  * algorithm default_sets: on_policy_runner.py:64 -> ["critic"];
    distillation_runner.py:61 -> ["teacher"] (teacher ONLY, no "student").
  * a missing default set is filled with the same-named env group if present,
    else a copy of the policy set's groups (utils.py:313-331).
  * "rnd_state" is appended only when the rnd feature flag is set
    (on_policy_runner.py:65-66).
"""
from omegaconf import OmegaConf

from motion_tracking_rl import compat
from pmt_tasks.derive import derive_obs_groups, derive_obs_groups_with_features

# obs axis: an obs-GROUP inventory + an explicit base set-map (mirrors the real
# env cfg dicts, e.g. rsl_rl_transformer_ppo_cfg.py:208).
OBS = OmegaConf.create(
    {
        "groups": ["policy", "critic", "teacher", "add_disc_obs", "add_disc_demo", "rnd_state"],
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "terms": ["joint_pos", "joint_vel", "last_action"],
    }
)
NET = OmegaConf.create({"name": "X"})
STAGE = OmegaConf.create({"lr": 1e-3})


def test_ppo_policy_and_critic():
    """ppo (on_policy) default_sets=["critic"] -> {policy, critic} group-maps.

    Evidence: on_policy_runner.py:64 default_sets=["critic"];
    explicit map ships policy/critic groups (rsl_rl_transformer_ppo_cfg.py:208).
    """
    groups = derive_obs_groups(NET, STAGE, compat.SPECS["ppo"], OBS)
    assert set(groups.keys()) == {"policy", "critic"}
    # values are GROUP names, not term names
    assert groups["policy"] == ["policy"]
    assert groups["critic"] == ["critic"]


def test_ppo_critic_defaults_to_same_named_group():
    """When 'critic' is not in the explicit map but a 'critic' env group exists,
    the resolver assigns ["critic"] (utils.py:316-317), not a copy of policy."""
    obs = OmegaConf.create(
        {"groups": ["policy", "critic"], "obs_groups": {"policy": ["policy", "proprio"]}}
    )
    groups = derive_obs_groups(NET, STAGE, compat.SPECS["ppo"], obs)
    assert groups["policy"] == ["policy", "proprio"]
    assert groups["critic"] == ["critic"]  # same-named group, not policy's copy


def test_critic_falls_back_to_policy_when_no_group():
    """If no 'critic' group exists in the inventory, critic copies the policy set's
    groups (utils.py:325)."""
    obs = OmegaConf.create(
        {"groups": ["policy", "proprio"], "obs_groups": {"policy": ["policy", "proprio"]}}
    )
    groups = derive_obs_groups(NET, STAGE, compat.SPECS["ppo"], obs)
    assert groups["critic"] == ["policy", "proprio"]


def test_add_ppo_adds_discriminator_sets():
    """add_ppo requires the discriminator obs set(s) from compat.required_obs_sets.

    The REAL ADD cfg names its discriminator groups 'add_disc_obs' /
    'add_disc_demo' (rl_cfg.py:433,436). derive surfaces whatever compat declares
    as required (single source of truth), so the produced keys must be a superset
    of compat.SPECS['add_ppo'].required_obs_sets.
    """
    spec = compat.SPECS["add_ppo"]
    groups = derive_obs_groups(NET, STAGE, spec, OBS)
    # policy + critic always; plus every required obs set the spec declares
    assert {"policy", "critic"} <= set(groups.keys())
    assert spec.required_obs_sets <= set(groups.keys())
    # group-name values, not term-name values
    for v in groups.values():
        assert all(isinstance(g, str) for g in v)


def test_distillation_adds_teacher_only_not_student():
    """distillation runner default_sets=["teacher"] — teacher ONLY, NOT student.

    Evidence: distillation_runner.py:61 `default_sets=["teacher"]` (the on_policy
    runner's student set is never added here). The 'teacher' env group exists in
    the inventory, so it resolves to ["teacher"] (utils.py:316-317).
    """
    groups = derive_obs_groups(NET, STAGE, compat.SPECS["distillation"], OBS)
    assert "policy" in groups
    assert "teacher" in groups
    assert "student" not in groups  # grounded: distillation default_sets is teacher-only
    assert groups["teacher"] == ["teacher"]


def test_rnd_flag_adds_rnd_state():
    """rnd_state is added only when the rnd feature flag is set.

    Evidence: on_policy_runner.py:65-66 appends 'rnd_state' to default_sets iff
    rnd_cfg is present. It resolves to the same-named group when available."""
    groups = derive_obs_groups_with_features(NET, STAGE, compat.SPECS["ppo"], OBS, rnd=True)
    assert "rnd_state" in groups
    assert groups["rnd_state"] == ["rnd_state"]  # same-named group present in inventory
    groups_no = derive_obs_groups_with_features(NET, STAGE, compat.SPECS["ppo"], OBS, rnd=False)
    assert "rnd_state" not in groups_no


def test_obs_groups_keys_satisfy_compat_required_sets():
    """The derived keys must satisfy compat.validate's required_obs_sets check
    (builder.py:170-174 passes set(obs_groups.keys()) into validate). Guards the
    derive<->compat contract for every spec."""
    for name, spec in compat.SPECS.items():
        groups = derive_obs_groups(NET, STAGE, spec, OBS)
        missing = set(spec.required_obs_sets) - set(groups.keys())
        assert not missing, f"{name}: derive missing required obs sets {missing}"
