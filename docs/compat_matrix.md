# PMT Compatibility Matrix (Phase 0.5 artifact)

Generated from `motion_tracking_rl/compat.py` SPECS via `compat.validate`.
Rows = algorithm (compat axis name). Cols = network (compat axis name).
Cell = derived **runner** name for the valid `(algorithm, network)` coupling, or **X** if the network is not in that algorithm's `compatible_networks`.

| algorithm \ network | mlp | transformer | vision_transformer | sonic | student_teacher | vision_student_latent_anchor | diffusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ppo** | on_policy | on_policy | on_policy | on_policy | X | X | X |
| **bpo** | on_policy | on_policy | on_policy | on_policy | X | X | X |
| **add_ppo** | on_policy | on_policy | X | X | X | X | X |
| **fpo** | X | X | X | X | X | X | on_policy |
| **fpo_plus** | X | X | X | X | X | X | on_policy |
| **distillation** | X | X | X | X | distillation | distillation | X |

## Feature support (per algorithm)

| algorithm | runner | rnd | symmetry | recurrent | requires_paired_command | required_obs_sets |
| --- | --- | --- | --- | --- | --- | --- |
| ppo | on_policy | True | True | True | False | {critic, policy} |
| bpo | on_policy | True | True | True | False | {critic, policy} |
| add_ppo | on_policy | False | False | False | False | {add_disc_demo, add_disc_obs, critic, policy} |
| fpo | on_policy | False | False | False | False | {critic, policy} |
| fpo_plus | on_policy | False | False | False | False | {critic, policy} |
| distillation | distillation | False | False | True | True | {policy, teacher} |

## Network backing-class status

| network (compat) | class name | registered? |
| --- | --- | --- |
| diffusion | DiffusionActorCritic | pending (Phase 1/2) |
| mlp | ActorCritic | yes |
| perceptive_motion_adapter | PerceptiveMotionAdapter | yes |
| perceptive_motion_adapter_tracker | PerceptiveMotionAdapterTracker | yes |
| perceptive_motion_token_tracker | PerceptiveMotionTokenTracker | yes |
| vision_ablation | VisionAblationActorCritic, VisionAblationRecurrentActorCritic | yes |
| sonic | SonicActorCritic | yes |
| student_teacher | StudentTeacher | yes |
| transformer | TransformerActorCritic | yes |
| vision_student_latent_anchor | VisionStudentTeacher | yes |
| vision_transformer | VisionTransformerActorCritic | yes |

## Notes

- A valid `(algorithm, network)` coupling can still fail to build if a feature flag (rnd/symmetry/recurrent) is set that the algorithm does not support, or if the provided `obs_sets` omit a `required_obs_sets` member.
- `fpo`/`fpo_plus` reject rnd/symmetry/recurrent (source: `sonic_rsl_rl/algorithms/fpo.py:77,79,84`).
- `distillation` routes to the `distillation` runner and requires a paired (teacher/student) command.
- `pending` networks have a real class but are not yet `@register_network`'d.
