"""PMT cluster-aware training entry (md_ai_kit / Ray multi-GPU).

A faithful PMT analog of the old whole_body_tracking
``scripts/rsl_rl/train_multi_motion.py`` distributed handling, but built on PMT's
config builder + registry instead of wandb/hydra:

  * env/agent cfgs come from PMT's gym factory closures
    (``load_cfg_from_registry`` on the registered PMT gym id), exactly like
    ``scripts/train.py``;
  * the runner CLASS is dispatched from ``agent_cfg.class_name`` via
    ``motion_tracking_rl.registry.get_runner`` (so distillation tasks route to the
    DistillationRunner);
  * distributed (RANK/WORLD_SIZE/LOCAL_RANK) handling, per-rank device/seed offset,
    rank-0-only logging, ISAACLAB_LOG_DIR honoring, and the end-of-run barrier +
    process-group teardown all mirror train_multi_motion.py lines ~160-340.

md_ai_kit launches this file with ``runpy.run_path`` (in-process on each Ray worker)
and auto-injects a fixed arg set (--task --log_dir --num_envs --distributed
--max_iterations --seed --experiment_name --run_name --resume --checkpoint
--load_task --load_run --headless --job_conf_path --task_index). We use
``parse_known_args`` so the injected args we do not model are tolerated, and read
ISAACLAB_LOG_DIR / RANK / WORLD_SIZE / LOCAL_RANK from the environment.

App-launch-first ordering (required): AppLauncher().app BEFORE importing isaaclab.envs.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# argparse — parse_known_args so md_ai_kit's extra injected flags are tolerated.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a PMT RL agent (cluster-aware).")
parser.add_argument("--task", type=str, required=True, help="PMT gym task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Per-GPU number of envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Base environment/agent seed.")
parser.add_argument(
    "--distributed", action="store_true", default=False,
    help="Enable multi-GPU training (RANK/WORLD_SIZE/LOCAL_RANK read from env).",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name override.")
parser.add_argument("--run_name", type=str, default=None, help="Run name override.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file regex/name (load_checkpoint).")
parser.add_argument("--load_task", type=str, default=None, help="Experiment dir to resume from (defaults to experiment_name).")
parser.add_argument("--load_run", type=str, default=None, help="Timestamped run dir to resume from.")
parser.add_argument("--profile", type=str, default=None, help="PMT paths profile (local|cluster). Default: $PMT_PROFILE.")

# AppLauncher adds --device (and --headless etc.); do NOT define --device ourselves
# or it raises "ArgParser already has the field 'device'".
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

# Default None so the YAML-set $PMT_PROFILE wins; only an explicit --profile overrides it.
if args_cli.profile is not None:
    os.environ["PMT_PROFILE"] = args_cli.profile

# md_ai_kit may inject hydra-style extras; clear sys.argv so nothing downstream re-parses.
sys.argv = [sys.argv[0]]

# Launch the Isaac Sim app FIRST (before importing isaaclab.envs).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from motion_tracking_rl import registry  # noqa: E402
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from pmt_tasks.registry_gym import register_pmt_tasks  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # --- debug: dump the REAL distributed env on each Ray worker (gated) ------
    if os.environ.get("PMT_DEBUG_DIST"):
        import torch.distributed as _dist

        print(
            "[PMT_DEBUG_DIST] "
            f"RANK={os.environ.get('RANK')} "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"device_count={torch.cuda.device_count()} "
            f"dist_initialized={_dist.is_initialized() if _dist.is_available() else 'NA'} "
            f"args_device={args_cli.device} "
            f"distributed_flag={args_cli.distributed}",
            flush=True,
        )

    registry.autoload()
    register_pmt_tasks()

    # Resolve env + agent cfgs through the registered PMT factory closures (§10).
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # --- non-distributed CLI overrides -------------------------------------
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run

    # Optional: streaming working-set swap cadence (only meaningful for the streaming
    # command). Setting it >0 makes the runner actually swap the resident clip set every
    # N PPO updates — needed to EXERCISE the streaming path (not just build it). Read from
    # env so we don't edit the shared runner cfg; inert (0) for non-streaming tasks.
    _resample_freq = os.environ.get("PMT_MOTION_RESAMPLE_FREQ")
    if _resample_freq is not None:
        agent_cfg.motion_resample_frequency = int(_resample_freq)
        print(f"[INFO] motion_resample_frequency set to {_resample_freq} (streaming swap).")

    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # --- distributed setup (mirror train_multi_motion.py:160-186) ----------
    rank = 0
    world_size = 1
    local_rank = 0
    if args_cli.distributed:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        print(f"[INFO] Distributed training: Rank {rank}/{world_size} (local rank {local_rank})")

        # Offset the seed per worker for distinct initialization.
        if env_cfg.seed is not None:
            env_cfg.seed = env_cfg.seed + rank
            agent_cfg.seed = agent_cfg.seed + rank
            print(f"[INFO] Worker {rank} using seed: {env_cfg.seed}")

        # Pin this worker to its local GPU.
        if torch.cuda.is_available():
            device_index = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_index)
            env_cfg.sim.device = f"cuda:{device_index}"
            print(f"[INFO] Worker {rank} using device: {env_cfg.sim.device}")
        else:
            env_cfg.sim.device = "cpu"

    # Keep the agent device aligned with the sim device.
    agent_cfg.device = env_cfg.sim.device

    # --- log dir (honor Ray-specified ISAACLAB_LOG_DIR) --------------------
    experiment_name = agent_cfg.experiment_name
    if "ISAACLAB_LOG_DIR" in os.environ:
        log_root_path = os.environ["ISAACLAB_LOG_DIR"]
        log_dir = log_root_path
        print(f"[INFO] Using Ray-specified log directory: {log_dir}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # --- build env ---------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # --- dispatch the runner CLASS from the agent cfg's class_name (§3b/§10) -
    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Only rank 0 logs/saves in distributed mode.
    if args_cli.distributed and rank != 0:
        runner.logger_type = None
        print(f"[INFO] Worker {rank}: logging disabled (only rank 0 logs)")

    runner.add_git_repo_to_log(__file__)

    # --- (smoke tooling) dump a shape-matching teacher stub then exit -------
    # When $PMT_DUMP_TEACHER_STUB is set, save the freshly-constructed teacher MLP's
    # state_dict (with the "actor." prefix StudentTeacher._load_teacher_from_checkpoint
    # expects) as a stub ckpt and exit BEFORE learn(). This lets a distillation smoke
    # validate the DistillationRunner path on the cluster without a real trained
    # teacher (the runner guards on policy.loaded_teacher and would otherwise refuse
    # to run). The stub weights are random -> distillation loss is meaningless.
    _stub_path = os.environ.get("PMT_DUMP_TEACHER_STUB")
    if _stub_path:
        teacher = runner.alg.policy.teacher
        stub = {f"actor.{k}": v.cpu() for k, v in teacher.state_dict().items()}
        os.makedirs(os.path.dirname(os.path.abspath(_stub_path)), exist_ok=True)
        torch.save({"model_state_dict": stub}, _stub_path)
        print(f"[PMT_STUB] wrote teacher stub ({len(stub)} tensors) to {_stub_path}", flush=True)
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        return

    # --- resume (mirror train_multi_motion.py:281-306) ---------------------
    if agent_cfg.resume:
        # In cluster mode ISAACLAB_LOG_DIR points at the NEW run dir, so the old run
        # lives in its parent; locally log_root_path is already the experiment dir.
        if "ISAACLAB_LOG_DIR" in os.environ:
            resume_base_dir = os.path.dirname(log_root_path.rstrip(os.sep))
        else:
            resume_base_dir = log_root_path
        # --load_task lets a job resume from a DIFFERENT experiment's dir.
        if args_cli.load_task is not None:
            resume_base_dir = os.path.join(os.path.dirname(resume_base_dir.rstrip(os.sep)), args_cli.load_task)
        resume_path = get_checkpoint_path(resume_base_dir, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    # --- train -------------------------------------------------------------
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # --- distributed barrier + teardown (mirror train_multi_motion.py:323-338)
    if args_cli.distributed:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            print(f"[INFO] Worker {rank}: training complete, synchronized.")

    # Training + checkpointing are DONE and all ranks are synced at the barrier above.
    # md_ai_kit runs this script as the Ray Train v2 *worker training function*, inside a
    # worker thread that Ray's controller polls over RPC. Ray records SUCCESS only when
    # that function RETURNS normally — so we must return here, NOT call os._exit(0).
    # Calling os._exit(0) from this thread kills the whole worker process before the
    # controller can poll the finished status, which Ray surfaces as a dead actor
    # ("connection error code 2 / End of file") and the job is reported FAILED even though
    # training fully succeeded and every checkpoint was written.
    #
    # Isaac Sim's native physx/CUDA teardown SEGFAULTS on Ray workers at process exit (a
    # hard C++ crash, uncatchable by try/except). We bypass it at TRUE process exit via the
    # atexit os._exit(0) hook registered in the __main__ block below — that fires only when
    # the worker process is actually shutting down, after Ray has already recorded success.
    print(f"[INFO] Worker {rank}: TRAINING_COMPLETE — checkpoints saved; returning so Ray "
          f"reports SUCCESS (Isaac teardown bypassed via atexit).", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return


if __name__ == "__main__":
    # Bypass Isaac Sim's segfaulting native teardown at process exit WITHOUT killing the
    # Ray worker thread early: register os._exit(0) as an atexit hook. It runs only when
    # the interpreter is already shutting down (after main() has returned and Ray has
    # recorded the run), and because atexit is LIFO it fires before Isaac's own teardown
    # hooks (registered at AppLauncher import time), so the crashy C++ teardown never runs.
    # On a hard ray.kill (SIGKILL) nothing runs and there is no segfault either way.
    import atexit

    atexit.register(lambda: os._exit(0))
    main()
