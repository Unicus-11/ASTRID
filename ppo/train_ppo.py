"""
ppo/train_ppo.py  (PATCHED v2 -- adds --resume)
===================
This version adds ONLY the resume-from-checkpoint feature on top of the
existing seed-determinism fix (unchanged, see ValidationCallback / the
_deterministic_seed usage below -- none of that logic was touched).

Why the plain command would have restarted from zero
------------------------------------------------------
`PPO("MlpPolicy", train_env, ...)` always constructs a FRESH network
with randomly-initialized weights. Nothing about re-running
`train_ppo.py --total-timesteps 100000 ...` ever looked at
`best_model/model.zip` -- so a second run was always a brand new
100k-step run from scratch, unrelated to the first one.

Why model.zip lets us continue
------------------------------
`model.save(...)` in Stable-Baselines3 serializes the full policy
network weights AND the training hyperparameters (learning rate
schedule, gamma, clip_range, etc.) into the zip. `PPO.load(path,
env=...)` reconstructs a PPO object from that state -- same weights,
same configured hyperparameters -- and re-attaches it to a (new) env
instance. Continuing `.learn()` on that loaded object genuinely
continues from the trained policy; it is not "loading weights into a
fresh model" in any hand-rolled sense.

Subtle SB3 timestep-accounting issue
-------------------------------------
`model.learn(total_timesteps=X, reset_num_timesteps=False)` does NOT
train until `model.num_timesteps == X`. Internally, SB3's
`_setup_learn()` does:

    if reset_num_timesteps:
        self.num_timesteps = 0
    else:
        total_timesteps += self.num_timesteps   # <-- adds already-done steps

So when resuming, `total_timesteps` passed to `.learn()` must be the
REMAINING steps (target - already_done), not the grand total -- SB3
adds the already-done count back in itself. Passing the grand total
directly (e.g. 100_000 when 50_000 is already done) would silently
train for 150_000 total steps, not 100_000. This file computes the
remaining amount explicitly and passes that.

A second, easy-to-miss issue: `ValidationCallback` starts with
`best_score = -inf` and `_last_eval_step = 0`. On a fresh run that's
correct. On a resumed run, if left as-is, the very next validation
call (at `num_timesteps` already >= eval_every_timesteps) would be
compared against `-inf` and treated as "the new best" even if it
scores WORSE than the already-saved 50k checkpoint -- silently
overwriting a better model with a worse one. This file seeds
`best_score` and `_last_eval_step` from the resumed checkpoint's own
saved `best_model/validation_metrics.json` and `validation_history.csv`
so the callback picks up exactly where it left off.

Crash/restart safety
---------------------
Previously, validation_history.json was only written ONCE, at
_on_training_end() -- if the process was killed (crash, VS Code
restart, power loss) mid-run, that write never happened and the
ENTIRE run's history was lost, not just the last few steps. Likewise
final_model.zip was only written once, at a clean finish -- after a
crash there was no "latest" checkpoint at all to resume from, only
whichever best_model happened to be last time score improved (which
could be far behind).

Both are now fixed:
  - validation_history.csv is appended to (with an immediate flush +
    fsync) after EVERY validation call, not just at the end. A crash
    loses at most the last eval_every_timesteps of training, never the
    whole run's record.
  - out_dir/latest_model/model.zip is saved after EVERY validation
    call too, regardless of whether it was a new best. This -- not
    final_model.zip -- is the checkpoint to --resume from after a
    crash, since it's always the most recent completed validation
    point rather than "whatever the best score happened to be" or
    "nothing, because training never finished cleanly."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import ppo_config as cfg
from ppo_env import ASTRIDSignalEnv


CSV_FIELDNAMES = [
    "timesteps", "score", "avg_queue_m", "avg_waiting_s",
    "avg_speed_mps", "avg_throughput", "collisions", "teleports",
]


def _deterministic_seed(scenario_id: str, base_seed: int) -> int:
    """Stable (process-independent) seed derived from scenario_id +
    base_seed. Do NOT use Python's built-in hash() here -- it is
    randomized per process via PYTHONHASHSEED and would silently break
    reproducibility across runs."""
    return zlib.crc32(f"{scenario_id}_{base_seed}".encode("utf-8")) % (2**31 - 1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO for ASTRID signal control.")
    p.add_argument("--total-timesteps", type=int, default=None,
                    help="TOTAL timesteps the model should have after this run finishes. "
                         "When --resume is used, this is the grand total (e.g. 100000), "
                         "NOT additional steps on top of the checkpoint.")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--warmup-seconds", type=int, default=None)
    p.add_argument("--episode-seconds", type=int, default=None,
                    help="Absolute SUMO simulation end time (must match your sumocfg's <end>).")
    p.add_argument("--eval-every-timesteps", type=int, default=20_000)
    p.add_argument("--sumo-binary", type=str, default=None, help="'sumo' or 'sumo-gui'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default="ppo_astrid_v1")
    p.add_argument("--model-out-dir", type=str, default=None,
                    help="Override the base output directory (default: ppo_models/, from "
                         "ppo_config.PPORunConfig.model_out_dir). Artifacts land in "
                         "<model-out-dir>/<run-name>/, same structure either way. Use this "
                         "to point a fresh training run somewhere entirely separate from "
                         "an existing run's artifacts, e.g. --model-out-dir ppo_models_again "
                         "so it never touches ppo_models/ppo_astrid_v1/.")
    p.add_argument("--allow-fresh-overwrite", action="store_true",
                    help="Required to start a brand-new (non-resumed) model under a "
                         "--run-name that already has a saved best_model/. Without this, "
                         "the script refuses to run rather than risk silently conflating "
                         "an unrelated fresh attempt with prior training history.")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a saved PPO model.zip to resume training from. For "
                         "resuming after a crash/restart, use "
                         "<out_dir>/latest_model/model.zip (updated after EVERY validation, "
                         "not just improvements or a clean finish -- this loses the least "
                         "progress). Use <out_dir>/best_model/model.zip if you specifically "
                         "want to roll back to the best-scoring checkpoint instead of "
                         "continuing the raw training trajectory. When set, --total-timesteps "
                         "must be the GRAND TOTAL you want the model to reach, and must be "
                         "greater than the checkpoint's own num_timesteps. NOTE: "
                         "hyperparameter override flags above (--learning-rate, --gamma, "
                         "etc.) are NOT re-applied to a resumed model -- PPO.load() restores "
                         "the hyperparameters that were saved with the checkpoint. Only "
                         "--total-timesteps, --eval-every-timesteps, and --seed (which only "
                         "affects the env's own RNG, not the loaded policy) take effect on "
                         "resume.")
    return p


def apply_overrides(run_cfg: cfg.PPORunConfig, args: argparse.Namespace) -> None:
    hp = run_cfg.hyperparams
    if args.total_timesteps is not None:
        hp.total_timesteps = args.total_timesteps
    if args.learning_rate is not None:
        hp.learning_rate = args.learning_rate
    if args.n_steps is not None:
        hp.n_steps = args.n_steps
    if args.batch_size is not None:
        hp.batch_size = args.batch_size
    if args.n_epochs is not None:
        hp.n_epochs = args.n_epochs
    if args.gamma is not None:
        hp.gamma = args.gamma
    if args.gae_lambda is not None:
        hp.gae_lambda = args.gae_lambda
    if args.clip_range is not None:
        hp.clip_range = args.clip_range
    if args.ent_coef is not None:
        hp.ent_coef = args.ent_coef
    if args.vf_coef is not None:
        hp.vf_coef = args.vf_coef
    if args.max_grad_norm is not None:
        hp.max_grad_norm = args.max_grad_norm
    if args.seed is not None:
        hp.seed = args.seed
    if args.warmup_seconds is not None:
        run_cfg.warmup_seconds = args.warmup_seconds
    if args.episode_seconds is not None:
        run_cfg.episode_seconds = args.episode_seconds
    if args.sumo_binary is not None:
        run_cfg.sumo_binary = args.sumo_binary
    if args.model_out_dir is not None:
        run_cfg.model_out_dir = Path(args.model_out_dir)


def composite_score(metrics: Dict[str, float]) -> float:
    """HIGHER is better. Queue first, then delay, then speed/throughput
    as tie-breakers. Kept separate from the PPO training reward."""
    return (
        -1.00 * metrics["avg_queue_m"]
        - 0.50 * metrics["avg_waiting_s"]
        + 0.10 * metrics["avg_speed_mps"]
        + 0.05 * metrics["avg_throughput"]
    )


def run_deterministic_episode(
    env: ASTRIDSignalEnv,
    model: PPO,
    scenario_id: str,
    sumo_seed: int,
    estimator_seed: int,
) -> Dict[str, float]:
    obs, _ = env.reset(
        options={
            "scenario_id": scenario_id,
            "sumo_seed": sumo_seed,
            "estimator_seed": estimator_seed,
        }
    )
    done = False
    queues, waitings, speeds, arrived = [], [], [], 0
    collisions, teleports = 0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        queues.append(info["avg_queue_m"])
        waitings.append(info["avg_waiting_s"])
        speeds.append(info["avg_speed_mps"])
        arrived += info["arrived_this_interval"]
        collisions += info["collisions"]
        teleports += info["teleports"]
    return {
        "avg_queue_m": float(np.mean(queues)),
        "avg_waiting_s": float(np.mean(waitings)),
        "avg_speed_mps": float(np.mean(speeds)),
        "avg_throughput": float(arrived),
        "switch_count": env._controller.switch_count,
        "collisions": collisions,
        "teleports": teleports,
    }


class ValidationCallback(BaseCallback):
    """Every `eval_every_timesteps`, runs exactly one deterministic
    episode per validation scenario, on a FIXED SUMO seed and FIXED
    sensor-noise seed per scenario (same seeds every single call), so
    checkpoints are compared on identical traffic. Averages the
    resulting traffic metrics across scenarios, scores them with
    composite_score(), and saves the model whenever that score improves.

    CRASH SAFETY: two things are written on EVERY validation call, not
    just at clean training end:
      1. A row appended to validation_history.csv (flushed + fsynced
         immediately) -- so a crash mid-run loses at most the LAST
         eval_every_timesteps worth of training, not the entire run's
         history. validation_history.json is still written too, at
         _on_training_end, purely for convenience -- the CSV is the
         durable source of truth resume reads from.
      2. The full model state saved to out_dir/latest_model/model.zip,
         REGARDLESS of whether this validation was a new best. This is
         the checkpoint to --resume from after a crash/restart: unlike
         best_model/ (only updated when score improves -- could be far
         behind) or final_model.zip (only written once, at a clean
         finish -- doesn't exist at all after a crash), latest_model/
         always reflects the most recent completed validation point.

    RESUME SUPPORT: initial_best_score / initial_last_eval_step /
    initial_history let a resumed run pick up exactly where a previous
    run left off, instead of starting from best_score=-inf (which would
    silently treat the first post-resume validation as "the new best"
    even if it's worse than what was already saved)."""

    def __init__(
        self,
        run_cfg: cfg.PPORunConfig,
        out_dir: Path,
        eval_every_timesteps: int,
        verbose: int = 1,
        initial_best_score: float = -np.inf,
        initial_last_eval_step: int = 0,
        initial_history: Optional[List[dict]] = None,
    ):
        super().__init__(verbose)
        self.run_cfg = run_cfg
        self.out_dir = out_dir
        self.eval_every_timesteps = eval_every_timesteps
        self.best_score = initial_best_score
        self._last_eval_step = initial_last_eval_step
        # seed=None here: we always pass explicit sumo_seed/estimator_seed
        # per call, so this env's own np_random is irrelevant.
        self._eval_env = ASTRIDSignalEnv(run_cfg, run_cfg.validation_scenarios, seed=run_cfg.hyperparams.seed + 1)
        self.history: List[dict] = list(initial_history) if initial_history else []

        self.csv_path = out_dir / "validation_history.csv"
        if not self.csv_path.is_file():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    def _append_csv_row(self, row: dict) -> None:
        """Appends one row and forces it to disk immediately (flush + fsync),
        so it survives a hard crash right after this call returns -- not just
        a normal process exit. This is the whole point of writing per-row
        instead of dumping the full history once at the end."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_every_timesteps:
            return True
        self._last_eval_step = self.num_timesteps

        base_seed = self.run_cfg.hyperparams.seed
        per_scenario = [
            run_deterministic_episode(
                self._eval_env,
                self.model,
                s,
                sumo_seed=_deterministic_seed(s, base_seed),
                estimator_seed=_deterministic_seed(s, base_seed + 1),
            )
            for s in self.run_cfg.validation_scenarios
        ]
        avg_metrics = {
            k: float(np.mean([m[k] for m in per_scenario]))
            for k in ("avg_queue_m", "avg_waiting_s", "avg_speed_mps", "avg_throughput", "collisions", "teleports")
        }
        score = composite_score(avg_metrics)
        record = {"timesteps": self.num_timesteps, "score": score, **avg_metrics}
        self.history.append(record)
        self._append_csv_row(record)

        if self.verbose:
            print(f"[validation @ {self.num_timesteps}] score={score:.4f} metrics={avg_metrics}")

        # Save the LATEST state every time, independent of whether it's a new
        # best -- this is what makes resuming after a crash lose minimal
        # progress instead of falling back to a possibly much older best_model.
        latest_dir = self.out_dir / "latest_model"
        latest_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(latest_dir / "model"))
        with open(latest_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if score > self.best_score:
            self.best_score = score
            best_dir = self.out_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(best_dir / "model"))
            with open(best_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        return True

    def _on_training_end(self) -> None:
        self._eval_env.close()
        # Convenience copy only -- validation_history.csv (written incrementally
        # above) is the durable record; this JSON dump is just easier to read
        # in one go once a run finishes cleanly.
        with open(self.out_dir / "validation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


def _load_resume_state(out_dir: Path) -> tuple[float, int, List[dict]]:
    """Reads the run's own best-so-far bookkeeping AND validation history so
    ValidationCallback doesn't forget either across a resume.

    History now comes from validation_history.csv, NOT validation_history.json
    -- the CSV is written incrementally after every single validation call, so
    it survives a crash mid-run; the JSON is only written at a clean finish
    and would simply not exist after a crash. Falls back to the JSON if no CSV
    is found (e.g. an older run from before this change), so old runs can
    still be resumed without losing their recorded history.

    best_score is still read from out_dir/best_model/validation_metrics.json
    specifically (not derived from history) since that's the one score that
    actually matters for the overwrite-protection check -- but as a second
    line of defense, if that file is missing, best_score falls back to the
    max score found in history/CSV instead of jumping straight to -inf."""
    best_score = -np.inf
    last_eval_step = 0
    history: List[dict] = []

    csv_path = out_dir / "validation_history.csv"
    json_path = out_dir / "validation_history.json"
    if csv_path.is_file():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    "timesteps": int(row["timesteps"]),
                    "score": float(row["score"]),
                    "avg_queue_m": float(row["avg_queue_m"]),
                    "avg_waiting_s": float(row["avg_waiting_s"]),
                    "avg_speed_mps": float(row["avg_speed_mps"]),
                    "avg_throughput": float(row["avg_throughput"]),
                    "collisions": float(row["collisions"]),
                    "teleports": float(row["teleports"]),
                })
        print(f"[resume] Loaded {len(history)} prior validation record(s) from {csv_path}.")
    elif json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"[resume] No CSV found; loaded {len(history)} prior validation record(s) "
              f"from {json_path} instead.")

    if history:
        last_eval_step = max(int(r["timesteps"]) for r in history)

    metrics_path = out_dir / "best_model" / "validation_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        best_score = float(saved["score"])
        print(f"[resume] Seeding best_score={best_score:.4f} from {metrics_path} "
              f"(saved at {int(saved['timesteps'])} timesteps).")
    elif history:
        best_score = max(r["score"] for r in history)
        print(f"[resume] WARNING: {metrics_path} not found. Falling back to "
              f"best_score={best_score:.4f} derived from history instead of -inf.")
    else:
        print(f"[resume] WARNING: no validation_metrics.json and no history found. "
              f"best_score will start at -inf, meaning the FIRST post-resume validation "
              f"will always be saved as 'best' even if it's worse than any prior result.")

    return best_score, last_eval_step, history


def main() -> None:
    args = build_arg_parser().parse_args()
    run_cfg = cfg.PPORunConfig()
    apply_overrides(run_cfg, args)
    hp = run_cfg.hyperparams

    out_dir = Path(run_cfg.model_out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resuming = args.resume is not None

    # --- Safety check 0: refuse to silently start a FRESH model on top of an
    # out_dir that already has trained artifacts, unless the caller explicitly
    # opted in. This is what causes runs to get conflated together (a
    # forgotten --resume flag creates a brand-new random model and, at
    # training end, overwrites validation_history.json with only ITS OWN
    # entries -- destroying the record of everything trained before it, even
    # though best_model/ itself stays protected by the best-score check
    # below). If you genuinely want a new experiment, use a new --run-name.
    existing_checkpoint = out_dir / "best_model" / "model.zip"
    if not resuming and existing_checkpoint.is_file() and not args.allow_fresh_overwrite:
        raise RuntimeError(
            f"Refusing to start a NEW model: {existing_checkpoint} already exists from a "
            f"previous run under --run-name '{args.run_name}'.\n"
            f"If you meant to continue training (e.g. after a crash/restart), pass "
            f"--resume {out_dir / 'latest_model' / 'model.zip'}\n"
            f"If you genuinely want to start a fresh, unrelated experiment, either pick a "
            f"different --run-name, or pass --allow-fresh-overwrite to proceed anyway "
            f"(this will NOT delete the existing best_model/latest_model, but will overwrite "
            f"validation_history.csv/.json with only this new run's entries)."
        )

    train_env = Monitor(ASTRIDSignalEnv(run_cfg, run_cfg.train_scenarios, seed=hp.seed))

    initial_best_score, initial_last_eval_step, initial_history = -np.inf, 0, []

    if resuming:
        resume_path = Path(args.resume)

        # --- Safety check 1: checkpoint file must exist ---
        if not resume_path.is_file():
            train_env.close()
            raise FileNotFoundError(
                f"--resume path does not exist: {resume_path}\n"
                f"Expected a saved PPO checkpoint, e.g. "
                f"ppo_models/{args.run_name}/best_model/model.zip"
            )

        # --- Safety check 2: --total-timesteps required and must be a real grand total ---
        if args.total_timesteps is None:
            train_env.close()
            raise ValueError(
                "--resume requires --total-timesteps to be set explicitly to the GRAND "
                "TOTAL you want the model to reach (e.g. 100000), not left as the config "
                "default, and not treated as 'additional steps'."
            )

        # --- Safety check 3: load the model (validates env/observation-space compatibility) ---
        try:
            model = PPO.load(str(resume_path), env=train_env)
        except Exception as exc:  # noqa: BLE001 -- re-raise with actionable context
            train_env.close()
            raise RuntimeError(
                f"Failed to load PPO checkpoint from {resume_path} against the current "
                f"environment. This usually means the observation/action space or a "
                f"policy_kwargs setting (e.g. net_arch) has changed since this checkpoint "
                f"was saved. Original error: {exc}"
            ) from exc

        already_done = model.num_timesteps
        print(f"[resume] Loaded checkpoint from {resume_path}: already trained for "
              f"{already_done} timesteps.")

        # --- Safety check 4: requested total must be strictly ahead of the checkpoint ---
        if args.total_timesteps <= already_done:
            train_env.close()
            raise ValueError(
                f"--total-timesteps ({args.total_timesteps}) must be greater than the "
                f"checkpoint's own num_timesteps ({already_done}). Nothing to train -- "
                f"did you mean to pass a larger --total-timesteps?"
            )

        remaining_timesteps = args.total_timesteps - already_done
        print(f"[resume] Target grand total: {args.total_timesteps}. "
              f"Will train for {remaining_timesteps} additional timesteps "
              f"({already_done} -> {args.total_timesteps}).")

        # --- Safety check 5: back up the checkpoint being resumed from before touching it ---
        # best_model/ is the SAME directory ValidationCallback saves new "best" checkpoints
        # into, so if this run's own validation ever beats the loaded score, that directory
        # gets overwritten with the (better) new one. Keep an explicit, timestamped copy of
        # what we loaded so the original 50k result is always recoverable even in that case.
        backup_dir = out_dir / f"best_model_pre_resume_{already_done}"
        if resume_path.parent.is_dir() and not backup_dir.exists():
            shutil.copytree(resume_path.parent, backup_dir)
            print(f"[resume] Backed up pre-resume checkpoint to {backup_dir}")

        initial_best_score, initial_last_eval_step, initial_history = _load_resume_state(out_dir)
        # Never let the checkpoint's own already-completed step count be treated as due
        # for a re-validation the instant training resumes -- next validation should land
        # at (already_done + eval_every_timesteps), not immediately at already_done again.
        initial_last_eval_step = max(initial_last_eval_step, already_done)

    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hp.learning_rate,
            n_steps=hp.n_steps,
            batch_size=hp.batch_size,
            n_epochs=hp.n_epochs,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
            clip_range=hp.clip_range,
            ent_coef=hp.ent_coef,
            vf_coef=hp.vf_coef,
            max_grad_norm=hp.max_grad_norm,
            policy_kwargs={"net_arch": hp.net_arch},
            seed=hp.seed,
            verbose=1,
            tensorboard_log=str(out_dir / "tb"),
        )
        remaining_timesteps = hp.total_timesteps

    validation_callback = ValidationCallback(
        run_cfg,
        out_dir,
        args.eval_every_timesteps,
        initial_best_score=initial_best_score,
        initial_last_eval_step=initial_last_eval_step,
        initial_history=initial_history,
    )

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=validation_callback,
        reset_num_timesteps=not resuming,
    )

    model.save(str(out_dir / "final_model"))
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        f.write(str(run_cfg))

    train_env.close()
    print(f"Done. Artifacts written to {out_dir}")
    print(f"Best validation checkpoint: {out_dir / 'best_model' / 'model.zip'} "
          f"(score={validation_callback.best_score:.4f})")
    print(f"Latest checkpoint (for resuming after a future crash/restart): "
          f"{out_dir / 'latest_model' / 'model.zip'}")
    print(f"Full validation history (durable, updated every validation): "
          f"{out_dir / 'validation_history.csv'}")


if __name__ == "__main__":
    main()