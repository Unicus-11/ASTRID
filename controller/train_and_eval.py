"""
train_and_eval.py
====================
Smallest possible training/eval harness for nn_controller.NNPolicy's
TrafficControllerMLP. Two stages, deliberately kept separate:

STAGE 1 -- supervised warm-start (implemented here)
    Trains the network to imitate a simple, already-safe default
    behavior -- NOT Webster, and not any tuned traffic-engineering
    rule. The imitation target reproduces astrid_controller.py's own
    placeholder_policy logic (request the next stage once minimum green
    has elapsed, when there is no real queue estimate yet). This exists
    ONLY to move the weights off pure random initialization before the
    network ever runs in the loop. It is explicitly not trained "to
    convergence" or treated as the finished controller (see main()'s
    small, fixed epoch count).

STAGE 2 -- traffic-performance objective (NOT implemented here, by design)
    The real goal -- minimize queue / queue growth / waiting time /
    spillback / unnecessary switching, beat Webster -- needs a reward
    computed from an actual closed-loop rollout, and the network's own
    actions change the traffic it later observes. That makes it a
    genuine RL problem, and the task is explicit that RL is not to be
    introduced yet. So this stage is left as a clearly documented
    extension point:
      - evaluate_policy() below already runs one full closed-loop
        episode with an NNPolicy plugged in and returns aggregate
        metrics computed purely from sumo_interface.py's own StepTrace
        history (which it already collects for its printed trace) --
        nothing here reads new ground truth to do this.
      - a future RL trainer only needs to (a) get a per-episode or
        per-step reward -- this project's reward.py is where ground
        truth is already legitimately used for that, per
        controller_state.py's docstring, and is intentionally not
        duplicated here -- and (b) replace
        TrafficControllerMLP.train_step()'s supervised cross-entropy
        update with a policy-gradient update. The network architecture
        (nn_model.py), the feature contract (nn_features.py), and the
        safety wrapper (nn_controller.NNPolicy -> actions.resolve_action)
        all stay exactly as they are.

POLICY INJECTION (this revision)
---------------------------------
evaluate_policy() previously swapped in a different policy by
monkeypatching the module-level `sumo_interface.placeholder_policy`
attribute for the duration of one episode, then restoring it. That is
no longer needed: sumo_interface.LoopConfig now accepts an optional
`policy_fn`, and SumoInterface resolves `self.policy_fn = config.policy_fn
or placeholder_policy` once in __init__. Passing `policy_fn=policy`
below is sufficient to run one evaluation episode with a different
policy. sumo_interface.placeholder_policy itself is never read or
written by this file anymore.
"""

from __future__ import annotations

from typing import List

import numpy as np

from controller_state import ControllerState, PlaceholderQueueEstimator
from nn_controller import NNPolicy
from nn_features import FEATURE_DIM, build_nn_features
from signal_config import MIN_GREEN_S, STAGE_INDICES


def _synthetic_warmstart_batch(n: int, rng: np.random.Generator):
    """Generates synthetic (feature, action) pairs whose target action
    reproduces astrid_controller.placeholder_policy's own rule (request
    next once min green has elapsed), so Stage 1 has something concrete
    and honest to imitate without needing a live SUMO connection just to
    initialize weights. Queue estimates are left at the placeholder's
    real value (None -> "no estimate" bits off), matching what the
    network will actually see until a real online HGB estimator is
    wired into controller_state.py."""
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    stage_phases = list(STAGE_INDICES)

    for i in range(n):
        phase = int(rng.choice(stage_phases))
        min_green = MIN_GREEN_S[phase]
        elapsed = float(rng.uniform(0.0, 2.0 * min_green))

        state = ControllerState(
            estimated_queue_m=PlaceholderQueueEstimator().estimate(),
            current_phase=phase,
            phase_elapsed_s=elapsed,
        )
        x[i] = build_nn_features(state)
        y[i] = 1 if elapsed >= min_green else 0  # 1 = REQUEST_NEXT, 0 = KEEP

    return x, y


def train_supervised_warmstart(
    policy: NNPolicy, epochs: int = 500, batch_size: int = 64, lr: float = 0.1, seed: int = 0
) -> List[float]:
    """Runs `epochs` SGD steps of the Stage-1 imitation objective and
    returns the per-epoch loss history. The defaults (500 steps, lr=0.1)
    are enough for this tiny 14-input network to reliably learn the
    simple imitation target (~99% held-out accuracy reproducing
    placeholder_policy's own rule) in well under a second -- this is
    still just a warm-start, not "trained to convergence" on anything
    resembling the real traffic-performance objective; see module
    docstring."""
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        x, y = _synthetic_warmstart_batch(batch_size, rng)
        loss = policy.model.train_step(x, y, lr=lr)
        losses.append(loss)
    return losses


def evaluate_policy(
    policy: NNPolicy,
    sumo_binary: str = "sumo",
    config_path: str = "sq.sumo.cfg",
    max_steps=None,
) -> dict:
    """Runs one full closed-loop episode with `policy` plugged in, using
    the exact same SumoInterface that runs the placeholder controller
    today, and reports simple aggregate metrics from the StepTrace
    history sumo_interface.py's run() already collects for its own
    printed trace.

    Requires a real SUMO/TraCI install (same requirement as
    sumo_interface.run_closed_loop_demo).

    INTEGRATION: sumo_interface.LoopConfig accepts an optional
    `policy_fn` (defaulting to placeholder_policy when omitted).
    SumoInterface.step() calls `self.policy_fn(state)`, resolved once in
    __init__ as `config.policy_fn or placeholder_policy`. Passing
    `policy_fn=policy` below runs this episode with `policy` instead of
    placeholder_policy -- no monkeypatching of
    sumo_interface.placeholder_policy is needed or performed here.

    For deeper traffic metrics (queue growth, waiting time, spillback),
    wire in this project's own reward.py, which is where ground truth is
    already legitimately used for evaluation (see controller_state.py's
    docstring) -- intentionally not duplicated here.
    """
    import sumo_interface as _si

    interface = _si.SumoInterface(
        _si.LoopConfig(
            sumo_binary=sumo_binary,
            config_path=config_path,
            max_steps=max_steps,
            policy_fn=policy,
        )
    )
    interface.start()
    try:
        traces = interface.run()
    finally:
        interface.close()

    switch_count = sum(
        1 for t in traces if t.resolved_action in ("BEGIN_TRANSITION", "FORCE_TRANSITION_MAX_GREEN")
    )
    avg_active_vehicles = float(np.mean([t.active_vehicle_count for t in traces])) if traces else 0.0

    return {
        "num_steps": len(traces),
        "switch_count": switch_count,
        "avg_active_vehicle_count": avg_active_vehicles,
    }


def main() -> None:
    policy = NNPolicy()
    print("Stage 1: supervised warm-start (imitating the existing safe default, not Webster)...")
    losses = train_supervised_warmstart(policy)
    print(f"warm-start loss: start={losses[0]:.4f} end={losses[-1]:.4f}")
    policy.model.save("nn_controller_warmstart.npz")
    print("Saved weights to nn_controller_warmstart.npz")
    print(
        "Stage 2 (traffic-performance / RL objective) is NOT run here by design -- "
        "see this file's module docstring for exactly where it plugs in later."
    )


if __name__ == "__main__":
    main()