"""
sumo_interface.py
====================
The smallest reliable closed-loop TraCI interface for traffic light
signal_config.TLS_ID, running against sq.sumo.cfg (source of truth for
how this simulation starts: net-file=sq.net.xml, route-files=sq.rou.xml,
begin=0, end=3600).

--------------------------------------------------------------------------
BUG FIX (earlier revision)
--------------------------------------------------------------------------
The previous revision maintained its OWN `_current_phase` /
`_phase_start_time` bookkeeping and called
`traci.trafficlight.setPhase(TLS_ID, current_phase)` every single step
for HOLD_STAGE and for "still inside a yellow phase" -- including when
the phase had not actually changed. `setPhase()` RESTARTS the named
phase's timer even when it is already the active phase, so that
per-step re-assertion was corrupting SUMO's own phase timing and could
prevent a phase from ever naturally completing.

Fixed by making SUMO itself the single source of truth for phase state,
every step, via:
    traci.trafficlight.getPhase(TLS_ID)          -- which phase is active
    traci.trafficlight.getSpentDuration(TLS_ID)  -- how long it's been active
No parallel Python-side clock is kept anymore. Concretely:
  * ACTION_KEEP / HOLD_STAGE issues NO TraCI call at all -- the static
    tlLogic program in sq.net.xml keeps running on its own.
  * A transition is applied with EXACTLY ONE `setPhase()` call, only
    when actions.resolve_action() returns BEGIN_TRANSITION or
    FORCE_TRANSITION_MAX_GREEN, and only after
    signal_config.is_legal_transition() has checked it.
  * While a yellow/transition phase (1, 3, 5, 7) is active,
    actions.resolve_action() already returns TransitionEffect.NONE (see
    actions.py -- unchanged), and this interface now issues NO TraCI
    call in that case either. SUMO's own static program is left to run
    the transition to completion and advance to the next phase by
    itself; this interface simply reads whatever phase SUMO reports on
    the next step.

CONTROLLER ACTUATOR AUTHORITY
-----------------------------
The controller explicitly programs each newly entered green stage with
MAX_GREEN_S using TraCI setPhaseDuration().

This gives ACTION_KEEP genuine authority to allow a green stage to remain
active beyond its base sq.net.xml duration, while ACTION_REQUEST_NEXT can
request the mandatory transition after MIN_GREEN_S has elapsed.

The controller never programs an arbitrary duration. MAX_GREEN_S remains
the safety ceiling defined in signal_config.py.

Transition/yellow phases remain fully controlled by the static SUMO
program and are not extended or skipped by the controller.

This revision does not touch that mechanism -- see _ensure_stage_duration()
below, unchanged.

--------------------------------------------------------------------------
POLICY INJECTION (this revision)
--------------------------------------------------------------------------
Previously, `step()` called the module-level `placeholder_policy` name
directly, and tests/eval code swapped in a different policy by
monkeypatching that module attribute (`sumo_interface.placeholder_policy
= some_other_policy`). That worked, but it mutated shared module state
and required careful save/restore around every use.

LoopConfig now accepts an optional `policy_fn`. SumoInterface resolves
the active policy once, in __init__, as:

    self.policy_fn = config.policy_fn or placeholder_policy

and step() calls `self.policy_fn(state)` instead of the bare
`placeholder_policy(state)`. When no policy_fn is supplied, behavior is
byte-for-byte identical to before: the module-level placeholder_policy
is used, and nothing about it is modified at runtime.

This is policy injection only. It does not change what a policy is
allowed to do: policy_fn must still return one of actions.ACTION_KEEP /
actions.ACTION_REQUEST_NEXT, that string still passes through
actions.resolve_action() as the sole legality/safety check below, and no
policy has (or gains) any direct path to setPhase()/setPhaseDuration().

`traci_module` remains injectable so the finite-state machine in this
class can be exercised without a real SUMO/TraCI install -- see
controller/results/mock_traci_selftest.py, which is a Python-only wiring
check, NOT a substitute traffic simulation, and NOT a claim that the
real SUMO/TraCI closed-loop test has been run (it has not been possible
to run real SUMO in the environment this revision was written in -- see
the accompanying report). Production use always passes the real `traci`
package (the default when `traci_module=None`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from actions import TransitionEffect, resolve_action
from astrid_controller import placeholder_policy
from controller_state import (
    ControllerState,
    PlaceholderQueueEstimator,
    QueueEstimator,
    build_controller_state,
)
from signal_config import (
    MAX_GREEN_S,
    PHASE_SEQUENCE,
    SIMULATION_END_S,
    STAGE_INDICES,
    TLS_ID,
    is_legal_transition,
)


@dataclass
class LoopConfig:
    sumo_binary: str = "sumo"  # use "sumo-gui" for visualization
    config_path: str = "sq.sumo.cfg"
    max_steps: Optional[int] = None  # None -> run until SIMULATION_END_S
    print_every_s: float = 5.0
    queue_estimator: Optional[QueueEstimator] = None  # defaults to PlaceholderQueueEstimator()
    policy_fn: Optional[Callable[[ControllerState], str]] = None  # defaults to placeholder_policy


@dataclass
class StepTrace:
    simulation_time: float
    sumo_current_phase: int
    sumo_spent_duration: float
    active_vehicle_count: int
    controller_action: str
    resolved_action: str
    state_summary: str


def format_trace(trace: StepTrace) -> str:
    return (
        f"simulation_time={trace.simulation_time:7.1f}s  "
        f"SUMO_current_phase={trace.sumo_current_phase}  "
        f"SUMO_spent_duration={trace.sumo_spent_duration:5.1f}s  "
        f"active_vehicle_count={trace.active_vehicle_count:4d}  "
        f"controller_action={trace.controller_action:<13} "
        f"resolved_action={trace.resolved_action:<26} "
        f"{trace.state_summary}"
    )


class SumoInterface:
    def __init__(self, config: LoopConfig, traci_module=None):
        self.config = config
        if traci_module is None:
            import traci as _traci  # imported lazily so this module is
            traci_module = _traci   # importable even without SUMO installed
        self.traci = traci_module
        self.queue_estimator: QueueEstimator = config.queue_estimator or PlaceholderQueueEstimator()
        # Policy injection point: falls back to the module-level
        # placeholder_policy when none is supplied via LoopConfig, so
        # every existing `LoopConfig(...)` call site (with no policy_fn)
        # keeps its exact current behavior. Resolved once here, not
        # re-resolved per step.
        self.policy_fn: Callable[[ControllerState], str] = config.policy_fn or placeholder_policy

        # No parallel phase/timing bookkeeping is kept anymore -- see
        # module docstring. This list is diagnostic only (for the
        # observed-vs-expected phase-sequence check in run()), never
        # used as a source of truth for control decisions.
        self._observed_phase_sequence: List[int] = []
        # Tracks which controllable stage has already had its SUMO duration
        # explicitly programmed. This is actuator bookkeeping only; SUMO
        # remains the source of truth for phase identity and elapsed time.
        self._duration_programmed_for_phase: Optional[int] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """1. start SUMO, 2. connect through TraCI, per sq.sumo.cfg."""
        sumo_cmd = [self.config.sumo_binary, "-c", self.config.config_path]
        self.traci.start(sumo_cmd)
        self._validate_traffic_light()
        # Read (never assume) whatever phase SUMO's own static program
        # is actually in at connection time.
        initial_phase = self.traci.trafficlight.getPhase(TLS_ID)
        self._duration_programmed_for_phase = None
        print(f"[start] SUMO reports initial phase = {initial_phase} (not assumed to be 0).")

    def close(self) -> None:
        """13. close SUMO cleanly."""
        self.traci.close()

    def _validate_traffic_light(self) -> None:
        """3.-5. confirm the traffic light this module controls actually
        exists in the running simulation, per PART 10's checklist."""
        tls_ids = self.traci.trafficlight.getIDList()
        if TLS_ID not in tls_ids:
            raise RuntimeError(f"Traffic light '{TLS_ID}' not found in SUMO. Available: {tls_ids}")

    # -- one control decision + one simulation step ---------------------

    def step(self) -> StepTrace:
        """4.-11. read SUMO's own current phase/timing, build state, call
        the active policy (self.policy_fn), validate its action, and
        apply AT MOST one setPhase() call -- only for an actual, legal
        transition -- before advancing the simulation by one step."""
        simulation_time = self.traci.simulation.getTime()

        # SUMO is the single source of truth for phase identity and
        # elapsed time -- no independent Python-side clock.
        current_phase = self.traci.trafficlight.getPhase(TLS_ID)
        spent_duration = self.traci.trafficlight.getSpentDuration(TLS_ID)

        if not self._observed_phase_sequence or self._observed_phase_sequence[-1] != current_phase:
            self._observed_phase_sequence.append(current_phase)

        self._ensure_stage_duration(current_phase)

        state = build_controller_state(
            queue_estimator=self.queue_estimator,
            current_phase=current_phase,
            phase_elapsed_s=spent_duration,
        )
        action = self.policy_fn(state)
        resolved = resolve_action(action, current_phase, spent_duration)

        if resolved.effect in (
            TransitionEffect.BEGIN_TRANSITION,
            TransitionEffect.FORCE_TRANSITION_MAX_GREEN,
        ):
            next_index = resolved.next_phase_index
            if not is_legal_transition(current_phase, next_index):
                raise RuntimeError(f"Illegal transition blocked: {current_phase} -> {next_index}")
            # Exactly one setPhase() call for this transition. Not
            # repeated on subsequent steps while next_index remains active.
            self.traci.trafficlight.setPhase(TLS_ID, next_index)

        # TransitionEffect.HOLD_STAGE -> no TraCI call: let SUMO's own
        # program continue the current phase.
        # TransitionEffect.NONE (inside a yellow phase) -> no TraCI call
        # either: let SUMO run the transition to completion on its own.

        trace = StepTrace(
            simulation_time=simulation_time,
            sumo_current_phase=current_phase,
            sumo_spent_duration=spent_duration,
            active_vehicle_count=self.traci.vehicle.getIDCount(),
            controller_action=action,
            resolved_action=resolved.effect.value,
            state_summary=state.summary(),
        )

        self.traci.simulationStep()
        return trace

    def _ensure_stage_duration(self, current_phase: int) -> None:
        """Give a newly entered green stage its full controller-authorized
        duration.

        SUMO's static program contains the base phase duration. For adaptive
        control, we explicitly extend each controllable green stage to its
        MAX_GREEN_S when that stage begins.

        This method is called at most once per stage entry, and is
        completely independent of which policy_fn is active -- it runs
        before the policy is even consulted for this step.
        """

        if current_phase not in STAGE_INDICES:
            return

        if self._duration_programmed_for_phase == current_phase:
            return

        max_green = MAX_GREEN_S[current_phase]

        self.traci.trafficlight.setPhaseDuration(
            TLS_ID,
            max_green,
        )

        self._duration_programmed_for_phase = current_phase

        print(
            f"[actuator] phase {current_phase}: "
            f"programmed MAX_GREEN_S={max_green:.1f}s"
        )

    def run(self) -> List[StepTrace]:
        """12. repeat, until SIMULATION_END_S or config.max_steps."""
        traces: List[StepTrace] = []
        steps = 0
        last_printed = -1.0
        while True:
            if self.config.max_steps is not None and steps >= self.config.max_steps:
                break
            if self.traci.simulation.getTime() >= SIMULATION_END_S:
                break
            trace = self.step()
            traces.append(trace)
            if trace.simulation_time - last_printed >= self.config.print_every_s:
                print(format_trace(trace))
                last_printed = trace.simulation_time
            steps += 1

        self._report_phase_sequence_check()
        return traces

    def _report_phase_sequence_check(self) -> None:
        """Explicitly verify the observed phase sequence against
        signal_config.PHASE_SEQUENCE (the 8-phase sequence defined in
        sq.net.xml), as required by validation."""
        print("\n[phase-sequence check]")
        print(f"expected cyclic sequence (sq.net.xml): {list(PHASE_SEQUENCE)}")
        print(f"observed sequence (deduped, in order): {self._observed_phase_sequence}")
        n = len(PHASE_SEQUENCE)
        mismatches = []
        for i in range(1, len(self._observed_phase_sequence)):
            prev = self._observed_phase_sequence[i - 1]
            curr = self._observed_phase_sequence[i]
            expected_next = PHASE_SEQUENCE[(PHASE_SEQUENCE.index(prev) + 1) % n]
            if curr != expected_next:
                mismatches.append((prev, curr, expected_next))
        if mismatches:
            print(f"MISMATCHES FOUND (prev, observed_next, expected_next): {mismatches}")
        else:
            print("OK: every observed transition matched sq.net.xml's own phase sequence.")


def run_closed_loop_demo(
    sumo_binary: str = "sumo",
    config_path: str = "sq.sumo.cfg",
    max_steps: Optional[int] = None,
) -> List[StepTrace]:
    """PART 10/11 entry point: run one existing SUMO scenario end to end,
    printing the required per-step trace, and return every StepTrace for
    inspection. Requires a real SUMO install with `sumo` on PATH and this
    directory's sq.net.xml / sq.rou.xml / sq.sumo.cfg present."""
    interface = SumoInterface(LoopConfig(sumo_binary=sumo_binary, config_path=config_path, max_steps=max_steps))
    interface.start()
    try:
        return interface.run()
    finally:
        interface.close()


if __name__ == "__main__":
    run_closed_loop_demo()