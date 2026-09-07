"""
mock_traci_selftest.py
====================
PYTHON-ONLY WIRING CHECK -- NOT A REAL SUMO/TraCI RUN, NOT A TRAFFIC
SIMULATION, AND NOT EVIDENCE THAT THE REQUIRED closed-loop test has
been performed.

This exists ONLY because no SUMO binary and no `traci` package are
installed (and no network access to install them) in the environment
this controller module was written in -- see the accompanying report.
It emulates just enough of `traci`'s API surface, driven by
sq.net.xml's OWN real phase durations, to exercise sumo_interface.py's
control-flow logic end to end and catch wiring bugs (e.g. the repeated
setPhase() bug this revision fixes) before anyone runs it against real
SUMO.

It does NOT simulate vehicle dynamics, arrivals, queues, or speeds in
any meaningful way; MockVehiclePool below is a deliberately crude
counter, not a traffic model. Nothing printed by this script should be
read as evidence about real traffic behavior.
"""

from __future__ import annotations

from typing import Dict, List

from signal_config import PHASE_BY_INDEX, PHASE_SEQUENCE, TLS_ID, SIMULATION_END_S


class _MockTrafficLight:
    """Emulates SUMO's static tlLogic auto-advance for TLS_ID, and
    records every setPhase() call so the self-test can assert the fixed
    bug (repeated setPhase on the same phase) never happens again."""

    def __init__(self):
        self._phase_index = 0
        self._phase_start_time = 0.0
        self.set_phase_call_log: List[tuple] = []  # (sim_time, requested_phase)

    def getIDList(self):
        return [TLS_ID]

    def getPhase(self, tls_id):
        assert tls_id == TLS_ID
        return self._phase_index

    def getSpentDuration(self, tls_id):
        assert tls_id == TLS_ID
        return self._current_time - self._phase_start_time

    def setPhase(self, tls_id, phase_index):
        assert tls_id == TLS_ID
        self.set_phase_call_log.append((self._current_time, phase_index))
        self._phase_index = phase_index
        self._phase_start_time = self._current_time

    def _advance_static_program_if_due(self):
        """Mimics SUMO's OWN static-program auto-advance -- happens
        regardless of whether the controller ever calls setPhase()."""
        phase_def = PHASE_BY_INDEX[self._phase_index]
        elapsed = self._current_time - self._phase_start_time
        if elapsed >= phase_def.duration_s:
            pos = PHASE_SEQUENCE.index(self._phase_index)
            self._phase_index = PHASE_SEQUENCE[(pos + 1) % len(PHASE_SEQUENCE)]
            self._phase_start_time = self._current_time

    def _set_time(self, t: float):
        self._current_time = t


class _MockVehiclePool:
    """Deliberately crude vehicle-count stand-in -- NOT a traffic model.
    Grows and shrinks on a fixed schedule so active_vehicle_count is
    visibly non-constant in the trace, nothing more."""

    def __init__(self):
        self._count = 0

    def getIDCount(self):
        return self._count

    def _set_time(self, t: float):
        # crude synthetic pattern: ramps up over each 90s window then
        # partially clears -- purely for exercising the print path.
        cycle_pos = t % 90.0
        self._count = int(cycle_pos // 3)


class _MockSimulation:
    def __init__(self):
        self._time = 0.0

    def getTime(self):
        return self._time


class MockTraci:
    """Minimal stand-in for the `traci` module's API surface used by
    sumo_interface.SumoInterface. See module docstring: NOT SUMO."""

    def __init__(self):
        self.trafficlight = _MockTrafficLight()
        self.vehicle = _MockVehiclePool()
        self.simulation = _MockSimulation()
        self._step_length = 1.0

    def start(self, cmd):
        self._advance_and_sync(0.0)

    def close(self):
        pass

    def simulationStep(self):
        self._advance_and_sync(self.simulation._time + self._step_length)

    def _advance_and_sync(self, t: float):
        self.simulation._time = t
        self.trafficlight._set_time(t)
        self.vehicle._set_time(t)
        self.trafficlight._advance_static_program_if_due()


def run_selftest(max_steps: int = 400) -> None:
    from sumo_interface import LoopConfig, SumoInterface

    print("=" * 78)
    print("MOCK-TRACI WIRING SELF-TEST -- NOT A REAL SUMO RUN, NOT TRAFFIC EVIDENCE")
    print("=" * 78)

    mock = MockTraci()
    interface = SumoInterface(LoopConfig(max_steps=max_steps, print_every_s=5.0), traci_module=mock)
    interface.start()
    traces = interface.run()
    interface.close()

    # -- assertions specific to the bug this revision fixes --
    repeated_same_phase_calls = [
        (t, p) for (t, p) in mock.trafficlight.set_phase_call_log
        if mock.trafficlight.set_phase_call_log.count((t, p)) > 1
    ]
    consecutive_same_phase = 0
    prev = None
    for (_, p) in mock.trafficlight.set_phase_call_log:
        if prev is not None and p == prev:
            consecutive_same_phase += 1
        prev = p

    print("\n[bug-fix assertions]")
    print(f"total setPhase() calls: {len(mock.trafficlight.set_phase_call_log)}")
    print(f"setPhase() calls that re-asserted the CURRENTLY ACTIVE phase (should be 0): "
          f"{consecutive_same_phase}")
    assert consecutive_same_phase == 0, "BUG REGRESSION: setPhase() re-asserted an unchanged phase."

    distinct_phases_seen = sorted(set(t.sumo_current_phase for t in traces))
    print(f"distinct phases observed: {distinct_phases_seen}")
    max_spent = max(t.sumo_spent_duration for t in traces)
    print(f"max SUMO_spent_duration observed in any single phase: {max_spent:.1f}s "
          f"(should exceed 1 step, i.e. genuinely accumulate rather than reset every step)")
    assert max_spent > 1.0, "BUG REGRESSION: spent_duration never accumulated past one step."

    print("\nSELF-TEST PASSED (wiring only -- see module docstring for what this does NOT prove).")


if __name__ == "__main__":
    run_selftest()