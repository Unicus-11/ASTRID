"""
ppo/ppo_controller.py
========================
Safety layer PPO's actions pass through. PPO NEVER calls
traci.trafficlight.setPhase()/setPhaseDuration() directly -- it only
chooses KEEP or SWITCH, and this class decides whether that's currently
legal and performs the actual traci call.

IMPORTANT, corrected from an earlier version: sq.net.xml's tlLogic is a
STATIC program -- SUMO will auto-advance to the next phase on its own
once a phase's configured duration elapses, with no traci call needed.
That means a naive "KEEP = do nothing" implementation is NOT actually a
KEEP -- it just lets the static program run out and switch on its own,
regardless of what PPO wanted. To make KEEP meaningful (extend the
current stage), this controller calls
traci.trafficlight.setPhaseDuration() every control tick while KEEPing,
which resets the phase's REMAINING time -- capped so the phase can never
run past its own MAX_GREEN_S. Only the four STAGE phases
(controller.STAGE_INDICES = 0, 2, 4, 6) are controllable at all; the
mandatory yellow/transition phases (1, 3, 5, 7) always run their fixed
sq.net.xml duration untouched -- PPO is never asked to act during them,
and apply_action() is a no-op if called while one is active.

MIN_GREEN_S[phase] is (per signal_config.py) exactly that phase's own
static duration -- so a SWITCH request is only ever legal at or after
the point the original fixed-time program would itself have switched.
PPO's actual lever is extending PAST that point (up to MAX_GREEN_S), not
cutting a stage short.
"""

from __future__ import annotations

import ppo_config as cfg

KEEP = 0
SWITCH = 1


class SignalSafetyController:
    """Owns all direct traci.trafficlight calls for one TLS."""

    def __init__(self, traci_module, tls_id: str):
        self._traci = traci_module
        self._tls_id = tls_id
        self._last_switch_time = 0.0
        self.switch_count = 0
        self.forced_switch_count = 0

    def reset(self, start_time: float) -> None:
        """Call once per episode, right after warm-up ends, so
        time_since_last_switch() is measured from the start of the
        reward-bearing control window rather than from t=0 of warm-up."""
        self._last_switch_time = start_time
        self.switch_count = 0
        self.forced_switch_count = 0

    def current_phase(self) -> int:
        return self._traci.trafficlight.getPhase(self._tls_id)

    def current_phase_elapsed_s(self) -> float:
        """Time spent in the CURRENT SUMO PHASE. Resets on every SUMO
        phase change (including the automatic yellow/all-red
        transitions), not only on a controller-requested switch -- do
        not read this as 'time since PPO last acted'; use
        time_since_last_switch() for that."""
        return self._traci.trafficlight.getSpentDuration(self._tls_id)

    def time_since_last_switch(self, current_time: float) -> float:
        """Time since THIS controller last issued a switch (PPO-requested
        or safety-forced)."""
        return current_time - self._last_switch_time

    def is_controllable_stage(self) -> bool:
        return self.current_phase() in cfg.STAGE_INDICES

    def apply_action(self, action: int, current_time: float, control_interval_s: float) -> bool:
        """Returns True if a phase switch was actually issued this call
        (either PPO-requested and legal, or safety-forced by max green).

        While in a controllable stage phase:
            - if elapsed >= MAX_GREEN_S[phase]: force the mandatory
              switch (safety cap), regardless of `action`.
            - elif action == SWITCH and elapsed >= MIN_GREEN_S[phase]
              (i.e. the original static program would already have
              switched by now): switch immediately.
            - otherwise (KEEP, or an early SWITCH that isn't legal yet):
              extend the phase's remaining duration via
              setPhaseDuration() so SUMO's static program does not
              auto-advance before the next decision point, capped at
              MAX_GREEN_S so the phase can never run longer than that
              regardless of how many times KEEP is chosen.

        While in a mandatory transition (yellow) phase: no-op. PPO's
        action this tick has no effect; the fixed yellow duration always
        runs to completion untouched.
        """
        phase = self.current_phase()
        if phase not in cfg.STAGE_INDICES:
            return False

        elapsed = self.current_phase_elapsed_s()
        min_green = cfg.MIN_GREEN_S[phase]
        max_green = cfg.MAX_GREEN_S[phase]

        forced = elapsed >= max_green
        requested = action == SWITCH and elapsed >= min_green

        if forced or requested:
            next_phase = cfg.TRANSITION_AFTER_STAGE[phase]
            self._traci.trafficlight.setPhase(self._tls_id, next_phase)
            self._last_switch_time = current_time
            self.switch_count += 1
            if forced:
                self.forced_switch_count += 1
            return True

        # KEEP (or an early, not-yet-legal SWITCH): actively extend so
        # the static program doesn't switch on its own before the next
        # control tick. +1s buffer against float/step-boundary rounding;
        # capped so the phase never exceeds max_green in total.
        remaining_budget = max_green - elapsed
        extend_to = min(control_interval_s + 1.0, remaining_budget)
        if extend_to > 0:
            self._traci.trafficlight.setPhaseDuration(self._tls_id, extend_to)
        return False