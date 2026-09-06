"""
rule_teacher.py
====================
Simple, documented rule-based "teacher" used to LABEL states for
supervised training of candidate classifiers (see collect_dataset.py) --
not itself proposed as the final ASTRID controller.

Rule (gap-out style, using ONLY estimated_queue_m -- the same
information ControllerState already legitimately exposes, never ground
truth):

  * Below MIN_GREEN_S for the current stage: always KEEP (matches
    actions.py's own enforcement -- REQUEST_NEXT would be denied there
    anyway).
  * At/above MAX_GREEN_S: always REQUEST_NEXT (matches actions.py's own
    safety cap -- labeling this as REQUEST_NEXT teaches the classifier
    to switch proactively instead of relying on being forced).
  * Otherwise: REQUEST_NEXT if the OTHER stage's total estimated queue
    exceeds the CURRENT stage's by more than `ratio*current + margin`;
    else KEEP.

ratio/margin are prototype defaults, tunable via RuleTeacherConfig.
"""
from __future__ import annotations

from dataclasses import dataclass

from actions import ACTION_KEEP, ACTION_REQUEST_NEXT
from controller_state import ControllerState
from signal_config import MAX_GREEN_S, MIN_GREEN_S, PHASE_BY_INDEX, STAGE_APPROACHES, STAGE_INDICES


@dataclass(frozen=True)
class RuleTeacherConfig:
    ratio: float = 0.5    # other stage must exceed current by ratio*current + margin
    margin: float = 5.0   # meters


def _stage_queue_m(estimated_queue_m, stage: str) -> float:
    return sum((estimated_queue_m.get(edge) or 0.0) for edge in STAGE_APPROACHES[stage])


def rule_based_teacher(state: ControllerState, config: RuleTeacherConfig = RuleTeacherConfig()) -> str:
    if state.current_phase not in STAGE_INDICES:
        return ACTION_KEEP

    min_green = MIN_GREEN_S[state.current_phase]
    max_green = MAX_GREEN_S[state.current_phase]
    if state.phase_elapsed_s < min_green:
        return ACTION_KEEP
    if state.phase_elapsed_s >= max_green:
        return ACTION_REQUEST_NEXT

    current_stage = PHASE_BY_INDEX[state.current_phase].stage
    other_stage = "EW" if current_stage == "NS" else "NS"
    current_q = _stage_queue_m(state.estimated_queue_m, current_stage)
    other_q = _stage_queue_m(state.estimated_queue_m, other_stage)

    if other_q > config.ratio * current_q + config.margin:
        return ACTION_REQUEST_NEXT
    return ACTION_KEEP