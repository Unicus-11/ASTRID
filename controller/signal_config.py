"""
signal_config.py
====================
Authoritative signal-program facts for traffic light "0", taken directly
from sq.net.xml's <tlLogic> and <connection ... linkIndex="N"> elements.

Nothing in this file invents a phase, a state string, a duration, or a
lane mapping. Every constant below is either copied verbatim from
sq.net.xml or derived from it through a documented, checkable step.

--------------------------------------------------------------------------
HOW THE 12-CHARACTER STATE STRING MAPS TO MOVEMENTS
--------------------------------------------------------------------------
SUMO's per-phase state string is ordered by <connection ... linkIndex="N">,
not by <edge> declaration order. Reading every tl="0" connection out of
sq.net.xml directly gives this link-index table:

    idx  from  to   dir
     0   4i    1o   r
     1   4i    3o   s
     2   4i    2o   l
     3   2i    4o   r
     4   2i    1o   s
     5   2i    3o   l
     6   3i    2o   r
     7   3i    4o   s
     8   3i    1o   l
     9   1i    3o   r
    10   1i    2o   s
    11   1i    4o   l

Also directly from sq.net.xml: every approach edge (1i/2i/3i/4i) has
exactly 3 lanes, and (from those same connections) fromLane 0 is always
the right turn, fromLane 1 the through movement, fromLane 2 the left
turn -- consistent across all four approaches.

--------------------------------------------------------------------------
GEOGRAPHIC LABELING (from <edge>/<junction> coordinates in sq.net.xml)
--------------------------------------------------------------------------
    edge 4i: from junction "4" at y=1010 (top)    -> North approach
    edge 3i: from junction "3" at y=10   (bottom) -> South approach
    edge 1i: from junction "1" at x=10   (left)   -> West approach
    edge 2i: from junction "2" at x=1010 (right)  -> East approach

--------------------------------------------------------------------------
DECODING EACH PHASE'S STATE STRING AGAINST THE LINK-INDEX TABLE ABOVE
--------------------------------------------------------------------------
    phase 0  dur=25  "GGrrrrGGrrrr"  -> G at idx 0,1,6,7
                                         = 4i(r,s) + 3i(r,s) green
                                         = North+South THROUGH/RIGHT green
    phase 1  dur=7   "yyrrrryyrrrr"  -> yellow at the same idx 0,1,6,7
                                         = transition OFF phase 0's green
    phase 2  dur=6   "rrGrrrrrGrrr"  -> G at idx 2,8
                                         = 4i(l) + 3i(l) green
                                         = North+South PROTECTED LEFT green
    phase 3  dur=7   "rryrrrrryrrr"  -> yellow at idx 2,8
                                         = transition OFF phase 2's green
    phase 4  dur=25  "rrrGGrrrrGGr"  -> G at idx 3,4,9,10
                                         = 2i(r,s) + 1i(r,s) green
                                         = East+West THROUGH/RIGHT green
    phase 5  dur=7   "rrryyrrrryyr"  -> yellow at idx 3,4,9,10
                                         = transition OFF phase 4's green
    phase 6  dur=6   "rrrrrGrrrrrG"  -> G at idx 5,11
                                         = 2i(l) + 1i(l) green
                                         = East+West PROTECTED LEFT green
    phase 7  dur=7   "rrrrryrrrrry"  -> yellow at idx 5,11
                                         = transition OFF phase 6's green

So the true STAGE grouping is:
    STAGE "NS" = phases {0 (through/right green), 2 (protected left green)}
                 serves edges 3i, 4i
    STAGE "EW" = phases {4 (through/right green), 6 (protected left green)}
                 serves edges 1i, 2i
with phases {1, 3, 5, 7} as the mandatory, non-optional yellow/transition
phases in between, run for their fixed durations.

--------------------------------------------------------------------------
CROSS-CHECK AGAINST dataset/feature_builder.py's PHASE_GREEN_GROUP
--------------------------------------------------------------------------
dataset/feature_builder.py (the ASTRID feature pipeline, a separate
module from this controller) defines:

    PHASE_GREEN_GROUP = {0: "EW", 1: "EW", 2: "EW", 3: "EW",
                          4: "NS", 5: "NS", 6: "NS", 7: "NS"}
    GROUP_EDGES = {"EW": ["1i", "2i"], "NS": ["3i", "4i"]}

GROUP_EDGES's edge-to-label assignment agrees with the decoding above
(EW = 1i,2i; NS = 3i,4i). PHASE_GREEN_GROUP's phase-to-label assignment
does NOT: this file's decoding shows phases 0/2 physically serve the
"NS" edges (3i, 4i) and phases 4/6 physically serve the "EW" edges
(1i, 2i) -- the exact opposite of what PHASE_GREEN_GROUP says. That
appears to be a real labeling bug in feature_builder.py's
is_green_for_approach() (it would mark 3i/4i as "not green" during their
own actual green phases, and vice versa for 1i/2i). This controller
module does not depend on feature_builder.py and is unaffected by that
bug, but it is flagged here, and in the final report, because it was
discovered while independently deriving this file's own phase/edge
mapping from the same sq.net.xml -- it is NOT fixed here, since
feature_builder.py is a different module outside this task's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================================
# Traffic light identity
# ============================================================================

TLS_ID = "0"  # <tlLogic id="0" ...> in sq.net.xml; also <junction id="0" type="traffic_light">

# ============================================================================
# Phases -- copied VERBATIM from sq.net.xml's <tlLogic id="0"> block.
# Index in this list == SUMO phase index (0-7).
# ============================================================================


@dataclass(frozen=True)
class PhaseDef:
    index: int
    duration_s: float          # the duration sq.net.xml itself assigns this phase
    state: str                 # verbatim state string from sq.net.xml
    is_transition: bool        # True for the four yellow phases (1,3,5,7)
    stage: "str | None"        # "NS" / "EW" for green phases, None for transitions


PHASES: Tuple[PhaseDef, ...] = (
    PhaseDef(0, 25.0, "GGrrrrGGrrrr", is_transition=False, stage="NS"),
    PhaseDef(1, 7.0,  "yyrrrryyrrrr", is_transition=True,  stage=None),
    PhaseDef(2, 6.0,  "rrGrrrrrGrrr", is_transition=False, stage="NS"),
    PhaseDef(3, 7.0,  "rryrrrrryrrr", is_transition=True,  stage=None),
    PhaseDef(4, 25.0, "rrrGGrrrrGGr", is_transition=False, stage="EW"),
    PhaseDef(5, 7.0,  "rrryyrrrryyr", is_transition=True,  stage=None),
    PhaseDef(6, 6.0,  "rrrrrGrrrrrG", is_transition=False, stage="EW"),
    PhaseDef(7, 7.0,  "rrrrryrrrrry", is_transition=True,  stage=None),
)

PHASE_BY_INDEX: Dict[int, PhaseDef] = {p.index: p for p in PHASES}

# Fixed, cyclic phase sequence exactly as sq.net.xml's tlLogic lists it.
# This is the ONLY legal order of phase indices; nothing may be skipped.
PHASE_SEQUENCE: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


def next_phase_index(current: int) -> int:
    """The single legal next phase index after `current`, cyclic. This is
    sq.net.xml's own sequence -- there is no other legal transition."""
    pos = PHASE_SEQUENCE.index(current)
    return PHASE_SEQUENCE[(pos + 1) % len(PHASE_SEQUENCE)]


# The stable, controllable "stage" indices (steady green -- these are the
# only phases at which ACTION_KEEP / ACTION_REQUEST_NEXT are meaningful).
STAGE_INDICES: Tuple[int, ...] = (0, 2, 4, 6)

# The mandatory transition phase that MUST run (for its full, fixed
# sq.net.xml duration) immediately after leaving a given stage.
TRANSITION_AFTER_STAGE: Dict[int, int] = {0: 1, 2: 3, 4: 5, 6: 7}

# ============================================================================
# Approach / lane mapping -- derived directly from sq.net.xml's
# tl="0" <connection> elements (see module docstring link-index table).
# ============================================================================

APPROACH_EDGES: Tuple[str, ...] = ("1i", "2i", "3i", "4i")

# GROUND-TRUTH edge -> geographic label, from junction coordinates.
APPROACH_DIRECTION: Dict[str, str] = {
    "4i": "north",
    "3i": "south",
    "1i": "west",
    "2i": "east",
}

# GROUND-TRUTH edge -> stage that serves it. Independently re-derived
# here (see docstring) rather than imported from feature_builder.py,
# specifically because that file's PHASE_GREEN_GROUP was found to
# disagree with this decoding.
APPROACH_STAGE: Dict[str, str] = {
    "3i": "NS",
    "4i": "NS",
    "1i": "EW",
    "2i": "EW",
}

STAGE_APPROACHES: Dict[str, Tuple[str, ...]] = {
    "NS": ("3i", "4i"),
    "EW": ("1i", "2i"),
}

# fromLane index -> movement, identical across all four approaches
# (confirmed directly from every tl="0" connection's fromLane/dir pair).
LANE_MOVEMENT: Dict[int, str] = {0: "right", 1: "through", 2: "left"}
LANES_PER_APPROACH = 3

# ============================================================================
# Green-time bounds -- ASTRID PROJECT ASSUMPTIONS, not sourced from
# sq.net.xml. sq.net.xml's own tlLogic is a STATIC (non-actuated) program
# with a single fixed duration per phase; it defines no min/max range at
# all. To let the controller extend or shorten a green stage (that is
# the entire point of ACTION_KEEP / ACTION_REQUEST_NEXT), *some* bounds
# are required, and these are ASTRID's own choice, not SUMO's:
#
#   MIN green  = sq.net.xml's own phase duration for that phase. i.e. the
#                controller may never end a stage earlier than the
#                original static program would have -- a conservative,
#                clearly-labeled floor, not a value invented from
#                nothing.
#   MAX green  = 2x sq.net.xml's own duration, a simple, documented cap
#                that prevents unbounded starvation of the other stage.
#                Not derived from any traffic-engineering source; purely
#                a prototype safety bound.
# ============================================================================

MIN_GREEN_S: Dict[int, float] = {p.index: p.duration_s for p in PHASES if not p.is_transition}
MAX_GREEN_S: Dict[int, float] = {idx: 2.0 * base for idx, base in MIN_GREEN_S.items()}

# ============================================================================
# Simulation window -- copied verbatim from sq.sumo.cfg.
# ============================================================================

SIMULATION_BEGIN_S = 0
SIMULATION_END_S = 3600

NET_FILE = "sq.net.xml"
ROUTE_FILE = "sq.rou.xml"
SUMO_CONFIG_FILE = "sq.sumo.cfg"


def is_legal_transition(from_index: int, to_index: int) -> bool:
    """True only for sq.net.xml's own single legal successor. Anything
    else (skipping a phase, jumping stages, reordering) is illegal."""
    return to_index == next_phase_index(from_index)