"""Hot-seat human-vs-archetype MVP — pure core (T1) + interactive game layer (T2).

**T1 (pure core, NO terminal I/O)** is the shared contract the rest of the hot-seat
feature builds on. It owns three pure pieces, all of them fully unit-testable with
``numpy`` + ``core.schema`` alone (no prompting, no printing, no env, no pettingzoo, no
file I/O):

1. **The observation-decoder SSOT** (:data:`OBS_INDEX` / :func:`obs_field`): every
   downstream reader looks up a NAMED ``OBSERVATION_SCHEMA`` field through this map
   instead of hard-coding an integer offset. The ~6 "scoreboard" field names a human
   sees are exposed as named constants (:data:`BOARD_RIVAL_FIELDS` + the SELF pair).

2. **The running-impression fold** (:class:`RunningImpression`): a deterministic
   accumulator that folds the PERCEIVED (noised) rival view into a per-field cumulative
   mean across the ticks of one episode. It reads the perceived obs fields ONLY — it
   never references a true / un-noised competitor source (the leakage invariant).

3. **The categorical mapping + confidence** (:func:`categorize_rival_field`,
   :func:`confidence_sigil`, :func:`field_confidence_sigil`): pure functions turning a
   running-mean rival scalar into a LOW / MED / HIGH posture bucket and an
   ``n_ticks``-keyed confidence sigil, PER FIELD (:data:`SOFT_RIVAL_FIELDS` caps
   promotion/share below the settled glyph — their buckets are measured unreliable even
   once fully folded). The renderer (T2) consumes these; T1 does NOT print.

**T2 (interactive game layer)** sits on top of T1 in this same module: the board
renderer, the human input → flat action prompt, the turn-loop driver, the end-game
reveal, and the debrief writer. T2 reuses the ZERO-economics seam (``RetailParallelEnv``
+ ``World.step``) and reimplements NO economics. Its human-interactive functions take
INJECTABLE ``input_fn`` / ``output_fn`` callables (default to stdin/print) so tests (T3)
drive a full game with scripted stdin and no real terminal. The env / pettingzoo import
is LAZY (inside the driver) so ``import retail_simulator.harness.hotseat`` stays cheap
and the ``harness`` import-linter contract (no cli/network) holds.

THE PERCEIVED-RIVAL SEATING (load-bearing): the board reads the rival ONLY through the
observer's PERCEIVED (draw-#4 noised) competitor obs block — but that block is populated
from ``_competitors`` (NPC seats only), so an all-LEARNING duopoly perceives an
all-zero rival. T2 therefore seats the chosen Phase-4 archetype as the SINGLE NPC seat
(seat 1) and drives the human as the lone learning seat (seat 0); the archetype's fixed
posture is wrapped in :class:`_ArchetypeNPCPolicy` so the seam noises its true state into
the human's perceived view each tick. This is what makes the rival LEGIBLE on the
perceived channel (the leakage invariant holds — the board never reads the rival's true
state until the end-game reveal).

Layer note: pure ``core`` reads only (``core.schema``) plus ``numpy`` — no envs, no RL
framework, no network, no terminal at import. The T2 driver lazily imports ``envs`` (the
seam) inside its body. Safe under the ``harness`` import-linter contract.

Naming note (spec vs code): the spec brief labels the rival marketing channel
"marketing(awareness)". The real ``OBSERVATION_SCHEMA`` COMPETITOR-group field is named
``competitor_awareness`` (the SEATED awareness stock the rival's demand reads this tick,
built from prior marketing spend — the lagged-investment model). There is no
``competitor_marketing`` obs field; awareness is the observable proxy for the rival's
marketing posture. This module uses the real field names and documents the mapping in
:data:`BOARD_RIVAL_FIELDS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from retail_simulator.core.schema import OBSERVATION_SCHEMA

# --- Obs-decoder SSOT ---------------------------------------------------------------
#
OBS_INDEX: dict[str, int] = {f.name: i for i, f in enumerate(OBSERVATION_SCHEMA)}


def obs_field(obs: npt.NDArray[np.float32], name: str) -> float:
    """Read one NAMED scalar out of a flat observation vector.

    The decoder SSOT: callers ask for ``"competitor_price"``, not ``obs[9]``. Raises
    ``KeyError`` if ``name`` is not a known schema field (a programming error — fail
    loud) and lets numpy raise on a wrong-length ``obs`` (a contract violation).
    """
    return float(obs[OBS_INDEX[name]])


# The PERCEIVED (noised) rival "scoreboard" fields a human sees in the hot-seat board.
# These are the COMPETITOR-group fields whose running mean identifies the rival posture.
# Each maps a human-facing label to its real ``OBSERVATION_SCHEMA`` field name (see the
# module-docstring naming note — the rival marketing channel is ``competitor_awareness``,
# the seated awareness stock, NOT a non-existent ``competitor_marketing``).
#
BOARD_RIVAL_FIELDS: dict[str, str] = {
    "price": "competitor_price",
    "marketing": "competitor_awareness",
    "promotion": "competitor_promotion",
    "share": "competitor_market_share",
}

# The SELF scoreboard fields a human also sees (own outcome, EXACT — never noised). Read
# through the same SSOT; surfaced by name so T2 never hard-codes offsets 1 / 3.
SELF_LAST_PROFIT_FIELD: str = "last_profit"
SELF_MARKET_SHARE_FIELD: str = "market_share"


# --- Running-impression fold --------------------------------------------------------


@dataclass
class RunningImpression:
    """Cumulative per-field mean of the PERCEIVED rival view across one episode.

    Folds the noised rival scoreboard fields (:data:`BOARD_RIVAL_FIELDS`) into a running
    mean so a human (and the T2 renderer) builds an impression of the rival posture that
    sharpens as evidence accumulates — the whole point of the perceived/noised channel.

    LEAKAGE INVARIANT: :meth:`update` reads ONLY the perceived obs fields named in
    :data:`BOARD_RIVAL_FIELDS`. It never references ``true_competitor_view`` or any
    un-noised source, and it never takes an archetype label as input. The impression is
    earned from noised observation alone.

    RESET-OBS CONTRACT (Lead review-hard item): the perceived view in the RESET
    observation is seeded to the TRUE rival values before any tick runs, so folding it
    would contaminate the impression with un-noised ground truth (a leak). The chosen
    contract is therefore **the caller folds POST-STEP observations only** — never the
    reset obs. This object is a pure accumulator with one job (fold what it is given);
    the "skip the reset obs" responsibility lives with the T2 driver, which folds inside
    its step loop and never on the reset return. :meth:`n_ticks` counts folded ticks, so
    a correctly-driven impression has ``n_ticks == ticks_stepped``.
    """

    # Per-field running sums and the shared fold count. A simple count+sum mean (not
    # Welford) is exact and sufficient: these are bounded, well-scaled scoreboard scalars
    # and we never need an online variance.
    _sums: dict[str, float] = field(default_factory=lambda: dict.fromkeys(BOARD_RIVAL_FIELDS, 0.0))
    _n: int = 0

    @property
    def n_ticks(self) -> int:
        """Number of post-step observations folded since construction/reset."""
        return self._n

    def update(self, obs: npt.NDArray[np.float32]) -> None:
        """Fold one POST-STEP observation's perceived rival values into the means.

        Reads only the :data:`BOARD_RIVAL_FIELDS` perceived fields (leakage invariant).
        Must NOT be called with the reset observation (see the reset-obs contract on the
        class docstring) — doing so would fold the un-noised seeded view.
        """
        for label, field_name in BOARD_RIVAL_FIELDS.items():
            self._sums[label] += obs_field(obs, field_name)
        self._n += 1

    def means(self) -> dict[str, float]:
        """Current per-field cumulative means, keyed by the human-facing label.

        Returns an empty-impression sentinel of all-``nan`` before the first fold
        (``n_ticks == 0``): there is no evidence yet, and ``nan`` makes that explicit
        rather than implying a real 0.0 reading. After the first fold every value is the
        ordinary running mean ``sum / n``.
        """
        if self._n == 0:
            return dict.fromkeys(BOARD_RIVAL_FIELDS, float("nan"))
        return {label: total / self._n for label, total in self._sums.items()}


# --- Categorical mapping + confidence -----------------------------------------------


class PostureBucket(Enum):
    """A coarse categorical posture label for one rival scoreboard field."""

    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


@dataclass(frozen=True)
class FieldThresholds:
    """The LOW|MED|HIGH cut points for one rival field.

    ``low_max`` is the inclusive top of the LOW band and ``high_min`` the inclusive
    bottom of the HIGH band; the open interval between them is MED. The cuts are MIDPOINTS
    between the three archetypes' true postures on that field so each archetype lands in a
    distinct bucket (see :data:`RIVAL_FIELD_THRESHOLDS`).
    """

    low_max: float
    high_min: float


# Per-field LOW|MED|HIGH cut points, chosen so the three Phase-4 archetypes land in
# DISTINCT buckets on each field. TUNED ON REAL PERCEIVED ROLLOUT OBS (T2 threshold
# validation), NOT on the raw archetype lever magnitudes.
#
# WHY THE TUNE (load-bearing): the hot-seat board reads the rival through the PERCEIVED
# (draw-#4 noised, stock-based) competitor obs block — the SAME channel a human sees in
# play — not the raw action's levers. The marketing channel especially is the SEATED
# ``competitor_awareness`` STOCK, which COMPOUNDS over ticks from the rival's marketing
# spend (the lagged-investment model): AGGRESSOR's 0.9 spend accumulates to a perceived
# running mean ~2.4, far above the 0.65 cut the old raw-lever thresholds assumed. The
# original constants were set on raw lever values (T1) and did NOT separate the three
# postures on perceived obs (all three collapsed to the same buckets — the marketing
# stock and price both landed outside their intended bands). These cuts restore the
# intended posture mapping ON THE ACTUAL OBSERVABLE.
#
#
# Measured running-impression means (10 seeds × 50 ticks — the GAME length, retailer_0 =
# hold-course defaults, retailer_1 = the calibrated archetype seated as the perceivable NPC
# rival, i.e. the T2 driver's exact configuration; see
# ``test_archetype_lands_in_intended_bucket_on_real_rollout``). Previous (partial-rival)
# means in brackets:
#
#   field        AGGRESSOR       PATIENT         LEAN            -> intended buckets
#   price          0.826 [0.854]   1.009 [1.044]   1.500 [1.538]    LOW  / MED / HIGH
#   marketing      2.395 [2.651]   1.243 [1.195]   0.214 [0.217]    HIGH / MED / LOW
#   promotion      0.660 [0.659]   0.244 [0.247]   0.124 [0.132]    HIGH / MED / LOW
#   share          0.810 [0.711]   0.703 [0.596]   0.400 [0.387]    HIGH / MED / LOW
#
# **SHARE is the channel that moved, and it is why this re-derivation was needed.** A rival
# playing assortment/promotion/service_level for real takes far more share off a
# hold-course human, and PATIENT gained most (0.596 → 0.703) — straight over the old
# ``high_min=0.654``, which read the measured re-investor as a max-growth AGGRESSOR. The
# re-derived cut restores its intended MED, so no archetype's INTENDED bucket is
# re-pointed: the map was right and the ruler had drifted.
#
# Each cut is still the MIDPOINT between the two adjacent archetype means. What that rule
# now buys differs sharply per channel, and the difference is itself the honest reading:
#
#
#
# AGGRESSOR reads uniformly HIGH (max-growth: cheap price, heavy marketing/promotion,
# winning share); LEAN reads HIGH price + LOW everything-else (defend-margin cost
# minimizer); PATIENT reads uniformly MED (the measured re-investor between them).
RIVAL_FIELD_THRESHOLDS: dict[str, FieldThresholds] = {
    "price": FieldThresholds(low_max=0.918, high_min=1.255),
    "marketing": FieldThresholds(low_max=0.728, high_min=1.819),
    "promotion": FieldThresholds(low_max=0.184, high_min=0.452),
    "share": FieldThresholds(low_max=0.551, high_min=0.756),
}


def categorize_rival_field(label: str, value: float) -> PostureBucket:
    """Map a running-mean rival scalar to a LOW | MED | HIGH posture bucket.

    ``label`` is one of :data:`BOARD_RIVAL_FIELDS` (``"price"``/``"marketing"``/
    ``"promotion"``/``"share"``); ``value`` is the field's current running mean. Uses the
    per-field midpoint cuts in :data:`RIVAL_FIELD_THRESHOLDS`. ``nan`` (the empty
    impression) maps to MED — a neutral "no evidence yet" posture rather than a spurious
    extreme. Raises ``KeyError`` for an unknown label (a programming error — fail loud).
    """
    cuts = RIVAL_FIELD_THRESHOLDS[label]
    if np.isnan(value):
        return PostureBucket.MED
    if value <= cuts.low_max:
        return PostureBucket.LOW
    if value >= cuts.high_min:
        return PostureBucket.HIGH
    return PostureBucket.MED


# Confidence sigils keyed to how many ticks of evidence the impression has folded. The
# perceived channel is noised, so a single tick is weak evidence; the sigil tells the
# human how settled the running mean is. Thresholds (per the T1 spec): forming below 2
# folds, tentative through 4, settled at 5+.
CONFIDENCE_FORMING: str = "??"
CONFIDENCE_TENTATIVE: str = "~"
CONFIDENCE_SETTLED: str = ">>"


def confidence_sigil(n_ticks: int) -> str:
    """Map a fold count to a confidence sigil: ``??`` (n<2) / ``~`` (2<=n<5) / ``>>`` (n>=5).

    The renderer (T2) prefixes the categorical posture with this so a human knows how
    much to trust it. Negative ``n_ticks`` is a programming error (counts never go
    negative); it is treated as forming rather than raising, since this is display-only.

    Unchanged by the per-field cap below (:func:`field_confidence_sigil`) — this stays
    the field-agnostic schedule :func:`field_confidence_sigil` builds on.
    """
    if n_ticks < 2:
        return CONFIDENCE_FORMING
    if n_ticks < 5:
        return CONFIDENCE_TENTATIVE
    return CONFIDENCE_SETTLED


SOFT_RIVAL_FIELDS: frozenset[str] = frozenset({"promotion", "share"})


def field_confidence_sigil(field: str, n_ticks: int) -> str:
    """Map a fold count to a confidence sigil, PER RIVAL FIELD (:data:`BOARD_RIVAL_FIELDS`
    keys).

    Identical to :func:`confidence_sigil` for every field except :data:`SOFT_RIVAL_FIELDS`
    (promotion, share): those never report more than :data:`CONFIDENCE_TENTATIVE`, because
    their bucket reads are the ones measured to still be wrong often enough that a settled
    `>>` would over-claim (see :data:`SOFT_RIVAL_FIELDS`'s comment for the numbers). Raises
    ``KeyError`` for an unknown label (a programming error — fail loud, mirrors
    :func:`categorize_rival_field`) rather than silently falling through to the uncapped
    schedule, which is exactly the pre-fix defect for any renamed/added channel.
    """
    if field not in BOARD_RIVAL_FIELDS:
        raise KeyError(field)
    sigil = confidence_sigil(n_ticks)
    if field in SOFT_RIVAL_FIELDS and sigil == CONFIDENCE_SETTLED:
        return CONFIDENCE_TENTATIVE
    return sigil


# ====================================================================================
# T2 — interactive game layer (board + prompt + driver + debrief). Reuses the seam;
# reimplements ZERO economics. Imports of the env / pettingzoo are LAZY (inside the
# driver) so importing this module stays cheap and harness-contract-safe.
# ====================================================================================

import json as _json  # noqa: E402 — kept local to T2; T1 needs neither json nor typing
from collections.abc import Callable, Mapping  # noqa: E402
from itertools import zip_longest  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

from retail_simulator.core.encoding import (  # noqa: E402
    StructuredAction,
    decode_action,
    encode_action,
)
from retail_simulator.core.schema import LEVER_REGISTRY, LeverKind  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover — typing only, never imported at runtime
    import numpy.typing as _npt


PLAYER_LEVERS: tuple[str, ...] = (
    "price_index",
    "marketing",
    "assortment",
    "service_level",
    "expansion",
    "promotion",
    "research",
    "wage_spend",
    "warehouse_invest",
)

# The full-registry default lever values (the "hold course" baseline). Every continuous
# lever except price_index/service_level defaults to 0.0 (see LEVER_REGISTRY), so a
# hold-course turn leaves assortment/research at 0.0 exactly as it always has.
_REGISTRY_DEFAULTS: dict[str, float] = {spec.name: spec.default for spec in LEVER_REGISTRY}

# The fixed canonical playstyle labels (mirrors spike_archetypes; duplicated as plain
# strings to avoid importing the archetype module at T1 import time — the driver imports
# the closures lazily). The intended counter-cycle is LEAN ≻ AGGRESSOR ≻ PATIENT ≻ LEAN.
PLAYSTYLES: tuple[str, ...] = ("lean", "aggressor", "patient")

COUNTERED_BY: dict[str, str] = {
    "aggressor": "lean",
    "patient": "aggressor",
    "lean": "patient",
}

# A one-line "why" for each counter edge — the teaching feedback shown at the reveal.
_COUNTER_RATIONALE: dict[str, str] = {
    "aggressor": (
        "LEAN counters AGGRESSOR: the aggressor over-extends on cash-hungry growth "
        "(expansion + heavy marketing/promotion), and the lean defender's fat margin "
        "out-survives the burn."
    ),
    "patient": (
        "AGGRESSOR counters PATIENT: the patient re-investor builds too slowly, so the aggressor's early aggressive expansion takes share the patient never recovers."
    ),
    "lean": (
        "PATIENT counters LEAN: the lean defender deliberately under-invests, and the "
        "patient re-investor compounds past it over the full horizon."
    ),
}


# --- Per-turn record (the debrief row) ----------------------------------------------


@dataclass
class TurnRecord:
    """One tick's logged state — the unit the debrief writer appends (DETERMINISTIC).

    Captures exactly what a replay needs: the human's nine chosen lever values, the raw
    perceived-rival reading this tick AND the running mean to that point, the human's own
    outcome (market_share + last_profit), and the expansion ``action_mask`` the human
    faced (so an illegal-move rejection is reproducible). NO wall-clock — the record is a
    pure function of (seed, archetype, the logged human inputs).
    """

    tick: int
    human_levers: dict[str, float]
    perceived_rival_raw: dict[str, float]
    perceived_rival_mean: dict[str, float]
    own_market_share: float
    own_last_profit: float
    expansion_mask: list[bool]

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON view (lists/floats only) for the ``.jsonl`` writer."""
        return {
            "tick": self.tick,
            "human_levers": {k: float(v) for k, v in self.human_levers.items()},
            "perceived_rival_raw": {k: float(v) for k, v in self.perceived_rival_raw.items()},
            "perceived_rival_mean": {k: float(v) for k, v in self.perceived_rival_mean.items()},
            "own_market_share": float(self.own_market_share),
            "own_last_profit": float(self.own_last_profit),
            "expansion_mask": [bool(b) for b in self.expansion_mask],
        }


@dataclass
class GameResult:
    """The full record of one finished hot-seat game — the debrief writer's input.

    Holds everything needed to reproduce the game from ``seed`` + the logged human
    inputs: the actual archetype (revealed only here), the per-tick records, and the
    end-game outcome (the human's guess, their counter guess, whether the guess was
    correct, and whether they won). DETERMINISTIC by construction (no timestamp).

    B1 (senior-review blocker): ``won`` is decided on the CYCLE'S metric — the cumulative
    stationary ``weighted_objective`` (``own_final_score`` vs ``rival_final_score``) — so
    the reveal AGREES with the counter teaching (the proven cycle is defined on this
    score, not on cash). ``own_final_cash``/``rival_final_cash`` are kept as SECONDARY
    recorded fields only (the binding Phase-4 resource, useful in the debrief) but are NOT
    the verdict.
    """

    seed: int
    n_ticks: int
    archetype_actual: str
    turns: list[TurnRecord]
    guess: str
    counter_guess: str
    correct: bool
    won: bool
    # The VERDICT metric (B1): cumulative weighted_objective per seat over the episode.
    own_final_score: float
    rival_final_score: float
    # SECONDARY (recorded, not the verdict): final seated cash per seat.
    own_final_cash: float
    rival_final_cash: float

    def to_dict(self) -> dict[str, Any]:
        """The single JSON object appended as one ``.jsonl`` line. NO wall-clock."""
        return {
            "seed": self.seed,
            "n_ticks": self.n_ticks,
            "archetype_actual": self.archetype_actual,
            "turns": [t.to_dict() for t in self.turns],
            "final": {
                "guess": self.guess,
                "counter_guess": self.counter_guess,
                "archetype_actual": self.archetype_actual,
                "correct": self.correct,
                "won": self.won,
                # The verdict metric (B1) and the secondary cash record.
                "own_final_score": float(self.own_final_score),
                "rival_final_score": float(self.rival_final_score),
                "own_final_cash": float(self.own_final_cash),
                "rival_final_cash": float(self.rival_final_cash),
            },
        }


# --- The perceivable-rival NPC adapter ----------------------------------------------


class _ArchetypeNPCPolicy:
    """Wrap a Phase-4 archetype ``predict`` closure as an :class:`NPCPolicy`-shaped seat.

    The hot-seat rival MUST be an NPC seat so the seam noises its true state into the
    human's PERCEIVED competitor obs block (``_competitors`` reads NPC seats only — an
    all-learning duopoly perceives an all-zero rival; see the module docstring). The
    Phase-4 archetypes are OPEN-LOOP fixed-posture closures, so this adapter ignores the
    obs and returns the archetype's frozen action each tick. Deterministic (no rng draw),
    matching the scripted core archetypes' draw-budget contract.
    """

    def __init__(self, predict: Callable[..., "_npt.NDArray[np.float32]"]) -> None:
        self._predict = predict

    def act(self, state: object, rng: object, seat_index: int = 1) -> "_npt.NDArray[np.float32]":
        """Return the archetype's fixed open-loop action (ignores state/rng/seat)."""
        del state, rng, seat_index
        return self._predict(None, episode_start=False)


# --- Human input → flat action ------------------------------------------------------


def hold_course_action() -> "_npt.NDArray[np.float32]":
    """The flat action that holds every lever at its registry default.

    The "hit Enter on everything" baseline — and the byte-identity self posture used by
    the threshold-validation rollout (hold-course self vs the archetype rival).
    """
    return encode_action(StructuredAction(levers=dict(_REGISTRY_DEFAULTS)))


def _parse_lever_input(raw: str, *, lever: str, default: float) -> tuple[bool, float, str]:
    """Validate one prompted lever value. Returns ``(ok, value, error_message)``.

    Empty input ⇒ accept the ``default`` ("hold course"). Otherwise parse a float and
    bound-check against the lever's schema range (continuous) or the discrete choice set
    (``expansion`` ∈ {0, 1}). On a parse / range error returns ``ok=False`` with a clear
    inline message so the caller re-prompts (never submits an out-of-range value).
    """
    text = raw.strip()
    if text == "":
        return True, default, ""
    try:
        value = float(text)
    except ValueError:
        return False, default, f"  ! {lever}: '{raw.strip()}' is not a number — try again."

    spec = next(s for s in LEVER_REGISTRY if s.name == lever)
    if spec.kind is LeverKind.CONTINUOUS:
        assert spec.low is not None and spec.high is not None
        if not (spec.low <= value <= spec.high):
            return (
                False,
                default,
                f"  ! {lever}: {value:g} is outside [{spec.low:g}, {spec.high:g}] — try again.",
            )
        return True, value, ""
    # Discrete lever (expansion): an integer choice in [0, n). ``spec.n`` is set for
    # discrete levers (None only for continuous, handled above).
    assert spec.n is not None
    choice = int(value)
    if choice != value or not (0 <= choice < spec.n):
        return (
            False,
            default,
            f"  ! {lever}: choice must be an integer in [0, {spec.n}) — try again.",
        )
    return True, float(choice), ""


def prompt_human_action(
    obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    info: Mapping[str, Any],
    *,
    tick: int,
    n_ticks: int,
    n_regions: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prior_profit: float | None = None,
) -> "_npt.NDArray[np.float32]":
    """Render the board, prompt the nine player levers (defaults pre-filled), build action.

    Renders the two-panel board (own snapshot + rival impression) via ``output_fn``, then
    collects + validates the nine :data:`PLAYER_LEVERS` (see :func:`_collect_levers` for
    the rules: Enter holds the default, bad input re-prompts, an out-of-world expansion
    choice or a cash-gated one is rejected — ``n_regions`` is this world's REAL region
    count, schema v12). The two non-exposed levers (automation, loyalty_spend) are pinned
    to their registry defaults. Returns the flat 44-D action.

    ``input_fn`` / ``output_fn`` are injected (default stdin/print at the driver) so T3
    can script a whole game without a real terminal.
    """
    output_fn(
        render_board_with_delta(
            obs, impression, tick=tick, n_ticks=n_ticks, prior_profit=prior_profit
        )
    )
    return _collect_levers(info, n_regions=n_regions, input_fn=input_fn, output_fn=output_fn)


def _collect_levers(
    info: Mapping[str, Any],
    *,
    n_regions: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> "_npt.NDArray[np.float32]":
    """Prompt + validate the nine player levers (board already rendered) → flat action.

    Empty input holds the lever's registry default ("hold course"); a non-numeric /
    out-of-range value re-prompts that lever inline. The EXPANSION lever is additionally
    checked in TWO layers, mirroring the web platform's ``web/session.py`` /
    ``_validate_levers`` (schema v12, senior-review M1): ``info['action_mask']
    ['expansion']`` is the seam's FIXED-CAPACITY (``N_REGIONS_MAX`` = 32-wide) array, so
    it is trimmed to ``n_regions`` (this world's REAL region count) FIRST; a choice
    beyond that trimmed width is a region that does not exist in this world at all
    (rejected with its own distinct message — never an ``IndexError``, never conflated
    with the cash-gate wording); an in-bounds but masked-False choice is cash-gated as
    before. The two non-exposed levers (automation, loyalty_spend) stay at their
    registry defaults (the byte-identity baseline).
    """
    expansion_mask = list(info["action_mask"]["expansion"])[:n_regions]
    chosen: dict[str, float] = dict(_REGISTRY_DEFAULTS)
    for lever in PLAYER_LEVERS:
        default = _REGISTRY_DEFAULTS[lever]
        while True:
            raw = input_fn(f"  {lever} [{default:g}]: ")
            ok, value, error = _parse_lever_input(raw, lever=lever, default=default)
            if not ok:
                output_fn(error)
                continue
            if lever == "expansion":
                choice = int(value)
                if choice >= len(expansion_mask):
                    output_fn(
                        f"  ! expansion: region {choice} does not exist in this world "
                        f"(it has {len(expansion_mask)} region(s)) — pick an allowed "
                        "choice."
                    )
                    continue
                if not expansion_mask[choice]:
                    output_fn(
                        f"  ! expansion: choice {choice} is cash-gated (the seam forbids "
                        "it this tick) — pick an allowed choice."
                    )
                    continue
            chosen[lever] = value
            break
    return encode_action(StructuredAction(levers=chosen))


# --- Board renderer ------------------------------------------------------------------

_BOARD_WIDTH: int = 78
# The own-panel label column width. Widened in TIER-B-3 to fit "staff happiness" (the
# longest own-panel label) while keeping every row — old and new — in the same
# left-padded-label-then-colon pattern, so the column stays aligned across all rows.
_OWN_LABEL_WIDTH: int = 16


def _fmt_delta(delta: float) -> str:
    """A compact up/down/flat trend marker for the profit turn-over-turn delta."""
    if delta > 0:
        return f"^ +{delta:,.0f}"
    if delta < 0:
        return f"v {delta:,.0f}"
    return "= flat"


def _rival_presence_label(obs: "_npt.NDArray[np.float32]") -> str:
    """Whether the rival is present per region, DERIVED from perceived share > 0.

    There is no competitor-presence obs field; the perceived per-region competitor
    market share is the only proxy. A share > 0 in a region means the rival is selling
    there. Reads ONLY perceived obs (leakage invariant).
    """
    share_0 = obs_field(obs, "competitor_market_share")
    share_1 = obs_field(obs, "competitor_market_share_region_1")
    regions = [str(r) for r, share in enumerate((share_0, share_1)) if share > 0.0]
    return "+".join(f"R{r}" for r in regions) if regions else "(none seen)"


def render_board(
    obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    *,
    tick: int,
    n_ticks: int,
) -> str:
    """Two-panel ~80-col board with no profit trend — see :func:`render_board_with_delta`."""
    return render_board_with_delta(obs, impression, tick=tick, n_ticks=n_ticks, prior_profit=None)


def render_board_with_delta(
    obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    *,
    tick: int,
    n_ticks: int,
    prior_profit: float | None,
) -> str:
    """Two-panel ~80-col board: LEFT own snapshot, RIGHT rival running-impression.

    LEFT reads the human's OWN obs (exact, never noised): market_share, last_profit (with
    a turn-over-turn trend when ``prior_profit`` is known), cash, stores, and the current
    research/intel fidelity (:data:`~retail_simulator.core.schema` ``research_fidelity``,
    the SELF field the research lever drives — "how clearly you currently see the rival").
    RIGHT reads ONLY the running impression of the PERCEIVED rival: each of the four
    categorical posture buckets (:func:`categorize_rival_field` on
    :meth:`RunningImpression.means`), prefixed by the :func:`field_confidence_sigil` for
    that field and the impression's tick count (promotion/share cap at tentative — see
    :data:`SOFT_RIVAL_FIELDS`), plus a presence read DERIVED from perceived per-region
    share.
    LEAKAGE INVARIANT: never reads ``true_competitor_view`` and never prints the archetype
    label.

    TIER-B-3: LEFT also carries the four Phase 5.0 wage/warehouse readouts (own state,
    EXACT — never noised, like everything else on this panel): ``staff happiness``
    (``employee_happiness``, the write-off risk gauge), ``staff pay``
    (``wage_level``, last tick's paid wage), ``warehouse used``
    (``warehouse_utilization``, 1.0 = the cap bound last tick), and ``lost sales``
    (``last_stockout_rate``, demand-weighted fraction lost to supplier/warehouse
    squeeze). These pair with the two new :data:`PLAYER_LEVERS` (``wage_spend``,
    ``warehouse_invest``) the way ``fidelity`` pairs with ``research``.
    """
    own_share = obs_field(obs, SELF_MARKET_SHARE_FIELD)
    own_profit = obs_field(obs, SELF_LAST_PROFIT_FIELD)
    own_cash = obs_field(obs, "cash")
    own_stores = obs_field(obs, "stores")
    own_fidelity = obs_field(obs, "research_fidelity")
    own_happiness = obs_field(obs, "employee_happiness")
    own_wage_level = obs_field(obs, "wage_level")
    own_warehouse_used = obs_field(obs, "warehouse_utilization")
    own_lost_sales = obs_field(obs, "last_stockout_rate")
    trend = "" if prior_profit is None else f"  ({_fmt_delta(own_profit - prior_profit)})"

    means = impression.means()
    rival_rows: list[tuple[str, str]] = []
    for label in BOARD_RIVAL_FIELDS:
        sigil = field_confidence_sigil(label, impression.n_ticks)
        bucket = categorize_rival_field(label, means[label]).value
        rival_rows.append((label, f"{sigil} {bucket}"))

    left = [
        "YOU (retailer_0)",
        f"  {'market share':<{_OWN_LABEL_WIDTH}}: {own_share:6.1%}",
        f"  {'last profit':<{_OWN_LABEL_WIDTH}}: {own_profit:>10,.0f}{trend}",
        f"  {'cash':<{_OWN_LABEL_WIDTH}}: {own_cash:>10,.0f}",
        f"  {'stores':<{_OWN_LABEL_WIDTH}}: {int(own_stores):>10d}",
        f"  {'fidelity':<{_OWN_LABEL_WIDTH}}: {own_fidelity:>10.0%}",
        f"  {'staff happiness':<{_OWN_LABEL_WIDTH}}: {own_happiness:>10.0%}",
        f"  {'staff pay':<{_OWN_LABEL_WIDTH}}: {own_wage_level:>10,.0f}",
        f"  {'warehouse used':<{_OWN_LABEL_WIDTH}}: {own_warehouse_used:>10.0%}",
        f"  {'lost sales':<{_OWN_LABEL_WIDTH}}: {own_lost_sales:>10.0%}",
    ]
    right = [
        "RIVAL (read of perceived signal)",
        *[f"  {label:<10}: {posture}" for label, posture in rival_rows],
        f"  presence  : {_rival_presence_label(obs)}",
    ]

    half = _BOARD_WIDTH // 2
    header = f"== TICK {tick}/{n_ticks} ".ljust(_BOARD_WIDTH, "=")
    body_lines = []
    for left_line, right_line in zip_longest(left, right, fillvalue=""):
        body_lines.append(f"{left_line:<{half}}{right_line}")
    return "\n".join([header, *body_lines, "=" * _BOARD_WIDTH])


# --- Turn-loop driver ----------------------------------------------------------------


def _pick_hidden_archetype(seed: int) -> str:
    """Pick the HIDDEN opponent playstyle from a SEEDED RNG (reproducible per seed)."""
    rng = np.random.default_rng(seed)
    return str(rng.choice(PLAYSTYLES))


def print_rps_primer(output_fn: Callable[[str], None]) -> None:
    """Teach the three playstyles + the abstract counter-cycle (NO opponent reference).

    The first-time primer the CLI shows before the game starts: it teaches what-beats-what
    ABSTRACTLY (no reference to the current opponent), so the human learns the cycle
    without being told which archetype they face.
    """
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "HOT-SEAT — know your opponents (rock / paper / scissors)".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                "Three playstyles, each beaten by exactly one other:",
                "",
                "  LEAN      defend margin: high price, minimal spend, no growth.",
                "  AGGRESSOR max growth: cheap price, heavy marketing/promotion, expand fast.",
                "  PATIENT   measured re-investor: moderate everything, build steadily.",
                "",
                "  LEAN beats AGGRESSOR   (fat margin out-survives the burn)",
                "  AGGRESSOR beats PATIENT (early aggressive expansion takes share)",
                "  PATIENT beats LEAN      (compounds past the under-investor)",
                "",
                "Watch the RIVAL panel, infer the playstyle, and play its counter.",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


def play_hotseat_game(
    *,
    config: Any,
    seed: int,
    n_ticks: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    archetype: str | None = None,
    action_fn: Callable[[Mapping[str, Any]], "_npt.NDArray[np.float32]"] | None = None,
) -> GameResult:
    """ """
    from retail_simulator.core.reward import weighted_objective
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.parallel_gate import _components_from_info
    from retail_simulator.harness.spike_archetypes import calibrated_archetypes

    chosen_archetype = archetype if archetype is not None else _pick_hidden_archetype(seed)
    if chosen_archetype not in PLAYSTYLES:
        raise ValueError(f"unknown archetype {chosen_archetype!r}; known: {PLAYSTYLES}")
    predict = calibrated_archetypes()[chosen_archetype]

    # Seat the human as the lone LEARNING seat (0) and the archetype as the NPC rival
    # (1). The placeholder archetype name only shapes the seat plan; the policy object is
    # replaced so the rival plays the Phase-4 posture (perceivable via the seam's noise).
    env = RetailParallelEnv(config, n_learning_agents=1, npc_archetypes=("discounter",))
    env._world._npc_policies[1] = _ArchetypeNPCPolicy(predict)  # noqa: SLF001 — documented seam

    observations, infos = env.reset(seed=seed)
    # Schema v12 (senior-review M1): the seam's `action_mask['expansion']` is
    # fixed-capacity (N_REGIONS_MAX=32-wide, `core.schema`), not sized to this world's
    # actual region count — mirrors `web/session.py`'s own read. Read the REAL count
    # once (fixed for the life of a game) so the human prompt's bound-check and the
    # per-turn debrief record both trim down to it, exactly like the web platform.
    reset_state = env._state  # noqa: SLF001 — documented seam (mirrors web/session.py)
    assert reset_state is not None
    n_regions = len(reset_state.regions)
    impression = RunningImpression()
    turns: list[TurnRecord] = []
    prior_profit: float | None = None
    reward_cfg = config.reward
    own_score = 0.0
    rival_score = 0.0

    for tick in range(1, n_ticks + 1):
        obs = observations["retailer_0"]
        info = infos["retailer_0"]
        if action_fn is None:
            # Interactive path: render the board + prompt the human's nine levers.
            action = prompt_human_action(
                obs,
                impression,
                info,
                tick=tick,
                n_ticks=n_ticks,
                n_regions=n_regions,
                input_fn=input_fn,
                output_fn=output_fn,
                prior_profit=prior_profit,
            )
        else:
            # Scripted-seat path (tests / a future agent): ``action_fn`` supplies seat-0's
            # full flat action from the seat's ``info`` (e.g. an archetype's ``predict``),
            # bypassing the prompt. The board is NOT rendered (no human to read it). This
            # is the seam that drives seat-0 with a fixed posture for the verdict-agrees
            # tests — the same A-vs-B geometry Phase-1 measured.
            action = action_fn(info)
        prior_profit = obs_field(obs, SELF_LAST_PROFIT_FIELD)

        observations, _rewards, _term, _trunc, infos = env.step({"retailer_0": action})

        # Score BOTH seats on this tick's stationary weighted_objective (B1). Seat 0
        # (human) from its public info; seat 1 (NPC rival) from the seam's raw stash —
        # the canonical ``reward_components(...).as_dict()`` for the NPC seat, identical
        # in shape to the human's and to what the cycle was measured on.
        own_score += weighted_objective(_components_from_info(infos["retailer_0"]), reward_cfg)
        raw_infos = env._last_raw_infos  # noqa: SLF001 — documented NPC-score seam
        assert raw_infos is not None  # set by env.step we just called
        rival_score += weighted_objective(_components_from_info(raw_infos[1]), reward_cfg)

        # Fold the POST-STEP obs ONLY (the T1 reset-obs contract — never the reset obs).
        post_obs = observations["retailer_0"]
        impression.update(post_obs)
        turns.append(
            _capture_turn(
                tick, action, post_obs, impression, info["action_mask"]["expansion"][:n_regions]
            )
        )

    guess, counter_guess = _prompt_end_game_guess(input_fn=input_fn, output_fn=output_fn)
    # End-game reveal only (NOT mid-game): read both seats' final seated cash from the
    # env's current WorldState — kept as a SECONDARY recorded field only (the debrief
    # records it; it is NOT the verdict). The VERDICT is the cumulative weighted_objective
    # accumulated above (B1).
    final_state = env._state
    assert final_state is not None  # n_ticks >= 1 ⇒ at least one step ran after reset
    own_cash = float(final_state.retailers[0].cash)
    rival_cash = float(final_state.retailers[1].cash)
    won = own_score > rival_score
    correct = guess == chosen_archetype
    _print_reveal(
        chosen_archetype,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_score=own_score,
        rival_score=rival_score,
        own_cash=own_cash,
        rival_cash=rival_cash,
        output_fn=output_fn,
    )
    return GameResult(
        seed=seed,
        n_ticks=n_ticks,
        archetype_actual=chosen_archetype,
        turns=turns,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_final_score=own_score,
        rival_final_score=rival_score,
        own_final_cash=own_cash,
        rival_final_cash=rival_cash,
    )


def _capture_turn(
    tick: int,
    action: "_npt.NDArray[np.float32]",
    post_obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    expansion_mask: Any,
) -> TurnRecord:
    """Build the per-tick debrief record from the action + post-step obs + impression."""
    levers = decode_action(action).levers
    human_levers = {lever: float(levers[lever]) for lever in PLAYER_LEVERS}
    raw = {
        label: obs_field(post_obs, field_name) for label, field_name in BOARD_RIVAL_FIELDS.items()
    }
    return TurnRecord(
        tick=tick,
        human_levers=human_levers,
        perceived_rival_raw=raw,
        perceived_rival_mean=impression.means(),
        own_market_share=obs_field(post_obs, SELF_MARKET_SHARE_FIELD),
        own_last_profit=obs_field(post_obs, SELF_LAST_PROFIT_FIELD),
        expansion_mask=[bool(b) for b in expansion_mask],
    )


def _prompt_playstyle(
    prompt: str, *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> str:
    """Prompt for one playstyle label until a valid one (or empty → first) is given."""
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw == "":
            return PLAYSTYLES[0]
        if raw in PLAYSTYLES:
            return raw
        output_fn(f"  ! pick one of {', '.join(PLAYSTYLES)} — try again.")


def _prompt_end_game_guess(
    *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> tuple[str, str]:
    """Prompt the human for (opponent-playstyle guess, the playstyle they think counters it)."""
    output_fn("\nGame over. Time to call it:")
    guess = _prompt_playstyle(
        f"  Which playstyle did you face? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    counter_guess = _prompt_playstyle(
        f"  Which playstyle counters it? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    return guess, counter_guess


def _print_reveal(
    archetype_actual: str,
    *,
    guess: str,
    counter_guess: str,
    correct: bool,
    won: bool,
    own_score: float,
    rival_score: float,
    own_cash: float,
    rival_cash: float,
    output_fn: Callable[[str], None],
) -> None:
    """Reveal the true archetype + win/lose + the teaching feedback (the counter rationale).

    B1: the verdict line reports the CYCLE'S score (cumulative ``weighted_objective``) —
    the metric the win/lose is decided on, so the result agrees with the counter teaching.
    Final cash is shown as a secondary line only.
    """
    true_counter = COUNTERED_BY[archetype_actual]
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "REVEAL".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                f"  You faced       : {archetype_actual.upper()}",
                f"  Your guess      : {guess.upper()}  ({'correct' if correct else 'wrong'})",
                f"  It is countered : {true_counter.upper()}  "
                f"(you said {counter_guess.upper()}: "
                f"{'right' if counter_guess == true_counter else 'not quite'})",
                f"  Score           : you {own_score:,.1f}  vs  rival {rival_score:,.1f}",
                f"  (final cash      : you {own_cash:,.0f}  vs  rival {rival_cash:,.0f})",
                f"  Result          : {'YOU WIN' if won else 'YOU LOSE'}",
                "",
                f"  {_COUNTER_RATIONALE[archetype_actual]}",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


# --- Debrief writer ------------------------------------------------------------------


def append_debrief(result: GameResult, path: Path) -> None:
    """Append the game record as ONE ``.jsonl`` line (deterministic — no wall-clock).

    Local file only (no network/db). The line is fully reproducible from ``seed`` + the
    logged human inputs: re-running the same seed with the same scripted inputs writes a
    byte-identical line. ``json.dumps`` with ``sort_keys`` makes the key order stable too.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(result.to_dict(), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
