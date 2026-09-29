"""Four CLOSED-LOOP geographic archetypes stressing the arena's geography axis (S5).

Every other lever (price/marketing/assortment/promotion/research/service_level/
automation/loyalty_spend) is a FIXED per-archetype constant, exactly like the spike
archetypes — only the ``expansion`` target is closed-loop. This keeps each posture
simple and auditable: what varies tick-to-tick is entirely the geography-targeting rule,
never the demand/marketing posture.

**Region 0 is never a candidate, by construction, in every rule below** (the arena's
deliberately STRANDED slot — ``harness/arena.py``'s module docstring; the expansion
lever's permanent hold/no-op index). Every candidate scan below starts at region 1 —
there is no special-case "skip 0" branch to audit because 0 is simply never enumerated.

**Numeric lever choices are borrowed from already-vetted values in
:mod:`retail_simulator.harness.spike_archetypes`** (the codebase's own established
"low/mid/premium price", "minimal/mid/max marketing-assortment-loyalty", and
"deep-squeezer/squeezer/partner service_level" anchors) rather than invented afresh —
this trivially satisfies "every lever stays within its :class:`~retail_simulator.core.
schema.LeverSpec` bounds" (the source values are already proven in-bounds) and keeps the
four postures' non-geographic axes comparable in *scale* to the calibrated 2-region
archetypes. Every lever the contract table above does not name (assortment/promotion/
loyalty_spend for CHERRY-PICKER and FAST-FOLLOWER) is held at PATIENT's calibrated
"moderate" seed value; ``research`` stays at 0.0 for all four (none of these rules read
the noised COMPETITOR block, so paying for perception fidelity would be pure wasted
opex) and ``automation`` stays at tier 0 (the registry no-op default) since no archetype
in this family is differentiated on that axis.

Layer note: pure ``core`` reads only (encoding/schema/config) — no envs, no RL
framework, no network. Safe under the ``harness`` import-linter contract, mirroring
:mod:`retail_simulator.harness.spike_archetypes`.

**Ladder/league allowlist extension (RLB-8b — done).** ``harness.league.
POOL_ARCHETYPE_LABELS`` and ``harness.ladder.LADDER_ARCHETYPE_LABELS`` are the union
of this module's four labels and their respective spike-only family
(``POOL_SPIKE_ARCHETYPE_LABELS``/``LADDER_SPIKE_ARCHETYPE_LABELS``), importing
:data:`GEO_ARCHETYPE_LABELS` rather than re-listing it — so ``--participants
archetype:<name>``/``--seed-archetypes`` can seat these four in the ladder/league CLI
and the ``phase4_rl_confirm.py`` driver, config-aware materialization (a region table
threaded through ``_materialize_predicts``/``FrozenPolicyPool`` and required whenever a
geo name is actually used) included.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import RegionConfig
from retail_simulator.core.encoding import StructuredAction, encode_action
from retail_simulator.core.schema import N_AUTOMATION_TIERS, observation_schema

# Canonical archetype labels (the tournament node ids for scripts/phase6_geo_tournament.py).
SPRAWLER: str = "sprawler"
FORTRESS: str = "fortress"
CHERRY_PICKER: str = "cherry-picker"
FAST_FOLLOWER: str = "fast-follower"

# The four labels in :func:`geo_archetypes`'s OWN dict-literal (insertion) order — the
# single source of truth for "what archetypes exist, in what canonical order", so a
# caller needing just the labels (e.g. ``scripts/phase6_geo_tournament.py``) can import
# this constant instead of re-hardcoding (and risking drifting) its own copy of the
# 4-tuple. Deliberately a STATIC tuple, not derived by calling :func:`geo_archetypes`
# and reading its ``.keys()`` — a follow-up to S5 (commit 91fd7db) specifically removed
# a throwaway ``geo_archetypes()`` construction used only to enumerate labels.
GEO_ARCHETYPE_LABELS: tuple[str, ...] = (SPRAWLER, FORTRESS, CHERRY_PICKER, FAST_FOLLOWER)

# The predict closure signature the head-to-head primitives drive (obs positional, an
# optional ``episode_start`` kwarg, a flat float32 action) — the SAME shape
# :data:`retail_simulator.harness.spike_archetypes.ArchetypePredict` returns, so the
# league/ladder materialize seam (and ``run_spike``) can seat either family
# interchangeably.
Predict = Callable[..., npt.NDArray[np.float32]]

# FORTRESS's target store count and its final tick eligible for an expansion attempt
# (inclusive — see _fortress_target's docstring for the exact tick-counting convention).
_FORTRESS_STORE_CAP: int = 3
_FORTRESS_LAST_ELIGIBLE_TICK: int = 5

# --- Non-geographic lever postures, borrowed from spike_archetypes' own established
# anchors (see the module docstring's rationale). ---
_PRICE_LOW: float = 0.75  # AGGRESSOR's seed "aggressive low price to grab share"
_PRICE_MID: float = 1.05  # PATIENT's seed "moderate price near baseline"
_PRICE_PREMIUM: float = 1.5  # LEAN's seed "high price to defend margin"
_MARKETING_MINIMAL: float = 0.05  # LEAN's seed
_MARKETING_MID: float = 0.4  # PATIENT's seed
_MARKETING_MAX: float = 0.9  # AGGRESSOR's seed
_ASSORTMENT_MINIMAL: float = 0.1  # LEAN's seed
_ASSORTMENT_MODERATE: float = 0.45  # PATIENT's seed (unspecified-lever baseline)
_ASSORTMENT_MAX: float = 0.9  # AGGRESSOR's seed
_PROMOTION_MINIMAL: float = 0.0  # LEAN's seed
_PROMOTION_MODERATE: float = 0.2  # PATIENT's seed (unspecified-lever baseline)
_SERVICE_LOW: float = 0.2  # calibrated LEAN's "deep squeezer"
_SERVICE_SQUEEZER: float = 0.5  # calibrated AGGRESSOR's "squeezer"
_SERVICE_PARTNER: float = 1.0  # calibrated PATIENT's "partner"
_LOYALTY_NONE: float = 0.0
_LOYALTY_MODERATE: float = 0.4  # PATIENT's seed (unspecified-lever baseline)
_LOYALTY_MAX: float = 0.8  # AGGRESSOR's seed "spend on loyalty early"
_RESEARCH_INERT: float = 0.0  # no rule below reads the noised COMPETITOR block
_AUTOMATION_TIER_INERT: int = 0  # the registry no-op default; undifferentiated here


# --------------------------------------------------------------------------- #
# Observation lookup + region-table derivation (construction-time, pure)        #
# --------------------------------------------------------------------------- #


def _name_to_index(n_regions: int) -> dict[str, int]:
    """Map every ``observation_schema(n_regions)`` field name to its flat obs index."""
    return {field.name: i for i, field in enumerate(observation_schema(n_regions))}


def _metro_indices(region_table: tuple[RegionConfig, ...]) -> tuple[int, ...]:
    """ """
    if len(region_table) < 2:
        return ()
    n_enterable = len(region_table) - 1
    k = (n_enterable + 3) // 4  # ceil(n_enterable / 4) via integer arithmetic.
    enterable = range(1, len(region_table))
    ranked = sorted(enterable, key=lambda r: (-region_table[r].base_regional_demand, r))
    return tuple(sorted(ranked[:k]))


# --------------------------------------------------------------------------- #
# Target-selection rules — PURE functions of (obs, tick, construction-time      #
# context), independently unit-testable on synthetic obs vectors.               #
# --------------------------------------------------------------------------- #


def _own_empty_regions(
    obs: npt.NDArray[np.float32], *, idx: dict[str, int], candidates: tuple[int, ...] | range
) -> Iterator[int]:
    """Yield ``candidates`` (in the given order) where the seat's own presence reads 0.0.

    ``stores_region_r`` is a SELF-group field — EXACT, never noised (only the
    COMPETITOR group carries perception noise) — so a plain float equality against
    0.0 is safe.
    """
    for r in candidates:
        if float(obs[idx[f"stores_region_{r}"]]) == 0.0:
            yield r


def _sprawler_target(obs: npt.NDArray[np.float32], *, idx: dict[str, int], n_regions: int) -> int:
    """Lowest-index own-empty enterable region (1..n_regions-1); 0 (no-op) once saturated."""
    for r in _own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)):
        return r
    return 0


def _fortress_target(
    obs: npt.NDArray[np.float32], tick: int, *, idx: dict[str, int], n_regions: int
) -> int:
    """ """
    if tick > _FORTRESS_LAST_ELIGIBLE_TICK:
        return 0
    if float(obs[idx["stores"]]) >= _FORTRESS_STORE_CAP:
        return 0
    return _sprawler_target(obs, idx=idx, n_regions=n_regions)


def _cherry_picker_target(
    obs: npt.NDArray[np.float32], *, idx: dict[str, int], metro_indices: tuple[int, ...]
) -> int:
    """Lowest-index own-empty region AMONG the top-quartile-by-pie (METRO) class only.

    ``metro_indices`` (from :func:`_metro_indices`) already excludes region 0 by
    construction — never a per-call re-check needed.
    """
    for r in _own_empty_regions(obs, idx=idx, candidates=metro_indices):
        return r
    return 0


def _fast_follower_target(
    obs: npt.NDArray[np.float32],
    *,
    idx: dict[str, int],
    n_regions: int,
    region_table: tuple[RegionConfig, ...],
) -> int:
    """Lowest-index contested own-empty region, else the largest-pie own-empty region.

    "Contested" = ``others_present_region_r > 0`` (a MARKET-group field — an EXACT
    count of other present retailers, no perception noise, unlike the COMPETITOR
    group). Ties in the fallback (more than one own-empty region sharing the max
    pie) are broken by lowest index (the scan is ascending and only replaces the
    running best on a STRICT improvement).
    """
    for r in range(1, n_regions):
        if (
            float(obs[idx[f"others_present_region_{r}"]]) > 0.0
            and float(obs[idx[f"stores_region_{r}"]]) == 0.0
        ):
            return r
    best_r = 0
    best_demand = -1.0
    for r in _own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)):
        demand = region_table[r].base_regional_demand
        if demand > best_demand:
            best_demand = demand
            best_r = r
    return best_r


# --------------------------------------------------------------------------- #
# StructuredAction assembly + the stateful predict-closure factory              #
# --------------------------------------------------------------------------- #


def _structured_action(
    *,
    price: float,
    marketing: float,
    assortment: float,
    promotion: float,
    expansion_choice: int,
    research: float,
    service_level: float,
    automation_tier: int,
    loyalty_spend: float,
    n_regions: int,
    wage_spend: float = 0.0,
    warehouse_invest: float = 0.0,
) -> StructuredAction:
    """Build a :class:`StructuredAction` over the full 11-lever registry.

    Mirrors :func:`retail_simulator.harness.spike_archetypes._structured_action`
    exactly, EXCEPT ``expansion_choice`` is bound-checked against ``n_regions`` (this
    WORLD's actual region count) rather than the fixed 2-region ``N_REGIONS`` constant
    — that function's own docstring explicitly reserves the wider check for "a later
    slice's ``competition_archetypes()``", which is this module. ``wage_spend``/
    ``warehouse_invest`` default to 0.0 (the registry byte-identity anchor); none of
    the four geo archetypes' postures need them.
    """
    if not (0 <= expansion_choice < n_regions):
        raise ValueError(f"expansion_choice must be in [0, {n_regions}); got {expansion_choice}")
    if not (0 <= automation_tier < N_AUTOMATION_TIERS):
        raise ValueError(
            f"automation_tier must be in [0, {N_AUTOMATION_TIERS}); got {automation_tier}"
        )
    return StructuredAction(
        levers={
            "price_index": price,
            "marketing": marketing,
            "assortment": assortment,
            "promotion": promotion,
            "expansion": float(expansion_choice),
            "research": research,
            "service_level": service_level,
            "automation": float(automation_tier),
            "loyalty_spend": loyalty_spend,
            "wage_spend": wage_spend,
            "warehouse_invest": warehouse_invest,
        }
    )


def _make_predict(
    target_fn: Callable[[npt.NDArray[np.float32], int], int],
    *,
    n_regions: int,
    n_obs: int,
    price: float,
    marketing: float,
    assortment: float,
    promotion: float,
    research: float,
    service_level: float,
    automation_tier: int,
    loyalty_spend: float,
) -> Predict:
    """Wrap a pure ``target_fn(obs, tick) -> expansion_choice`` into a stateful,
    deterministic ``predict`` closure carrying its own tick counter.

    Mirrors the mutable-cell idiom :meth:`retail_simulator.harness.league.
    FrozenPolicyPool.materialize` uses for the LSTM checkpoint closure's carried
    hidden state (``state_cell: list[Any] = [None]``) — a one-element list closed
    over by ``predict``, since resetting a plain closure-captured int on
    ``episode_start`` needs ``nonlocal`` either way; a cell reads identically and
    composes with helper calls. ``tick_cell[0]`` counts calls since construction or
    the last ``episode_start=True`` reset — the first call (or the first after a
    reset) computes its action at ``tick == 0``.

    Every call is deterministic in (obs, tick): the non-geographic levers are fixed
    constants closed over here; only ``expansion_choice`` (via ``target_fn``) varies.

    ``predict`` validates ``len(obs) == n_obs`` (``n_obs`` = ``len(observation_schema
    (n_regions))``, computed once at construction from the SAME ``idx`` every
    ``_make_*`` builder already has) — a wrong-width obs (e.g. a caller passing an
    ``n_regions``-mismatched world's observation) would otherwise fail as a silent
    wrong-field read via ``idx``, or not at all, rather than a loud, diagnosable error.
    ``**_kwargs`` is absorbed-and-ignored (mirrors :meth:`retail_simulator.harness.
    league.FrozenPolicyPool.materialize`'s MLP checkpoint closures, e.g.
    ``_checkpoint_predict_mlp(obs, **_kwargs)``) so a caller that uniformly passes
    other keyword args to every roster member's ``predict`` (present or future) does
    not need this family to special-case them.
    """

    def predict(
        obs: npt.NDArray[np.float32], *, episode_start: bool = False, **_kwargs: Any
    ) -> npt.NDArray[np.float32]:
        if len(obs) != n_obs:
            raise ValueError(f"predict expected an observation of length {n_obs}; got {len(obs)}")
        if episode_start:
            tick_cell[0] = 0
        tick = tick_cell[0]
        tick_cell[0] = tick + 1
        expansion_choice = target_fn(obs, tick)
        structured = _structured_action(
            price=price,
            marketing=marketing,
            assortment=assortment,
            promotion=promotion,
            expansion_choice=expansion_choice,
            research=research,
            service_level=service_level,
            automation_tier=automation_tier,
            loyalty_spend=loyalty_spend,
            n_regions=n_regions,
        )
        return encode_action(structured)

    tick_cell: list[int] = [0]
    return predict


# --------------------------------------------------------------------------- #
# Per-archetype builders + the public entrypoint                                #
# --------------------------------------------------------------------------- #


def _make_sprawler(n_regions: int, idx: dict[str, int]) -> Predict:
    def target_fn(obs: npt.NDArray[np.float32], _tick: int) -> int:
        return _sprawler_target(obs, idx=idx, n_regions=n_regions)

    return _make_predict(
        target_fn,
        n_regions=n_regions,
        n_obs=len(idx),
        price=_PRICE_LOW,
        marketing=_MARKETING_MINIMAL,
        assortment=_ASSORTMENT_MINIMAL,
        promotion=_PROMOTION_MINIMAL,
        research=_RESEARCH_INERT,
        service_level=_SERVICE_LOW,
        automation_tier=_AUTOMATION_TIER_INERT,
        loyalty_spend=_LOYALTY_NONE,
    )


def _make_fortress(n_regions: int, idx: dict[str, int]) -> Predict:
    def target_fn(obs: npt.NDArray[np.float32], tick: int) -> int:
        return _fortress_target(obs, tick, idx=idx, n_regions=n_regions)

    return _make_predict(
        target_fn,
        n_regions=n_regions,
        n_obs=len(idx),
        price=_PRICE_MID,
        marketing=_MARKETING_MAX,
        assortment=_ASSORTMENT_MAX,
        promotion=_PROMOTION_MINIMAL,
        research=_RESEARCH_INERT,
        service_level=_SERVICE_PARTNER,
        automation_tier=_AUTOMATION_TIER_INERT,
        loyalty_spend=_LOYALTY_MAX,
    )


def _make_cherry_picker(
    n_regions: int, idx: dict[str, int], metro_indices: tuple[int, ...]
) -> Predict:
    def target_fn(obs: npt.NDArray[np.float32], _tick: int) -> int:
        return _cherry_picker_target(obs, idx=idx, metro_indices=metro_indices)

    return _make_predict(
        target_fn,
        n_regions=n_regions,
        n_obs=len(idx),
        price=_PRICE_PREMIUM,
        marketing=_MARKETING_MID,
        assortment=_ASSORTMENT_MODERATE,
        promotion=_PROMOTION_MODERATE,
        research=_RESEARCH_INERT,
        service_level=_SERVICE_PARTNER,
        automation_tier=_AUTOMATION_TIER_INERT,
        loyalty_spend=_LOYALTY_MODERATE,
    )


def _make_fast_follower(
    n_regions: int, idx: dict[str, int], region_table: tuple[RegionConfig, ...]
) -> Predict:
    def target_fn(obs: npt.NDArray[np.float32], _tick: int) -> int:
        return _fast_follower_target(obs, idx=idx, n_regions=n_regions, region_table=region_table)

    return _make_predict(
        target_fn,
        n_regions=n_regions,
        n_obs=len(idx),
        price=_PRICE_MID,
        marketing=_MARKETING_MID,
        assortment=_ASSORTMENT_MODERATE,
        promotion=_PROMOTION_MODERATE,
        research=_RESEARCH_INERT,
        service_level=_SERVICE_SQUEEZER,
        automation_tier=_AUTOMATION_TIER_INERT,
        loyalty_spend=_LOYALTY_MODERATE,
    )


def geo_archetypes(n_regions: int, region_table: tuple[RegionConfig, ...]) -> dict[str, Predict]:
    """Build the four closed-loop geographic archetypes for a world of ``n_regions``.

    ``region_table`` is the ``DemandConfig.regions`` tuple the world these predict
    closures will play in was built from (e.g. ``harness.arena.arena_regions(n_regions)``
    — CHERRY-PICKER's METRO-class targets and FAST-FOLLOWER's pie-size tie-break are
    both derived from its ``base_regional_demand`` values, never a hardcoded class
    pattern). Each closure carries its OWN independent tick counter (construction, not
    call, is where fresh state is allocated) — calling this function twice yields two
    fully independent rosters, e.g. for a self-play matchup of the same archetype
    against itself.

    Raises ``ValueError`` if ``region_table``'s length disagrees with ``n_regions``.
    """
    if len(region_table) != n_regions:
        raise ValueError(
            f"region_table has {len(region_table)} entries, expected n_regions={n_regions}"
        )
    idx = _name_to_index(n_regions)
    metro_indices = _metro_indices(region_table)
    return {
        SPRAWLER: _make_sprawler(n_regions, idx),
        FORTRESS: _make_fortress(n_regions, idx),
        CHERRY_PICKER: _make_cherry_picker(n_regions, idx, metro_indices),
        FAST_FOLLOWER: _make_fast_follower(n_regions, idx, region_table),
    }
