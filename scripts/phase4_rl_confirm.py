#!/usr/bin/env python
"""Phase 4 RL-confirmation driver — do TRAINED agents form the non-transitive cycle?

The experiment
--------------
1. **Train** a self-play LEAGUE under :func:`calibrated_spike_config` via
   :func:`run_league_training` with a PFSP opponent distribution — PFSP grows a diverse
   pool, which is the natural way to explore a non-transitive landscape (a uniform/latest
   sampler would not pressure the learner to find counters). Each round snapshots a PPO
   checkpoint into the pool under ``runs/phase4_rl_confirm/<run>/`` (git-ignored).
2. **Tournament** the resulting DISTINCT checkpoints (deduped by policy-weight hash — two
   rounds can converge to byte-identical weights) head-to-head UNDER the calibrated config,
   position-controlled over a fixed shared seed set (the ladder / spike fairness contract),
   collecting continuous per-seed score margins.
3. **Measure** non-transitivity on the trained-agent tournament: build the dominance graph,
   count 3-cycles, detect a Condorcet winner, and flag DECISIVE cycles (every edge's
   bootstrap CI excludes 0). REUSES the spike's cycle measurement + the ladder bootstrap
   (no re-implementation).

Success interpretation (the honest validate-first read)
-------------------------------------------------------
* ``rl_confirms = True`` — the trained-agent tournament has >=1 DECISIVE cycle AND no
  Condorcet winner. RL CONFIRMS the non-transitivity: learned policies form the
  rock-paper-scissors. The redesign generalizes from scripted postures to discovered ones.
* ``rl_confirms = False`` — the tournament is all-transitive / has a Condorcet winner. The
  GAME has counters (the scripted spike proved that) but RL collapses to a corner. This is
  a REAL, publishable finding — report it honestly; do NOT force the cycle.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.harness.ladder import _bootstrap_margin_ci
from retail_simulator.harness.spike import PairMargin, _measure_tournament, calibrated_spike_config

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "runs" / "phase4_rl_confirm"
DOCS_DATA = REPO / "docs" / "data"

logger = logging.getLogger("phase4_rl_confirm")

# Defaults (full run). PFSP grows a diverse pool over R rounds of S steps each.
DEFAULT_ROUNDS = 3
DEFAULT_STEPS_PER_ROUND = 300_000
DEFAULT_PFSP_ALPHA = 2.0
DEFAULT_PFSP_MIX_UNIFORM: float = 0.0
# The (league, curriculum, core) seed triple threaded into run_league_training.
DEFAULT_SEEDS: tuple[int, int, int] = (0, 0, 42)
# PPO entropy-bonus coefficient (RL-DIVERSE H2) — SB3's own PPO/RecurrentPPO default;
# passing this value through --ent-coef reproduces every prior run byte-for-byte.
DEFAULT_ENT_COEF: float = 0.0
# PPO/RecurrentPPO GAE lambda (RLB-6, the pre-registered >=24-region arena RL recipe)
# — SB3's own PPO/RecurrentPPO default; passing this value through --gae-lambda
# reproduces every prior run byte-for-byte.
DEFAULT_GAE_LAMBDA: float = 0.95
# Tournament: fixed shared seed set 0..K-1 (the spike/ladder fairness contract).
DEFAULT_TOURNEY_SEEDS = 24
DEFAULT_TOURNEY_EPISODES = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 1000

# Smoke budgets — exercise EVERY code path (train -> snapshot -> dedup -> tournament ->
# cycle measurement) fast enough to validate the driver end-to-end in ~1-2 min.
SMOKE_ROUNDS = 2
SMOKE_STEPS_PER_ROUND = 2_000
SMOKE_TOURNEY_SEEDS = 3
SMOKE_BOOTSTRAP_RESAMPLES = 200

# Senior-review fix-round (Major-2): --mask-expansion scopes to the LEARNER'S
# TRAINING rollouts only (threaded through run_league_training's
# `_single_learner_factory` closure, S8) -- it is NEVER threaded to any
# evaluation/scoring surface. Named explicitly (rather than left to be inferred
# from a --help sentence) because five separate RetailParallelEnv constructions
# roll out unmasked regardless of this flag: the `tournament()` call below (this
# script), harness/ladder.py's `run_ladder`, harness/league.py's
# `evaluate_pool_snapshot_improvement` (snapshot-improvement eval) and
# `_run_round_end_evaluation` (both the checkpoint/archetype AND scripted-opponent
# branches, feeding PFSP weights), and harness/arena.py's `arena_env`. A run whose
# summary claims `training.mask_expansion: true` therefore has its share/margin
# numbers measured by rolling the masked-trained policy through an UNMASKED
# pipeline -- this constant makes that legible in the artifact itself, not just in
# --help. A fixed fact about the mechanism, not a per-run recipe value: always
# present in the summary regardless of --resume or the mask_expansion value.
MASK_EXPANSION_SCOPE_NOTE = (
    "mask_expansion, when true, applies only to the learner's TRAINING rollouts; "
    "the post-training tournament (this run), harness/ladder.py's ladder, "
    "harness/league.py's snapshot-improvement + round-end evaluations, and "
    "harness/arena.py's arena all roll out UNMASKED regardless of this flag"
)


# --------------------------------------------------------------------------- #
# Step 1 — train a PFSP self-play league under the calibrated config           #
# --------------------------------------------------------------------------- #


def train_league(
    *,
    pool_dir: Path,
    n_rounds: int,
    train_steps_per_round: int,
    pfsp_alpha: float,
    seeds: tuple[int, int, int],
    ent_coef: float = DEFAULT_ENT_COEF,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    seed_archetypes: tuple[str, ...] = (),
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion: bool = False,
    max_episode_steps: int | None = None,
    pfsp_mix_uniform: float = DEFAULT_PFSP_MIX_UNIFORM,
    warm_start: bool = False,
) -> Any:
    """ """
    from retail_simulator.harness.league import (  # noqa: PLC0415
        POOL_SPIKE_ARCHETYPE_LABELS,
        FrozenPolicyPool,
        OpponentDistribution,
        PFSPOpponentDistribution,
        make_seeded_pool,
        run_league_training,
    )

    effective_config = config if config is not None else calibrated_spike_config()
    # The H5 fix round: only the NOTABLE "weighted" mode is ever
    # recorded on the pool's manifest -- "profit" is the byte-identical universal
    # default across calibrated_spike_config/arena_lock_config/arena_balanced_config
    # (and this function's own `config is None` fallback above), so recording it for
    # every pool would grow every untouched pool's manifest for no reader benefit,
    # exactly the byte-identity regression MANIFEST_VERSION_ARENA_SEATING's own
    # docstring already measured once for home_regions. This decision is made HERE
    # (not inside run_league_training/FrozenPolicyPool.save) because
    # run_league_training's OWN standalone default differs (competition_config()'s
    # mode="weighted") -- see FrozenPolicyPool.record_reward_mode's own docstring.
    reward_mode_to_record = (
        effective_config.reward.mode if effective_config.reward.mode != "profit" else None
    )
    is_calibrated_shaped = (
        effective_config.demand.regions == calibrated_spike_config().demand.regions
    )
    if is_calibrated_shaped:
        pool = make_seeded_pool(pool_dir, spike_archetypes=seed_archetypes)
    else:
        offending = tuple(name for name in seed_archetypes if name in POOL_SPIKE_ARCHETYPE_LABELS)
        if offending:
            raise ValueError(
                f"train_league: seed_archetypes {offending} are calibrated spike "
                "posture(s) with a FIXED expansion_choice — outside the calibrated "
                "lock's own region table (e.g. under the arena's, it only ever "
                "opens region 1), that fixed choice can no longer be assumed to "
                "target a meaningful region, which would silently corrupt the "
                "measurement; pass geo archetypes "
                "(harness.geo_archetypes.GEO_ARCHETYPE_LABELS) instead"
            )
        pool = make_seeded_pool(
            pool_dir, geo_archetypes=seed_archetypes, region_table=effective_config.demand.regions
        )
    dist = PFSPOpponentDistribution(alpha=pfsp_alpha, mix_uniform=pfsp_mix_uniform)

    def _distribution_factory(_pool: FrozenPolicyPool) -> OpponentDistribution:
        return dist

    logger.info(
        "training PFSP league: rounds=%d, steps/round=%d, pfsp_alpha=%s, "
        "pfsp_mix_uniform=%s, seeds=%s, "
        "ent_coef=%s, gae_lambda=%s, seed_archetypes=%s, home_regions=%s, "
        "mask_expansion=%s, max_episode_steps=%s, warm_start=%s, pool_dir=%s",
        n_rounds,
        train_steps_per_round,
        pfsp_alpha,
        pfsp_mix_uniform,
        seeds,
        ent_coef,
        gae_lambda,
        seed_archetypes,
        home_regions,
        mask_expansion,
        max_episode_steps,
        warm_start,
        pool_dir,
    )
    return run_league_training(
        initial_pool=pool,
        n_rounds=n_rounds,
        distribution_factory=_distribution_factory,
        train_steps_per_round=train_steps_per_round,
        seeds=seeds,
        config=effective_config,
        distribution=dist,
        pfsp_alpha=pfsp_alpha,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        home_regions=home_regions,
        mask_expansion=mask_expansion,
        reward_mode=reward_mode_to_record,
        max_episode_steps=max_episode_steps,
        warm_start=warm_start,
    )


# --------------------------------------------------------------------------- #
# Step 2 — dedupe distinct checkpoints by policy-weight hash                    #
# --------------------------------------------------------------------------- #


def _policy_weight_hash(zip_path: Path) -> str:
    """Hash an SB3 checkpoint's policy weights (stdlib zipfile — NEVER imports SB3).

    Two league rounds can converge to byte-identical policies (e.g. a round whose training
    barely moved the weights); a tournament over duplicates wastes rollouts and pollutes
    the dominance graph with degenerate 0-margin edges. We dedupe on the SHA-256 of the
    zip's ``policy.pth`` member bytes (the serialized weights) — the same weight-fingerprint
    approach the transitivity probe used. STDLIB-only so the lazy-SB3 contract holds.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        member = "policy.pth" if "policy.pth" in names else next(iter(sorted(names)))
        payload = zf.read(member)
    return hashlib.sha256(payload).hexdigest()[:16]


def distinct_checkpoints(pool_dir: Path, checkpoints: tuple[Any, ...]) -> dict[str, Path]:
    """Map ``label -> checkpoint zip path`` for the DISTINCT trained checkpoints.

    ``checkpoints`` is the ``LeagueTrainingResult.checkpoints`` tuple (kind=="checkpoint"
    PoolMembers, in round order). We resolve each member's ``checkpoint_path`` against
    ``pool_dir``, hash its weights, and keep the FIRST member per distinct hash (earliest
    round wins the slot). Labels are ``round_<n>`` so the tournament + dominance graph read
    cleanly. A round-0 random-init snapshot is included (it is a legitimate, distinct policy
    — a genuine corner of the strategy space the trained rounds may or may not beat).
    """
    by_hash: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for member in checkpoints:
        if member.checkpoint_path is None:
            continue
        zip_path = pool_dir / member.checkpoint_path
        weight_hash = _policy_weight_hash(zip_path)
        if weight_hash in by_hash:
            logger.info(
                "dedup: round_%d checkpoint is weight-identical to an earlier round; skipping",
                member.round_index,
            )
            continue
        by_hash[weight_hash] = zip_path
        labels[f"round_{member.round_index}"] = zip_path
    return labels


# --------------------------------------------------------------------------- #
# Step 3 — tournament the distinct checkpoints under the calibrated config      #
# --------------------------------------------------------------------------- #


def _materialize(label_to_zip: dict[str, Path]) -> dict[str, Callable[..., Any]]:
    """Load each distinct checkpoint's predict closure ONCE (reuses the ladder loader).

    Delegates to :func:`retail_simulator.harness.ladder._materialize_predicts` — the
    EXISTING checkpoint-load path (one-member ``FrozenPolicyPool`` + ``materialize``, with
    MLP/LSTM auto-detect). No re-implementation of ``PPO.load``. Lazy import keeps module
    load cheap + SB3-free.
    """
    from retail_simulator.harness.ladder import _materialize_predicts  # noqa: PLC0415

    return _materialize_predicts({label: str(path) for label, path in label_to_zip.items()})


def tournament(
    label_to_zip: dict[str, Path],
    *,
    n_seeds: int,
    n_episodes: int,
    bootstrap_resamples: int,
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
) -> tuple[PairMargin, ...]:
    """Position-controlled head-to-head over every distinct-checkpoint pair; return PairMargins.

    Mirrors :func:`run_spike`'s rollout + bootstrap exactly, but over TRAINED checkpoints:
    one shared ``RetailParallelEnv(config, n_learning_agents=2)`` reused across every
    (pair, seed); per-seed continuous margin ``mean(scores_a)-mean(scores_b)``; the
    ladder's seed-axis bootstrap CI applied CONSISTENTLY across pairs (one resample draw,
    reused). Heavy imports (pettingzoo env + the league head-to-head primitive) are lazy.

    ``config`` (RLB-8a) defaults (``None``) to :func:`calibrated_spike_config` — every
    prior call's hardcoded world, byte-identical; the caller passes the SAME config
    :func:`train_league` trained on (``--env arena`` ⇒
    :func:`~retail_simulator.harness.arena.arena_lock_config`) so the tournament rates
    the checkpoints on the world they were actually trained under. ``home_regions``
    (default ``None``) forwards to the shared env's construction the same way.
    """
    from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: PLC0415
    from retail_simulator.harness.league import (  # noqa: PLC0415
        _position_controlled_head_to_head,
    )

    effective_config = config if config is not None else calibrated_spike_config()
    labels = tuple(sorted(label_to_zip))
    predicts = _materialize(label_to_zip)
    seed_set = tuple(range(n_seeds))

    env = RetailParallelEnv(
        effective_config, n_learning_agents=2, seed=seed_set[0], home_regions=home_regions
    )
    per_seed_by_pair: dict[tuple[str, str], list[float]] = {}
    try:
        for label_a, label_b in combinations(labels, 2):
            per_seed: list[float] = []
            for seed in seed_set:
                scores_a, scores_b = _position_controlled_head_to_head(
                    env,
                    effective_config.reward,
                    predict_a=predicts[label_a],
                    predict_b=predicts[label_b],
                    n_episodes=n_episodes,
                    seed=seed,
                )
                per_seed.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
            per_seed_by_pair[(label_a, label_b)] = per_seed
    finally:
        env.close()

    # Seed-axis bootstrap, drawn ONCE and applied consistently (the ladder coherence
    # contract — reuses _bootstrap_margin_ci exactly as run_spike does).
    rng = np.random.default_rng(0)
    resampled_seed_indices = rng.integers(0, n_seeds, size=(bootstrap_resamples, n_seeds))
    pair_per_seed: dict[tuple[str, str], np.ndarray] = {
        pair: np.asarray(vals, dtype=np.float64) for pair, vals in per_seed_by_pair.items()
    }

    pairwise: list[PairMargin] = []
    for label_a, label_b in combinations(labels, 2):
        arr = pair_per_seed[(label_a, label_b)]
        ci_low, ci_high = _bootstrap_margin_ci(
            label_a,
            [label_b],
            pair_per_seed,
            n_seeds=n_seeds,
            resampled_seed_indices=resampled_seed_indices,
        )
        pairwise.append(
            PairMargin(
                label_a=label_a,
                label_b=label_b,
                per_seed_margins=tuple(float(v) for v in arr),
                mean_margin=float(arr.mean()),
                margin_se=float(np.std(arr, ddof=0)) / sqrt(n_seeds),
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return tuple(pairwise)


# --------------------------------------------------------------------------- #
# Summary assembly + reporting                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfirmResult:
    """The trained-agent non-transitivity measurement (the experiment's verdict)."""

    labels: tuple[str, ...]
    pairwise: tuple[PairMargin, ...]
    dominance: dict[str, tuple[str, ...]]
    n_cyclic_triples: int
    total_triples: int
    has_condorcet_winner: bool
    condorcet_winner: str | None
    decisive_cycle: bool
    rl_confirms: bool


def measure(pairwise: tuple[PairMargin, ...], labels: tuple[str, ...]) -> ConfirmResult:
    """ """
    (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        _cycle_direction,
        decisive,
    ) = _measure_tournament(pairwise, labels)
    total_triples = len(list(combinations(labels, 3)))
    rl_confirms = n_cyclic_triples >= 1 and not has_condorcet_winner and decisive
    return ConfirmResult(
        labels=labels,
        pairwise=pairwise,
        dominance=dominance,
        n_cyclic_triples=n_cyclic_triples,
        total_triples=total_triples,
        has_condorcet_winner=has_condorcet_winner,
        condorcet_winner=condorcet_winner,
        decisive_cycle=decisive,
        rl_confirms=rl_confirms,
    )


def _summary_payload(
    result: ConfirmResult,
    *,
    rounds: int | None,
    train_steps_per_round: int | None,
    pfsp_alpha: float | None,
    pfsp_mix_uniform: float | None,
    seeds: tuple[int, int, int] | None,
    ent_coef: float | None,
    gae_lambda: float | None,
    mask_expansion: bool | None,
    max_episode_steps: int | None,
    warm_start: bool | None,
    reward_mode: str,
    seed_archetypes: tuple[str, ...],
    env: str,
    home_regions: tuple[int, ...] | None,
    arena_region_table_hash: str | None,
    arena_lock_knobs: dict[str, float] | None,
    label: str | None,
    tourney_seeds: int,
    tourney_episodes: int,
    smoke: bool,
    resumed: bool,
) -> dict[str, Any]:
    """ """
    return {
        "experiment": "phase4_rl_confirm",
        "question": "Do TRAINED self-play agents form the non-transitive cycle, "
        "or collapse to a corner?",
        "smoke": smoke,
        "label": label,
        "config": (
            "calibrated_spike_config"
            if env == "calibrated"
            else (
                "arena_lock_config (candidate lock, RLB-8a)"
                if env == "arena"
                else "arena_balanced_config (credit-line-and-rebalance: working-"
                "capital credit line + smaller warehouse footprint)"
            )
        ),
        "training": {
            # senior-review M1: True only when --resume skipped train_league
            # entirely — tells a reader WHY every field below except
            # seed_archetypes/env/home_regions/arena_region_table_hash is
            # null, rather than leaving null ambiguous between "unknown" and
            # "off".
            "resumed": resumed,
            "n_rounds": rounds,
            "train_steps_per_round": train_steps_per_round,
            "max_episode_steps": max_episode_steps,
            "distribution": "pfsp",
            "pfsp_alpha": pfsp_alpha,
            "pfsp_mix_uniform": pfsp_mix_uniform,
            "warm_start": warm_start,
            "seeds": list(seeds) if seeds is not None else None,
            "ent_coef": ent_coef,
            "gae_lambda": gae_lambda,
            # S8: null under --resume (train_league skipped, mirrors ent_coef/
            # gae_lambda above); otherwise the as-typed --mask-expansion CLI value.
            "mask_expansion": mask_expansion,
            # Major-2 (senior review, S8 fix round): unlike mask_expansion above,
            # this is a FIXED fact about the mechanism (see MASK_EXPANSION_SCOPE_NOTE),
            # not a per-run recipe value -- always present, never nulled under
            # --resume.
            "mask_expansion_scope": MASK_EXPANSION_SCOPE_NOTE,
            "reward_mode": reward_mode,
            # H4: always recorded — empty when the pool has no archetype
            # members, so a summary JSON's absence of the key never has to be
            # interpreted as "unknown" vs "off". The caller's `seed_archetypes`
            # reflects --seed-archetypes on a fresh train, or (senior-review
            # m1) the RESUMED pool's actual manifest under --resume — either
            # way this always describes what the pool truly contains.
            "seed_archetypes": list(seed_archetypes),
            # RLB-8a: which world this run actually trained/tournamented on —
            # "calibrated", "arena", or "arena-balanced" — derived from the
            # resumed pool's own manifest under --resume (never nulled),
            # exactly like seed_archetypes above.
            "env": env,
            "home_regions": list(home_regions) if home_regions is not None else None,
            "arena_region_table_hash": arena_region_table_hash,
            "arena_lock_knobs": arena_lock_knobs,
        },
        "tournament": {
            "n_distinct_checkpoints": len(result.labels),
            "labels": list(result.labels),
            "n_seeds": tourney_seeds,
            "n_episodes": tourney_episodes,
        },
        "measurement": {
            "n_cyclic_triples": result.n_cyclic_triples,
            "total_triples": result.total_triples,
            "has_condorcet_winner": result.has_condorcet_winner,
            "condorcet_winner": result.condorcet_winner,
            "decisive_cycle": result.decisive_cycle,
            "dominance": {k: list(v) for k, v in result.dominance.items()},
            "pairwise": [
                {
                    "label_a": pm.label_a,
                    "label_b": pm.label_b,
                    "mean_margin": pm.mean_margin,
                    "ci_low": pm.ci_low,
                    "ci_high": pm.ci_high,
                    "ci_excludes_zero": pm.ci_low > 0.0 or pm.ci_high < 0.0,
                }
                for pm in result.pairwise
            ],
        },
        "rl_confirms": result.rl_confirms,
        "interpretation": (
            "RL CONFIRMS the non-transitivity: trained self-play policies form the "
            "rock-paper-scissors (>=1 decisive cycle, no Condorcet winner)."
            if result.rl_confirms
            else "RL does NOT confirm: the trained tournament is transitive / has a "
            "Condorcet winner. The GAME has counters (proven by the scripted spike) but "
            "RL collapsed to a corner — a real, publishable finding, reported honestly."
        ),
    }


def _print_table(result: ConfirmResult) -> None:
    """Print a compact human-readable table of the trained-agent tournament."""
    print("\n=== Phase 4 RL-confirmation — trained-agent tournament ===")
    print(f"distinct checkpoints: {len(result.labels)}  ({', '.join(result.labels)})")
    print(f"{'pair':<28}{'mean_margin':>14}{'ci_low':>12}{'ci_high':>12}{'decisive':>10}")
    for pm in result.pairwise:
        decisive_edge = pm.ci_low > 0.0 or pm.ci_high < 0.0
        print(
            f"{pm.label_a + ' vs ' + pm.label_b:<28}"
            f"{pm.mean_margin:>14.2f}{pm.ci_low:>12.2f}{pm.ci_high:>12.2f}"
            f"{('yes' if decisive_edge else 'no'):>10}"
        )
    print(f"\ndominance: {dict(result.dominance)}")
    print(
        f"n_cyclic_triples={result.n_cyclic_triples}/{result.total_triples}  "
        f"has_condorcet_winner={result.has_condorcet_winner} "
        f"(winner={result.condorcet_winner})  decisive_cycle={result.decisive_cycle}"
    )
    verdict = "RL CONFIRMS non-transitivity" if result.rl_confirms else "RL does NOT confirm"
    print(f"\nVERDICT: rl_confirms={result.rl_confirms}  ({verdict})\n")


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def _load_existing_pool_state(
    pool_dir: Path,
) -> tuple[
    tuple[Any, ...],
    tuple[str, ...],
    tuple[int, ...] | None,
    str | None,
    dict[str, float] | None,
    str | None,
]:
    """Reload the trained checkpoints + the training provenance from an existing pool dir (--resume).

    Every returned value is derived from the SAME on-disk manifest (senior-review m1,
    extended by RLB-8a, extended again by the H5 fix round): under
    ``--resume``, ``train_league`` — the only path to ``make_seeded_pool``/
    ``run_league_training`` — is SKIPPED, so this invocation's ``--seed-archetypes``/
    ``--env``/``--reward`` flags reflect nothing about what the RESUMED pool actually
    contains or was trained under. Trusting them would let ``--resume
    --seed-archetypes`` claim seeding that never happened, ``--resume --env arena``
    claim seating a calibrated-trained pool never had, or ``--resume`` (with
    ``--reward`` omitted) silently re-score a weighted-trained pool on the profit
    objective. Reading the archetype names AND the seating AND reward-mode provenance
    straight off the manifest (``kind == "archetype"``; :attr:`FrozenPolicyPool.
    home_regions`/:attr:`FrozenPolicyPool.arena_region_table_hash`/
    :attr:`FrozenPolicyPool.arena_lock_knobs`, persisted by :meth:`FrozenPolicyPool.
    record_arena_seating`; :attr:`FrozenPolicyPool.reward_mode`, persisted by
    :meth:`FrozenPolicyPool.record_reward_mode`) makes the summary JSON's
    ``training.seed_archetypes``/``training.home_regions``/
    ``training.arena_region_table_hash``/``training.arena_lock_knobs``/
    ``training.reward_mode`` describe reality regardless of what this invocation's
    flags say.

    Returns ``(checkpoints, seed_archetypes, home_regions, arena_region_table_hash,
    arena_lock_knobs, reward_mode)``. ``reward_mode`` is ``None`` for every pool that
    never trained under an explicit ``"weighted"`` mode (every pre-H5 pool, and every
    H5 pool trained under the "profit" default) — the caller (``run()``) treats
    ``None`` exactly like ``--reward``'s own "leave it untouched" default, since
    "profit" IS that default.
    """
    from retail_simulator.harness.league import FrozenPolicyPool  # noqa: PLC0415

    pool = FrozenPolicyPool.load(pool_dir)
    members = pool.members()
    checkpoints = tuple(m for m in members if m.kind == "checkpoint")
    seed_archetypes = tuple(m.archetype_name for m in members if m.kind == "archetype")
    return (
        checkpoints,
        seed_archetypes,
        pool.home_regions,
        pool.arena_region_table_hash,
        pool.arena_lock_knobs,
        pool.reward_mode,
    )


def _resolve_env(
    env: str,
) -> tuple[CoreConfig | None, tuple[int, ...] | None, str | None, dict[str, float] | None]:
    """Resolve ``--env`` into ``(config, home_regions, arena_region_table_hash,
    arena_lock_knobs)`` (RLB-8a; credit-line-and-rebalance extends it to a second
    arena variant).

    ``"calibrated"`` (the default) -> ``(None, None, None, None)`` —
    :func:`train_league`'s own default (:func:`calibrated_spike_config`), no explicit
    seating, byte-identical to every run before this flag existed. ``"arena"`` -> the
    25-region candidate-lock arena (:func:`~retail_simulator.harness.arena.
    arena_lock_config`); ``"arena-balanced"`` -> the same 25-region table with a
    working-capital credit line + a smaller warehouse footprint
    (:func:`~retail_simulator.harness.arena.arena_balanced_config`). Both arena
    variants seat both seats (:func:`~retail_simulator.harness.arena.
    arena_home_regions`\\ ``(2)`` — a league game is two seats: the learner plus one
    opponent, the exact seat count :class:`~retail_simulator.harness.league.
    LeagueEnvWrapper`'s ``single_learner`` mode always builds), and record that
    config's region-table fairness hash plus (senior-review M3, extended to five
    knobs by the credit-line-and-rebalance fix round, then to six by
    arm-employee-satisfaction) the economics knobs the hash alone cannot
    discriminate — READ OFF the resolved config, so
    ``arena``/``arena-balanced`` are distinguishable by any of
    ``provided_capacity_per_store`` (700 vs 300), ``credit_limit`` (0 vs 5000), or
    ``overdraft_rate`` (0 vs 0.03) even though they share the exact same region
    table and therefore the exact same hash. ``--resume`` (below) disambiguates
    on ``credit_limit`` specifically rather than ``provided_capacity_per_store``
    — a future amendment to the CANDIDATE LOCK could move its own cap to 300
    (colliding with the rebalance's), but never touches ``credit_limit``/
    ``overdraft_rate``, which exist only on the rebalance. Computed ONCE here
    (rather than four separate lookups) so ``config``/``home_regions``/the
    hash/the knobs can never silently diverge for the SAME ``env`` value.
    """
    if env in ("arena", "arena-balanced"):
        from retail_simulator.harness.arena import (  # noqa: PLC0415
            arena_balanced_config,
            arena_home_regions,
            arena_lock_config,
            arena_region_table_hash,
        )

        config = arena_balanced_config() if env == "arena-balanced" else arena_lock_config()
        home_regions = arena_home_regions(2)
        arena_lock_knobs = {
            "overhead_reference_stores": config.wage.overhead_reference_stores,
            "overhead_exponent": config.wage.overhead_exponent,
            "provided_capacity_per_store": config.warehouse.provided_capacity_per_store,
            "credit_limit": config.wage.credit_limit,
            "overdraft_rate": config.wage.overdraft_rate,
            "writeoff_frac_max": config.wage.writeoff_frac_max,
        }
        return config, home_regions, arena_region_table_hash(config), arena_lock_knobs
    return None, None, None, None


def _apply_reward_override(config: CoreConfig | None, reward: str | None) -> CoreConfig | None:
    """ """
    if reward is None:
        return config
    effective = config if config is not None else calibrated_spike_config()
    return replace(effective, reward=replace(effective.reward, mode=reward))


def _effective_reward_mode(config: CoreConfig | None) -> str:
    """The reward mode ``config`` will actually train/tournament under (H5).

    Mirrors :func:`train_league`'s and :func:`tournament`'s own ``config if config is not None
    else calibrated_spike_config()`` fallback, so the summary's ``training.reward_mode`` always
    names the SAME mode those two functions resolve to -- never a separate guess. Always a
    concrete ``"profit"``/``"weighted"``, in every run including ``--resume`` and ``--smoke``:
    unlike the train-only recipe knobs (``ent_coef``, ``gae_lambda``, ...), the reward mode also
    governs the tournament step below, which DOES run under ``--resume`` -- so, like
    ``seed_archetypes``/``env``, it is always derivable and never nulled out.

    This function trusts ``config`` completely -- the correctness of what it reports under
    ``--resume`` lives entirely in ``run()``'s own resume branch (senior-review MAJOR, H5 fix
    round: ``league_config`` there is now built from the resumed pool's manifest-derived reward
    mode when ``--reward`` is omitted, not merely the resolved env's generic default), not here.
    """
    effective = config if config is not None else calibrated_spike_config()
    return effective.reward.mode


def run(args: argparse.Namespace) -> int:
    rounds = SMOKE_ROUNDS if args.smoke else args.rounds
    steps = SMOKE_STEPS_PER_ROUND if args.smoke else args.train_steps_per_round
    tourney_seeds = SMOKE_TOURNEY_SEEDS if args.smoke else args.tourney_seeds
    resamples = SMOKE_BOOTSTRAP_RESAMPLES if args.smoke else args.bootstrap_resamples
    seeds = args.seeds
    ent_coef = args.ent_coef
    gae_lambda = args.gae_lambda
    mask_expansion = args.mask_expansion
    max_episode_steps = args.league_max_episode_steps
    warm_start = args.warm_start
    run_label = args.label
    resumed = args.resume

    seed_archetypes: tuple[str, ...]
    if args.seed_archetypes:
        if args.env in ("arena", "arena-balanced"):
            from retail_simulator.harness.geo_archetypes import (  # noqa: PLC0415
                GEO_ARCHETYPE_LABELS,
            )

            seed_archetypes = GEO_ARCHETYPE_LABELS
        else:
            from retail_simulator.harness.league import (  # noqa: PLC0415
                POOL_SPIKE_ARCHETYPE_LABELS,
            )

            seed_archetypes = POOL_SPIKE_ARCHETYPE_LABELS
    else:
        seed_archetypes = ()

    run_name = "smoke" if args.smoke else args.run_name
    pool_dir = Path(args.pool_dir) if args.pool_dir else RUNS_ROOT / run_name
    pool_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # senior-review m1 (extended by RLB-8a): override every CLI-flag-derived
        # guess above with the RESUMED pool's actual manifest — --resume never
        # calls train_league (the only path to make_seeded_pool /
        # run_league_training), so args.seed_archetypes/args.env reflect nothing
        # about what this pool was actually seeded with or trained under.
        # --env is deliberately IGNORED here (like --seed-archetypes already
        # was) — the pool's own recorded seating is authoritative, not
        # whatever `--env` happens to be typed alongside `--resume`.
        (
            checkpoints,
            seed_archetypes,
            home_regions,
            region_table_hash,
            arena_lock_knobs,
            stored_reward_mode,
        ) = _load_existing_pool_state(pool_dir)
        if home_regions is None:
            env_choice = "calibrated"
        else:
            # credit-line-and-rebalance: "arena" and "arena-balanced" share the
            # exact same region table (arena_lock_config/arena_balanced_config
            # both build on arena_regions() at the SAME ARENA_N_REGIONS), so
            # home_regions/arena_region_table_hash alone cannot tell them apart
            # any more (the pre-existing "home_regions is not None => arena"
            # shortcut this replaces). arena_lock_knobs (M3) CAN — it is read
            # straight off whichever config actually trained the pool
            # (harness/league.py::run_league_training).
            #
            # MAJOR fix (senior review, fix round): disambiguate on `credit_limit`
            # (0 for "arena", ARENA_BALANCED_CREDIT_LIMIT for "arena-balanced"),
            # NOT `provided_capacity_per_store` (700 vs 300) as originally shipped.
            # `provided_capacity_per_store` is one of the CANDIDATE LOCK's own
            # lock-owned knobs — an amendment moving
            # the lock's own cap to 300 would make arena_lock_config() collide
            # with the rebalance's cap, so every "arena" pool would silently
            # resume as "arena-balanced" and tournament on a credit-and-overdraft
            # -armed world its checkpoints never trained in. `credit_limit`/
            # `overdraft_rate` exist ONLY on the rebalance and are never touched
            # by a lock amendment, so they stay a safe disambiguation axis
            # regardless of where the lock's own knobs move.
            from retail_simulator.harness.arena import (  # noqa: PLC0415
                ARENA_BALANCED_CREDIT_LIMIT,
            )

            is_balanced = (
                arena_lock_knobs is not None
                and arena_lock_knobs.get("credit_limit") == ARENA_BALANCED_CREDIT_LIMIT
            )
            env_choice = "arena-balanced" if is_balanced else "arena"
        # m2 (senior review): the precedence above (the pool's manifest wins)
        # is correct; it was just SILENT. `--resume --env arena` on a
        # calibrated pool used to exit 0, write `env: "calibrated"`, and say
        # nothing about why the flag was overridden.
        if args.env != env_choice:
            logger.warning(
                "--resume: --env=%s was given but the resumed pool's manifest "
                "says %s; using %s (the pool's recorded seating is authoritative, "
                "not the --env flag)",
                args.env,
                env_choice,
                env_choice,
            )
        # Only the config half is needed here — home_regions/region_table_hash/
        # arena_lock_knobs already came from the pool's own manifest just above
        # (the resumed pool's ACTUAL recorded seating, not a re-derivation from
        # env_choice) — EXCEPT fresh_region_table_hash, captured below to guard
        # against a moved lock (M4).
        league_config, _, fresh_region_table_hash, _ = _resolve_env(env_choice)
        # MAJOR fix (senior review, H5 fix round): --reward has nothing to override
        # FROM under --resume in the sense that train_league never runs -- but
        # `_resolve_env(env_choice)` above only ever returns env_choice's GENERIC
        # "profit" default, not what this pool actually trained under. Before this
        # fix, `--reward` omitted meant "leave that generic default alone", so
        # resuming a weighted-trained pool without retyping the flag silently
        # re-scored its checkpoints on the profit objective. `--seed-archetypes`/
        # `--env` are already ignored on resume in favour of the manifest (m1/m2
        # above); `--reward` gets the SAME manifest-wins precedence when OMITTED,
        # via `stored_reward_mode` (`None` for every pre-H5 pool and every pool
        # trained under "profit" -- see FrozenPolicyPool.record_reward_mode). Unlike
        # `--env`, an EXPLICIT `--reward` on resume still WINS over the manifest
        # (never merely ignored-with-a-warning): re-scoring an already-trained
        # pool's tournament under a different objective without retraining is a
        # deliberate, documented use of the flag (see the non-resume branch's own
        # comment below) -- only the SILENT-DEFAULT case was the bug. A mismatch
        # between an EXPLICIT `--reward` and the pool's recorded history still gets
        # a warning, mirroring `--env`'s own m2 fix.
        effective_reward = args.reward if args.reward is not None else stored_reward_mode
        if args.reward is not None:
            historical_reward_mode = (
                stored_reward_mode if stored_reward_mode is not None else "profit"
            )
            if args.reward != historical_reward_mode:
                logger.warning(
                    "--resume: --reward=%s was given but the resumed pool's manifest "
                    "records training under reward_mode=%s; the tournament will score "
                    "these checkpoints under %s, which is NOT what they were trained under",
                    args.reward,
                    historical_reward_mode,
                    args.reward,
                )
        league_config = _apply_reward_override(league_config, effective_reward)
        # M4 (senior review): on resume, home_regions/region_table_hash came
        # from the manifest but league_config was rebuilt by calling TODAY's
        # arena_lock_config() above — if the candidate lock has moved since
        # this pool was trained, the tournament would silently roll out on the
        # NEW world while the summary still reports the OLD (stored) hash.
        # `fresh_region_table_hash` is arena_region_table_hash(league_config)
        # for the env_choice resolved just above (`_resolve_env`'s own third
        # return value), so this is exactly the reviewer's guard without a
        # second, redundant computation.
        if region_table_hash is not None and region_table_hash != fresh_region_table_hash:
            logger.error(
                "--resume: pool was trained under a different arena region table; refusing"
            )
            return 1
        if not checkpoints:
            logger.error(
                "--resume: no trained checkpoints found in %s; run without --resume", pool_dir
            )
            return 1
        logger.info("--resume: reusing %d trained checkpoint(s) in %s", len(checkpoints), pool_dir)
    else:
        env_choice = args.env
        league_config, home_regions, region_table_hash, arena_lock_knobs = _resolve_env(env_choice)
        # H5: apply --reward AFTER _resolve_env returns (never before), so a
        # monkey-patched resolver that already set mode="weighted"
        # (arena_train_player.py) is never silently overridden back to
        # "profit" by this flag's own default -- see _apply_reward_override.
        league_config = _apply_reward_override(league_config, args.reward)
        training_result = train_league(
            pool_dir=pool_dir,
            n_rounds=rounds,
            train_steps_per_round=steps,
            pfsp_alpha=args.pfsp_alpha,
            seeds=seeds,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            seed_archetypes=seed_archetypes,
            config=league_config,
            home_regions=home_regions,
            mask_expansion=mask_expansion,
            max_episode_steps=max_episode_steps,
            pfsp_mix_uniform=args.pfsp_mix_uniform,
            warm_start=warm_start,
        )
        checkpoints = training_result.checkpoints

    # H5: the reward mode league_config ends up carrying (post-override, both
    # branches above) -- always a concrete "profit"/"weighted", never null; see
    # _effective_reward_mode.
    reward_mode = _effective_reward_mode(league_config)

    label_to_zip = distinct_checkpoints(pool_dir, checkpoints)
    if len(label_to_zip) < 3:
        logger.warning(
            "only %d distinct trained checkpoint(s) — a tournament needs >=3 for a 3-cycle; "
            "the dominance graph is reported but cannot exhibit non-transitivity. "
            "(Expected with --smoke's tiny round count; increase --rounds for a real run.)",
            len(label_to_zip),
        )

    # RLB-8a: the tournament rolls out on the SAME (config, home_regions) the
    # checkpoints were actually trained under — resolved above either from
    # --env (a fresh run) or from the resumed pool's own manifest. NOT the same
    # under --mask-expansion, though (senior-review fix-round, Major-2): this
    # rollout is always UNMASKED, even when the checkpoints above were trained
    # under masking — see MASK_EXPANSION_SCOPE_NOTE.
    pairwise = tournament(
        label_to_zip,
        n_seeds=tourney_seeds,
        n_episodes=args.tourney_episodes,
        bootstrap_resamples=resamples,
        config=league_config,
        home_regions=home_regions,
    )
    result = measure(pairwise, tuple(sorted(label_to_zip)))
    _print_table(result)

    # senior-review M1: --resume never calls train_league, and the on-disk pool
    # manifest carries no recipe to recover instead (unlike seed_archetypes/
    # env/home_regions/arena_region_table_hash/reward_mode, which ARE derived from the
    # manifest above) — so these as-typed CLI values describe nothing about how
    # the resumed pool's checkpoints were actually trained. Null them out rather
    # than let `--resume --gae-lambda 0.98` (etc.) claim a training recipe this
    # invocation never ran; `resumed` records why they're null.
    payload = _summary_payload(
        result,
        rounds=None if resumed else rounds,
        train_steps_per_round=None if resumed else steps,
        pfsp_alpha=None if resumed else args.pfsp_alpha,
        pfsp_mix_uniform=None if resumed else args.pfsp_mix_uniform,
        seeds=None if resumed else seeds,
        ent_coef=None if resumed else ent_coef,
        gae_lambda=None if resumed else gae_lambda,
        mask_expansion=None if resumed else mask_expansion,
        max_episode_steps=None if resumed else max_episode_steps,
        warm_start=None if resumed else warm_start,
        reward_mode=reward_mode,
        seed_archetypes=seed_archetypes,
        env=env_choice,
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
        label=run_label,
        tourney_seeds=tourney_seeds,
        tourney_episodes=args.tourney_episodes,
        smoke=args.smoke,
        resumed=resumed,
    )
    if args.summary:
        summary_path = Path(args.summary)
    else:
        today = _dt.date.today().isoformat()
        smoke_suffix = "_smoke" if args.smoke else ""
        label_suffix = f"_{run_label}" if run_label else ""
        summary_path = DOCS_DATA / f"phase4_rl_confirm_{today}{smoke_suffix}{label_suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"summary written to {summary_path}")
    return 0


def _parse_seeds(value: str) -> tuple[int, int, int]:
    """Parse ``--seeds league,curriculum,core`` into the 3-tuple ``run_league_training`` wants.

    Boundary validation (argparse ``type=``): exactly 3 comma-separated ints, else
    :class:`argparse.ArgumentTypeError` with the offending value echoed back —
    argparse renders that as a clean CLI usage error, never a stack trace.
    """
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        )
    try:
        parsed_seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        ) from exc
    league_seed, curriculum_seed, core_seed = parsed_seeds
    return (league_seed, curriculum_seed, core_seed)


def _parse_label(value: str) -> str:
    """ """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            f"--label must match [A-Za-z0-9_-]+ (a safe filename component); got {value!r}"
        )
    return value


def _parse_gae_lambda(value: str) -> float:
    """Validate ``--gae-lambda`` as a float in ``(0, 1]`` (argparse ``type=``).

    ``gae_lambda`` is PPO's GAE bias-variance trade-off: 0 is one-step TD
    (maximally biased), 1 is the unbiased Monte-Carlo return, and SB3 does not
    clamp it — a value of 0 or below, or above 1, degrades the advantage
    estimator silently. Rejecting it here at the CLI boundary (the same
    discipline ``_parse_seeds``/``_parse_label`` apply) turns a typo into a
    clean usage error instead of a silently-degenerate training run.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--gae-lambda must be a float in (0, 1]; got {value!r}"
        ) from exc
    if not (0.0 < parsed <= 1.0):
        raise argparse.ArgumentTypeError(f"--gae-lambda must be a float in (0, 1]; got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny train+tournament to validate the driver end-to-end",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip training; tournament the existing pool's checkpoints",
    )
    parser.add_argument(
        "--run-name", default="run", help="sub-dir under runs/phase4_rl_confirm/ (full runs)"
    )
    parser.add_argument(
        "--pool-dir",
        default=None,
        help="override the pool dir (default: runs/phase4_rl_confirm/<run-name>)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="league rounds (full run)"
    )
    parser.add_argument(
        "--train-steps-per-round",
        type=int,
        default=DEFAULT_STEPS_PER_ROUND,
        help="PPO steps per round (full run)",
    )
    parser.add_argument(
        "--pfsp-alpha", type=float, default=DEFAULT_PFSP_ALPHA, help="PFSP weighting exponent"
    )
    parser.add_argument(
        "--pfsp-mix-uniform",
        type=float,
        default=DEFAULT_PFSP_MIX_UNIFORM,
        help=(
            f"""decision 10 (taken 2026-09-19): uniform-mixing floor passed to PFSPOpponentDistribution(mix_uniform=...) -- guarantees every eligible pool member keeps at least mix_uniform/n draw probability each round, preventing the single-opponent collapse measured (default:{DEFAULT_PFSP_MIX_UNIFORM}, no floor; must be in [0.0, 1.0])"""
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "'league,curriculum,core' seed triple threaded into run_league_training "
            f"(comma-separated ints; default: {','.join(str(s) for s in DEFAULT_SEEDS)})"
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=DEFAULT_ENT_COEF,
        help=(
            "PPO/RecurrentPPO entropy-bonus coefficient, passed straight through to "
            f"_make_shared_ppo (SB3 default: {DEFAULT_ENT_COEF})"
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=_parse_gae_lambda,
        default=DEFAULT_GAE_LAMBDA,
        help=(
            "PPO/RecurrentPPO GAE lambda, passed straight through to "
            "_make_shared_ppo; raising it lengthens the advantage estimator's "
            "horizon 1/(1-gamma*lambda) so a learner can see slower-payoff "
            "levers like expansion (the pre-registered arena recipe uses 0.98) "
            f"(SB3 default: {DEFAULT_GAE_LAMBDA}; must be in (0, 1])"
        ),
    )
    parser.add_argument(
        "--mask-expansion",
        action="store_true",
        help=(
            "S8 (/ row S8): mask the learner's illegal expansion logits (already-open/unaffordable regions, and slots beyond n_regions) to -inf before its argmax decode, so it can only ever make a legal choice (hold is always legal but is not itself a region); the frozen opponent is never masked. Applies to the learner's TRAINING rollouts only -- the post-training tournament below and every evaluation/scoring surface (the ladder, league snapshot/round-end evaluations, the arena) always roll out UNMASKED; see the summary JSON's training.mask_expansion_scope field. Recorded in training.mask_expansion. Default (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--league-max-episode-steps",
        type=int,
        default=None,
        help=(
            "decision 8 (taken 2026-09-18): the league training-episode horizon, passed straight through to run_league_training(max_episode_steps=...) -> every LeagueEnvWrapper it builds. Default (unset -> None) never truncates -- one opponent per round, byte-identical to every prior run. A positive int truncates each round's game at that many ticks so the SuperSuit bridge auto-resets mid-round and draws a NEW opponent per EPISODE instead (the arena recipe uses 1000, the arena match horizon). Recorded in training.max_episode_steps; null under --resume (train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "owner decision 12a (taken 2026-09-19): round k>=2 continues from round "
            "(k-1)'s just-trained checkpoint's policy PARAMETERS instead of an "
            "independent from-scratch init, passed straight through to "
            "run_league_training(warm_start=...) -- see that function's own "
            "docstring for the mechanism (only the policy weights transfer; the "
            "optimizer stays fresh). Round 1 is unaffected either way -- it always "
            "starts from round 0's random init. Default (off) is byte-identical to "
            "every prior run. Recorded in training.warm_start; null under --resume "
            "(train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--seed-archetypes",
        action="store_true",
        help=(
            "seed the pool with a fixed-opponent archetype roster alongside the default scripted seed, following --env (H4, addendum; RLB-8b): under --env calibrated (default), the three calibrated Phase 4 spike postures (lean, aggressor, patient); under --env arena, the four closed-loop geo postures (sprawler, fortress, cherry-picker, fast-follower) instead — the spike postures' fixed expansion_choice can only ever open region 1 there. Default behavior (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--env",
        choices=["calibrated", "arena", "arena-balanced"],
        default="calibrated",
        help=(
            "which world train_league trains on and the tournament rates on "
            "(RLB-8a; credit-line-and-rebalance): 'calibrated' (default) is "
            "calibrated_spike_config, byte-identical to every prior run; 'arena' is "
            "the 25-region candidate-lock arena (harness.arena.arena_lock_config); "
            "'arena-balanced' is the same 25-region arena with a working-capital "
            "credit line + a smaller warehouse footprint (harness.arena."
            "arena_balanced_config) — both arena envs seat both seats via "
            "arena_home_regions(2). Ignored under --resume (the resumed pool's own "
            "recorded seating is authoritative, like --seed-archetypes already is)."
        ),
    )
    parser.add_argument(
        "--reward",
        choices=["profit", "weighted"],
        default=None,
        help=(
            "override the resolved --env's reward mode (core/reward.py; H5) for BOTH training and the tournament: 'profit' is raw profit, 'weighted' is the DESIGN-locked per-component z-normalized blend (RewardConfig's own weights). Default (unset) leaves the resolved env's own reward mode untouched -- calibrated/arena/arena-balanced all default to 'profit', byte-identical to every prior run, and a caller that already resolved a 'weighted' config (arena_train_player.py's monkey-patched env resolver) is never silently reset to 'profit' by this flag's own default. Applied AFTER --env resolves, so it always wins when given explicitly. Recorded in training.reward_mode (never null, even under --resume, since it also governs the tournament step)."
        ),
    )
    parser.add_argument(
        "--label",
        type=_parse_label,
        default=None,
        help=(
            "run label recorded in the summary JSON and appended to the default "
            "summary filename (phase4_rl_confirm_<date>_<label>.json); must match "
            "[A-Za-z0-9_-]+"
        ),
    )
    parser.add_argument(
        "--tourney-seeds",
        type=int,
        default=DEFAULT_TOURNEY_SEEDS,
        help="shared seed-set size K for the tournament",
    )
    parser.add_argument(
        "--tourney-episodes",
        type=int,
        default=DEFAULT_TOURNEY_EPISODES,
        help="episodes per head-to-head seed",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap CI resamples",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="summary JSON path (default:)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
