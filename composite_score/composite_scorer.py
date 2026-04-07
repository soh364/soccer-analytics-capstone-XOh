"""
composite_scorer.py
Combines all signals into a single team readiness score for 2026 World Cup prediction.

Components:
    - Player quality (35%) — from player_aggregator.py
    - Tactical archetype (20%) — from tc_pipeline.py 
    - FIFA ranking (15%) — from external_factors.py
    - Club cohesion (10%) — from club_cohesion.py
    - Coach tenure (5%) — from external_factors.py
    - Squad age (5%) — from external_factors.py
    - Tournament experience (5%) — from external_factors.py
    + Confederation bonus — multiplier for host/regional advantage

Usage:
    from composite_scorer import get_team_readiness_scores
    df = get_team_readiness_scores(scored)
"""

import sys
import math
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "player_score"))

from external_factors import (
    FIFA_RANKINGS_SORTED,
    COACH_TENURE,
    WC_APPEARANCES,
    normalize_fifa_rank,
    normalize_tenure,
    normalize_wc_experience,
    compute_squad_age_score,
)
from club_cohesion import compute_club_cohesion
from player_score.player_aggregator import compute_player_quality_score
from rosters_2026 import rosters_2026

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

COMPONENT_WEIGHTS = {
    "player_quality":     0.35,
    "tactical_archetype": 0.20,
    "fifa_score":         0.15,
    "club_cohesion":      0.10,
    "squad_age":          0.05,  # new
    "coach_tenure":       0.05,
    "tournament_exp":     0.05,
    "confederation":      0.05,
}


CONFEDERATION_BONUS = {
    "United States": 1.05,  # down from 1.10
    "Canada":        1.05,  # down from 1.10
    "Mexico":        1.05,  # down from 1.10
    "Argentina":     1.03,  # down from 1.05
    "Brazil":        1.03,  # down from 1.05
    "Uruguay":       1.02,  # down from 1.04
    "Colombia":      1.02,  # down from 1.04
    "Ecuador":       1.01,  # down from 1.03
    "Paraguay":      1.01,  # down from 1.02
}

ARCHETYPE_SCORES = {
    "High Press / High Output": 85,
    "Possession Dominant":      75,
    "Compact Transition":       65,
    "Mid-Block Reactive":       60,
    "Moderate Possession":      50,
    "Low Intensity":            40,
}

# ---------------------------------------------------------------------------
# Weight redistribution when archetype is missing
# ---------------------------------------------------------------------------

def get_weights(has_archetype: bool) -> dict:
    """
    If no archetype data, redistribute tactical_archetype weight
    proportionally across remaining components.
    """
    if has_archetype:
        return COMPONENT_WEIGHTS.copy()

    remaining = {
        k: v for k, v in COMPONENT_WEIGHTS.items()
        if k != "tactical_archetype"
    }
    total = sum(remaining.values())
    return {k: round(v / total, 4) for k, v in remaining.items()}


# ---------------------------------------------------------------------------
# Individual component scorers
# ---------------------------------------------------------------------------

def get_fifa_score(country: str) -> float:
    rank = FIFA_RANKINGS_SORTED.get(country, 80)
    return normalize_fifa_rank(rank)


def get_coach_tenure_score(country: str) -> float:
    years = COACH_TENURE.get(country, {}).get("years", 1.0)
    return normalize_tenure(years)


def get_tournament_exp_score(country: str) -> float:
    apps = WC_APPEARANCES.get(country, 0)
    return normalize_wc_experience(apps)


def get_cohesion_score(country: str) -> float:
    return compute_club_cohesion(country).get("cohesion_score", 0.0)


def get_archetype_score(
    country: str,
    archetype_df: pd.DataFrame = None,
) -> tuple[float | None, float]:
    if archetype_df is None or len(archetype_df) == 0:
        return None, 0.0

    match = archetype_df[archetype_df["team"] == country]
    if len(match) == 0:
        return None, 0.0

    row = match.iloc[0]
    archetype = row.get("archetype")
    score = row.get("archetype_score", ARCHETYPE_SCORES.get(archetype, 60.0))
    confidence = row.get("gmm_confidence", 1.0)
    return float(score), float(confidence)


# ---------------------------------------------------------------------------
# Main composite scorer
# ---------------------------------------------------------------------------

def compute_team_score(
    country: str,
    scored_df: pd.DataFrame,
    archetype_df: pd.DataFrame = None,
) -> dict:
    """
    Compute composite readiness score for a single country.
    """
    # Player quality
    player_result = compute_player_quality_score(country, scored_df)
    player_score = player_result.get("effective_score")
    player_confidence = player_result.get("player_coverage_confidence", 0.0)
    n_scored = player_result.get("n_players_scored", 0)

    # External factors
    fifa_score = get_fifa_score(country)
    coach_score = get_coach_tenure_score(country)
    exp_score = get_tournament_exp_score(country)
    cohesion_score = get_cohesion_score(country)

    # Tactical archetype
    archetype_score, gmm_confidence = get_archetype_score(country, archetype_df)
    has_archetype = archetype_score is not None

    age_score = compute_squad_age_score(country)

    # Get weights
    weights = get_weights(has_archetype)

    # Build component dict
    components = {
        "player_quality": player_score if player_score is not None else fifa_score * 0.8,
        "fifa_score":     fifa_score,
        "club_cohesion":  cohesion_score,
        "coach_tenure":   coach_score,
        "tournament_exp": exp_score,
        "squad_age":      age_score,
    }

    if has_archetype:
        if gmm_confidence == 0.0:
            # Boundary team — use archetype score directly
            components["tactical_archetype"] = archetype_score
        else:
            components["tactical_archetype"] = (
                gmm_confidence * archetype_score +
                (1 - gmm_confidence) * fifa_score * 0.8
            )

    # Weighted composite
    composite = sum(
        weights[k] * v
        for k, v in components.items()
        if k in weights
    )

    # Confederation bonus
    conf_bonus = CONFEDERATION_BONUS.get(country, 1.0)
    final_score = round(composite * conf_bonus, 2)

    return {
        "country":              country,
        "final_score":          final_score,
        "composite_score":      round(composite, 2),
        "confederation_bonus":  conf_bonus,
        # Components
        "player_quality_score": round(components["player_quality"], 2),
        "player_confidence":    player_confidence,
        "n_players_scored":     n_scored,
        "tactical_archetype":   archetype_score,
        "gmm_confidence":       gmm_confidence if has_archetype else None,
        "fifa_score":           round(fifa_score, 2),
        "club_cohesion_score":  round(cohesion_score, 2),
        "coach_tenure_score":   round(coach_score, 2),
        "tournament_exp_score": round(exp_score, 2),
        "squad_age_score": round(age_score, 2),
        # Weights used
        "archetype_available":  has_archetype,
    }


def get_team_readiness_scores(
    scored_df: pd.DataFrame,
    archetype_df: pd.DataFrame = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute composite readiness scores for all rostered countries.

    Args:
        scored_df:    Player scores from get_player_scores()
        archetype_df: Tactical clustering output from get_team_archetypes()
                      Pass None to run without clustering (placeholder mode)
        verbose:      Print summary

    Returns:
        DataFrame sorted by final_score descending
    """
    if verbose:
        print("=" * 60)
        print("COMPOSITE TEAM READINESS SCORER")
        print("=" * 60)
        if archetype_df is None:
            print("  ⚠ No archetype data — running without tactical clustering")
            print("    Tactical weight redistributed to other components")
        else:
            n_matched = sum(
                1 for c in rosters_2026
                if c in archetype_df["team"].values
            )
            print(f"  ✓ Archetype data loaded — {n_matched} countries matched")

    rows = [
        compute_team_score(country, scored_df, archetype_df)
        for country in rosters_2026.keys()
    ]

    df = pd.DataFrame(rows).sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"

    if verbose:
        print(f"\n  {len(df)} countries scored")
        print(f"\n  TOP 10:")
        print(df[["country", "final_score", "player_quality_score",
                   "fifa_score", "archetype_available"]].head(10).to_string())

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "player_score"))
    from player_score_pipeline import get_player_scores

    scored = get_player_scores(verbose=False)
    df = get_team_readiness_scores(scored, archetype_df=None, verbose=True)
    print(df.to_string())