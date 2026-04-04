"""
player_aggregator.py
Aggregates player scores to country level for the composite scorer.
Uses top players with positional balance guarantee.
Confidence based on absolute number of scored players vs starting 11 baseline.
Low coverage countries blended with FIFA ranking as fallback.
"""

import sys
import math
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "player_score"))

from rosters_2026 import rosters_2026
from external_factors import FIFA_RANKINGS_SORTED, normalize_fifa_rank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFENDERS   = ["CB", "FB"]
MIDFIELDERS = ["DM", "CM", "AM"]
ATTACKERS   = ["FW", "W"]

MIN_DEFENDERS    = 3
MIN_MIDFIELDERS  = 3
MIN_ATTACKERS    = 2
MIN_PLAYER_WEIGHT = 0.40  # player score always contributes at least 40%

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_coverage_confidence(
    country: str,
    scored_df: pd.DataFrame,
    baseline: int = 11,
) -> float:
    """
    Confidence based on how many of the starting 11 we have scores for.
    3/11 = 0.27, 6/11 = 0.55, 11/11 = 1.0
    Same baseline for all countries regardless of roster size.
    """
    roster_names = list(rosters_2026.get(country, {}).keys())
    matched = scored_df[scored_df['player'].isin(roster_names)]
    n_scored = len(matched)
    confidence = min(n_scored / baseline, 1.0)
    return round(confidence, 3)


def get_top_players_balanced(
    country: str,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get top players for a country with positional balance.
    top_n scales with roster size: large roster → 15, small → 8.
    Falls back gracefully if coverage is sparse.
    """
    roster_names = list(rosters_2026.get(country, {}).keys())
    roster_size = len(roster_names)
    top_n = max(8, min(15, int(roster_size * 0.6)))

    drop_cols = [c for c in ['seasons_present'] if c in scored_df.columns]
    matched = scored_df[scored_df['player'].isin(roster_names)].drop(
        columns=drop_cols
    ).copy()

    if len(matched) == 0:
        return matched

    # Guarantee minimums from each position group where possible
    defenders = matched[
        matched['position_archetype'].isin(DEFENDERS)
    ].nlargest(MIN_DEFENDERS, 'final_score')

    midfielders = matched[
        matched['position_archetype'].isin(MIDFIELDERS)
    ].nlargest(MIN_MIDFIELDERS, 'final_score')

    attackers = matched[
        matched['position_archetype'].isin(ATTACKERS)
    ].nlargest(MIN_ATTACKERS, 'final_score')

    guaranteed = pd.concat([defenders, midfielders, attackers]).drop_duplicates()

    remaining_slots = top_n - len(guaranteed)
    if remaining_slots > 0:
        remaining = matched[
            ~matched.index.isin(guaranteed.index)
        ].nlargest(remaining_slots, 'final_score')
        selected = pd.concat([guaranteed, remaining])
    else:
        selected = guaranteed.nlargest(top_n, 'final_score')

    return selected.head(top_n)


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def compute_player_quality_score(
    country: str,
    scored_df: pd.DataFrame,
) -> dict:
    """
    Compute country-level player quality score.
    Blends top-5 average with FIFA fallback weighted by coverage confidence.
    Player score always contributes at least MIN_PLAYER_WEIGHT (40%).
    """
    roster_names = list(rosters_2026.get(country, {}).keys())
    drop_cols = [c for c in ['seasons_present'] if c in scored_df.columns]
    matched = scored_df[scored_df['player'].isin(roster_names)].drop(
        columns=drop_cols
    ).copy()

    confidence = compute_coverage_confidence(country, scored_df)

    # FIFA fallback — discounted proxy for unknown player quality
    fifa_rank = FIFA_RANKINGS_SORTED.get(country, 60)
    fifa_fallback = normalize_fifa_rank(fifa_rank) * 0.75

    if len(matched) == 0:
        return {
            "country": country,
            "player_quality_score": None,
            "effective_score": round(fifa_fallback, 2),
            "player_coverage_confidence": 0.0,
            "n_players_scored": 0,
            "top_player": None,
            "top_player_score": None,
        }

    # Use top 5 for quality score — elite core matters most
    top5 = matched.nlargest(5, 'final_score')
    avg_score = top5['final_score'].mean()
    top_player = matched.nlargest(1, 'final_score').iloc[0]

    # Blend: player score always contributes at least MIN_PLAYER_WEIGHT
    effective_confidence = max(confidence, MIN_PLAYER_WEIGHT)
    effective = round(
        effective_confidence * avg_score +
        (1 - effective_confidence) * fifa_fallback,
        2
    )

    return {
        "country": country,
        "player_quality_score": round(avg_score, 2),
        "effective_score": effective,
        "player_coverage_confidence": confidence,
        "n_players_scored": len(matched),
        "top_player": top_player['player'],
        "top_player_score": round(top_player['final_score'], 2),
    }


def build_player_quality_table(
    scored_df: pd.DataFrame,
    export_path: str = None,
) -> pd.DataFrame:
    """
    Build country-level player quality table for all rostered countries.
    Always exports to CSV in outputs/ folder next to this file.
    Also exports full player details per country.
    """
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    if export_path is None:
        export_path = outputs_dir / "player_quality_2026.csv"

    rows = []
    all_player_rows = []

    for country in rosters_2026.keys():
        # Country-level summary
        result = compute_player_quality_score(country, scored_df)
        rows.append(result)

        # Player-level details for this country
        roster_names = list(rosters_2026.get(country, {}).keys())
        drop_cols = [c for c in ['seasons_present'] if c in scored_df.columns]
        matched = scored_df[scored_df['player'].isin(roster_names)].drop(
            columns=drop_cols
        ).copy()

        if len(matched) > 0:
            matched['country'] = country
            all_player_rows.append(matched)

    # Country-level summary export
    df = pd.DataFrame(rows).sort_values(
        "player_quality_score", ascending=False,
        na_position='last'
    ).reset_index(drop=True)
    df.to_csv(export_path, index=False)
    print(f"Exported player quality table → {export_path}")

    # Player-level detail export
    if all_player_rows:
        players_df = pd.concat(all_player_rows, ignore_index=True)

        # Reorder — country first
        priority_cols = ['country', 'player', 'team', 'position_archetype',
                         'coverage_tier', 'age', 'composite_score', 'final_score']
        other_cols = [c for c in players_df.columns if c not in priority_cols]
        players_df = players_df[
            [c for c in priority_cols if c in players_df.columns] + other_cols
        ].sort_values(['country', 'final_score'], ascending=[True, False])

        players_export = outputs_dir / "player_details_2026.csv"
        players_df.to_csv(players_export, index=False)
        print(f"Exported player details → {players_export} ({len(players_df)} players)")

    return df

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "player_score"))
    from player_score_pipeline import get_player_scores
    scored = get_player_scores(verbose=False)
    df = build_player_quality_table(scored)
    print(df.to_string(index=False))