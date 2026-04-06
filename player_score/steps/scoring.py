"""
Step 7 & 8: Intra-Archetype Percentile Scoring + Composite Score Assembly

Order of operations:
1. Join all files on (player, season_name) to build a unified player-season table
2. Decay-weighted collapse: for each metric, compute weighted mean across seasons
   → one row per player
3. Intra-archetype percentile rank for each metric (0-100)
4. Trait category scores: mean of metric percentiles within each category
5. Composite score: position-weighted mean of 4 trait category scores
6. Club tier discount
7. Age multiplier (penalise players over 29)
8. Coverage tier classification
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict
from player_metrics_config import PLAYER_METRICS, TRAIT_CATEGORIES
from player_score.rosters_2026 import rosters_2026
from player_score.player_position_map import get_player_position_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_NORMALISE = {
    "2022": "2021/2022",
    "2023": "2022/2023",
    "2024": "2023/2024",
}

ARCHETYPE_WEIGHTS = {
    "FW": {"Mobility_Intensity": 0.05, "Progression": 0.20, "Control": 0.05, "Final_Third_Output": 0.70},
    "W":  {"Mobility_Intensity": 0.15, "Progression": 0.40, "Control": 0.10, "Final_Third_Output": 0.35},
    "AM": {"Mobility_Intensity": 0.10, "Progression": 0.35, "Control": 0.20, "Final_Third_Output": 0.35},
    "CM": {"Mobility_Intensity": 0.15, "Progression": 0.35, "Control": 0.40, "Final_Third_Output": 0.10},
    "DM": {"Mobility_Intensity": 0.10, "Progression": 0.35, "Control": 0.50, "Final_Third_Output": 0.05},
    "CB": {"Mobility_Intensity": 0.40, "Progression": 0.35, "Control": 0.25, "Final_Third_Output": 0.00},
    "FB": {"Mobility_Intensity": 0.25, "Progression": 0.50, "Control": 0.20, "Final_Third_Output": 0.05},
}

CLUB_TIERS = {
    "Tier_1": {
        "Barcelona", "Real Madrid", "Manchester City", "Liverpool",
        "Arsenal", "Chelsea", "Bayern Munich", "Borussia Dortmund",
        "Paris Saint-Germain", "Inter Milan", "AC Milan", "Juventus",
        "Atlético Madrid", "Manchester United", "Bayer Leverkusen",
        "RB Leipzig",
    },
    "Tier_2": {
        "Tottenham Hotspur", "Newcastle United", "Aston Villa",
        "West Ham United", "Brighton & Hove Albion", "VfB Stuttgart",
        "Eintracht Frankfurt", "Werder Bremen", "Real Sociedad",
        "Athletic Club", "Villarreal", "Sevilla", "Real Betis",
        "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina",
        "Benfica", "Porto", "Sporting CP", "Ajax", "PSV Eindhoven",
        "Feyenoord", "Monaco", "Lille", "Lyon", "Marseille", "Nice",
    },
    "Tier_3": {
        "Celta Vigo", "Getafe", "Rennes", "Lens", "Strasbourg",
        "Wolfsburg", "Freiburg", "Augsburg", "Hoffenheim", "Mainz",
        "Crystal Palace", "Fulham", "Everton", "Leicester City",
        "Celtic", "Rangers", "Galatasaray", "Fenerbahçe",
    },
    "Tier_4": {
        "Al Hilal", "Al Nassr", "Al Ittihad", "Al Ahli",
        "Inter Miami", "LA Galaxy", "New York City",
        "Flamengo", "Boca Juniors", "Santos",
    },
    "Tier_5": {
        "Mumbai City", "Hyderabad", "Kerala Blasters", "Bengaluru",
        "ATK Mohun Bagan", "Chennaiyin", "Jamshedpur", "Odisha",
    },
}

CLUB_MULTIPLIERS = {
    "Tier_1": 1.00,
    "Tier_2": 0.95,
    "Tier_3": 0.88,
    "Tier_4": 0.75,
    "Tier_5": 0.60,
    "__default__": 0.80,
}

COMPETITION_MULTIPLIERS = {
    "La Liga":               1.00,
    "Premier League":        1.00,
    "Serie A":               1.00,
    "1. Bundesliga":         1.00,
    "Ligue 1":               1.00,
    "Champions League":      1.00,
    "UEFA Europa League":    0.85,
    "Copa del Rey":          0.80,
    "Liga Profesional":      0.80,
    "Major League Soccer":   0.70,
    "North American League": 0.65,
    "Indian Super league":   0.55,
    "FIFA World Cup":        0.75,
    "UEFA Euro":             0.75,
    "Copa America":          0.70,
    "African Cup of Nations": 0.65,
    "FIFA U20 World Cup":    0.50,
    "__default__":           0.65,
}

# ---------------------------------------------------------------------------
# Module-level lookups — built once at import
# ---------------------------------------------------------------------------

def build_player_club_map() -> dict:
    player_club = {}
    for country, players in rosters_2026.items():
        for player_name, info in players.items():
            player_club[player_name] = info["club"]
    return player_club


def build_player_age_map() -> dict:
    player_age = {}
    for country, players in rosters_2026.items():
        for player_name, info in players.items():
            player_age[player_name] = info.get("age", 27)
    return player_age


PLAYER_CLUB_MAP = build_player_club_map()
PLAYER_AGE_MAP = build_player_age_map()

# Position map is lazy-loaded on first use via get_player_position_map()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_club_multiplier(player_name: str, competition_name: str = None) -> float:
    """Club quality multiplier — roster lookup first, competition fallback."""
    club = PLAYER_CLUB_MAP.get(player_name)
    if club is not None:
        for tier, clubs in CLUB_TIERS.items():
            if club in clubs:
                return CLUB_MULTIPLIERS[tier]
        return CLUB_MULTIPLIERS["__default__"]
    if competition_name:
        return COMPETITION_MULTIPLIERS.get(competition_name, COMPETITION_MULTIPLIERS["__default__"])
    return COMPETITION_MULTIPLIERS["__default__"]


def get_age_multiplier(age: int) -> float:
    """Only penalise players past peak (30+). Never penalise youth."""
    if age <= 29:
        return 1.00
    elif age <= 31:
        return 0.97
    elif age <= 33:
        return 0.93
    elif age <= 35:
        return 0.88
    else:
        return 0.82


def _compute_coverage_tier(row: pd.Series) -> str:
    seasons = row.get("seasons_present", [])
    competition = row.get("competition_name", "")
    n_seasons = len(seasons) if isinstance(seasons, list) else 1

    CLUB_COMPETITIONS = {
        "La Liga", "Premier League", "Serie A", "1. Bundesliga",
        "Ligue 1", "Champions League", "UEFA Europa League",
        "Liga Profesional", "Major League Soccer",
    }
    TOURNAMENT_COMPETITIONS = {
        "FIFA World Cup", "UEFA Euro", "Copa America",
        "African Cup of Nations", "FIFA U20 World Cup",
    }

    if n_seasons >= 2 and competition in CLUB_COMPETITIONS:
        return "A"
    elif n_seasons >= 1 and competition in CLUB_COMPETITIONS:
        return "B"
    elif competition in TOURNAMENT_COMPETITIONS:
        return "C"
    else:
        return "D"


def _compute_confidence(row: pd.Series) -> float:
    tier = row.get("coverage_tier", "D")
    return {"A": 1.00, "B": 0.97, "C": 0.90, "D": 0.80}.get(tier, 0.80)

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _normalise_seasons(df: pd.DataFrame) -> pd.DataFrame:
    if "season_name" in df.columns:
        df = df.copy()
        df["season_name"] = df["season_name"].replace(SEASON_NORMALISE)
    return df


def _build_unified_table(segmented: Dict[str, pl.DataFrame]) -> pd.DataFrame:
    base = None

    for filename, df in segmented.items():
        norm_cols = [
            f"{k}_norm" for k, v in PLAYER_METRICS.items()
            if v["file"] == filename and f"{k}_norm" in df.columns
        ]
        if not norm_cols:
            continue

        keep_cols = ["player", "team", "season_name", "position_archetype",
                     "archetype_label", "decay_weight", "shrinkage_flag",
                     "competition_name"] + norm_cols
        keep_cols = [c for c in keep_cols if c in df.columns]
        sub = df.select(keep_cols).to_pandas()
        sub = _normalise_seasons(sub)

        if base is None:
            base = sub
        else:
            meta_cols = ["position_archetype", "archetype_label",
                         "decay_weight", "shrinkage_flag"]

            join_on = [c for c in ["player", "season_name"]
                       if c in base.columns and c in sub.columns]

            right_cols = ["player", "season_name", "team"] + norm_cols
            right_cols = [c for c in right_cols if c in sub.columns]

            base = base.merge(
                sub[right_cols],
                on=join_on,
                how="outer",
                suffixes=("", "_right"),
            )

            if "team_right" in base.columns:
                base["team"] = base["team"].fillna(base["team_right"])
                base = base.drop(columns=["team_right"])

            for col in meta_cols:
                if f"{col}_x" in base.columns:
                    base[col] = base[f"{col}_x"].fillna(base.get(f"{col}_y", None))
                    base = base.drop(columns=[c for c in [f"{col}_x", f"{col}_y"]
                                              if c in base.columns])
    return base


def _decay_collapse(unified: pd.DataFrame) -> pd.DataFrame:
    # Fill null archetypes from events position map
    player_position_map = get_player_position_map()
    unified = unified.copy()
    unified["position_archetype"] = unified["position_archetype"].fillna(
        unified["player"].map(player_position_map)
    )

    # Drop rows still missing archetype
    unified = unified[unified["position_archetype"].notna()].copy()

    norm_cols = [c for c in unified.columns if c.endswith("_norm")]

    if "decay_weight" not in unified.columns:
        unified["decay_weight"] = 1.0

    def weighted_mean(group):
        result = {}
        for col in norm_cols:
            vals = group[col]
            weights = group["decay_weight"]
            mask = vals.notna()
            if mask.sum() == 0:
                result[col] = np.nan
            else:
                result[col] = np.average(vals[mask], weights=weights[mask])
        return pd.Series(result)

    def modal(series):
        mode = series.mode()
        return mode.iloc[0] if len(mode) > 0 else series.iloc[0]

    meta_agg = (
        unified.groupby("player")
        .agg(
            team=("team", modal),
            position_archetype=("position_archetype", modal),
            archetype_label=("archetype_label", modal),
            **({ "competition_name": ("competition_name", modal) }
               if "competition_name" in unified.columns else {}),
            seasons_present=("season_name", lambda x: sorted(x.dropna().unique().tolist())),
            decay_weight_max=("decay_weight", "max"),
        )
        .reset_index()
    )

    # Override team with current club from roster
    meta_agg["team"] = meta_agg["player"].map(
        lambda p: PLAYER_CLUB_MAP.get(p, None)
    ).fillna(meta_agg["team"])

    # Club weight
    meta_agg["club_weight"] = meta_agg.apply(
        lambda row: get_club_multiplier(row["player"], row.get("competition_name")),
        axis=1,
    )

    # Age multiplier
    meta_agg["age"] = meta_agg["player"].map(lambda p: PLAYER_AGE_MAP.get(p, 27))
    meta_agg["age_multiplier"] = meta_agg["age"].apply(get_age_multiplier)

    norm_agg = (
        unified.groupby("player")
        .apply(weighted_mean)
        .reset_index()
    )

    return meta_agg.merge(norm_agg, on="player", how="left")


def _intra_archetype_percentile(collapsed: pd.DataFrame, norm_col: str) -> pd.Series:
    result = pd.Series(np.nan, index=collapsed.index)

    for archetype in collapsed["position_archetype"].dropna().unique():
        mask = collapsed["position_archetype"] == archetype
        vals = collapsed.loc[mask, norm_col]
        valid = vals.notna()

        if valid.sum() <= 1:
            result.loc[mask & valid] = 50.0
            continue

        ranks = vals[valid].rank(method="dense")
        max_rank = ranks.max()
        result.loc[mask & valid] = (ranks / max_rank) * 100

    return result


def _compute_category_score(scored: pd.DataFrame, category: str, metric_keys: list) -> pd.Series:
    percentile_cols = [f"{k}_pct" for k in metric_keys if f"{k}_pct" in scored.columns]

    if not percentile_cols:
        return pd.Series(np.nan, index=scored.index)

    result_df = scored[percentile_cols].copy()
    if "packing_pct" in result_df.columns and "packing_norm" in scored.columns:
        no_packing = scored["packing_norm"].isna() | (scored["packing_norm"] == 0)
        result_df.loc[no_packing, "packing_pct"] = np.nan

    return result_df.mean(axis=1, skipna=True)


def _compute_composite(row: pd.Series, category_scores: list) -> float:
    archetype = row.get("position_archetype")
    weights = ARCHETYPE_WEIGHTS.get(archetype, None)

    if weights is None:
        vals = [row[s] for s in category_scores if pd.notna(row[s])]
        return np.mean(vals) if vals else np.nan

    category_map = {
        "Mobility_Intensity_score": "Mobility_Intensity",
        "Progression_score":        "Progression",
        "Control_score":            "Control",
        "Final_Third_Output_score": "Final_Third_Output",
    }

    total_weight = 0.0
    weighted_sum = 0.0
    for score_col in category_scores:
        cat = category_map.get(score_col)
        if cat is None:
            continue
        val = row[score_col]
        if pd.isna(val):
            continue
        w = weights.get(cat, 0.25)
        weighted_sum += val * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else np.nan

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_scores(
    segmented: Dict[str, pl.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7 & 8: SCORING + COMPOSITE ASSEMBLY")
        print("=" * 60)

    # Step 1: Unified table
    if verbose:
        print("  [1/5] Building unified player-season table...")
    unified = _build_unified_table(segmented)
    if verbose:
        print(f"         {len(unified):,} player-season rows, "
              f"{unified['player'].nunique():,} unique players")

    # Step 2: Decay-weighted collapse
    if verbose:
        print("  [2/5] Collapsing seasons with decay weighting...")
    collapsed = _decay_collapse(unified)
    if verbose:
        print(f"         {len(collapsed):,} players after collapse")

    # Step 3: Intra-archetype percentile per metric
    if verbose:
        print("  [3/5] Computing intra-archetype percentiles...")
    norm_cols = [c for c in collapsed.columns if c.endswith("_norm")]
    for norm_col in norm_cols:
        metric_key = norm_col.replace("_norm", "")
        collapsed[f"{metric_key}_pct"] = _intra_archetype_percentile(collapsed, norm_col)
    if verbose:
        print(f"         {len([c for c in collapsed.columns if c.endswith('_pct')])} percentile columns")

    # Step 4: Position-weighted category scores
    if verbose:
        print("  [4/5] Computing category scores...")
    category_scores = []
    for category, metric_keys in TRAIT_CATEGORIES.items():
        score_col = f"{category}_score"
        collapsed[score_col] = _compute_category_score(collapsed, category, metric_keys)
        category_scores.append(score_col)
        if verbose:
            valid = collapsed[score_col].notna().sum()
            median = collapsed[score_col].median()
            print(f"         {category}: {valid:,} players, median={median:.1f}")


    # Step 5: Composite + discounts
    if verbose:
        print("  [5/5] Computing composite scores...")

    collapsed["composite_score"] = collapsed.apply(
        lambda row: _compute_composite(row, category_scores), axis=1
    )

    # Club tier discount
    collapsed["composite_score"] = collapsed["composite_score"] * collapsed["club_weight"]

    # Age multiplier
    collapsed["composite_score"] = collapsed["composite_score"] * collapsed["age_multiplier"]

    # Coverage tier
    collapsed["coverage_tier"] = collapsed.apply(_compute_coverage_tier, axis=1)

    # Confidence penalty
    collapsed["confidence"] = collapsed.apply(_compute_confidence, axis=1)
    collapsed["composite_score"] = collapsed["composite_score"] * collapsed["confidence"]

    if verbose:
        print(f"\n  Composite score: {len(collapsed):,} players")
        print(f"  min={collapsed['composite_score'].min():.1f}, "
              f"median={collapsed['composite_score'].median():.1f}, "
              f"max={collapsed['composite_score'].max():.1f}")

    # Sort and rank
    collapsed = collapsed.sort_values("composite_score", ascending=False).reset_index(drop=True)
    collapsed.index = collapsed.index + 1
    collapsed.index.name = "rank"

    # Require at least 3 out of 4 category scores
    category_coverage = collapsed[category_scores].notna().sum(axis=1)
    collapsed = collapsed[category_coverage >= 3].copy()

    return collapsed


def scoring_summary(scored: pd.DataFrame, top_n: int = 20) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {top_n} PLAYERS — COMPOSITE SCORE")
    print("=" * 60)

    # Sort by final_score if available, else composite_score
    sort_col = 'final_score' if 'final_score' in scored.columns else 'composite_score'
    scored_display = scored.sort_values(sort_col, ascending=False)

    display_cols = ["player", "team", "position_archetype", "coverage_tier", "age",
                    "composite_score", "final_score", "Mobility_Intensity_score", 
                    "Progression_score", "Control_score", "Final_Third_Output_score"]
    display_cols = [c for c in display_cols if c in scored.columns]
    print(scored_display[display_cols].head(top_n).to_string())


def apply_guardian_blend(
    scored: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    from guardians_2025 import TOP_100

    def guardian_rank_to_score(rank: int) -> float:
        """Tiered scoring: top 10 get higher signal."""
        if rank <= 10:
            return 90 - (rank - 1) * (7 / 9)
        elif rank <= 30:
            return 82 - (rank - 11) * (10 / 19)
        else:
            return 71 - (rank - 31) * (16 / 69)

    def guardian_only_score(rank: int) -> float:
        """Guardian-only players capped below model players."""
        if rank <= 10:
            return 65 - (rank - 1) * (5 / 9)
        else:
            return 40 - (rank - 11) * (10 / 89)

    guardian_lookup = {p['player']: p for p in TOP_100}

    def get_final_score(row):
        player = row['player']
        composite = row['composite_score']
        
        if pd.isna(composite) or composite is None:
            return None
        
        if player in guardian_lookup:
            guardian_rank = guardian_lookup[player]['rank']
            guardian_score = guardian_rank_to_score(guardian_rank)
            if guardian_rank <= 10:
                weight = 0.40
            elif guardian_rank <= 30:
                weight = 0.25
            else:
                weight = 0.15
            return (1 - weight) * composite + weight * guardian_score
        
        # Not in guardian list — cap at 70
        return min(composite, 70.0)


    scored = scored.copy()
    scored['final_score'] = scored.apply(get_final_score, axis=1)

    # Add guardian-only players
    scored_players = set(scored['player'].tolist())
    guardian_only_rows = []

    for p in TOP_100:
        if p['player'] not in scored_players:
            guardian_only_rows.append({
                'player': p['player'],
                'team': p['team'],
                'coverage_tier': 'D',
                'composite_score': None,
                'final_score': guardian_only_score(p['rank']),
                'age': None,
            })

    if guardian_only_rows:
        scored = pd.concat([scored, pd.DataFrame(guardian_only_rows)], ignore_index=True)

    scored = scored.sort_values('final_score', ascending=False).reset_index(drop=True)
    scored.index = scored.index + 1
    scored.index.name = 'rank'

    if verbose:
        in_both = sum(1 for p in TOP_100 if p['player'] in scored_players)
        print(f"\n  Guardian blend applied:")
        print(f"    In both (blended):      {in_both}")
        print(f"    Guardian only (capped): {len(guardian_only_rows)}")
        print(f"    Model only (unchanged): {len(scored) - in_both - len(guardian_only_rows)}")

    return scored