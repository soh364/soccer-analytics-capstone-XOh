"""
Step 7 & 8: Intra-Archetype Percentile Scoring + Composite Score Assembly

Order of operations:
1. Join all files on (player, season_name) to build a unified player-season table
2. Decay-weighted collapse: for each metric, compute weighted mean across seasons
3. Intra-archetype percentile rank for each metric (0-100)
4. Position-weighted category scores
5. Club tier discount applied to composite score
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Optional
from player_metrics_config import PLAYER_METRICS, TRAIT_CATEGORIES
from rosters_2026 import rosters_2026

# Metric key → norm column name
METRIC_NORM_COLS = {k: f"{k}_norm" for k in PLAYER_METRICS}

# Which file each norm column lives in
METRIC_FILE = {k: v["file"] for k, v in PLAYER_METRICS.items()}

JOIN_KEYS = ["player", "team", "season_name", "position_archetype",
             "archetype_label", "decay_weight", "shrinkage_flag"]

SEASON_NORMALISE = {
    "2022": "2021/2022",
    "2023": "2022/2023",
    "2024": "2023/2024",
}

# Position-specific category weights
ARCHETYPE_WEIGHTS = {
    "W":  {"Mobility_Intensity": 0.20, "Progression": 0.35, "Control": 0.10, "Final_Third_Output": 0.35},
    "FW": {"Mobility_Intensity": 0.15, "Progression": 0.30, "Control": 0.10, "Final_Third_Output": 0.45},
    "CB": {"Mobility_Intensity": 0.35, "Progression": 0.25, "Control": 0.30, "Final_Third_Output": 0.10},
    "DM": {"Mobility_Intensity": 0.30, "Progression": 0.25, "Control": 0.35, "Final_Third_Output": 0.10},
    "CM": {"Mobility_Intensity": 0.20, "Progression": 0.30, "Control": 0.30, "Final_Third_Output": 0.20},
    "AM": {"Mobility_Intensity": 0.15, "Progression": 0.30, "Control": 0.20, "Final_Third_Output": 0.35},
    "FB": {"Mobility_Intensity": 0.25, "Progression": 0.35, "Control": 0.20, "Final_Third_Output": 0.20},
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
    "African Cup of Nations":0.65,
    "FIFA U20 World Cup":    0.50,
    "__default__":           0.65,
}


def get_club_multiplier(player_name: str, competition_name: str = None) -> float:
    # Priority 1 — roster lookup (current club)
    club = PLAYER_CLUB_MAP.get(player_name)
    if club is not None:
        for tier, clubs in CLUB_TIERS.items():
            if club in clubs:
                return CLUB_MULTIPLIERS[tier]
        return CLUB_MULTIPLIERS["__default__"]

    # Priority 2 — competition fallback
    if competition_name:
        return COMPETITION_MULTIPLIERS.get(
            competition_name,
            COMPETITION_MULTIPLIERS["__default__"]
        )

    # Priority 3 — unknown everything
    return COMPETITION_MULTIPLIERS["__default__"]


def build_player_club_map() -> dict:
    """Flatten rosters_2026 into player -> current club lookup."""
    player_club = {}
    for country, players in rosters_2026.items():
        for player_name, info in players.items():
            player_club[player_name] = info["club"]
    return player_club

PLAYER_CLUB_MAP = build_player_club_map()


def get_club_multiplier(player_name: str, competition_name: str = None) -> float:
    # Priority 1 — roster lookup (current club)
    club = PLAYER_CLUB_MAP.get(player_name)
    if club is not None:
        for tier, clubs in CLUB_TIERS.items():
            if club in clubs:
                return CLUB_MULTIPLIERS[tier]
        return CLUB_MULTIPLIERS["__default__"]

    # Priority 2 — competition fallback
    if competition_name:
        return COMPETITION_MULTIPLIERS.get(
            competition_name,
            COMPETITION_MULTIPLIERS["__default__"]
        )

    # Priority 3 — unknown everything
    return COMPETITION_MULTIPLIERS["__default__"]


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
                suffixes=("", "_right")
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

    # Club weight — roster lookup first, competition fallback for unrostered players
    meta_agg["club_weight"] = meta_agg.apply(
        lambda row: get_club_multiplier(
            row["player"],
            row.get("competition_name")
        ),
        axis=1
    )

    norm_agg = (
        unified.groupby("player")
        .apply(weighted_mean)
        .reset_index()
    )

    collapsed = meta_agg.merge(norm_agg, on="player", how="left")
    return collapsed


def _intra_archetype_percentile(
    collapsed: pd.DataFrame,
    norm_col: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=collapsed.index)

    for archetype in collapsed["position_archetype"].dropna().unique():
        mask = collapsed["position_archetype"] == archetype
        vals = collapsed.loc[mask, norm_col]
        valid = vals.notna()

        if valid.sum() <= 1:
            result.loc[mask & valid] = 50.0
            continue

        ranks = vals[valid].rank(pct=True) * 100
        result.loc[mask & valid] = ranks.values

    return result


def _compute_category_score(
    scored: pd.DataFrame,
    category: str,
    metric_keys: list,
) -> pd.Series:
    percentile_cols = [f"{k}_pct" for k in metric_keys if f"{k}_pct" in scored.columns]

    if not percentile_cols:
        return pd.Series(np.nan, index=scored.index)

    result_df = scored[percentile_cols].copy()
    if "packing_pct" in result_df.columns and "packing_norm" in scored.columns:
        no_packing = scored["packing_norm"].isna() | (scored["packing_norm"] == 0)
        result_df.loc[no_packing, "packing_pct"] = np.nan

    return result_df.mean(axis=1, skipna=True)


def _compute_composite(row: pd.Series, category_scores: list) -> float:
    """Compute position-weighted composite score for a single player."""
    archetype = row.get("position_archetype")
    weights = ARCHETYPE_WEIGHTS.get(archetype, None)

    if weights is None:
        # Equal weights fallback
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


def build_scores(
    segmented: Dict[str, pl.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7 & 8: SCORING + COMPOSITE ASSEMBLY")
        print("=" * 60)

    # Step 1: Build unified table
    if verbose:
        print("  [1/4] Building unified player-season table...")
    unified = _build_unified_table(segmented)
    if verbose:
        print(f"         {len(unified):,} player-season rows, "
              f"{unified['player'].nunique():,} unique players")

    # Step 2: Decay-weighted collapse
    if verbose:
        print("  [2/4] Collapsing seasons with decay weighting...")
    collapsed = _decay_collapse(unified)
    if verbose:
        print(f"         {len(collapsed):,} players after collapse")

    # Step 3: Intra-archetype percentile per metric
    if verbose:
        print("  [3/4] Computing intra-archetype percentiles...")
    norm_cols = [c for c in collapsed.columns if c.endswith("_norm")]
    for norm_col in norm_cols:
        metric_key = norm_col.replace("_norm", "")
        collapsed[f"{metric_key}_pct"] = _intra_archetype_percentile(collapsed, norm_col)

    if verbose:
        pct_cols = [c for c in collapsed.columns if c.endswith("_pct")]
        print(f"         {len(pct_cols)} percentile columns computed")

    # Step 4: Position-weighted category scores
    if verbose:
        print("  [4/4] Computing position-weighted category scores...")

    category_scores = []
    for category, metric_keys in TRAIT_CATEGORIES.items():
        score_col = f"{category}_score"
        collapsed[score_col] = _compute_category_score(collapsed, category, metric_keys)
        category_scores.append(score_col)
        if verbose:
            valid = collapsed[score_col].notna().sum()
            median = collapsed[score_col].median()
            print(f"         {category}: {valid:,} players, median={median:.1f}")

    # Step 5: Position-weighted composite
    collapsed["composite_score"] = collapsed.apply(
        lambda row: _compute_composite(row, category_scores), axis=1
    )

    # Step 6: Apply club tier discount last
    if "club_weight" in collapsed.columns:
        collapsed["composite_score"] = collapsed["composite_score"] * collapsed["club_weight"]

    if verbose:
        print(f"\n  Composite score: {len(collapsed):,} players")
        print(f"  min={collapsed['composite_score'].min():.1f}, "
              f"median={collapsed['composite_score'].median():.1f}, "
              f"max={collapsed['composite_score'].max():.1f}")

    # Sort and rank
    collapsed = collapsed.sort_values("composite_score", ascending=False).reset_index(drop=True)
    collapsed.index = collapsed.index + 1
    collapsed.index.name = "rank"

    # Require at least all 4 category scores
    min_categories = 4
    category_coverage = collapsed[category_scores].notna().sum(axis=1)
    collapsed = collapsed[category_coverage >= min_categories].copy()

    return collapsed


def scoring_summary(scored: pd.DataFrame, top_n: int = 20) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {top_n} PLAYERS — COMPOSITE SCORE")
    print("=" * 60)

    display_cols = ["player", "team", "position_archetype", "composite_score",
                    "Mobility_Intensity_score", "Progression_score",
                    "Control_score", "Final_Third_Output_score"]
    display_cols = [c for c in display_cols if c in scored.columns]
    print(scored[display_cols].head(top_n).to_string())

    print("\n" + "=" * 60)
    print("ARCHETYPE BREAKDOWN — MEDIAN COMPOSITE SCORE")
    print("=" * 60)
    breakdown = (
        scored.groupby("position_archetype")["composite_score"]
        .agg(["count", "median", "mean", "std"])
        .round(1)
        .sort_values("median", ascending=False)
    )
    print(breakdown.to_string())