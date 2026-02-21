"""
step1_tactical.py
-----------------
Step 1 of the 2026 World Cup Prediction Framework.

Uses TacticalClustering from clustering_analysis.py for all
sklearn operations (K-Means, PCA, validation, archetype display).

This module handles the framework-specific logic:
    1. Building the 8-dimension feature matrix
    2. Tournament-adjusted residuals (xG_Diff - mu_tournament)
    3. Opponent-adjusted schedule correction
    4. Stage-weighted Cluster Success Scores (Pressure Multiplier)
    5. Tactical Volatility / Risk Profile

Public API
----------
    run_step1(data_dir, metadata_path, n_clusters, archetype_names, random_state)
        returns dict with keys:
            "team_matches"      : full annotated DataFrame (one row per team-match)
            "cluster_profiles"  : archetype centers in original feature space
            "cluster_success"   : weighted success + volatility per archetype
            "clustering"        : fitted TacticalClustering instance (scaler, kmeans, pca inside)
            "characterization"  : output of tc.characterize_archetypes()
            "feature_cols"      : list of feature columns actually used
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_team_metrics, attach_tournament_metadata
from clustering_analysis import TacticalClustering


# ── Constants ─────────────────────────────────────────────────────────────────

# The 8 dimensions from the framework
CLUSTER_FEATURES = [
    "ppda",                    # Pressing intensity  (lower = more aggressive)
    "field_tilt_pct",          # Territorial dominance
    "pass_to_carry_ratio",     # Sequence style      (higher = more passing)
    "epr",                     # Efficiency per possession
    "defensive_line_height",   # Defensive shape
    "total_xg",                # Threat creation
    "possession_pct",          # Ball retention
    "progressive_actions",     # Forward momentum
]

STAGE_WEIGHTS = {
    "Group Stage":   1.0,
    "Round of 16":   1.5,
    "Quarter-Final": 2.0,
    "Semi-Final":    2.5,
    "Final":         3.0,
}


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the 8 cluster features, drop rows with too many nulls,
    median-impute any remaining gaps.
    Returns (feature_df, used_feature_cols).
    """
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing = set(CLUSTER_FEATURES) - set(available)
    if missing:
        print(f"[step1] Warning: missing feature columns: {missing}")
        print(f"        Clustering on {len(available)}/8 dimensions.\n")

    feat_df = df[["match_id", "team"] + available].copy()

    # Drop rows where more than half the features are null
    threshold = len(available) // 2
    feat_df = feat_df.dropna(thresh=2 + threshold)  # 2 = match_id + team cols

    # Median-impute remaining nulls
    for col in available:
        if feat_df[col].isnull().any():
            feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    print(f"[step1] Feature matrix: {len(feat_df)} rows x {len(available)} features after cleaning.")
    return feat_df, available


# ── xG Diff & residuals ───────────────────────────────────────────────────────

def compute_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    xG_Diff = team_xg - opponent_xg for each team-match.
    Requires both teams in a match to appear as rows with the same match_id.
    """
    df = df.copy()

    opp = (
        df[["match_id", "team", "total_xg"]]
        .rename(columns={"team": "opponent", "total_xg": "opp_xg"})
    )
    df = df.merge(opp, on="match_id", how="left")
    df = df[df["team"] != df["opponent"]].copy()
    df["xg_diff"] = df["total_xg"] - df["opp_xg"]
    df = df.drop_duplicates(subset=["match_id", "team"])
    return df


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    residual_raw = xg_diff - mu_tournament
    residual_adj = residual_raw - opponent's avg xg_diff in that tournament
                   (reduces schedule bias from farming weak opponents)
    """
    df = df.copy()

    tourn_mean = df.groupby("tournament")["xg_diff"].transform("mean")
    df["residual_raw"] = df["xg_diff"] - tourn_mean

    opp_avg = (
        df.groupby(["tournament", "team"])["xg_diff"]
        .mean()
        .reset_index()
        .rename(columns={"team": "opponent", "xg_diff": "opp_avg_xg_diff"})
    )
    df = df.merge(opp_avg, on=["tournament", "opponent"], how="left")
    df["residual_adj"] = df["residual_raw"] - df["opp_avg_xg_diff"].fillna(0)
    return df


# ── Stage-weighted cluster success ────────────────────────────────────────────

def compute_cluster_success(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted average residual per cluster using the Pressure Multiplier.

    Returns DataFrame with columns:
        cluster, archetype,
        weighted_success_score,
        volatility_sigma, knockout_volatility_sigma,
        risk_profile,
        n_matches, n_knockout_matches
    """
    df = df.copy()
    df["weighted_residual"] = df["residual_adj"] * df["stage_weight"]

    knockout_stages = {"Round of 16", "Quarter-Final", "Semi-Final", "Final"}
    df["is_knockout"] = df["stage"].isin(knockout_stages)

    records = []
    for cid, grp in df.groupby("cluster"):
        archetype = grp["archetype"].iloc[0] if "archetype" in grp.columns else f"Cluster {cid}"
        n = len(grp)

        w_sum   = grp["weighted_residual"].sum()
        w_denom = grp["stage_weight"].sum()
        success = w_sum / w_denom if w_denom > 0 else np.nan

        sigma    = grp["residual_adj"].std()
        ko       = grp[grp["is_knockout"]]
        sigma_ko = ko["residual_adj"].std() if len(ko) >= 3 else np.nan

        records.append({
            "cluster":                    cid,
            "archetype":                  archetype,
            "weighted_success_score":     round(success, 4),
            "volatility_sigma":           round(sigma, 4),
            "knockout_volatility_sigma":  round(sigma_ko, 4) if not np.isnan(sigma_ko) else None,
            "n_matches":                  n,
            "n_knockout_matches":         len(ko),
        })

    result = pd.DataFrame(records).sort_values("weighted_success_score", ascending=False)
    median_vol = result["volatility_sigma"].median()
    result["risk_profile"] = result["volatility_sigma"].apply(
        lambda s: "High Risk / High Upset" if s > median_vol else "Robust / Predictable"
    )
    return result.reset_index(drop=True)


# ── Master runner ─────────────────────────────────────────────────────────────

def run_step1(
    data_dir:        Path = Path("../output/raw_metrics/men_tourn_2022_24"),
    metadata_path:   Path = None,
    n_clusters:      int  = 5,
    archetype_names: dict = None,
    random_state:    int  = 42,
) -> dict:
    """
    Full Step 1 pipeline.

    Parameters
    ----------
    data_dir        : path to your raw metrics folder
    metadata_path   : optional CSV with (match_id, tournament, stage) columns
    n_clusters      : number of tactical archetypes (4-6 recommended)
    archetype_names : dict {cluster_int: "Label"} — inspect cluster profiles
                      on first run, then pass your labels on the second run
    random_state    : for reproducibility

    Returns
    -------
    dict with keys:
        "team_matches"      -> full annotated team-match DataFrame
        "cluster_profiles"  -> cluster centers in original feature space (+ archetype name)
        "cluster_success"   -> weighted success + volatility per archetype
        "clustering"        -> fitted TacticalClustering instance (tc.scaler, tc.kmeans, tc.pca)
        "characterization"  -> output of tc.characterize_archetypes()
        "feature_cols"      -> feature columns used
    """
    print("=" * 60)
    print("STEP 1: TACTICAL DNA & TOURNAMENT-ADJUSTED RESIDUALS")
    print("=" * 60)

    # 1. Load data
    df = load_team_metrics(data_dir)
    df = attach_tournament_metadata(df, metadata_path)

    # 2. Build feature matrix
    feat_df, feature_cols = build_feature_matrix(df)

    # 3. TacticalClustering — prepare → cluster → PCA
    tc = TacticalClustering(dimensions=feature_cols)
    tc.prepare_data(feat_df)

    cluster_result = tc.run_kmeans(k=n_clusters, random_state=random_state)
    labels         = cluster_result["labels"]
    centers        = cluster_result["centers"]   # DataFrame, original scale, col "cluster" = int

    pca_result = tc.run_pca(n_components=2)
    feat_df = feat_df.copy()
    feat_df["cluster"] = labels
    feat_df["pca_x"]   = pca_result["coords"][:, 0]
    feat_df["pca_y"]   = pca_result["coords"][:, 1]

    # 4. Characterize archetypes (uses your existing method)
    characterization = tc.characterize_archetypes(
        profiles_df     = feat_df,
        cluster_centers = centers,
        labels          = labels,
        archetype_names = archetype_names,
    )

    # Map cluster int -> archetype name string
    cluster_to_name = {cid: info["name"] for cid, info in characterization.items()}
    feat_df["archetype"] = feat_df["cluster"].map(cluster_to_name)

    # Build cluster_profiles table
    profiles = centers.copy()
    profiles["archetype"] = profiles["cluster"].map(cluster_to_name)
    size_map = feat_df.groupby("cluster").size().rename("n_matches")
    profiles = profiles.merge(size_map, on="cluster", how="left")

    # 5. Merge cluster labels back onto full df
    df = df.merge(
        feat_df[["match_id", "team", "cluster", "archetype", "pca_x", "pca_y"]],
        on=["match_id", "team"],
        how="left",
    )

    # 6. xG Diff & residuals
    df = compute_xg_diff(df)
    df = compute_residuals(df)

    # 7. Cluster success scores
    success = compute_cluster_success(df)

    # ── Summary print ─────────────────────────────────────────────────────
    print("\n-- Cluster Success Scores ------------------------------------------")
    print(
        success[["archetype", "weighted_success_score", "volatility_sigma", "risk_profile", "n_matches"]]
        .to_string(index=False)
    )
    print("\n-- Cluster Profiles (feature centers) ------------------------------")
    print(profiles[["archetype"] + feature_cols].to_string(index=False))
    print("=" * 60)
    print("Step 1 complete.\n")

    return {
        "team_matches":     df,
        "cluster_profiles": profiles,
        "cluster_success":  success,
        "clustering":       tc,               # tc.scaler, tc.kmeans, tc.pca all in here
        "characterization": characterization,
        "feature_cols":     feature_cols,
    }