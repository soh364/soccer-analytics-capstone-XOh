"""
Player Score Pipeline
Runs the full player scoring pipeline and returns the final scored DataFrame.

Usage:
    from player_score.player_score_pipeline import get_player_scores
    scored = get_player_scores()
"""

import sys
from pathlib import Path

# Path setup
PLAYER_SCORE_DIR = Path(__file__).parent
PROJECT_ROOT = PLAYER_SCORE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "Statsbomb"

sys.path.insert(0, str(PLAYER_SCORE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import pandas as pd

from loader import load_player_data_for_scoring
from aggregation import aggregate_all
from steps.filter import filter_all
from steps.decay import apply_decay
from steps.normalization import normalize_all
from steps.shrinkage import apply_shrinkage
from steps.segmentation import apply_segmentation
from steps.scoring import build_scores, apply_guardian_blend


def get_player_scores(verbose: bool = False) -> pd.DataFrame:
    """
    Run full player scoring pipeline.
    Returns final scored DataFrame with composite_score and final_score.
    """
    if verbose:
        print("=" * 60)
        print("PLAYER SCORE PIPELINE")
        print("=" * 60)

    # Step 1 — Load raw data
    if verbose:
        print("\n[1/8] Loading raw data...")
    matches = pl.from_pandas(pd.read_parquet(DATA_DIR / "matches.parquet"))
    raw = load_player_data_for_scoring("recent_club_players")

    # Step 2 — Aggregate match → season level
    if verbose:
        print("[2/8] Aggregating to player-season level...")
    clean = aggregate_all(raw, matches, verbose=False)

    # Step 3 — Filter
    if verbose:
        print("[3/8] Filtering...")
    filtered = filter_all(clean, verbose=False)

    # Step 4 — Decay weighting
    if verbose:
        print("[4/8] Applying decay weights...")
    decayed = apply_decay(filtered, verbose=False)

    # Step 5 — Normalization
    if verbose:
        print("[5/8] Normalizing metrics...")
    normalized = normalize_all(decayed, verbose=False)

    # Step 6 — Bayesian shrinkage
    if verbose:
        print("[6/8] Applying shrinkage...")
    shrunk = apply_shrinkage(
        normalized,
        lineups_path=DATA_DIR / "lineups.parquet",
        verbose=False,
    )

    # Step 7 — Segmentation
    if verbose:
        print("[7/8] Segmenting by archetype...")
    segmented = apply_segmentation(shrunk, verbose=False)

    # Step 8 — Scoring + guardian blend
    if verbose:
        print("[8/8] Computing scores...")
    scored_raw = build_scores(segmented, verbose=False)
    scored = apply_guardian_blend(scored_raw, verbose=False)

    if verbose:
        print(f"\n✓ Done — {len(scored)} players scored")
        print(f"  Coverage tiers: {scored['coverage_tier'].value_counts().to_dict()}")

    return scored


if __name__ == "__main__":
    scored = get_player_scores(verbose=True)
    print(scored[['player', 'team', 'position_archetype',
                  'coverage_tier', 'final_score']].head(20).to_string())