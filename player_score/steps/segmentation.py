"""
Step 6: Archetype Segmentation
- Removes GK rows from all files
- Validates position_archetype coverage
- Provides peer group context for Step 7 percentile scoring
- No new columns added beyond cleaning — position_archetype from Step 5 is used directly
- 7 archetypes: CB, FB, DM, CM, AM, W, FW
"""

import polars as pl
from typing import Dict

VALID_ARCHETYPES = ["CB", "FB", "DM", "CM", "AM", "W", "FW"]

ARCHETYPE_LABELS = {
    "CB": "Centre-Back",
    "FB": "Fullback / Wing-Back",
    "DM": "Defensive Midfielder",
    "CM": "Central Midfielder",
    "AM": "Attacking Midfielder",
    "W":  "Winger",
    "FW": "Forward / Striker",
}


def apply_segmentation(
    shrunk: Dict[str, pl.DataFrame],
    verbose: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Apply archetype segmentation to all shrunk DataFrames.

    - Drops GK rows
    - Drops rows with unrecognised archetypes
    - Adds archetype_label column (human-readable)

    Args:
        shrunk: dict of {filename: DataFrame} from shrinkage step
        verbose: print per-file summary

    Returns:
        dict of {filename: DataFrame} with GKs removed and archetype_label added
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: ARCHETYPE SEGMENTATION")
        print(f"Archetypes: {VALID_ARCHETYPES}")
        print("=" * 60)

    segmented = {}

    for filename, df in shrunk.items():
        n_before = len(df)

        if "position_archetype" not in df.columns:
            if verbose:
                print(f"  [seg] {filename}: no position_archetype column, passing through")
            segmented[filename] = df
            continue

        # Drop GKs and any unrecognised archetypes
        df = df.filter(pl.col("position_archetype").is_in(VALID_ARCHETYPES))

        # Add human-readable label
        df = df.with_columns(
            pl.col("position_archetype")
            .replace(ARCHETYPE_LABELS)
            .alias("archetype_label")
        )

        n_after = len(df)
        n_dropped = n_before - n_after

        if verbose:
            breakdown = (
                df.group_by("position_archetype")
                .agg(pl.len().alias("count"))
                .sort("position_archetype")
            )
            print(f"  [seg] {filename}: {n_before:,} → {n_after:,} rows ({n_dropped} dropped)")
            for row in breakdown.iter_rows(named=True):
                label = ARCHETYPE_LABELS.get(row["position_archetype"], row["position_archetype"])
                print(f"         {row['position_archetype']} ({label}): {row['count']:,}")

        segmented[filename] = df

    return segmented


def segmentation_summary(segmented: Dict[str, pl.DataFrame]) -> None:
    """Print archetype distribution across all files."""
    print("\n" + "=" * 60)
    print("SEGMENTATION SUMMARY")
    print("=" * 60)

    # Aggregate archetype counts across all files
    all_counts = {}
    for filename, df in segmented.items():
        if "position_archetype" not in df.columns:
            continue
        counts = (
            df.group_by("position_archetype")
            .agg(pl.len().alias("count"))
            .to_dicts()
        )
        for row in counts:
            arch = row["position_archetype"]
            all_counts[arch] = all_counts.get(arch, {})
            all_counts[arch][filename] = row["count"]

    # Print per-archetype breakdown
    for arch in VALID_ARCHETYPES:
        if arch not in all_counts:
            continue
        label = ARCHETYPE_LABELS[arch]
        print(f"\n  {arch} — {label}")
        for filename, count in sorted(all_counts[arch].items()):
            print(f"    {filename}: {count:,} rows")