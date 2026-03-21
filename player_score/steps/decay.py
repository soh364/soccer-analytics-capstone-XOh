"""
Step 3: Time Decay Weighting
- Weights each player-season row by recency
- 2023/24 → 1.0, 2022/23 → 0.75, 2021/22 → 0.50
- Adds a `decay_weight` column to each DataFrame
- Does NOT yet collapse multi-season rows — that happens in Step 4/5
  when we have normalised values to combine
"""
import polars as pl
from typing import Dict

# Season name → decay weight
# Handles both club season format and tournament single-year format
DECAY_WEIGHTS = {
    # Club seasons
    "2023/2024": 1.00,
    "2022/2023": 0.75,
    "2021/2022": 0.50,
    # Tournament equivalents
    "2024": 1.00,
    "2023": 0.75,
    "2022": 0.50,
}


def apply_decay(
    filtered: Dict[str, pl.DataFrame],
    decay_weights: Dict[str, float] = None,
    verbose: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Add a decay_weight column to each filtered DataFrame based on season_name.

    Args:
        filtered: dict of {filename: DataFrame} from filtering step
        decay_weights: optional override of season → weight mapping
        verbose: print per-file summary

    Returns:
        dict of {filename: DataFrame} with decay_weight column added
    """
    weights = decay_weights or DECAY_WEIGHTS

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: TIME DECAY WEIGHTING")
        print("Weights:", {k: v for k, v in weights.items() if "/" in k})
        print("=" * 60)

    decayed = {}

    for filename, df in filtered.items():
        if "season_name" not in df.columns:
            if verbose:
                print(f"  [decay] {filename}: no season_name column, assigning weight 1.0")
            decayed[filename] = df.with_columns(pl.lit(1.0).alias("decay_weight"))
            continue

        # Map season_name → decay_weight
        # Unknown seasons default to 1.0 with a warning
        season_values = df["season_name"].unique().to_list()
        unknown = [s for s in season_values if s not in weights]
        if unknown and verbose:
            print(f"  [decay] {filename}: unknown seasons {unknown}, defaulting to 1.0")

        weight_map = {s: weights.get(s, 1.0) for s in season_values}

        df = df.with_columns(
            pl.col("season_name")
            .replace(weight_map)
            .cast(pl.Float64)
            .alias("decay_weight")
        )

        if verbose:
            breakdown = (
                df.group_by("season_name")
                .agg(
                    pl.len().alias("rows"),
                    pl.first("decay_weight").alias("weight")
                )
                .sort("season_name")
            )
            print(f"  [decay] {filename}:")
            for row in breakdown.iter_rows(named=True):
                print(f"           {row['season_name']}: {row['rows']:,} rows × {row['weight']}")

        decayed[filename] = df

    return decayed


def decay_summary(decayed: Dict[str, pl.DataFrame]) -> None:
    """Print a concise summary of decay-weighted DataFrames."""
    print("\n" + "=" * 60)
    print("DECAY SUMMARY")
    print("=" * 60)
    for filename, df in decayed.items():
        n_players = df["player"].n_unique() if "player" in df.columns else "N/A"
        weights = sorted(df["decay_weight"].unique().to_list()) if "decay_weight" in df.columns else []
        print(f"  {filename}")
        print(f"    {len(df):,} rows | {n_players} players | weights present: {weights}")