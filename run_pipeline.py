"""
run_pipeline.py
───────────────
2026 FIFA World Cup Readiness Framework — Full Pipeline

Single entry point. Run this script to reproduce all outputs from scratch.

Usage:
    python run_pipeline.py                    # full run, verbose
    python run_pipeline.py --quiet            # suppress step-level logs
    python run_pipeline.py --skip-mc          # skip Monte Carlo (saves ~30s)
    python run_pipeline.py --mc-sims 1000     # fewer simulations (for testing)

Outputs produced:
    player_score/outputs/player_quality_2026.csv   — country-level player scores
    player_score/outputs/player_details_2026.csv   — individual player scores
    tactical_clustering/outputs/team_archetypes.csv — tactical archetype assignments
    composite_score/outputs/team_readiness_2026.csv — composite readiness scores
    composite_score/outputs/monte_carlo_2026.csv    — Monte Carlo simulation results

Prerequisites:
    - StatsBomb data directory at data/Statsbomb/ (parquet files)
    - pip install polars==1.3.0 pandas scikit-learn scipy rapidfuzz

Authors: Soomi Oh, Yoo Mi Oh — GT OMSA Capstone 2026
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "player_score"))
sys.path.insert(0, str(PROJECT_ROOT / "composite_score"))
sys.path.insert(0, str(PROJECT_ROOT / "tactical_clustering"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _step(msg: str) -> None:
    print(f"  ▸ {msg}")


def _ok(msg: str, elapsed: float = None) -> None:
    suffix = f"  ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"  ✓ {msg}{suffix}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


# ── Stage 1: Player Scoring ───────────────────────────────────────────────────

def run_player_scoring(verbose: bool = True) -> "pd.DataFrame":
    """
    Run the full 8-step player scoring pipeline.

    Steps:
        1. Load raw StatsBomb player metrics (club seasons 2021/22 – 2023/24)
        2. Aggregate from match-level to player × season
        3. Filter: hard threshold 270 min, shrinkage floor 180 min
        4. Time decay: 2023/24 → 1.0, 2022/23 → 0.90, 2021/22 → 0.80
        5. Per-season normalization (log / log1p / rank / zscore by metric)
        6. Bayesian shrinkage toward positional mean for low-minute players
        7. Archetype segmentation: CB / FB / DM / CM / AM / W / FW
        8. Intra-archetype percentile → composite score → Guardian blend

    Returns:
        pd.DataFrame — one row per player with composite_score and final_score
    """
    _header("STAGE 1 — PLAYER SCORING")
    t0 = time.time()

    from player_score.player_score_pipeline import get_player_scores
    from player_score.player_aggregator import build_player_quality_table

    _step("Running 8-step scoring pipeline...")
    scored = get_player_scores(verbose=False)
    _ok(f"{len(scored):,} players scored", time.time() - t0)

    _step("Aggregating to country level...")
    t1 = time.time()
    quality_df = build_player_quality_table(scored)
    _ok(f"{len(quality_df)} countries — outputs written to player_score/outputs/",
        time.time() - t1)

    if verbose:
        print()
        top10 = quality_df.head(10)[
            ["country", "effective_score", "player_coverage_confidence",
             "n_players_scored", "top_player"]
        ]
        print(top10.to_string(index=False))

    return scored


# ── Stage 2: Tactical Clustering ──────────────────────────────────────────────

def run_tactical_clustering(verbose: bool = True) -> "pd.DataFrame":
    """
    Run the tactical clustering pipeline.

    Steps:
        1. Load 8 team-level metrics from 2022–2024 tournaments
           (WC 2022, Euro 2024, Copa América 2024, AFCON 2023)
        2. Aggregate to one row per team (mean + volatility features)
        3. Exclude Georgia & Slovenia (extreme PPDA outliers, not in WC 2026)
        4. Cap PPDA + EPR at 95th percentile, then StandardScaler
        5. KMeans k=6 (n_init=20, random_state=42)
        6. GMM cross-validation → per-team confidence scores
        7. Team name reconciliation against rosters_2026.py
        8. Export to tactical_clustering/outputs/team_archetypes.csv

    Returns:
        pd.DataFrame — one row per team with archetype and gmm_confidence
    """
    _header("STAGE 2 — TACTICAL CLUSTERING")
    t0 = time.time()

    from tactical_clustering.tc_pipeline import export_archetypes_csv

    _step("Running clustering pipeline (k=6, 69 teams)...")
    archetype_df = export_archetypes_csv()
    _ok(f"{len(archetype_df)} teams clustered — output written to tactical_clustering/outputs/",
        time.time() - t0)

    if verbose:
        print()
        dist = archetype_df["archetype"].value_counts()
        for arch, count in dist.items():
            score = archetype_df.loc[
                archetype_df["archetype"] == arch, "archetype_score"
            ].iloc[0]
            print(f"    [{score:>2}] {arch:<30} {count} teams")

    return archetype_df


# ── Stage 3: Composite Scoring ────────────────────────────────────────────────

def run_composite_scoring(
    scored: "pd.DataFrame",
    archetype_df: "pd.DataFrame",
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Combine all signals into a single team readiness score.

    Component weights:
        Player quality      35%  — top-5 average, confidence-weighted FIFA fallback
        Tactical archetype  20%  — archetype score × GMM confidence
        FIFA ranking        15%  — log-scaled, rank 1 → 100, rank 90 → 20
        Club cohesion       10%  — % of squad from same club (log-scaled)
        Squad age            5%  — peak at 26–29, penalties outside
        Coach tenure         5%  — sweet spot 3–7 years, staleness penalty >10
        Tournament exp       5%  — log-scaled WC appearances
        Confederation bonus  5%  — host nation multiplier (US/CAN/MEX: ×1.05)

    Returns:
        pd.DataFrame — one row per country with final_score and all components
    """
    _header("STAGE 3 — COMPOSITE SCORING")
    t0 = time.time()

    import pandas as pd
    from composite_score.composite_scorer import get_team_readiness_scores

    _step("Computing composite readiness scores...")
    readiness_df = get_team_readiness_scores(
        scored_df=scored,
        archetype_df=archetype_df,
        verbose=False,
    )

    # Export
    out_path = PROJECT_ROOT / "composite_score" / "outputs" / "team_readiness_2026.csv"
    readiness_df.to_csv(out_path, index=True)
    _ok(f"{len(readiness_df)} countries scored — output written to composite_score/outputs/",
        time.time() - t0)

    if verbose:
        print()
        cols = ["country", "final_score", "player_quality_score",
                "tactical_archetype", "fifa_score", "archetype_available"]
        print(readiness_df[cols].head(15).to_string())

    return readiness_df


# ── Stage 4: Monte Carlo Simulation ──────────────────────────────────────────

def run_monte_carlo(
    readiness_df: "pd.DataFrame",
    n_simulations: int = 10_000,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Simulate the 2026 World Cup tournament structure 10,000 times.

    Tournament structure:
        12 groups × 4 teams → round-robin (3 matches each)
        32 qualifiers: top 2 from each group + 8 best third-place
        Knockout: R32 → R16 → QF → SF → Final

    Match model:
        Win probability via logistic function on score difference (σ=15)
        Goals via Poisson (λ adjusted by relative team strength)
        Draws allowed in group stage; knockout uses extra time / penalties

    Returns:
        pd.DataFrame — per-team probabilities for each tournament stage
    """
    _header("STAGE 4 — MONTE CARLO SIMULATION")
    t0 = time.time()

    from composite_score.monte_carlo import run_monte_carlo as _run_mc

    _step(f"Running {n_simulations:,} simulations...")
    mc_df = _run_mc(readiness_df, n_simulations=n_simulations, verbose=False)
    _ok(f"Done — results written to composite_score/outputs/monte_carlo_2026.csv",
        time.time() - t0)

    if verbose:
        print()
        cols = ["team", "group", "readiness_score",
                "p_champion", "p_semi_final", "p_quarter_final", "p_group_exit"]
        print(mc_df[cols].head(15).to_string())

    return mc_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2026 World Cup Readiness Framework — Full Pipeline"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output tables"
    )
    parser.add_argument(
        "--skip-mc", action="store_true",
        help="Skip Monte Carlo simulation"
    )
    parser.add_argument(
        "--mc-sims", type=int, default=10_000,
        help="Number of Monte Carlo simulations (default: 10000)"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    t_total = time.time()

    print("\n" + "═" * 60)
    print("  2026 FIFA WORLD CUP READINESS FRAMEWORK")
    print("  GT OMSA Capstone — Soomi Oh, Yoo Mi Oh")
    print("═" * 60)

    # ── Run all stages ────────────────────────────────────────────────────────
    scored       = run_player_scoring(verbose=verbose)
    archetype_df = run_tactical_clustering(verbose=verbose)
    readiness_df = run_composite_scoring(scored, archetype_df, verbose=verbose)

    if not args.skip_mc:
        mc_df = run_monte_carlo(readiness_df, n_simulations=args.mc_sims, verbose=verbose)
    else:
        _warn("Monte Carlo skipped (--skip-mc flag)")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'═' * 60}")
    print(f"  ALL STAGES COMPLETE  ({elapsed:.1f}s total)")
    print(f"{'═' * 60}")
    print(f"\n  Outputs:")
    print(f"    player_score/outputs/player_quality_2026.csv")
    print(f"    player_score/outputs/player_details_2026.csv")
    print(f"    tactical_clustering/outputs/team_archetypes.csv")
    print(f"    composite_score/outputs/team_readiness_2026.csv")
    if not args.skip_mc:
        print(f"    composite_score/outputs/monte_carlo_2026.csv")
    print()


if __name__ == "__main__":
    main()
