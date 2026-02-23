"""
Soccer Analytics EDA Template V2 - Succinct Configuration-Driven Analysis.

Uses Polars LazyFrames for memory-efficient processing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl
import psutil

warnings.filterwarnings("ignore")

# Constants
TOP_N = 10
WIDTH = 80
BASE_PATH = Path.cwd().parent
DATA_DIR = BASE_PATH / "data"
POLYMARKET_DIR = DATA_DIR / "Polymarket"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"

# Memory tracking
_process = psutil.Process()
_peak_memory_mb = 0.0


def get_memory_mb() -> float:
    """Get current process memory in MB (RSS - Resident Set Size)."""
    return _process.memory_info().rss / 1024**2


def update_peak() -> float:
    """Update and return peak memory."""
    global _peak_memory_mb
    current = get_memory_mb()
    _peak_memory_mb = max(_peak_memory_mb, current)
    return current


def header(title: str) -> None:
    print("\n" + "=" * WIDTH + f"\n  {title}\n" + "=" * WIDTH)


def mem_report() -> str:
    """Return current and peak memory usage."""
    current = update_peak()
    return f"Memory: {current:.1f} MB (peak: {_peak_memory_mb:.1f} MB)"


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def dist(lf: pl.LazyFrame, col: str, n: int = TOP_N) -> pl.DataFrame:
    """Print and return distribution for a column."""
    r = (
        lf.group_by(col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(n)
        .collect()
    )
    print(r)
    return r


def desc(lf: pl.LazyFrame, col: str) -> None:
    """Print describe stats for a column."""
    print(lf.select(col).collect()[col].describe())


def top(lf: pl.LazyFrame, cols: list[str], sort_col: str, n: int = TOP_N) -> None:
    """Print top N rows sorted by a column."""
    print(lf.select(cols).sort(sort_col, descending=True).head(n).collect())


def safe_run(func, name: str) -> dict[str, Any] | None:
    """Run analysis with error handling."""
    try:
        return func()
    except FileNotFoundError:
        print(f"\n[SKIP] {name}: File not found")
    except Exception as e:
        print(f"\n[ERROR] {name}: {e}")
    return None


# ============ POLYMARKET ANALYZERS ============


def analyze_pm_markets() -> dict[str, Any]:
    header("POLYMARKET: MARKETS")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_markets.parquet")

    total = lf.select(pl.len()).collect()[0, 0]
    stats = lf.select(
        [
            pl.col("active").sum().alias("active"),
            pl.col("closed").sum().alias("closed"),
            pl.col("volume").sum().alias("volume"),
        ]
    ).collect()

    print(
        f"Total: {total:,} | Active: {stats['active'][0]:,} | Closed: {stats['closed'][0]:,}"
    )
    print(f"Total volume: ${stats['volume'][0]:,.2f}")

    sub("Category Distribution")
    dist(lf, "category")

    sub("Top Markets by Volume")
    top(lf, ["question", "volume", "active"], "volume")

    return {"total": total, "active": stats["active"][0], "volume": stats["volume"][0]}


def analyze_pm_tokens() -> dict[str, Any]:
    header("POLYMARKET: TOKENS")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_tokens.parquet")

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("token_id").n_unique().alias("tokens"),
        ]
    ).collect()

    print(
        f"Total: {stats['total'][0]:,} | Markets: {stats['markets'][0]:,} | Tokens: {stats['tokens'][0]:,}"
    )

    sub("Outcome Distribution")
    dist(lf, "outcome")

    return {"total": stats["total"][0], "markets": stats["markets"][0]}


def analyze_pm_trades() -> dict[str, Any]:
    header("POLYMARKET: TRADES")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_trades.parquet").with_columns(
        pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
    )

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("size").sum().alias("size"),
        ]
    ).collect()

    print(f"Total trades: {stats['total'][0]:,} | Markets: {stats['markets'][0]:,}")
    print(f"Total size: {stats['size'][0]:,.2f}")

    sub("Size Statistics")
    desc(lf, "size")

    sub("Price Statistics")
    desc(lf, "price")

    sub("Side Distribution")
    dist(lf, "side")

    times = lf.select(
        [
            pl.col("timestamp").min().alias("first"),
            pl.col("timestamp").max().alias("last"),
        ]
    ).collect()
    print(f"\nDate range: {times['first'][0]} to {times['last'][0]}")

    return {"total": stats["total"][0], "size": stats["size"][0]}


def analyze_pm_odds() -> dict[str, Any]:
    header("POLYMARKET: ODDS HISTORY")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_odds_history.parquet").with_columns(
        pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
    )

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("token_id").n_unique().alias("tokens"),
        ]
    ).collect()

    print(
        f"Snapshots: {stats['total'][0]:,} | Markets: {stats['markets'][0]:,} | Tokens: {stats['tokens'][0]:,}"
    )

    sub("Price Statistics")
    desc(lf, "price")

    return {"snapshots": stats["total"][0]}


def analyze_pm_events() -> dict[str, Any]:
    header("POLYMARKET: EVENT STATS")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_event_stats.parquet")

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("market_count").sum().alias("markets"),
            pl.col("total_volume").sum().alias("volume"),
        ]
    ).collect()

    print(f"Events: {stats['total'][0]:,} | Total markets: {stats['markets'][0]:,}")
    print(f"Total volume: ${stats['volume'][0]:,.2f}")

    sub("Top Events by Volume")
    top(lf, ["event_slug", "market_count", "total_volume"], "total_volume")

    return {"events": stats["total"][0], "volume": stats["volume"][0]}


def analyze_pm_summary() -> dict[str, Any]:
    header("POLYMARKET: SUMMARY")
    lf = pl.scan_parquet(POLYMARKET_DIR / "soccer_summary.parquet").with_columns(
        [
            pl.col("first_trade").cast(pl.Int64).cast(pl.Datetime("ms")),
            pl.col("last_trade").cast(pl.Int64).cast(pl.Datetime("ms")),
        ]
    )

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("trade_count").sum().alias("trades"),
            pl.col("active").sum().alias("active"),
        ]
    ).collect()

    print(
        f"Markets: {stats['total'][0]:,} | Trades: {stats['trades'][0]:,} | Active: {stats['active'][0]:,}"
    )

    sub("Top Markets by Trades")
    top(lf, ["question", "trade_count", "volume"], "trade_count")

    return {"markets": stats["total"][0], "trades": stats["trades"][0]}


# ============ STATSBOMB ANALYZERS ============


def analyze_sb_matches() -> dict[str, Any]:
    header("STATSBOMB: MATCHES")
    lf = pl.scan_parquet(STATSBOMB_DIR / "matches.parquet")

    total = lf.select(pl.len()).collect()[0, 0]
    print(f"Total matches: {total:,}")

    sub("Competition Distribution")
    dist(lf, "competition_name")

    sub("Season Distribution")
    dist(lf, "season_name")

    sub("Score Statistics")
    goals = lf.select(
        (pl.col("home_score") + pl.col("away_score")).alias("total")
    ).collect()
    print(
        f"Goals per match: mean={goals['total'].mean():.2f}, median={goals['total'].median():.1f}"
    )

    sub("Match Results")
    results = lf.select(
        [
            pl.when(pl.col("home_score") > pl.col("away_score"))
            .then(pl.lit("Home"))
            .when(pl.col("away_score") > pl.col("home_score"))
            .then(pl.lit("Away"))
            .otherwise(pl.lit("Draw"))
            .alias("result")
        ]
    )
    dist(results, "result")

    return {"matches": total}


def analyze_sb_events() -> dict[str, Any]:
    header("STATSBOMB: EVENTS")
    lf = pl.scan_parquet(STATSBOMB_DIR / "events.parquet")

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("match_id").n_unique().alias("matches"),
            pl.col("type").n_unique().alias("types"),
        ]
    ).collect()

    print(
        f"Events: {stats['total'][0]:,} | Matches: {stats['matches'][0]:,} | Types: {stats['types'][0]:,}"
    )

    sub("Event Type Distribution")
    dist(lf, "type", 15)

    sub("Shot Analysis")
    shots = lf.filter(pl.col("type") == "Shot")
    shot_count = shots.select(pl.len()).collect()[0, 0]
    print(f"Total shots: {shot_count:,}")

    sub("Pass Analysis")
    passes = lf.filter(pl.col("type") == "Pass")
    pass_stats = passes.select(
        [
            pl.len().alias("total"),
            pl.col("pass_outcome").is_null().sum().alias("successful"),
        ]
    ).collect()
    pct = pass_stats["successful"][0] / pass_stats["total"][0] * 100
    print(f"Passes: {pass_stats['total'][0]:,} | Success rate: {pct:.1f}%")

    sub("Most Active Players")
    top(
        lf.group_by("player").agg(pl.len().alias("count")), ["player", "count"], "count"
    )

    return {
        "events": stats["total"][0],
        "shots": shot_count,
        "passes": pass_stats["total"][0],
    }


def analyze_sb_lineups() -> dict[str, Any]:
    header("STATSBOMB: LINEUPS")
    lf = pl.scan_parquet(STATSBOMB_DIR / "lineups.parquet")

    # 1. Basic Stats & Quality (Duplicates)
    stats = lf.select([
        pl.len().alias("total"),
        pl.col("match_id").n_unique().alias("matches"),
        pl.col("player_name").n_unique().alias("players"),
    ]).collect()
    
    total_rec = stats['total'][0]
    total_matches = stats['matches'][0]
    
    # Calculate duplicates (position changes)
    dup_count = (
        lf.group_by(["match_id", "player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.col("len").sum())
        .collect()[0, 0] or 0
    )

    print(f"Records: {total_rec:,} | Matches: {total_matches:,} | Unique Players: {stats['players'][0]:,}")
    print(f"Quality: {dup_count:,} duplicate (match_id, player_id) pairs ({(dup_count/total_rec)*100:.1f}%), likely position changes.")

    # 2. Time Parsing Logic
    def parse_min(col_name):
        return (
            pl.col(col_name).str.split(":").list.get(0).cast(pl.Float64) + 
            pl.col(col_name).str.split(":").list.get(1).cast(pl.Float64) / 60
        ).fill_null(90.0)

    # Calculate Duration and Starter status
    lf_enriched = lf.with_columns([
        (parse_min("to_time") - parse_min("from_time")).alias("duration"),
        (pl.col("from_time") == "00:00").alias("is_starter")
    ])

    # Aggregate by player-match (collapsing position changes)
    pm_stats = lf_enriched.group_by(["match_id", "player_id"]).agg([
        pl.col("duration").sum(),
        pl.col("is_starter").any().alias("started"),
        pl.col("player_name").first()
    ]).collect()

    # 3. Participation & Subs
    played_df = pm_stats.filter(pl.col("duration") > 0)
    played_count = played_df.height
    print(f"Participation: Only {(played_count/total_rec)*100:.1f}% actually played ({played_count:,}).")

    avg_subs = (played_df.filter(~pl.col("started")).height / total_matches)
    starter_avg = played_df.filter(pl.col("started")).select(pl.col("duration").mean())[0,0]
    sub_avg = played_df.filter(~pl.col("started")).select(pl.col("duration").mean())[0,0]
    print(f"Substitutions: {avg_subs:.1f} avg per match. Starters {starter_avg:.1f} min avg, subs {sub_avg:.1f} min avg.")

    # 4. Tables (The original tables you wanted to keep)
    sub("Position Distribution")
    dist(lf, "position_name")

    sub("Playing Time Distribution")
    time_bins = played_df.with_columns(
        pl.when(pl.col("duration") >= 90).then(pl.lit("90+ min"))
        .when(pl.col("duration") >= 60).then(pl.lit("60-90 min"))
        .when(pl.col("duration") >= 30).then(pl.lit("30-60 min"))
        .otherwise(pl.lit("< 30 min")).alias("bin")
    )
    dist(time_bins.lazy(), "bin")

    # 5. Cards
    sub("Card Analysis")
    card_lf = lf.filter(pl.col("card_type").is_not_null())
    card_results = card_lf.collect()
    total_cards = card_results.height
    yellow_pct = (card_results.filter(pl.col("card_type").str.contains("Yellow")).height / total_cards) * 100
    
    print(f"Total cards: {total_cards:,} ({ (total_cards/total_rec)*100:.1f}% of records)")
    print(f"Yellow card frequency: {yellow_pct:.1f}%")
    
    top_carded = card_results.group_by("player_name").len().sort("len", descending=True).head(5)
    print("\nTop Carded Players:")
    print(top_carded)

    return {
        "records": total_rec,
        "played": played_count,
        "cards": total_cards
    }

def analyze_sb_360() -> dict[str, Any]:
    header("STATSBOMB: THREE SIXTY")
    lf = pl.scan_parquet(STATSBOMB_DIR / "three_sixty.parquet")

    stats = lf.select(
        [
            pl.len().alias("total"),
            pl.col("event_uuid").n_unique().alias("events"),
            pl.col("match_id").n_unique().alias("matches"),
        ]
    ).collect()

    print(
        f"Records: {stats['total'][0]:,} | Events: {stats['events'][0]:,} | Matches: {stats['matches'][0]:,}"
    )

    sub("Spatial Distribution")
    print("X coords:")
    desc(lf, "location_x")
    print("\nY coords:")
    desc(lf, "location_y")

    return {"records": stats["total"][0], "events": stats["events"][0]}


def analyze_sb_reference() -> dict[str, Any]:
    header("STATSBOMB: REFERENCE")
    lf = pl.scan_parquet(STATSBOMB_DIR / "reference.parquet")

    # 1. Entity Distribution (Now at the top)
    sub("Entity Distribution")
    dist(lf, "table_name")

    # 3. Name Collisions (Different IDs, Same Name)
    sub("Entity Name Collisions")
    # Group by table and name to find different IDs sharing a name
    collisions = (
        lf.group_by(["table_name", "name"])
        .agg(pl.col("id").n_unique().alias("id_count"))
        .filter(pl.col("id_count") > 1)
        .collect()
    )
    
    p_dupes = collisions.filter(pl.col("table_name") == "player").height
    t_dupes = collisions.filter(pl.col("table_name") == "team").height
    
    print(f"- Players: {p_dupes} duplicate names (different entities)")
    print(f"- Teams: {t_dupes} duplicate names (different entities)")
    if not collisions.is_empty():
        print(collisions.sort("id_count", descending=True).head(5))

    # 4. Team Metadata (Gender Table)
    sub("Team Gender Analysis (Top 3)")
    gender_sample = (
        lf.filter((pl.col("table_name") == "team") & (pl.col("extra_info").is_not_null()))
        .select(["name", "extra_info"])
        .head(3)
        .collect()
    )
    print(gender_sample)

    return {
        "player_collisions": p_dupes,
        "team_collisions": t_dupes
    }


def cross_analysis() -> dict[str, Any]:
    header("CROSS-DATASET ANALYSIS")

    pm = pl.scan_parquet(POLYMARKET_DIR / "soccer_markets.parquet")
    sb = pl.scan_parquet(STATSBOMB_DIR / "matches.parquet")

    pm_stats = pm.select(
        [
            pl.len().alias("n"),
            pl.col("created_at").min().alias("min"),
            pl.col("created_at").max().alias("max"),
        ]
    ).collect()
    sb_stats = sb.select(
        [
            pl.len().alias("n"),
            pl.col("match_date").min().alias("min"),
            pl.col("match_date").max().alias("max"),
        ]
    ).collect()

    print(
        f"Polymarket: {pm_stats['n'][0]:,} markets ({pm_stats['min'][0]} to {pm_stats['max'][0]})"
    )
    print(
        f"Statsbomb: {sb_stats['n'][0]:,} matches ({sb_stats['min'][0]} to {sb_stats['max'][0]})"
    )

    sub("File Sizes")
    for name, dir_path in [
        ("Polymarket", POLYMARKET_DIR),
        ("Statsbomb", STATSBOMB_DIR),
    ]:
        print(f"\n{name}:")
        for f in dir_path.glob("*.parquet"):
            count = pl.scan_parquet(f).select(pl.len()).collect()[0, 0]
            print(f"  {f.name}: {count:,}")

    return {"pm_markets": pm_stats["n"][0], "sb_matches": sb_stats["n"][0]}


def main() -> None:
    global _peak_memory_mb
    _peak_memory_mb = get_memory_mb()  # Initialize with baseline
    baseline = _peak_memory_mb

    header("SOCCER ANALYTICS EDA (V2)")
    print(f"Baseline memory: {baseline:.1f} MB")

    if POLYMARKET_DIR.exists():
        for fn in [
            analyze_pm_markets,
            analyze_pm_tokens,
            analyze_pm_trades,
            analyze_pm_odds,
            analyze_pm_events,
            analyze_pm_summary,
        ]:
            safe_run(fn, fn.__name__)
            update_peak()
        print(f"\n[Polymarket complete] {mem_report()}")
    else:
        print("\n[SKIP] Polymarket directory not found")

    if STATSBOMB_DIR.exists():
        for fn in [
            analyze_sb_matches,
            analyze_sb_events,
            analyze_sb_lineups,
            analyze_sb_360,
            analyze_sb_reference,
        ]:
            safe_run(fn, fn.__name__)
            update_peak()
        print(f"\n[Statsbomb complete] {mem_report()}")
    else:
        print("\n[ERROR] Statsbomb directory not found")

    if POLYMARKET_DIR.exists() and STATSBOMB_DIR.exists():
        safe_run(cross_analysis, "cross_analysis")
        update_peak()

    header("EDA COMPLETE")
    final = get_memory_mb()
    print(f"Baseline memory: {baseline:.2f} MB")
    print(f"Final memory: {final:.2f} MB")
    print(f"Peak memory: {_peak_memory_mb:.2f} MB")
    print(f"Memory used above baseline: {_peak_memory_mb - baseline:.2f} MB")


if __name__ == "__main__":
    main()
