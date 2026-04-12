"""
monte_carlo.py
Monte Carlo simulation of the 2026 FIFA World Cup.
Uses team readiness scores to determine match win probabilities.
Simulates 10,000 tournaments and tracks outcomes per team.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
np.random.seed(42)

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ---------------------------------------------------------------------------
# Match simulation
# ---------------------------------------------------------------------------

def get_win_prob(score_a: float, score_b: float) -> float:
    """
    Convert team scores to win probability using logistic function.
    Score difference of 10 points → ~60% win probability.
    Score difference of 25 points → ~75% win probability.
    """
    diff = score_a - score_b
    prob = 1 / (1 + np.exp(-diff / 15))
    return prob


def simulate_match(
    team_a: str,
    team_b: str,
    scores: dict,
    knockout: bool = False,
) -> tuple[str, str, int, int]:
    """
    Simulate a single match.
    Returns (winner, loser, goals_a, goals_b).
    In group stage draws are possible.
    In knockout stage, simulate extra time/penalties if draw.
    """
    score_a = scores.get(team_a, 50.0)
    score_b = scores.get(team_b, 50.0)
    win_prob_a = get_win_prob(score_a, score_b)

    # Simulate goals — Poisson distributed
    lambda_a = 0.8 + (score_a - 50) / 100
    lambda_b = 0.8 + (score_b - 50) / 100
    lambda_a = max(0.3, lambda_a)
    lambda_b = max(0.3, lambda_b)

    goals_a = np.random.poisson(lambda_a)
    goals_b = np.random.poisson(lambda_b)

    if goals_a > goals_b:
        return team_a, team_b, goals_a, goals_b
    elif goals_b > goals_a:
        return team_b, team_a, goals_b, goals_a
    else:
        # Draw
        if knockout:
            # Extra time / penalties — use win probability
            winner = team_a if np.random.random() < win_prob_a else team_b
            loser = team_b if winner == team_a else team_a
            return winner, loser, goals_a, goals_b
        else:
            # Group stage draw allowed
            return None, None, goals_a, goals_b


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def simulate_group(group_teams: list, scores: dict) -> pd.DataFrame:
    """
    Simulate group stage — each team plays 3 matches.
    Returns standings DataFrame sorted by points, GD, GF.
    """
    records = {t: {"points": 0, "gd": 0, "gf": 0} for t in group_teams}

    # Round robin matches
    matches = [
        (group_teams[0], group_teams[1]),
        (group_teams[0], group_teams[2]),
        (group_teams[0], group_teams[3]),
        (group_teams[1], group_teams[2]),
        (group_teams[1], group_teams[3]),
        (group_teams[2], group_teams[3]),
    ]

    for team_a, team_b in matches:
        winner, loser, ga, gb = simulate_match(team_a, team_b, scores)

        if winner is None:
            # Draw
            records[team_a]["points"] += 1
            records[team_b]["points"] += 1
            records[team_a]["gd"] += ga - gb
            records[team_b]["gd"] += gb - ga
            records[team_a]["gf"] += ga
            records[team_b]["gf"] += gb
        else:
            records[winner]["points"] += 3
            records[loser]["points"] += 0
            records[winner]["gd"] += ga - gb
            records[loser]["gd"] += gb - ga
            records[winner]["gf"] += ga
            records[loser]["gf"] += gb

    standings = pd.DataFrame([
        {"team": t, **v} for t, v in records.items()
    ]).sort_values(
        ["points", "gd", "gf"],
        ascending=False
    ).reset_index(drop=True)

    return standings


def simulate_all_groups(scores: dict) -> dict:
    """
    Simulate all 12 groups.
    Returns dict of group → standings DataFrame.
    """
    return {
        group: simulate_group(teams, scores)
        for group, teams in GROUPS_2026.items()
    }


# ---------------------------------------------------------------------------
# Third place qualification
# ---------------------------------------------------------------------------

def get_best_third_place(group_results: dict, n: int = 8) -> list:
    """
    Get the 8 best third-place teams across all 12 groups.
    Ranked by points, then GD, then GF.
    """
    third_place = []
    for group, standings in group_results.items():
        if len(standings) >= 3:
            third = standings.iloc[2]
            third_place.append({
                "team": third["team"],
                "points": third["points"],
                "gd": third["gd"],
                "gf": third["gf"],
            })

    third_df = pd.DataFrame(third_place).sort_values(
        ["points", "gd", "gf"],
        ascending=False
    ).head(n)

    return third_df["team"].tolist()


# ---------------------------------------------------------------------------
# Knockout stage
# ---------------------------------------------------------------------------

def simulate_knockout_round(teams: list, scores: dict) -> list:
    """
    Simulate one knockout round.
    Teams paired sequentially: [0 vs 1, 2 vs 3, ...].
    Returns list of winners.
    """
    winners = []
    for i in range(0, len(teams), 2):
        if i + 1 < len(teams):
            winner, _, _, _ = simulate_match(
                teams[i], teams[i + 1], scores, knockout=True
            )
            winners.append(winner)
        else:
            winners.append(teams[i])  # bye
    return winners


def simulate_tournament(scores: dict) -> dict:
    """
    Simulate one full tournament.
    Returns dict of team → furthest round reached.
    """
    results = {team: "Group Stage" for group in GROUPS_2026.values()
               for team in group}

    # Group stage
    group_results = simulate_all_groups(scores)

    # Collect qualifiers
    qualifiers = []
    for group, standings in group_results.items():
        qualifiers.append(standings.iloc[0]["team"])  # 1st
        qualifiers.append(standings.iloc[1]["team"])  # 2nd

    # Best 8 third place teams
    best_third = get_best_third_place(group_results, n=8)
    qualifiers.extend(best_third)

    # Update results
    for team in qualifiers:
        results[team] = "Round of 32"

    # Knockout rounds
    rounds = ["Round of 16", "Quarter-final",
              "Semi-final", "Runner-up", "Champion"]
    current_teams = qualifiers

    for i, round_name in enumerate(rounds):
        if len(current_teams) <= 1:
            break
        winners = simulate_knockout_round(current_teams, scores)
        for team in winners:
            results[team] = round_name
        current_teams = winners

    # Champion
    if current_teams:
        results[current_teams[0]] = "Champion"

    return results


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(
    readiness_df: pd.DataFrame,
    n_simulations: int = 10000,
    verbose: bool = True,
    export_path: str = None,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation of the 2026 World Cup.

    Args:
        readiness_df: Output from get_team_readiness_scores()
        n_simulations: Number of tournaments to simulate
        verbose: Print progress
        export_path: Optional custom export path

    Returns:
        DataFrame with win/QF/SF probabilities per team
    """
    # Build scores lookup
    scores = dict(zip(
        readiness_df["country"],
        readiness_df["final_score"]
    ))

    # Fill missing teams with default score
    all_teams = [t for group in GROUPS_2026.values() for t in group]
    for team in all_teams:
        if team not in scores:
            scores[team] = 45.0
            if verbose:
                print(f"  ⚠ {team!r} not in readiness scores — using default 45.0")

    # Track outcomes
    outcome_counts = defaultdict(lambda: defaultdict(int))

    if verbose:
        print(f"\nRunning {n_simulations:,} simulations...")

    for sim in range(n_simulations):
        if verbose and sim % 1000 == 0:
            print(f"  Simulation {sim:,}/{n_simulations:,}...")

        results = simulate_tournament(scores)
        for team, result in results.items():
            outcome_counts[team][result] += 1

    # Build output DataFrame
    rows = []
    for team in all_teams:
        counts = outcome_counts[team]
        total = n_simulations
        rows.append({
            "team":            team,
            "group":           next(g for g, teams in GROUPS_2026.items()
                                   if team in teams),
            "readiness_score": round(scores.get(team, 45.0), 2),
            "p_champion":      round(counts.get("Champion", 0) / total * 100, 1),
            "p_runner_up":     round(counts.get("Runner-up", 0) / total * 100, 1),
            "p_semi_final":    round(counts.get("Semi-final", 0) / total * 100, 1),
            "p_quarter_final": round(counts.get("Quarter-final", 0) / total * 100, 1),
            "p_round_of_16":   round(counts.get("Round of 16", 0) / total * 100, 1),
            "p_round_of_32":   round(counts.get("Round of 32", 0) / total * 100, 1),
            "p_group_exit":    round(counts.get("Group Stage", 0) / total * 100, 1),
        })

    result_df = pd.DataFrame(rows).sort_values(
        "p_champion", ascending=False
    ).reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "rank"

    if verbose:
        print(f"\n✓ Done — {n_simulations:,} simulations complete")
        print(f"\nTOP 10 CHAMPION PROBABILITIES:")
        print(result_df[["team", "group", "readiness_score",
                         "p_champion", "p_semi_final",
                         "p_quarter_final"]].head(10).to_string())

    # Export
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    if export_path is None:
        export_path = outputs_dir / "monte_carlo_2026.csv"

    result_df.to_csv(export_path, index=True)
    print(f"Exported Monte Carlo results → {export_path}")

    return result_df


if __name__ == "__main__":
    print("Run from composite_score notebook using:")
    print("  from monte_carlo import run_monte_carlo")
    print("  mc_df = run_monte_carlo(df, n_simulations=10000)")