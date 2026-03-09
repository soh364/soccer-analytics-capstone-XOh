"""
tc_outcome_validation.py
────────────────────────
Maps 2022 World Cup results onto cluster assignments to validate
whether tactical archetypes predict tournament outcomes.
"""

import pandas as pd


# ── 2022 WC Results ───────────────────────────────────────────────────────────
WC2022_OUTCOMES = {
    'Argentina'    : 'Winner',
    'France'       : 'Runner-up',
    'Croatia'      : 'Third',
    'Morocco'      : 'Fourth',
    'Netherlands'  : 'Quarter-final',
    'England'      : 'Quarter-final',
    'Brazil'       : 'Quarter-final',
    'Portugal'     : 'Quarter-final',
    'Japan'        : 'Round of 16',
    'South Korea'  : 'Round of 16',
    'Australia'    : 'Round of 16',
    'Switzerland'  : 'Round of 16',
    'Spain'        : 'Round of 16',
    'United States': 'Round of 16',
    'Poland'       : 'Round of 16',
    'Senegal'      : 'Round of 16',
    'Germany'      : 'Group Stage',
    'Belgium'      : 'Group Stage',
    'Uruguay'      : 'Group Stage',
    'Denmark'      : 'Group Stage',
    'Mexico'       : 'Group Stage',
    'Serbia'       : 'Group Stage',
    'Cameroon'     : 'Group Stage',
    'Ghana'        : 'Group Stage',
    'Ecuador'      : 'Group Stage',
    'Canada'       : 'Group Stage',
    'Tunisia'      : 'Group Stage',
    'Saudi Arabia' : 'Group Stage',
    'Iran'         : 'Group Stage',
    'Wales'        : 'Group Stage',
    'Costa Rica'   : 'Group Stage',
    'Qatar'        : 'Group Stage',
}

OUTCOME_RANK = {
    'Winner'       : 7,
    'Runner-up'    : 6,
    'Third'        : 5,
    'Fourth'       : 4,
    'Quarter-final': 3,
    'Round of 16'  : 2,
    'Group Stage'  : 1,
}


# ── Functions ─────────────────────────────────────────────────────────────────
def merge_outcomes(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge WC 2022 outcomes into the results DataFrame.

    Returns
    -------
    (results_with_outcomes, wc_teams) : pd.DataFrame, pd.DataFrame
        wc_teams — filtered to only the 32 WC 2022 participants
    """
    results = results.copy()
    results['wc2022_outcome'] = results['team'].map(WC2022_OUTCOMES)
    results['outcome_rank']   = results['wc2022_outcome'].map(OUTCOME_RANK)

    wc_teams = results[results['wc2022_outcome'].notna()].copy()
    print(f'WC 2022 teams matched: {len(wc_teams)}/32')

    return results, wc_teams


def compute_outcome_summary(wc_teams: pd.DataFrame,
                            archetype_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-archetype outcome distribution and average outcome rank.

    Returns
    -------
    (outcome_by_team, avg_rank) : pd.DataFrame, pd.DataFrame
    """
    # Per-team breakdown
    print('\n=== Outcome distribution by archetype ===')
    for archetype in archetype_map.values():
        subset = wc_teams[wc_teams['archetype'] == archetype]
        if len(subset) == 0:
            continue
        print(f'\n{archetype} ({len(subset)} WC teams):')
        print(subset[['team', 'wc2022_outcome']].sort_values('wc2022_outcome').to_string(index=False))

    # Average rank per archetype
    avg_rank = (
        wc_teams.groupby('archetype')['outcome_rank']
        .agg(['mean', 'max', 'count'])
        .round(2)
        .reset_index()
        .sort_values('mean', ascending=False)
    )
    avg_rank.columns = ['archetype', 'avg_outcome_rank', 'best_finish_rank', 'n_teams']
    avg_rank['best_finish'] = avg_rank['best_finish_rank'].map(
        {v: k for k, v in OUTCOME_RANK.items()}
    )

    print('\n=== Average outcome rank by archetype (higher = better) ===')
    print(avg_rank[['archetype', 'n_teams', 'avg_outcome_rank', 'best_finish']].to_string(index=False))

    return wc_teams, avg_rank
