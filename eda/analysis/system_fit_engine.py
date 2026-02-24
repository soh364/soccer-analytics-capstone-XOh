"""
2026 World Cup readiness scoring engine.
Combines squad quality, manager stability, star power, and travel factors.
"""

import pandas as pd
import numpy as np
import unicodedata
import polars as pl
from rosters_2026 import MANAGER_TENURE

# North American clubs (2026 hosts) for recovery edge
NA_CLUBS = [
    'Inter Miami', 'LAFC', 'LA Galaxy', 'San Diego FC', 'New York City FC',
    'Seattle Sounders', 'Atlanta United', 'Toronto FC', 'Club América',
    'Chivas', 'Cruz Azul', 'Monterrey', 'Tigres UANL', 'Toluca', 'Pachuca'
]

# Confederation-based travel penalty — applied per country, not per player
CONFEDERATION = {
    'France': 'UEFA', 'Germany': 'UEFA', 'Spain': 'UEFA',
    'England': 'UEFA', 'Netherlands': 'UEFA', 'Portugal': 'UEFA',
    'Croatia': 'UEFA', 'Switzerland': 'UEFA', 'Serbia': 'UEFA',
    'Turkey': 'UEFA', 'Denmark': 'UEFA', 'Belgium': 'UEFA',
    'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Colombia': 'CONMEBOL',
    'Uruguay': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Morocco': 'CAF', 'Nigeria': 'CAF', 'Senegal': 'CAF', 'Ghana': 'CAF',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC',
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
}

TRAVEL_PENALTY = {
    'UEFA':     2.5,   # long haul + time zone shift after grueling European season
    'CONMEBOL': 1.0,   # closer geography, more familiar conditions
    'CAF':      2.0,   # long haul from Africa
    'AFC':      2.5,   # worst time zone shift
    'CONCACAF': 0.0,   # already there
}


def _normalize_name(text):
    """Strip accents, lowercase, collapse whitespace."""
    text = unicodedata.normalize('NFD', str(text))
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return ' '.join(text.lower().split())


def _tokens(name):
    return set(_normalize_name(name).split())


def _find_best_match(query, df_p):
    """
    Multi-strategy matching, in priority order:
    1. Exact normalized match
    2. All query tokens present in DB name (Luis Chavez → Luis Gerardo Chávez Magallón)
    3. All DB name tokens present in query (reverse subset)
    4. Majority token overlap (≥ 60% of query tokens match)
    """
    q_norm = _normalize_name(query)
    q_tokens = _tokens(query)

    # 1. Exact
    exact = df_p[df_p['player_norm'] == q_norm]
    if not exact.empty:
        return exact['overall_quality'].idxmax()

    # 2. Query tokens ⊆ DB name tokens (short name → full legal name)
    subset_fwd = df_p[df_p['player_tokens'].apply(lambda t: q_tokens <= t)]
    if not subset_fwd.empty:
        return subset_fwd['overall_quality'].idxmax()

    # 3. DB tokens ⊆ query tokens (full legal name → short name)
    subset_rev = df_p[df_p['player_tokens'].apply(lambda t: t <= q_tokens)]
    if not subset_rev.empty:
        return subset_rev['overall_quality'].idxmax()

    # 4. Fuzzy token overlap — handles middle name mismatches
    def _overlap(t):
        if not t:
            return 0.0
        return len(q_tokens & t) / len(q_tokens)

    df_p_copy = df_p.copy()
    df_p_copy['_overlap'] = df_p_copy['player_tokens'].apply(_overlap)
    best = df_p_copy[df_p_copy['_overlap'] >= 0.60]
    if not best.empty:
        return best['overall_quality'].idxmax()

    return None

def _get_archetype_bonus(country, df_success, df_archetypes):
    """
    Returns (bonus, archetype_name) based on historical tournament success
    of the team's tactical cluster.
    """
    row = df_success[df_success['team'].str.lower() == country.lower()]
    if row.empty:
        return 0.0, 'Unknown'
    
    cluster_id = int(row.iloc[0]['cluster'])
    arch = df_archetypes[df_archetypes['cluster'] == cluster_id]
    if arch.empty:
        return 0.0, 'Unknown'
    
    arch = arch.iloc[0]
    # weighted blend: winning matters most, but r16+ shows consistent quality
    bonus = (arch['winner_pct'] / 100 * 3.0) + (arch['qf_plus_pct'] / 100 * 2.0)
    return round(bonus, 2), arch['archetype_name']

def _to_pandas(df):
    """Convert polars to pandas if needed."""
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def run_system_fit_engine(
    df_players_input,
    df_success_input,        # ← add back
    df_archetypes_input,     # ← add back
    rosters_2026,
    CLUB_MAPPING_2026,
    manager_tenure,
    config=None,
):
    df_p = _to_pandas(df_players_input).copy()
    df_success = _to_pandas(df_success_input)
    df_archetypes = _to_pandas(df_archetypes_input)
    df_p['player_norm']   = df_p['player'].apply(_normalize_name)
    df_p['player_tokens'] = df_p['player_norm'].apply(lambda n: set(n.split()))

    results = []

    for country, squad_dict in rosters_2026.items():
        found_indices = []

        for common_name in squad_dict.keys():
            idx = _find_best_match(common_name, df_p)
            if idx is not None:
                found_indices.append(idx)

        squad_df = df_p.loc[found_indices].drop_duplicates(subset=['player']).copy()
        num_found = len(squad_df)
        squad_size = len(squad_dict)
        coverage = num_found / squad_size if squad_size > 0 else 0

        if num_found == 0:
            continue
        # replace your current low_coverage_penalty with this
        if num_found == 1:
            low_coverage_penalty = 8.0   # single player is not a squad reading
        elif num_found < 4:
            low_coverage_penalty = 3.0   # thin but usable
        else:
            low_coverage_penalty = 0.0

        # ── Manager stability ────────────────────────────────────────────
        start_year = manager_tenure.get(country, 2025)
        tenure_years = 2026 - start_year
        stability_mult = min(1.05, 0.90 + (tenure_years * 0.02))

        # ── Star power (top 3 found players) ────────────────────────────
        top_3_avg = squad_df.nlargest(3, 'overall_quality')['overall_quality'].mean()
        star_bonus = (top_3_avg - 58) * 0.25 if top_3_avg > 58 else 0

        # ── Recovery edge (NA-based players) ────────────────────────────
        squad_df['club_2026'] = squad_df['player'].map(CLUB_MAPPING_2026)
        local_count = squad_df['club_2026'].isin(NA_CLUBS).sum()
        recovery_edge = (local_count / squad_size) * 5.0

        # ── Host bonus (additive nudge, not multiplicative shove) ─────────
        host_bonus = 2.0 if country in ['United States', 'Mexico', 'Canada'] else 0.0

        # ── Travel penalty (per country, not per player) ──────────────────
        confederation = CONFEDERATION.get(country, 'UEFA')
        travel_penalty = TRAVEL_PENALTY.get(confederation, 2.0)

        # ── Squad quality ────────────────────────────────────────────────
        GLOBAL_MEAN_QUALITY = 55.0  # run df_players['overall_quality'].mean() and set this

        # Bayesian-style shrinkage toward global mean when sample is small
        if num_found < 5:
            weight = num_found / 5  # 1 player = 20% trust, 4 players = 80% trust
            avg_quality = (weight * squad_df['overall_quality'].mean()) + ((1 - weight) * GLOBAL_MEAN_QUALITY)
        else:
            avg_quality = squad_df['overall_quality'].mean()

        # ── Cohesion ─────────────────────────────────────────────────────
        valid_clubs = squad_df['club_2026'].dropna()
        cohesion = (sum(valid_clubs.value_counts()**2) / (num_found**2)) * 10 \
                   if num_found >= 3 else 0

        # ── Archetype bonus ──────────────────────────────────────────────────
        archetype_bonus, archetype_name = _get_archetype_bonus(country, df_success, df_archetypes)

        # ── Final score ──────────────────────────────────────────────────────
        readiness = (
            (avg_quality * stability_mult)
            + host_bonus
            + cohesion
            + star_bonus
            + recovery_edge
            + archetype_bonus       # ← add
            - travel_penalty
            - low_coverage_penalty
        )

        results.append({
            'National_Team':   country,
            'Readiness_Score': round(readiness, 2),
            'Archetype':        archetype_name,
            'Archetype_Bonus':  archetype_bonus,
            'Squad_Quality':   round(avg_quality, 2),
            'Star_Power':      round(star_bonus, 2),
            'Stability':       round(stability_mult, 2),
            'Manager_Tenure':  tenure_years,
            'Cohesion':        round(cohesion, 2),
            'Recovery':        round(recovery_edge, 2),
            'Travel_Penalty':  travel_penalty,
            'Confederation':   confederation,
            'Host_Adv':        'Yes' if host_bonus > 0 else 'No',
            'Players_Found':   num_found,
            'Squad_Size':      squad_size,
            'Coverage_Pct':    round(coverage * 100, 1),
        })

    return (
        pd.DataFrame(results)
        .sort_values('Readiness_Score', ascending=False)
        .reset_index(drop=True)
    )

