"""
2026 World Cup readiness scoring engine.
Combines squad quality, manager stability, star power, and travel factors.
"""

import pandas as pd
import numpy as np
import unicodedata
import polars as pl
from difflib import get_close_matches
from rosters_2026 import MANAGER_TENURE

# North American clubs (2026 hosts) for travel advantage
NA_CLUBS = [
    'Inter Miami', 'LAFC', 'LA Galaxy', 'San Diego FC', 'New York City FC', 
    'Seattle Sounders', 'Atlanta United', 'Toronto FC', 'Club América', 
    'Chivas', 'Cruz Azul', 'Monterrey', 'Tigres UANL', 'Toluca', 'Pachuca'
]


def _normalize_name(text):
    """Normalize player names for matching (ASCII lowercase)."""
    return unicodedata.normalize('NFD', str(text)).encode('ascii', 'ignore').decode('utf-8').strip().lower()


def _to_pandas(df):
    """Convert polars to pandas if needed."""
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def run_system_fit_engine(df_players_input, df_success_input, df_archetypes_input, rosters_2026, CLUB_MAPPING_2026):
    """
    Score national teams on 2026 World Cup readiness.
    
    Factors:
    - Squad quality (avg + depth)
    - Manager tenure/stability
    - Star power (top 3 players)
    - Travel/recovery (NA-based players)
    - Host nation advantage (USA/Mexico/Canada)
    - Club cohesion (teammates playing together)
    """
    df_p = _to_pandas(df_players_input).copy()
    df_p['player_norm'] = df_p['player'].apply(_normalize_name)
    db_names_list = df_p['player_norm'].tolist()

    results = []

    for country, squad_dict in rosters_2026.items():
        found_indices = []
        
        for common_name in squad_dict.keys():
            n_common = _normalize_name(common_name)
            
            # Match: exact/contains first, then fuzzy fallback
            match = df_p[df_p['player_norm'].str.contains(n_common, na=False)]
            if not match.empty:
                found_indices.append(match['overall_quality'].idxmax())
            else:
                fuzzy_match = get_close_matches(n_common, db_names_list, n=1, cutoff=0.7)
                if fuzzy_match:
                    idx = df_p[df_p['player_norm'] == fuzzy_match[0]].index[0]
                    found_indices.append(idx)

        squad_df = df_p.loc[found_indices].drop_duplicates(subset=['player']).copy()
        num_found = len(squad_df)
        
        if num_found == 0:
            continue

        # Manager stability multiplier
        start_year = MANAGER_TENURE.get(country, 2025)
        tenure_years = 2026 - start_year
        stability_mult = min(1.05, 0.90 + (tenure_years * 0.02))

        # Star power bonus (top 3 players)
        top_3_avg = squad_df.nlargest(3, 'overall_quality')['overall_quality'].mean()
        star_bonus = (top_3_avg - 75) * 0.20 if top_3_avg > 75 else 0

        # Recovery/travel advantage (players in North America)
        squad_df['club_2026'] = squad_df['player'].map(CLUB_MAPPING_2026)
        local_count = squad_df['club_2026'].isin(NA_CLUBS).sum()
        recovery_edge = (local_count / num_found) * 5.0

        # Host nation bonus
        host_bonus = 1.12 if country in ['USA', 'Mexico', 'Canada'] else 1.00

        # Squad quality
        avg_quality = squad_df['overall_quality'].mean()
        depth_mult = min(1.0, 0.5 + (num_found * 0.05))
        
        # Club cohesion (teammates)
        valid_clubs = squad_df['club_2026'].dropna()
        cohesion = (sum(valid_clubs.value_counts()**2) / (num_found**2)) * 10 if num_found >= 3 else 0

        # Final readiness score
        readiness = ((avg_quality * depth_mult * stability_mult) * host_bonus) + cohesion + star_bonus + recovery_edge

        results.append({
            'National_Team': country,
            'Players_Found': num_found,
            'Stability': round(stability_mult, 2),
            'Star_Power': round(star_bonus, 2),
            'Recovery': round(recovery_edge, 2),
            'Host_Adv': "Yes" if host_bonus > 1 else "No",
            'Readiness_Score': round(readiness, 2)
        })

    return pd.DataFrame(results).sort_values('Readiness_Score', ascending=False).reset_index(drop=True)