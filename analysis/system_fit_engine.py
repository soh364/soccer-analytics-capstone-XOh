import pandas as pd
import numpy as np
import unicodedata
import polars as pl
from difflib import get_close_matches

# Dictionary of Manager Start Years for 2026 Context
# Long tenure = high stability. New (2025/2026) = instability.
MANAGER_TENURE = {
    'France': 2012, 'Argentina': 2018, 'Spain': 2022, 'Germany': 2023,
    'USA': 2024, 'England': 2025, 'Brazil': 2024, 'Portugal': 2023,
    'Canada': 2024, 'Mexico': 2024
}

def run_system_fit_engine_v3(df_players_input, df_success_input, df_archetypes_input, rosters_2026, CLUB_MAPPING_2026):
    def to_pd(df): return df.to_pandas() if isinstance(df, pl.DataFrame) else df
    df_p = to_pd(df_players_input).copy()
    
    def norm(t): return unicodedata.normalize('NFD', str(t)).encode('ascii', 'ignore').decode('utf-8').strip().lower()
    df_p['player_norm'] = df_p['player'].apply(norm)
    db_names_list = df_p['player_norm'].tolist()

    results = []

    for country, squad_dict in rosters_2026.items():
        found_indices = []
        for common_name in squad_dict.keys():
            n_common = norm(common_name)
            match = df_p[df_p['player_norm'].str.contains(n_common, na=False)]
            if not match.empty:
                found_indices.append(match['overall_quality'].idxmax())
            else:
                fuzzy_match = get_close_matches(n_common, db_names_list, n=1, cutoff=0.7)
                if fuzzy_match:
                    idx = df_p[df_p['player_norm'] == fuzzy_match[0]].index[0]
                    found_indices.append(idx)

        squad_df = df_p.loc[found_indices].drop_duplicates(subset=['player']).copy()
        if len(squad_df) == 0: continue

        # --- NERD METRIC 1: MANAGER STABILITY ---
        # Stability score: 1.0 (long term) to 0.85 (new manager)
        start_year = MANAGER_TENURE.get(country, 2025)
        tenure_years = 2026 - start_year
        stability_mult = min(1.05, 0.90 + (tenure_years * 0.02))

        # --- NERD METRIC 2: STAR POWER ---
        # Top 3 players determine the "ceiling" of the team
        top_3_avg = squad_df.nlargest(3, 'overall_quality')['overall_quality'].mean()
        star_bonus = (top_3_avg - 75) * 0.15 if top_3_avg > 75 else 0

        # --- NERD METRIC 3: TRAVEL & RECOVERY ---
        # Players in MLS/Liga MX/USL have 0 jet lag. 
        # Players in Europe have high fatigue.
        squad_df['club_2026'] = squad_df['player'].map(CLUB_MAPPING_2026)
        
        # Simple North American League check
        na_leagues = ['Inter Miami', 'LAFC', 'Club América', 'LA Galaxy', 'San Diego FC', 'Minnesota United']
        local_players = squad_df['club_2026'].isin(na_leagues).sum()
        recovery_edge = (local_players / len(squad_df)) * 5.0 # Up to 5 points bonus

        # --- CORE LOGIC ---
        avg_quality = squad_df['overall_quality'].mean()
        depth_mult = min(1.0, 0.4 + (len(squad_df) * 0.06))
        
        # Cohesion
        valid_clubs = squad_df['club_2026'].dropna()
        cohesion = (sum(valid_clubs.value_counts()**2) / (len(squad_df)**2)) * 10 if len(squad_df) >= 4 else 0

        # FINAL CALCULATION
        # Quality * Depth * Stability + Bonuses
        readiness = (avg_quality * depth_mult * stability_mult) + cohesion + star_bonus + recovery_edge

        results.append({
            'National_Team': country,
            'Players_Found': len(squad_df),
            'Stability_Bonus': round(stability_mult, 2),
            'Star_Power': round(star_bonus, 2),
            'Recovery_Edge': round(recovery_edge, 2),
            'Readiness_Score': round(readiness, 2)
        })

    return pd.DataFrame(results).sort_values(by='Readiness_Score', ascending=False).reset_index(drop=True)