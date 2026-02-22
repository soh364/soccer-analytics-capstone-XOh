import pandas as pd
import numpy as np
import unicodedata
import polars as pl
from difflib import get_close_matches

# Import your external configs
from rosters_2026 import MANAGER_TENURE

def run_system_fit_engine(df_players_input, df_success_input, df_archetypes_input, rosters_2026, CLUB_MAPPING_2026):
    """
    Refined 2026 Readiness Engine.
    Accounts for Manager Tenure, Star Power, Travel Fatigue, and Host Advantage.
    """
    def to_pd(df): return df.to_pandas() if isinstance(df, pl.DataFrame) else df
    df_p = to_pd(df_players_input).copy()
    
    def norm(t): return unicodedata.normalize('NFD', str(t)).encode('ascii', 'ignore').decode('utf-8').strip().lower()
    df_p['player_norm'] = df_p['player'].apply(norm)
    db_names_list = df_p['player_norm'].tolist()

    results = []

    # Comprehensive North American Club List for Recovery Edge
    NA_CLUBS = [
        'Inter Miami', 'LAFC', 'LA Galaxy', 'San Diego FC', 'New York City FC', 
        'Seattle Sounders', 'Atlanta United', 'Toronto FC', 'Club América', 
        'Chivas', 'Cruz Azul', 'Monterrey', 'Tigres UANL', 'Toluca', 'Pachuca'
    ]

    for country, squad_dict in rosters_2026.items():
        found_indices = []
        for common_name in squad_dict.keys():
            n_common = norm(common_name)
            # Match Logic (Priority: Exact/Contains -> Fuzzy)
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
        
        if num_found == 0: continue

        # --- 1. MANAGER STABILITY ---
        # Long-term (France/Arg) gets 1.05x. Brand new (England/Brazil) starts at 0.90x.
        start_year = MANAGER_TENURE.get(country, 2025)
        tenure_years = 2026 - start_year
        stability_mult = min(1.05, 0.90 + (tenure_years * 0.02))

        # --- 2. STAR POWER (Ceiling) ---
        # Bonus based on the quality of the top 3 match-winners
        top_3_avg = squad_df.nlargest(3, 'overall_quality')['overall_quality'].mean()
        star_bonus = (top_3_avg - 75) * 0.20 if top_3_avg > 75 else 0

        # --- 3. RECOVERY & TRAVEL ---
        # Players already in NA time zones avoid jet lag
        squad_df['club_2026'] = squad_df['player'].map(CLUB_MAPPING_2026)
        local_count = squad_df['club_2026'].isin(NA_CLUBS).sum()
        recovery_edge = (local_count / num_found) * 5.0

        # --- 4. HOST NATION ADVANTAGE ---
        # Statistical boost for home support and familiarity
        host_bonus = 1.12 if country in ['USA', 'Mexico', 'Canada'] else 1.00

        # --- CORE LOGIC ---
        avg_quality = squad_df['overall_quality'].mean()
        
        # NORMALIZATION: 
        # We use a depth multiplier that caps quickly so smaller squads 
        # with high quality aren't punished too harshly compared to 23-man squads.
        depth_mult = min(1.0, 0.5 + (num_found * 0.05))
        
        # Cohesion (Club teammates)
        valid_clubs = squad_df['club_2026'].dropna()
        cohesion = (sum(valid_clubs.value_counts()**2) / (num_found**2)) * 10 if num_found >= 3 else 0

        # FINAL CALCULATION
        # (Quality Base * Stability * Host) + Bonuses
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

    return pd.DataFrame(results).sort_values(by='Readiness_Score', ascending=False).reset_index(drop=True)