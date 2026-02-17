# tournament_progression.py

import pandas as pd
import numpy as np

def get_tournament_results():
    """
    Tournament progression data: best result per team across 2022-24 tournaments.
    
    Scoring:
        0 = Group stage exit
        1 = Round of 16
        2 = Quarter-final
        3 = Semi-final
        4 = Final (runner-up)
        5 = Winner
    
    Returns:
        dict of dicts: {tournament_name: {team: score}}
    """
    wc_2022 = {
        'Argentina': 5, 'France': 4, 'Croatia': 3, 'Morocco': 3,
        'Netherlands': 2, 'England': 2, 'Brazil': 2, 'Portugal': 2,
        'Japan': 1, 'Australia': 1, 'South Korea': 1, 'United States': 1,
        'Poland': 1, 'Senegal': 1, 'Switzerland': 1, 'Spain': 1,
        'Germany': 0, 'Belgium': 0, 'Canada': 0, 'Mexico': 0,
        'Uruguay': 0, 'Ghana': 0, 'Cameroon': 0, 'Serbia': 0,
        'Qatar': 0, 'Ecuador': 0, 'Saudi Arabia': 0, 'Tunisia': 0,
        'Costa Rica': 0, 'Iran': 0, 'Wales': 0, 'Denmark': 0
    }
    
    euro_2024 = {
        'Spain': 5, 'England': 4, 'France': 3, 'Netherlands': 3,
        'Germany': 2, 'Portugal': 2, 'Switzerland': 2, 'Turkey': 2,
        'Austria': 1, 'Belgium': 1, 'Denmark': 1, 'Georgia': 1,
        'Italy': 1, 'Romania': 1, 'Slovakia': 1, 'Slovenia': 1,
        'Albania': 0, 'Croatia': 0, 'Czech Republic': 0, 'Hungary': 0,
        'Poland': 0, 'Scotland': 0, 'Serbia': 0, 'Ukraine': 0
    }
    
    copa_2024 = {
        'Argentina': 5, 'Colombia': 4, 'Canada': 3, 'Uruguay': 3,
        'Venezuela': 2, 'Ecuador': 2, 'Panama': 2, 'Brazil': 2,
        'Mexico': 0, 'United States': 0, 'Bolivia': 0, 'Chile': 0,
        'Costa Rica': 0, 'Jamaica': 0, 'Paraguay': 0, 'Peru': 0
    }
    
    return {
        'World Cup 2022': wc_2022,
        'Euro 2024': euro_2024,
        'Copa América 2024': copa_2024
    }


def get_progression_df():
    """
    Build progression DataFrame with best result per team across all tournaments.
    
    Returns:
        DataFrame with columns: team, progression_score, best_tournament
    """
    tournaments = get_tournament_results()
    
    # Collect all team scores
    team_scores = {}
    team_best_tournament = {}
    
    for tournament_name, results in tournaments.items():
        for team, score in results.items():
            if team not in team_scores or score > team_scores[team]:
                team_scores[team] = score
                team_best_tournament[team] = tournament_name
    
    df = pd.DataFrame([
        {
            'team': team,
            'progression_score': score,
            'best_tournament': team_best_tournament[team]
        }
        for team, score in team_scores.items()
    ])
    
    return df.sort_values('progression_score', ascending=False).reset_index(drop=True)


def merge_progression(tournament_pd, archetype_names=None):
    """
    Merge progression scores with tournament profiles.
    
    Args:
        tournament_pd: Tournament profiles DataFrame with 'team' column
        archetype_names: Optional dict mapping cluster_id to name
    
    Returns:
        DataFrame with progression data merged in
    """
    progression_df = get_progression_df()
    
    merged = tournament_pd.merge(progression_df, on='team', how='left')
    
    # Fill missing with 0
    missing = merged[merged['progression_score'].isna()]['team'].tolist()
    if missing:
        print(f"  {len(missing)} teams missing scores (filling with 0): {', '.join(missing)}")
    merged['progression_score'] = merged['progression_score'].fillna(0)
    
    # Add stage labels
    stage_map = {0: 'Group Stage', 1: 'Round of 16', 2: 'Quarter-final',
                 3: 'Semi-final', 4: 'Final', 5: 'Winner'}
    merged['best_stage'] = merged['progression_score'].map(stage_map)
    
    return merged


def print_progression_summary(merged_df, archetype_names):
    """
    Display archetype success summary as styled HTML table.
    
    Args:
        merged_df: DataFrame from merge_progression with 'assigned_archetype', 'progression_score'
        archetype_names: dict mapping cluster_id to name
    """
    from IPython.display import display, HTML
    
    rows_html = ""
    for cid in sorted(archetype_names.keys()):
        mask = merged_df['assigned_archetype'] == cid
        subset = merged_df[mask]
        name = archetype_names[cid]
        n = len(subset)
        avg = subset['progression_score'].mean()
        median = subset['progression_score'].median()
        best_team = subset.loc[subset['progression_score'].idxmax()]
        sf_plus = (subset['progression_score'] >= 3).sum()
        sf_pct = sf_plus / n * 100 if n > 0 else 0
        final_plus = (subset['progression_score'] >= 4).sum()
        final_pct = final_plus / n * 100 if n > 0 else 0
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{name}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{n}</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:#2d6a4f;text-align:center;">{avg:.2f}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{median:.1f}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{sf_pct:.1f}%</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{final_pct:.1f}%</td>
            <td style="font-size:12px;">{best_team['team']} ({best_team['best_stage']})</td>
        </tr>"""
    
    html = f"""
    <style>
        .prog {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .prog th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:center; font-weight:600; }}
        .prog th:first-child {{ text-align:left; }}
        .prog td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .prog tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        Tournament Success by Archetype
    </h4>
    <table class="prog">
        <tr>
            <th style="text-align:left;">Archetype</th>
            <th>Teams</th>
            <th>Avg Progression</th>
            <th>Median</th>
            <th>Semi-final+</th>
            <th>Final+</th>
            <th style="text-align:left;">Best Result</th>
        </tr>
        {rows_html}
    </table>
    """
    display(HTML(html))