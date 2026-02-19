# tournament_progression.py

import pandas as pd
import numpy as np

from IPython.display import display, HTML
from io import BytesIO
import base64
import matplotlib.pyplot as plt

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

# tournament_compression.py

def analyze_archetype_shift(baseline_df, tournament_df, archetype_names,
                            baseline_col='archetype_name', tournament_col='assigned_archetype'):
    """
    Calculate archetype distribution shift between two datasets.
    
    Args:
        baseline_df: Club profiles DataFrame
        tournament_df: Tournament profiles DataFrame
        archetype_names: dict mapping cluster_id to name
        baseline_col: column name for archetype in baseline
        tournament_col: column name for archetype in tournament
    
    Returns:
        DataFrame with shift analysis
    """
    baseline_counts = baseline_df[baseline_col].value_counts().sort_index()
    tournament_counts = tournament_df[tournament_col].value_counts().sort_index()
    
    baseline_pct = baseline_counts / len(baseline_df) * 100
    tournament_pct = tournament_counts / len(tournament_df) * 100
    
    rows = []
    for cid in sorted(archetype_names.keys()):
        b_pct = baseline_pct.get(cid, 0)
        t_pct = tournament_pct.get(cid, 0)
        rows.append({
            'archetype_id': cid,
            'archetype': archetype_names[cid],
            'baseline_pct': b_pct,
            'tournament_pct': t_pct,
            'shift': t_pct - b_pct
        })
    
    return pd.DataFrame(rows)


def print_archetype_shift_table(shift_df, baseline_label='Club', tournament_label='Tournament'):
    """
    Display archetype shift as styled HTML table.
    """
    from IPython.display import display, HTML
    
    rows_html = ""
    for _, row in shift_df.iterrows():
        shift = row['shift']
        
        if shift > 10:
            color, arrow = '#d62828', '↑↑'
        elif shift > 0:
            color, arrow = '#e76f51', '↑'
        elif shift > -10:
            color, arrow = '#2d6a4f', '↓'
        else:
            color, arrow = '#1a6b3c', '↓↓'
        
        bar_width = min(int(abs(shift) * 4), 200)
        bar_color = '#d62828' if shift > 0 else '#2d6a4f'
        bar = f'<div style="background:{bar_color};width:{bar_width}px;height:14px;border-radius:3px;display:inline-block;"></div>'
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{row['archetype']}</td>
            <td style="font-family:'SF Mono',monospace;font-weight:600;text-align:center;">{row['baseline_pct']:.1f}%</td>
            <td style="font-family:'SF Mono',monospace;font-weight:600;text-align:center;">{row['tournament_pct']:.1f}%</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:{color};text-align:center;">{arrow} {shift:+.1f}pp</td>
            <td>{bar}</td>
        </tr>"""
    
    html = f"""
    <style>
        .shift {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .shift th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:center; font-weight:600; }}
        .shift th:first-child {{ text-align:left; }}
        .shift td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .shift tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        Archetype Distribution Shift
    </h4>
    <table class="shift">
        <tr>
            <th style="text-align:left;">Archetype</th>
            <th>{baseline_label}</th>
            <th>{tournament_label}</th>
            <th>Shift</th>
            <th></th>
        </tr>
        {rows_html}
    </table>
    """
    display(HTML(html))


def plot_archetype_shift(shift_df, baseline_label='Club 2022/23',
                         tournament_label='Tournament 2022-24', figsize=(9, 5), ax=None):
    """
    Grouped bar chart of archetype distribution shift.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    x = np.arange(len(shift_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, shift_df['baseline_pct'], width,
                   label=baseline_label, color='#2E86AB', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, shift_df['tournament_pct'], width,
                   label=tournament_label, color='#F18F01', edgecolor='white', linewidth=0.5)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color='#333')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color='#333')
    
    ax.set_xticks(x)
    ax.set_xticklabels(shift_df['archetype'], fontsize=10)
    ax.set_ylabel('Percentage of Teams (%)', fontsize=10)
    ax.set_title('Archetype Distribution: Club vs Tournament', fontsize=13, weight='bold')
    ax.legend(fontsize=10, framealpha=0.9, edgecolor='#ddd')
    ax.set_ylim(0, max(shift_df[['baseline_pct', 'tournament_pct']].max()) * 1.15)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(axis='y', alpha=0.12)
    ax.set_axisbelow(True)
    
    if ax is None:
        plt.tight_layout()
    return fig, ax

def assign_to_archetypes(df, cluster_centers, dimensions):
    """
    Assign teams to nearest baseline archetype by euclidean distance.
    
    Args:
        df: DataFrame with team profiles
        cluster_centers: DataFrame with archetype centers and 'cluster' column
        dimensions: list of D12 dimension names
    
    Returns:
        DataFrame with 'assigned_archetype' and 'distance_to_center' columns added
    """
    df = df.copy()
    
    assignments = []
    distances = []
    
    for _, row in df.iterrows():
        team_vector = row[dimensions].values.astype(float)
        
        best_dist = np.inf
        best_cluster = None
        
        for _, center_row in cluster_centers.iterrows():
            center_vector = center_row[dimensions].values.astype(float)
            dist = np.sqrt(((team_vector - center_vector) ** 2).sum())
            
            if dist < best_dist:
                best_dist = dist
                best_cluster = int(center_row['cluster'])
        
        assignments.append(best_cluster)
        distances.append(best_dist)
    
    df['assigned_archetype'] = assignments
    df['distance_to_center'] = distances
    
    return df


def calculate_cmi(baseline_profiles, tournament_profiles, dimensions):
    """
    Calculate Complexity Maintenance Index (CMI). 
    Quantifies how much tactical variance is lost between club and country.
    """
    cmi_results = []
    
    for dim in dimensions:
        # Baseline Stats (Club)
        b_mean = baseline_profiles[dim].mean()
        b_std = baseline_profiles[dim].std()
        
        # Tournament Stats (International)
        t_mean = tournament_profiles[dim].mean()
        t_std = tournament_profiles[dim].std()
        
        # CMI Calculation (Ratio of standard deviations)
        # Using Std instead of Var keeps the units on the same scale as the data.
        cmi = t_std / b_std if b_std > 0 else 1.0
        compression_pct = (1 - cmi) * 100
        
        cmi_results.append({
            'dimension': dim,
            'baseline_mean': b_mean,
            'tournament_mean': t_mean,
            'cmi': cmi,
            'compression_pct': compression_pct
        })
    
    cmi_df = pd.DataFrame(cmi_results).sort_values('compression_pct', ascending=False)
    
    overall_cmi = cmi_df['cmi'].mean()
    print(f"\nOverall CMI: {overall_cmi:.3f}")
    print(f"Overall Compression: {(1 - overall_cmi) * 100:.1f}%")
    
    return cmi_df

def analyze_archetype_distribution_shift(baseline_profiles, tournament_profiles, archetype_names):
    """
    Compare archetype distribution between baseline and tournament.
    
    Args:
        baseline_profiles: With 'cluster' column
        tournament_profiles: With 'assigned_archetype' column
        archetype_names: Dict mapping cluster_id to name
        
    Returns:
        DataFrame comparing distributions
    """
    baseline_dist = baseline_profiles['cluster'].value_counts(normalize=True).sort_index() * 100
    tournament_dist = tournament_profiles['assigned_archetype'].value_counts(normalize=True).sort_index() * 100
    
    comparison = pd.DataFrame({
        'Archetype': [archetype_names[i] for i in baseline_dist.index],
        'Baseline_%': baseline_dist.values,
        'Tournament_%': tournament_dist.values
    })
    
    comparison['Shift'] = comparison['Tournament_%'] - comparison['Baseline_%']
    
    return comparison


def calculate_archetype_success(tournament_profiles, progression_data, archetype_col='assigned_archetype'):
    """
    Calculate success metrics by archetype.
    
    Args:
        tournament_profiles: With archetype assignments
        progression_data: DataFrame with team progression scores
        archetype_col: Column name for archetype assignment
        
    Returns:
        DataFrame with success metrics by archetype
    """
    # Merge profiles with progression
    merged = tournament_profiles.merge(progression_data, on='team', how='left')
    
    success_stats = merged.groupby(archetype_col).agg({
        'progression_score': ['mean', 'std', 'count'],
        'team': 'count'
    }).round(2)
    
    success_stats.columns = ['avg_progression', 'std_progression', 'count', 'n_teams']
    success_stats = success_stats.reset_index()
    
    return success_stats 

def print_cmi_table(cmi_results):
    """
    Display CMI results as a styled HTML table with colored bars.
    
    Args:
        cmi_results: DataFrame with 'dimension', 'cmi', 'compression_pct' columns
    """
    from IPython.display import display, HTML
    
    cmi_sorted = cmi_results.sort_values('cmi')
    
    rows_html = ""
    for _, row in cmi_sorted.iterrows():
        dim_name = row['dimension'].replace('_', ' ').title()
        cmi_val = row['cmi']
        comp_pct = row['compression_pct']
        
        if comp_pct > 15:
            color, tag = '#d62828', 'HIGH'
        elif comp_pct > 5:
            color, tag = '#e76f51', 'MOD'
        else:
            color, tag = '#2d6a4f', 'LOW'
        
        bar_width = min(int(abs(comp_pct) * 3), 200)
        bar = f'<div style="background:{color};width:{bar_width}px;height:14px;border-radius:3px;display:inline-block;"></div>'
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{dim_name}</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:#2d6a4f;">{cmi_val:.3f}</td>
            <td style="font-family:'SF Mono',monospace;color:{color};font-weight:700;">{comp_pct:+.1f}%</td>
            <td>{bar}</td>
            <td style="font-size:11px;font-weight:700;color:{color};">{tag}</td>
        </tr>"""
    
    html = f"""
    <style>
        .cmi {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .cmi th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:left; font-weight:600; }}
        .cmi td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .cmi tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        Compression by Dimension (Same-Era: 2022/23 Clubs → 2022-24 Tournaments)
    </h4>
    <table class="cmi">
        <tr><th>Dimension</th><th>CMI</th><th>Compression</th><th></th><th>Level</th></tr>
        {rows_html}
    </table>
    """
    display(HTML(html))