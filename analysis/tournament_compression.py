"""
Tournament compression analysis: CMI calculation and archetype success.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


def assign_to_archetypes(tournament_profiles, baseline_centers, dimensions):
    """
    Assign tournament teams to nearest baseline archetype.
    
    Args:
        tournament_profiles: DataFrame with tournament team profiles
        baseline_centers: DataFrame with baseline cluster centers
        dimensions: List of dimension names
        
    Returns:
        tournament_profiles with 'assigned_archetype' and 'distance_to_center'
    """
    tournament_profiles = tournament_profiles.copy()
    
    assignments = []
    distances = []
    
    for idx, team_profile in tournament_profiles.iterrows():
        team_vector = team_profile[dimensions].values
        
        min_distance = float('inf')
        assigned_cluster = None
        
        for cluster_id, center in baseline_centers.iterrows():
            center_vector = center[dimensions].values
            dist = euclidean(team_vector, center_vector)
            
            if dist < min_distance:
                min_distance = dist
                assigned_cluster = cluster_id
        
        assignments.append(assigned_cluster)
        distances.append(min_distance)
    
    tournament_profiles['assigned_archetype'] = assignments
    tournament_profiles['distance_to_center'] = distances
    
    return tournament_profiles


def calculate_cmi(baseline_profiles, tournament_profiles, dimensions):
    """
    Calculate Complexity Maintenance Index (CMI) for each dimension.
    
    CMI = tournament_variance / baseline_variance
    Lower CMI = more compression
    
    Args:
        baseline_profiles: DataFrame with baseline team profiles
        tournament_profiles: DataFrame with tournament team profiles
        dimensions: List of dimension names
        
    Returns:
        DataFrame with CMI for each dimension
    """
    cmi_results = []
    
    for dim in dimensions:
        baseline_var = baseline_profiles[dim].var()
        baseline_std = baseline_profiles[dim].std()
        baseline_mean = baseline_profiles[dim].mean()
        
        tournament_var = tournament_profiles[dim].var()
        tournament_std = tournament_profiles[dim].std()
        tournament_mean = tournament_profiles[dim].mean()
        
        cmi = tournament_std / baseline_std if baseline_std > 0 else 1.0
        compression_pct = (1 - cmi) * 100
        
        cmi_results.append({
            'dimension': dim,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'tournament_mean': tournament_mean,
            'tournament_std': tournament_std,
            'cmi': cmi,
            'compression_pct': compression_pct
        })
    
    cmi_df = pd.DataFrame(cmi_results)
    
    # Overall CMI
    overall_cmi = cmi_df['cmi'].mean()
    overall_compression = (1 - overall_cmi) * 100
    
    print(f"\nOverall CMI: {overall_cmi:.3f}")
    print(f"Overall Compression: {overall_compression:.1f}%")
    
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


def analyze_archetype_shift(baseline_df, tournament_df, archetype_names,
                            baseline_label='Club 2015/16', 
                            tournament_label='Tournament 2022-24'):
    """
    Calculate and display archetype distribution shift between club and tournament.
    
    Args:
        baseline_df: Club profiles with 'cluster' or 'assigned_archetype' column
        tournament_df: Tournament profiles with 'assigned_archetype' column
        archetype_names: dict mapping cluster_id to name
        baseline_label: Label for baseline data
        tournament_label: Label for tournament data
    
    Returns:
        shift_df: DataFrame with shift analysis
    """
    # Determine column names
    baseline_col = 'cluster' if 'cluster' in baseline_df.columns else 'assigned_archetype'
    tournament_col = 'assigned_archetype'
    
    # Calculate distributions
    baseline_counts = baseline_df[baseline_col].value_counts().sort_index()
    tournament_counts = tournament_df[tournament_col].value_counts().sort_index()
    
    baseline_pct = baseline_counts / len(baseline_df) * 100
    tournament_pct = tournament_counts / len(tournament_df) * 100
    
    # Build shift dataframe
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
    
    shift_df = pd.DataFrame(rows)
    
    return shift_df


def plot_archetype_shift(shift_df, baseline_label='Club 2015/16',
                         tournament_label='Tournament 2022-24', figsize=(9, 5)):
    """
    Grouped bar chart of archetype distribution shift.
    
    Args:
        shift_df: DataFrame from analyze_archetype_shift
        baseline_label: Label for baseline bars
        tournament_label: Label for tournament bars
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(shift_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, shift_df['baseline_pct'], width,
                   label=baseline_label, color='#2E86AB', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, shift_df['tournament_pct'], width,
                   label=tournament_label, color='#F18F01', edgecolor='white', linewidth=0.5)
    
    # Value labels
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
    
    plt.tight_layout()
    return fig, ax


def print_archetype_shift_table(shift_df):
    """
    Display archetype shift as a styled HTML table.
    
    Args:
        shift_df: DataFrame from analyze_archetype_shift
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
            <td style="font-family:'SF Mono',monospace;font-weight:600;">{row['baseline_pct']:.1f}%</td>
            <td style="font-family:'SF Mono',monospace;font-weight:600;">{row['tournament_pct']:.1f}%</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:{color};">{arrow} {shift:+.1f}pp</td>
            <td>{bar}</td>
        </tr>"""
    
    html = f"""
    <style>
        .shift {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .shift th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:left; font-weight:600; }}
        .shift td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .shift tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        Archetype Distribution Shift
    </h4>
    <table class="shift">
        <tr><th>Archetype</th><th>Club</th><th>Tournament</th><th>Shift</th><th></th></tr>
        {rows_html}
    </table>
    """
    display(HTML(html))
