# tournament_progression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_tournament_results():
    """Tournament progression data: best result per team across 2022-24 tournaments.
    
    Scoring: 0=Group, 1=R16, 2=QF, 3=SF, 4=Final, 5=Winner
    Returns dict of dicts: {tournament_name: {team: score}}
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
    """Build progression DataFrame with best result per team across all tournaments."""
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
    """Merge progression scores with tournament profiles."""
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
    """Display archetype success summary as styled HTML table."""
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

def plot_paradox_scatter(merged_df, archetype_names, figsize=(8, 6), ax=None):
    """Chart: Prevalence vs Success scatter showing the paradox."""
    colors = {0: '#4895C4', 1: '#A23B72', 2: '#F18F01'}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for cid, name in archetype_names.items():
        mask = merged_df['assigned_archetype'] == cid
        subset = merged_df[mask]
        n = len(subset)
        prevalence = n / len(merged_df) * 100
        avg_prog = subset['progression_score'].mean()
        
        ax.scatter(prevalence, avg_prog, s=n * 40, c=colors[cid],
                   alpha=0.7, edgecolors='black', linewidth=2, zorder=3)
        ax.annotate(name, (prevalence, avg_prog),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 17), textcoords='offset points',
                    weight='bold')
    
    # Annotations
    ax.annotate('Most common\nLeast successful',
                xy=(70, 1.1), xytext=(55, 0.3),
                fontsize=9, style='italic', color='#d62828',
                arrowprops=dict(arrowstyle='->', color='#d62828', lw=1.5))
    ax.annotate('Rare but\nDominant',
                xy=(8, 3.0), xytext=(20, 2.8),
                fontsize=9, style='italic', color='#2d6a4f',
                arrowprops=dict(arrowstyle='->', color='#2d6a4f', lw=1.5))
    
    ax.set_xlabel('Prevalence in Tournaments (%)', fontsize=11)
    ax.set_ylabel('Average Progression Score', fontsize=11)
    ax.set_title('The Paradox: Prevalence vs Success', fontsize=13, weight='bold')
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 3.5)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    return fig, ax


def plot_success_by_archetype(merged_df, archetype_names, figsize=(6, 3), ax=None):
    """Chart: Average progression by archetype with reference lines."""
    colors = {0: '#4895C4', 1: '#A23B72', 2: '#F18F01'}

    # Calculate stats sorted by success
    stats = []
    for cid, name in archetype_names.items():
        mask = merged_df['assigned_archetype'] == cid
        subset = merged_df[mask]
        stats.append({
            'cid': cid, 'name': name,
            'avg': subset['progression_score'].mean(),
            'std': subset['progression_score'].std(),
            'n': len(subset)
        })
    stats = sorted(stats, key=lambda x: x['avg'], reverse=True)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    names = [s['name'] for s in stats]
    avgs = [s['avg'] for s in stats]
    stds = [s['std'] for s in stats]
    bar_colors = [colors[s['cid']] for s in stats]
    
    bars = ax.bar(range(len(stats)), avgs, color=bar_colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=8, error_kw={'linewidth': 2, 'alpha': 0.5})
    
    # Value labels
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, val/2,
                f'{val:.2f}', ha='center', va='center',
                fontsize=12, weight='bold', color='white')
    
    # Reference lines
    ax.axhline(y=3, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(len(stats) - 0.4, 3.1, 'Semi-final', fontsize=9, color='green', style='italic')
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(len(stats) - 0.4, 2.1, 'Quarter-final', fontsize=9, color='orange', style='italic')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(len(stats) - 0.4, 1.1, 'Round of 16', fontsize=9, color='red', style='italic')
    
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Average Progression Score', fontsize=11)
    ax.set_title('Tournament Success by Archetype', fontsize=13, weight='bold')
    ax.set_ylim(0, 5)
    ax.grid(axis='y', alpha=0.15)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax


def plot_progression_by_round(merged_df, archetype_names, figsize=(6, 3), ax=None):
    """Chart: Stacked archetype composition at each tournament stage."""
    colors = {0: '#4895C4', 1: '#A23B72', 2: '#F18F01'}
    
    stages = [
        ('Group Stage', 0),
        ('Round of 16', 1),
        ('Quarter-final', 2),
        ('Semi-final', 3),
        ('Final/Winner', 4)
    ]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # For each stage, calculate % of teams at or beyond that stage by archetype
    x = np.arange(len(stages))
    width = 0.7
    
    bottom = np.zeros(len(stages))
    for cid in sorted(archetype_names.keys()):
        name = archetype_names[cid]
        pcts = []
        for stage_name, threshold in stages:
            at_stage = merged_df[merged_df['progression_score'] >= threshold]
            if len(at_stage) > 0:
                arch_at_stage = (at_stage['assigned_archetype'] == cid).sum()
                pcts.append(arch_at_stage / len(at_stage) * 100)
            else:
                pcts.append(0)
        
        ax.bar(x, pcts, width, bottom=bottom, label=name,
               color=colors[cid], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += np.array(pcts)
    
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax.set_ylabel('Percentage of Teams (%)', fontsize=11)
    ax.set_title('Archetype Composition by Tournament Stage', fontsize=13, weight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#ddd')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.12)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    return fig, ax


def build_match_results(matches_df, team_archetypes_df):
    """Build match-level results with archetype assignments for both teams."""
    # Filter to tournament competitions
    tournament_comps = ['FIFA World Cup', "FIFA Women's World Cup", 
                        'UEFA Euro', 'Copa America']
    
    # Try to filter — adjust column names to match your data
    if 'competition_name' in matches_df.columns:
        tournament_matches = matches_df[
            matches_df['competition_name'].str.contains('|'.join(tournament_comps), case=False, na=False)
        ].copy()
    else:
        tournament_matches = matches_df.copy()
    
    # Build archetype lookup
    arch_lookup = dict(zip(team_archetypes_df['team'], team_archetypes_df['assigned_archetype']))
    name_lookup = dict(zip(team_archetypes_df['team'], team_archetypes_df['archetype_name']))
    
    # Assign archetypes to both teams
    tournament_matches['home_archetype'] = tournament_matches['home_team'].map(arch_lookup)
    tournament_matches['away_archetype'] = tournament_matches['away_team'].map(arch_lookup)
    tournament_matches['home_arch_name'] = tournament_matches['home_team'].map(name_lookup)
    tournament_matches['away_arch_name'] = tournament_matches['away_team'].map(name_lookup)
    
    # Determine result
    tournament_matches['result'] = 'draw'
    tournament_matches.loc[
        tournament_matches['home_score'] > tournament_matches['away_score'], 'result'] = 'home_win'
    tournament_matches.loc[
        tournament_matches['home_score'] < tournament_matches['away_score'], 'result'] = 'away_win'
    
    # Drop matches where either team has no archetype
    tournament_matches = tournament_matches.dropna(subset=['home_archetype', 'away_archetype'])
    
    return tournament_matches


def archetype_vs_archetype(match_results, archetype_names):
    """Calculate win/draw/loss rates for each archetype matchup."""
    matchups = []
    
    for cid_a, name_a in archetype_names.items():
        for cid_b, name_b in archetype_names.items():
            # Matches where A is home vs B away
            home = match_results[
                (match_results['home_archetype'] == cid_a) & 
                (match_results['away_archetype'] == cid_b)
            ]
            # Matches where A is away vs B home
            away = match_results[
                (match_results['away_archetype'] == cid_a) & 
                (match_results['home_archetype'] == cid_b)
            ]
            
            total = len(home) + len(away)
            if total == 0:
                continue
            
            # A wins
            a_wins = ((home['result'] == 'home_win').sum() + 
                      (away['result'] == 'away_win').sum())
            # B wins
            b_wins = ((home['result'] == 'away_win').sum() + 
                      (away['result'] == 'home_win').sum())
            # Draws
            draws = ((home['result'] == 'draw').sum() + 
                     (away['result'] == 'draw').sum())
            
            matchups.append({
                'archetype_a': name_a,
                'archetype_b': name_b,
                'matches': total,
                'a_wins': a_wins,
                'b_wins': b_wins,
                'draws': draws,
                'a_win_pct': a_wins / total * 100,
                'b_win_pct': b_wins / total * 100,
                'draw_pct': draws / total * 100
            })
    
    return pd.DataFrame(matchups)


def print_matchup_matrix(matchup_df, archetype_names):
    """Display archetype vs archetype win rates as styled HTML matrix."""
    from IPython.display import display, HTML
    
    names = list(archetype_names.values())
    
    rows_html = ""
    for name_a in names:
        cells = ""
        for name_b in names:
            row = matchup_df[
                (matchup_df['archetype_a'] == name_a) & 
                (matchup_df['archetype_b'] == name_b)
            ]
            if len(row) == 0 or name_a == name_b:
                cells += '<td style="text-align:center;color:#ccc;">—</td>'
            else:
                r = row.iloc[0]
                win_pct = r['a_win_pct']
                n = int(r['matches'])
                
                if win_pct >= 50:
                    color = '#2d6a4f'
                elif win_pct >= 35:
                    color = '#e76f51'
                else:
                    color = '#d62828'
                
                cells += f'''<td style="text-align:center;">
                    <span style="font-family:'SF Mono',monospace;font-weight:700;color:{color};">{win_pct:.0f}%</span>
                    <br><span style="font-size:10px;color:#999;">n={n}</span>
                </td>'''
        
        rows_html += f'<tr><td style="font-weight:600;color:#1a1a2e;white-space:nowrap;">{name_a}</td>{cells}</tr>'
    
    header_cells = "".join([f'<th style="white-space:nowrap;">{n}</th>' for n in names])
    
    html = f"""
    <style>
        .mm {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .mm th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:center; font-weight:600; }}
        .mm td {{ padding:8px 14px; border-bottom:1px solid #e0e0e0; }}
        .mm tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:4px;">
        Archetype vs Archetype: Win Rates in Tournaments
    </h4>
    <p style="font-family:-apple-system,sans-serif;font-size:11px;color:#888;margin-top:0;">
        Row team win % when facing column team
    </p>
    <table class="mm">
        <tr><th>vs →</th>{header_cells}</tr>
        {rows_html}
    </table>
    """
    display(HTML(html))