# wc_prediction.py

import pandas as pd
import numpy as np


def predict_team_archetypes(rosters, club_archetype_map, archetype_names):
    """
    Map player clubs to archetypes and predict national team tactical identity.
    
    Args:
        rosters: dict of {team: {player: club}}
        club_archetype_map: dict of {club: archetype_name}
        archetype_names: dict of {cluster_id: name} (used for initializing counts)
    
    Returns:
        dict of team predictions
    """
    name_list = list(archetype_names.values())
    predictions = {}
    
    for team, roster in rosters.items():
        archetype_counts = {name: 0 for name in name_list}
        mapped_players = []
        unmapped_players = []
        
        for player, club in roster.items():
            if club in club_archetype_map:
                archetype = club_archetype_map[club]
                archetype_counts[archetype] += 1
                mapped_players.append((player, club, archetype))
            else:
                unmapped_players.append((player, club))
        
        total_mapped = sum(archetype_counts.values())
        
        if total_mapped > 0:
            archetype_pcts = {k: (v / total_mapped) * 100 for k, v in archetype_counts.items()}
            predicted = max(archetype_pcts.items(), key=lambda x: x[1])[0]
            coverage = total_mapped / len(roster) * 100
            
            predictions[team] = {
                'predicted_archetype': predicted,
                'archetype_distribution': archetype_pcts,
                'archetype_counts': archetype_counts,
                'mapped_count': total_mapped,
                'unmapped_count': len(unmapped_players),
                'total_players': len(roster),
                'coverage_pct': coverage,
                'mapped_players': mapped_players,
                'unmapped_players': unmapped_players,
                'confidence': 'HIGH' if coverage >= 70 else 'MEDIUM' if coverage >= 50 else 'LOW'
            }
    
    return predictions


def predictions_to_df(predictions, archetype_names):
    """
    Convert predictions dict to a clean DataFrame.
    
    Args:
        predictions: dict from predict_team_archetypes
        archetype_names: dict of {cluster_id: name}
    
    Returns:
        DataFrame sorted by coverage
    """
    rows = []
    for team, pred in predictions.items():
        row = {
            'team': team,
            'predicted_archetype': pred['predicted_archetype'],
            'coverage_pct': pred['coverage_pct'],
            'mapped': pred['mapped_count'],
            'total': pred['total_players'],
            'confidence': pred['confidence']
        }
        for name in archetype_names.values():
            row[name + '_pct'] = pred['archetype_distribution'].get(name, 0)
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('coverage_pct', ascending=False)


def print_prediction_table(predictions, archetype_names):
    """
    Display predictions as styled HTML table.
    """
    from IPython.display import display, HTML
    
    df = predictions_to_df(predictions, archetype_names)
    name_list = list(archetype_names.values())
    colors = {'HIGH': '#2d6a4f', 'MEDIUM': '#e76f51', 'LOW': '#d62828'}
    
    rows_html = ""
    for _, row in df.iterrows():
        conf_color = colors[row['confidence']]
        
        # Mini distribution bar
        bar_parts = ""
        arch_colors = {name_list[0]: '#4895C4', name_list[1]: '#A23B72', name_list[2]: '#F18F01'}  
        for name in name_list:
            pct = row[name + '_pct']
            if pct > 0:
                bar_parts += f'<div style="background:{arch_colors[name]};width:{pct}%;height:16px;display:inline-block;" title="{name}: {pct:.0f}%"></div>'
        
        bar = f'<div style="width:150px;display:flex;border-radius:3px;overflow:hidden;">{bar_parts}</div>'
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{row['team']}</td>
            <td style="font-weight:600;">{row['predicted_archetype']}</td>
            <td>{bar}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{row['mapped']}/{row['total']}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{row['coverage_pct']:.0f}%</td>
            <td style="font-weight:700;color:{conf_color};text-align:center;">{row['confidence']}</td>
        </tr>"""
    
    html = f"""
    <style>
        .pred {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .pred th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:left; font-weight:600; }}
        .pred td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .pred tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:8px;">
        2026 World Cup: Predicted Tactical Identity
    </h4>
    <table class="pred">
        <tr>
            <th>Team</th>
            <th>Predicted Archetype</th>
            <th>Distribution</th>
            <th>Coverage</th>
            <th>%</th>
            <th>Confidence</th>
        </tr>
        {rows_html}
    </table>
    """
    display(HTML(html))


def print_team_detail(predictions, team):
    """
    Display detailed breakdown for a single team.
    """
    from IPython.display import display, HTML
    
    pred = predictions[team]
    
    # Group mapped players by archetype
    by_archetype = {}
    for player, club, arch in pred['mapped_players']:
        if arch not in by_archetype:
            by_archetype[arch] = []
        by_archetype[arch].append(f"{player} ({club})")
    
    sections = ""
    for arch, players in sorted(by_archetype.items(), key=lambda x: len(x[1]), reverse=True):
        pct = pred['archetype_distribution'][arch]
        player_list = ", ".join(players)
        sections += f"""
        <div style="margin-bottom:8px;">
            <span style="font-weight:700;font-size:12px;">{arch}</span>
            <span style="font-family:'SF Mono',monospace;font-size:11px;color:#666;"> — {len(players)} players ({pct:.0f}%)</span>
            <div style="font-size:11px;color:#555;margin-top:2px;">{player_list}</div>
        </div>"""
    
    # Unmapped
    unmapped_html = ""
    if pred['unmapped_players']:
        unmapped_list = ", ".join([f"{p} ({c})" for p, c in pred['unmapped_players']])
        unmapped_html = f"""
        <div style="margin-top:8px;padding-top:8px;border-top:1px solid #e0e0e0;">
            <span style="font-weight:700;font-size:12px;color:#999;">Unmapped</span>
            <span style="font-family:'SF Mono',monospace;font-size:11px;color:#999;"> — {pred['unmapped_count']} players</span>
            <div style="font-size:11px;color:#999;margin-top:2px;">{unmapped_list}</div>
        </div>"""
    
    conf_colors = {'HIGH': '#2d6a4f', 'MEDIUM': '#e76f51', 'LOW': '#d62828'}
    
    html = f"""
    <div style="font-family:-apple-system,sans-serif;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;max-width:500px;">
        <div style="background:#1a1a2e;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
            <span style="color:white;font-weight:700;font-size:14px;">{team}</span>
            <span style="color:{conf_colors[pred['confidence']]};font-weight:700;font-size:11px;background:white;padding:2px 8px;border-radius:4px;">
                {pred['confidence']} ({pred['coverage_pct']:.0f}%)
            </span>
        </div>
        <div style="padding:12px 14px;">
            <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;">Predicted Archetype</div>
            <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-bottom:12px;">{pred['predicted_archetype']}</div>
            {sections}
            {unmapped_html}
        </div>
    </div>
    """
    display(HTML(html))

# Add to wc_prediction.py

def calculate_tri(predictions, archetype_success_rates):
    """
    Tactical Readiness Index: composite score predicting 
    tournament compression resistance.
    
    Args:
        predictions: dict from predict_team_archetypes
        archetype_success_rates: dict of {archetype_name: avg_progression_score}
            e.g. {'Elite Attackers': 3.0, 'Proactive Organizers': 1.78, 'Balanced Mainstream': 1.05}
    
    Returns:
        dict of {team: {tri, coherence, quality, coverage, components}}
    """
    results = {}
    
    for team, pred in predictions.items():
        shares = [v / 100 for v in pred['archetype_distribution'].values()]
        
        # 1. Coherence (Herfindahl: 0.33 to 1.0, normalized to 0-1)
        hhi = sum(s ** 2 for s in shares)
        coherence = (hhi - 1/3) / (1 - 1/3)  # normalize: 0 = perfectly split, 1 = all same
        
        # 2. Quality (weighted avg of archetype success rates)
        quality = 0
        for arch, pct in pred['archetype_distribution'].items():
            rate = archetype_success_rates.get(arch, 1.0)
            quality += (pct / 100) * rate
        # Normalize to 0-1 (max possible is highest success rate)
        max_rate = max(archetype_success_rates.values())
        quality_norm = quality / max_rate
        
        # 3. Coverage (already 0-100, normalize to 0-1)
        coverage_norm = pred['coverage_pct'] / 100
        
        # TRI = weighted combination
        tri = (0.40 * coherence) + (0.40 * quality_norm) + (0.20 * coverage_norm)
        
        results[team] = {
            'tri': tri,
            'coherence': coherence,
            'quality': quality,
            'quality_norm': quality_norm,
            'coverage': coverage_norm,
            'predicted_archetype': pred['predicted_archetype'],
            'confidence': pred['confidence']
        }
    
    return results


def print_tri_table(tri_results):
    """Display TRI results as styled HTML table."""
    from IPython.display import display, HTML
    
    sorted_teams = sorted(tri_results.items(), key=lambda x: x[1]['tri'], reverse=True)
    
    rows_html = ""
    for rank, (team, data) in enumerate(sorted_teams, 1):
        tri = data['tri']
        
        # Color by TRI tier
        if tri >= 0.55:
            tier, tier_color = 'PRIME', '#2d6a4f'
        elif tri >= 0.40:
            tier, tier_color = 'CONTENDER', '#457b9d'
        elif tri >= 0.30:
            tier, tier_color = 'VULNERABLE', '#e76f51'
        else:
            tier, tier_color = 'AT RISK', '#d62828'
        
        # TRI bar
        bar_width = int(tri * 250)
        bar = f'<div style="background:{tier_color};width:{bar_width}px;height:16px;border-radius:3px;display:inline-block;"></div>'
        
        rows_html += f"""<tr>
            <td style="font-family:'SF Mono',monospace;text-align:center;color:#999;">{rank}</td>
            <td style="font-weight:600;color:#1a1a2e;">{team}</td>
            <td style="font-family:'SF Mono',monospace;font-weight:700;color:{tier_color};text-align:center;">{tri:.3f}</td>
            <td>{bar}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{data['coherence']:.2f}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{data['quality']:.2f}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{data['coverage']:.0%}</td>
            <td style="font-weight:700;color:{tier_color};text-align:center;font-size:11px;">{tier}</td>
        </tr>"""
    
    html = f"""
    <style>
        .tri {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .tri th {{ background:#1a1a2e; color:white; padding:8px 12px; text-align:center; font-weight:600; }}
        .tri th:nth-child(2) {{ text-align:left; }}
        .tri td {{ padding:6px 12px; border-bottom:1px solid #e0e0e0; }}
        .tri tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:4px;">
        Tactical Readiness Index: 2026 World Cup
    </h4>
    <p style="font-family:-apple-system,sans-serif;font-size:11px;color:#888;margin-top:0;">
        TRI = 40% Coherence + 40% Archetype Quality + 20% Coverage
    </p>
    <table class="tri">
        <tr>
            <th>#</th>
            <th style="text-align:left;">Team</th>
            <th>TRI</th>
            <th></th>
            <th>Coherence</th>
            <th>Quality</th>
            <th>Coverage</th>
            <th>Tier</th>
        </tr>
        {rows_html}
    </table>
    """
    display(HTML(html))

# Add to wc_prediction.py

def analyze_matchup(team_a, team_b, predictions, tri_results, 
                    matchup_df, men_tournament_pd, dimensions):
    """
    Generate detailed matchup analysis between two teams.
    """
    from IPython.display import display, HTML
    
    pred_a = predictions[team_a]
    pred_b = predictions[team_b]
    tri_a = tri_results[team_a]
    tri_b = tri_results[team_b]
    
    arch_a = pred_a['predicted_archetype']
    arch_b = pred_b['predicted_archetype']
    
    # Historical matchup rate
    row = matchup_df[
        (matchup_df['archetype_a'] == arch_a) & 
        (matchup_df['archetype_b'] == arch_b)
    ]
    
    if len(row) > 0:
        hist = row.iloc[0]
        hist_html = f"""
        <div style="margin:12px 0;padding:10px;background:#f8f8f8;border-radius:6px;">
            <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;">
                Historical: {arch_a} vs {arch_b} (n={int(hist['matches'])})
            </div>
            <div style="display:flex;gap:20px;margin-top:6px;">
                <span style="font-family:'SF Mono',monospace;font-weight:700;color:#2d6a4f;">{arch_a}: {hist['a_win_pct']:.0f}%</span>
                <span style="font-family:'SF Mono',monospace;color:#999;">Draw: {hist['draw_pct']:.0f}%</span>
                <span style="font-family:'SF Mono',monospace;font-weight:700;color:#d62828;">{arch_b}: {hist['b_win_pct']:.0f}%</span>
            </div>
        </div>"""
    else:
        hist_html = ""
    
    # D12 comparison
    d12_rows = ""
    # Get tournament profiles if available
    profile_a = men_tournament_pd[men_tournament_pd['team'] == team_a]
    profile_b = men_tournament_pd[men_tournament_pd['team'] == team_b]
    
    if len(profile_a) > 0 and len(profile_b) > 0:
        for dim in dimensions:
            val_a = profile_a[dim].values[0]
            val_b = profile_b[dim].values[0]
            diff = val_a - val_b
            
            if abs(diff) > 0:
                winner_color = '#2E86AB' if diff > 0 else '#F18F01'
                bar_width = min(int(abs(diff) * 20), 100)
                direction = 'right' if diff > 0 else 'left'
                
                d12_rows += f"""<tr>
                    <td style="font-size:12px;">{dim.replace('_', ' ').title()}</td>
                    <td style="font-family:'SF Mono',monospace;text-align:center;font-size:11px;">{val_a:.2f}</td>
                    <td style="text-align:center;">
                        <div style="background:{winner_color};width:{bar_width}px;height:10px;border-radius:2px;display:inline-block;"></div>
                    </td>
                    <td style="font-family:'SF Mono',monospace;text-align:center;font-size:11px;">{val_b:.2f}</td>
                </tr>"""
    
    # TRI comparison
    tri_winner = team_a if tri_a['tri'] > tri_b['tri'] else team_b
    tri_gap = abs(tri_a['tri'] - tri_b['tri'])
    
    html = f"""
    <style>
        .mu {{ font-family:-apple-system,sans-serif; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden; max-width:600px; }}
        .mu-header {{ background:#1a1a2e; padding:14px; display:flex; justify-content:space-between; align-items:center; }}
        .mu-team {{ color:white; font-weight:700; font-size:16px; }}
        .mu-vs {{ color:#888; font-size:12px; }}
        .mu-body {{ padding:14px; }}
        .mu-row {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #f0f0f0; }}
        .mu-label {{ font-size:11px; color:#999; }}
        .mu-val {{ font-family:'SF Mono',monospace; font-weight:700; font-size:13px; }}
        .d12 {{ border-collapse:collapse; font-size:12px; width:100%; margin-top:8px; }}
        .d12 td {{ padding:4px 8px; border-bottom:1px solid #f0f0f0; }}
    </style>
    <div class="mu">
        <div class="mu-header">
            <span class="mu-team" style="color:#2E86AB;">{team_a}</span>
            <span class="mu-vs">VS</span>
            <span class="mu-team" style="color:#F18F01;">{team_b}</span>
        </div>
        <div class="mu-body">
            <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                <div>
                    <div class="mu-label">Archetype</div>
                    <div class="mu-val" style="color:#2E86AB;">{arch_a}</div>
                </div>
                <div style="text-align:right;">
                    <div class="mu-label">Archetype</div>
                    <div class="mu-val" style="color:#F18F01;">{arch_b}</div>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <div>
                    <div class="mu-label">TRI</div>
                    <div class="mu-val" style="color:#2E86AB;">{tri_a['tri']:.3f}</div>
                </div>
                <div style="text-align:center;">
                    <div class="mu-label">Advantage</div>
                    <div class="mu-val">{tri_winner} (+{tri_gap:.3f})</div>
                </div>
                <div style="text-align:right;">
                    <div class="mu-label">TRI</div>
                    <div class="mu-val" style="color:#F18F01;">{tri_b['tri']:.3f}</div>
                </div>
            </div>
            {hist_html}
            {"<h4 style='font-size:12px;color:#1a1a2e;margin:12px 0 4px;'>Dimension Comparison</h4><table class='d12'><tr><td></td><td style='text-align:center;font-weight:600;color:#2E86AB;'>" + team_a + "</td><td></td><td style='text-align:center;font-weight:600;color:#F18F01;'>" + team_b + "</td></tr>" + d12_rows + "</table>" if d12_rows else ""}
        </div>
    </div>
    """
    display(HTML(html))

# Add to wc_prediction.py

def apply_compression(predictions, men_tournament_pd, recent_club_pd,
                      cluster_centers, dimensions, archetype_names):
    """
    Adjust team predictions by applying known tournament compression rates.
    
    Uses dimension-level compression ratios (tournament std / club std)
    to shift each team's predicted profile toward the tournament mean.
    
    Args:
        predictions: dict from predict_team_archetypes
        men_tournament_pd: tournament profiles (for tournament means)
        recent_club_pd: same-era club profiles (for compression ratios)
        cluster_centers: archetype centers
        dimensions: list of D12 dimension names
        archetype_names: dict of {cid: name}
    
    Returns:
        DataFrame with pre and post compression archetype assignments
    """
    name_to_id = {v: k for k, v in archetype_names.items()}
    
    # Compression ratios per dimension
    club_stds = recent_club_pd[dimensions].std()
    tourn_stds = men_tournament_pd[dimensions].std()
    compression_ratios = tourn_stds / club_stds  # < 1 means compression
    
    # Tournament means (what teams compress toward)
    tourn_means = men_tournament_pd[dimensions].mean()
    
    results = []
    
    for team, pred in predictions.items():
        # Build predicted club profile (weighted avg of archetype centers)
        club_profile = np.zeros(len(dimensions))
        for arch_name, pct in pred['archetype_distribution'].items():
            cid = name_to_id[arch_name]
            center = cluster_centers[cluster_centers['cluster'] == cid][dimensions].values[0]
            club_profile += (pct / 100) * center
        
        # Apply compression: pull each dimension toward tournament mean
        # compressed = tournament_mean + compression_ratio * (club_value - tournament_mean)
        compressed_profile = np.zeros(len(dimensions))
        for i, dim in enumerate(dimensions):
            ratio = compression_ratios[dim]
            compressed_profile[i] = tourn_means[dim] + ratio * (club_profile[i] - tourn_means[dim])
        
        # Assign compressed profile to nearest archetype
        best_dist = np.inf
        best_arch = None
        for cid, name in archetype_names.items():
            center = cluster_centers[cluster_centers['cluster'] == cid][dimensions].values[0]
            dist = np.sqrt(((compressed_profile - center) ** 2).sum())
            if dist < best_dist:
                best_dist = dist
                best_arch = name
        
        results.append({
            'team': team,
            'pre_compression': pred['predicted_archetype'],
            'post_compression': best_arch,
            'shifted': pred['predicted_archetype'] != best_arch,
            'compression_distance': np.sqrt(((club_profile - compressed_profile) ** 2).sum()),
            'confidence': pred['confidence']
        })
    
    return pd.DataFrame(results)


def print_compression_predictions(compression_df):
    """Display pre vs post compression predictions."""
    from IPython.display import display, HTML
    
    rows_html = ""
    for _, row in compression_df.iterrows():
        shifted = row['shifted']
        arrow_color = '#d62828' if shifted else '#2d6a4f'
        arrow = '→' if shifted else '='
        shift_label = 'SHIFTED' if shifted else 'HELD'
        
        rows_html += f"""<tr>
            <td style="font-weight:600;color:#1a1a2e;">{row['team']}</td>
            <td style="font-family:'SF Mono',monospace;">{row['pre_compression']}</td>
            <td style="text-align:center;font-weight:700;color:{arrow_color};font-size:16px;">{arrow}</td>
            <td style="font-family:'SF Mono',monospace;">{row['post_compression']}</td>
            <td style="font-family:'SF Mono',monospace;text-align:center;">{row['compression_distance']:.1f}</td>
            <td style="font-weight:700;color:{arrow_color};text-align:center;font-size:11px;">{shift_label}</td>
        </tr>"""
    
    html = f"""
    <style>
        .comp {{ border-collapse:collapse; font-family:-apple-system,sans-serif; font-size:13px; }}
        .comp th {{ background:#1a1a2e; color:white; padding:8px 14px; text-align:left; font-weight:600; }}
        .comp td {{ padding:6px 14px; border-bottom:1px solid #e0e0e0; }}
        .comp tr:hover {{ background:#f5f5f5; }}
    </style>
    <h4 style="font-family:-apple-system,sans-serif;color:#1a1a2e;margin-bottom:4px;">
        Compression-Adjusted Predictions: 2026 World Cup
    </h4>
    <p style="font-family:-apple-system,sans-serif;font-size:11px;color:#888;margin-top:0;">
        Applying tournament compression rates to club-based predictions
    </p>
    <table class="comp">
        <tr>
            <th>Team</th>
            <th>Club DNA</th>
            <th></th>
            <th>Tournament Predicted</th>
            <th>Compression Distance</th>
            <th>Status</th>
        </tr>
        {rows_html}
    </table>
    """
    display(HTML(html))

import random

def calculate_match_probability(team_a, team_b):
    """
    Calculates the probability of Team A beating Team B.
    Formula: P(A) = (TRI_A * Coherence_A) / [(TRI_A * Coherence_A) + (TRI_B * Coherence_B)]
    """
    # We use Coherence as a multiplier because in a 'Universal Press' 
    # environment, stability is the primary driver of execution.
    power_a = team_a['tri'] * (1 + team_a['coherence'])
    power_b = team_b['tri'] * (1 + team_b['coherence'])
    
    prob_a = power_a / (power_a + power_b)
    return prob_a

def simulate_knockout(team_a_name, team_b_name, tri_results):
    """
    Simulates a single knockout match.
    """
    a_stats = tri_results[team_a_name]
    b_stats = tri_results[team_b_name]
    
    prob_a = calculate_match_probability(a_stats, b_stats)
    
    # Random simulation based on the calculated probability
    winner = team_a_name if random.random() < prob_a else team_b_name
    
    return {
        "Matchup": f"{team_a_name} vs {team_b_name}",
        f"{team_a_name}_Win_Prob": f"{round(prob_a * 100, 1)}%",
        f"{team_b_name}_Win_Prob": f"{round((1 - prob_a) * 100, 1)}%",
        "Predicted_Winner": winner
    }