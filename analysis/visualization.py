"""
Compact, publication-ready visualizations for tactical analysis.
All functions create multi-panel figures to minimize notebook clutter.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def project_to_pca(baseline_df, new_dfs, dimensions):
    """
    Fit PCA on baseline data and project all dataframes into same space.
    Standalone function — no dependency on clustering object state.
    
    Args:
        baseline_df: 2015/16 club profiles (must have 'cluster' column)
        new_dfs: list of DataFrames to project (modified in place)
        dimensions: list of D12 dimension names
    
    Returns:
        pca: fitted PCA object
        scaler: fitted StandardScaler
    """
    scaler = StandardScaler()
    scaled_baseline = scaler.fit_transform(baseline_df[dimensions])
    
    pca = PCA(n_components=2)
    baseline_coords = pca.fit_transform(scaled_baseline)
    
    # Add to baseline if not already there
    if 'PC1' not in baseline_df.columns:
        baseline_df['PC1'] = baseline_coords[:, 0]
        baseline_df['PC2'] = baseline_coords[:, 1]
    
    # Project each new df
    for df in new_dfs:
        scaled = scaler.transform(df[dimensions])
        coords = pca.transform(scaled)
        df['PC1'] = coords[:, 0]
        df['PC2'] = coords[:, 1]
    
    return pca, scaler


def prepare_compression_plot(baseline_df, recent_club_df, tournament_df,
                             cluster_centers, dimensions, archetype_names):
    """
    Full pipeline: assign archetypes, project PCA, return ready-to-plot dataframes.
    
    Args:
        baseline_df: 2015/16 club profiles with 'cluster' column
        recent_club_df: same-era club profiles
        tournament_df: tournament profiles
        cluster_centers: archetype centers with 'cluster' column
        dimensions: list of D12 dimension names
        archetype_names: dict {cluster_id: name}
    
    Returns:
        baseline_df, recent_club_df, tournament_df (all with PC1, PC2, archetype columns)
    """
    from analysis.clustering_analysis import TacticalClustering
    
    # Assign archetypes to new data if not already done
    for df in [recent_club_df, tournament_df]:
        if 'assigned_archetype' not in df.columns:
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
        
        if 'archetype_name' not in df.columns:
            df['archetype_name'] = df['assigned_archetype'].map(archetype_names)
    
    # Project all into same PCA space
    project_to_pca(baseline_df, [recent_club_df, tournament_df], dimensions)
    
    return baseline_df, recent_club_df, tournament_df

def plot_clustering_validation_compact(optimization_results, figsize=(12, 5)):
    """
    Create compact validation figure with elbow + silhouette side-by-side.
    
    Args:
        optimization_results: DataFrame from optimize_k()
        
    Returns:
        fig, axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    ax1.plot(optimization_results['k'], optimization_results['inertia'], 
             'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia', fontsize=11)
    ax1.set_title('Elbow Method', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(optimization_results['k'])
    
    # Silhouette plot
    ax2.plot(optimization_results['k'], optimization_results['silhouette'], 
             'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Analysis', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(optimization_results['k'])
    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def save_figure(fig, filename, dpi=300):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}', dpi=dpi, bbox_inches='tight')
    print(f"Saved: outputs/figures/{filename}")

def plot_tactical_pca(profiles_df, dimensions, labels, archetype_names,
                      highlight_teams=None, figsize=(6, 6)):
    """
    PCA tactical map — the anchor visual. Separate from radars.
    
    Args:
        highlight_teams: Optional list of team names to annotate on the plot
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    colors = ['#4895C4', '#A23B72', '#F18F01']
    
    # PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles_df[dimensions])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each archetype
    for cluster_id in range(len(archetype_names)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[cluster_id], alpha=0.5, s=60, edgecolors='white',
                   linewidth=0.5, label=archetype_names[cluster_id], zorder=2)
        
        # Centroid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, c=colors[cluster_id], s=150, edgecolors='black',
                   linewidth=1, marker='D', zorder=3)
    
    # Annotate specific teams
    if highlight_teams and 'team' in profiles_df.columns:
        for team in highlight_teams:
            idx = profiles_df.index[profiles_df['team'] == team]
            if len(idx) > 0:
                i = idx[0]
                x, y = coords[i, 0], coords[i, 1]
                ax.annotate(team, (x, y), fontsize=8, 
                            xytext=(6, 6), textcoords='offset points',
                            arrowprops=dict(arrowstyle='-', color='#999', lw=0.8),
                            color='#1a1a2e', zorder=7)
    
    # Axis labels with variance explained
    ax.set_title("PCA: Tactical Space", fontsize=12, weight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    loadings = pca.components_
    top_pc1 = [dimensions[i].replace('_', ' ').title() for i in np.argsort(np.abs(loadings[0]))[-2:]]
    top_pc2 = [dimensions[i].replace('_', ' ').title() for i in np.argsort(np.abs(loadings[1]))[-2:]]
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%}) — {', '.join(top_pc1)}", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%}) — {', '.join(top_pc2)}", fontsize=9)
    
    plt.tight_layout()
    return fig, pca

def plot_archetype_radars(cluster_centers, dimensions, archetype_names, 
                          profiles_df=None, figsize=(15, 4)):
    """
    Clean radar charts only — one per archetype, side by side.
    """
    k = len(archetype_names)
    colors = ['#4895C4', '#A23B72', '#F18F01'] 

    fig, axes = plt.subplots(1, k, figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=-0.75)
    if k == 1:
        axes = [axes]
    
    for cluster_id in range(k):
        ax = axes[cluster_id]
        center = cluster_centers.iloc[cluster_id]
        
        # Normalize values to 0-1 using global min/max from profiles
        values = []
        for dim in dimensions:
            if profiles_df is not None:
                val = (center[dim] - profiles_df[dim].min()) / (profiles_df[dim].max() - profiles_df[dim].min())
            else:
                val = center[dim]
            values.append(val)
        values.append(values[0])  # close the polygon
        
        # Angles
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[cluster_id], markersize=4)
        ax.fill(angles, values, alpha=0.2, color=colors[cluster_id])
        
        # Labels
        labels_clean = [d.replace('_', '\n').title() for d in dimensions]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_clean, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=8, color='#999')
        
        # Count teams in this archetype
        name = archetype_names[cluster_id]
        size = int(center.get('size', 0)) if 'size' in center.index else ''
        title = f"{name}\n(n={size})" if size else name
        ax.set_title(title, fontsize=11, weight='bold', pad=20, color=colors[cluster_id])
    
    plt.tight_layout()
    return fig


    Python
import matplotlib.pyplot as plt
import pandas as pd

def run_tournament_analysis(tri_data):
    """
    Processes TRI data and generates the 'G-Force of Convergence' plot.
    """
    # 1. Prepare data for the DataFrame
    processed_data = []
    for nation, stats in tri_data.items():
        processed_data.append({
            'Nation': nation,
            'TRI': stats['tri'],
            'Quality': stats['quality'],
            'Coherence': stats['coherence'],
            'Archetype': stats['predicted_archetype']
        })
    
    df = pd.DataFrame(processed_data)
    
    # 2. Setup Visualization
    plt.figure(figsize=(12, 8))
    
    # Calculate bubble size based on TRI score
    sizes = df['TRI'] * 1000 
    
    # Scatter Plot: Quality vs. Coherence
    scatter = plt.scatter(
        df['Quality'], 
        df['Coherence'], 
        s=sizes, 
        alpha=0.6, 
        c=df['TRI'], 
        cmap='viridis',
        edgecolors='black'
    )
    
    # Annotations
    for i, row in df.iterrows():
        plt.annotate(
            row['Nation'], 
            (row['Quality'], row['Coherence']),
            xytext=(10, 5), 
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )
    
    # 3. Add Contextual Boundaries
    plt.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Stability Threshold')
    plt.axvspan(1.5, 2.5, color='gray', alpha=0.1, label='High G-Force Zone')
    
    plt.title("2026 World Cup: The G-Force of Convergence\n(Quality vs. Tactical Stability)", fontsize=15)
    plt.xlabel("Squad Quality (Club DNA Complexity)", fontsize=12)
    plt.ylabel("Coherence (Tactical Dialect Synchronization)", fontsize=12)
    plt.colorbar(scatter, label='Tactical Readiness Index (TRI)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 4. Final Output
    plt.tight_layout()
    plt.show()
    print(">>> Analysis Complete: G-Force Convergence Plot generated.")