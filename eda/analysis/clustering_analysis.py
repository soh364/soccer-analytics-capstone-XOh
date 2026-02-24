"""
Clustering analysis for tactical archetypes.
Handles k-means, validation, and archetype characterization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    silhouette_samples,
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import fcluster
from itertools import permutations
from IPython.display import display, HTML



class TacticalClustering:
    """Handle all clustering operations for tactical analysis"""
    
    def __init__(self, dimensions):
        """Initialize with dimension names."""
        self.dimensions = dimensions
        self.scaler = StandardScaler()
        self.kmeans = None
        self.scaled_data = None
        self.pca = None
        
    def prepare_data(self, profiles_df):
        """Standardize tactical dimensions."""
        tactical_data = profiles_df[self.dimensions]
        self.scaled_data = self.scaler.fit_transform(tactical_data)
        return self.scaled_data
    
    def run_kmeans(self, k=3, random_state=42, n_init=100):
        """Run k-means clustering."""
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = self.kmeans.fit_predict(self.scaled_data)
        
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=self.dimensions
        )
        centers['cluster'] = range(k)
        
        return {
            'labels': labels,
            'centers': centers,
            'inertia': self.kmeans.inertia_,
            'k': k
        }
    
    def optimize_k(self, k_range=range(3, 11)):
        """Find optimal k using elbow and silhouette methods."""
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
            labels = kmeans.fit_predict(self.scaled_data)
            
            results.append({
                'k': k,
                'inertia': kmeans.inertia_,
                'silhouette': silhouette_score(self.scaled_data, labels),
                'calinski_harabasz': calinski_harabasz_score(self.scaled_data, labels),
                'davies_bouldin': davies_bouldin_score(self.scaled_data, labels)
            })
        
        return pd.DataFrame(results)
    
    def validate_clusters(self, labels, linkage_matrix=None, random_state=42):
        """Calculate comprehensive validation metrics."""
        validation = {}
        
        validation['silhouette_avg'] = silhouette_score(self.scaled_data, labels)
        validation['calinski_harabasz'] = calinski_harabasz_score(self.scaled_data, labels)
        validation['davies_bouldin'] = davies_bouldin_score(self.scaled_data, labels)
        validation['inertia'] = self.kmeans.inertia_ if self.kmeans else None
        
        silhouette_vals = silhouette_samples(self.scaled_data, labels)
        validation['silhouette_by_cluster'] = {}
        for cluster_id in np.unique(labels):
            cluster_sil = silhouette_vals[labels == cluster_id].mean()
            validation['silhouette_by_cluster'][cluster_id] = cluster_sil
        
        total_variance = np.var(self.scaled_data, axis=0).sum()
        within_variance = self.kmeans.inertia_ / len(self.scaled_data) if self.kmeans else 0
        between_variance = total_variance - within_variance
        validation['variance_explained'] = between_variance / total_variance
        
        gmm = GaussianMixture(n_components=len(np.unique(labels)), random_state=random_state)
        gmm_labels = gmm.fit_predict(self.scaled_data)
        validation['kmeans_vs_gmm_ari'] = adjusted_rand_score(labels, gmm_labels)
        
        if linkage_matrix is not None:
            k = len(np.unique(labels))
            hier_labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
            
            best_ari = -1
            for perm in permutations(range(k)):
                mapping = {i: perm[i] for i in range(k)}
                hier_relabeled = np.array([mapping[label] for label in hier_labels])
                ari = adjusted_rand_score(labels, hier_relabeled)
                if ari > best_ari:
                    best_ari = ari
            
            validation['kmeans_vs_hierarchical_ari'] = best_ari
        
        return validation
    
    def run_pca(self, n_components=2):
        """Run PCA for visualization."""
        self.pca = PCA(n_components=n_components)
        coords = self.pca.fit_transform(self.scaled_data)
        
        return {
            'coords': coords,
            'variance_explained': self.pca.explained_variance_ratio_,
            'total_variance': self.pca.explained_variance_ratio_.sum()
        }
    
    def print_k_comparison(self, k_range=range(2, 9), random_state=42):
        """Display k-selection metrics in clean text format"""
        results = self.optimize_k(k_range)
        
        # Calculate GMM ARI for each k
        ari_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_labels = km.fit_predict(self.scaled_data)
            
            gmm = GaussianMixture(n_components=k, random_state=random_state)
            gmm_labels = gmm.fit_predict(self.scaled_data)
            
            ari = adjusted_rand_score(km_labels, gmm_labels)
            ari_scores.append(ari)
        
        results['gmm_ari'] = ari_scores
        
        # Identify optimal k
        best_sil = results.loc[results['silhouette'].idxmax(), 'k']
        best_ch = results.loc[results['calinski_harabasz'].idxmax(), 'k']
        best_db = results.loc[results['davies_bouldin'].idxmin(), 'k']
        best_ari = results.loc[results['gmm_ari'].idxmax(), 'k']
        
        print("\n" + "="*90)
        print("K-SELECTION: TACTICAL IDENTITY CONVERGENCE")
        print("="*90)
        
        print(f"{'k':<4} {'Inertia':<10} {'Silhouette':<12} {'Calinski':<12} {'DB Index':<12} {'GMM ARI':<12}")
        print(f"{'':4} {'↓':<10} {'↑':<12} {'↑':<12} {'↓':<12} {'↑':<12}")
        print("-"*90)
        
        for _, row in results.iterrows():
            k = int(row['k'])
            
            sil_str = f"{row['silhouette']:.3f}" + (" *" if k == best_sil else "  ")
            ch_str = f"{row['calinski_harabasz']:.2f}" + (" *" if k == best_ch else "  ")
            db_str = f"{row['davies_bouldin']:.3f}" + (" *" if k == best_db else "  ")
            ari_str = f"{row['gmm_ari']:.3f}" + (" *" if k == best_ari else "  ")
            
            print(f"{k:<4} {row['inertia']:<10.0f} {sil_str:<12} {ch_str:<12} {db_str:<12} {ari_str:<12}")
        
        print("="*90)
        
        return results


    def render_tactical_dna(self, clustering_results):
        """Render 8 tactical dimensions in a clean text table."""
        centers = clustering_results['centers']
        
        print("\n" + "="*110)
        print("TACTICAL DNA: 8-DIMENSIONAL CLUSTER CENTERS")
        print("="*110)
        
        print(f"{'Cluster':<10} {'D1:Press':<10} {'D2:Terr':<10} {'D3:Ctrl':<10} {'D4:Eff':<10} "
            f"{'D5:Pos':<10} {'D6:Threat':<10} {'D7:Style':<10} {'D8:Build':<10} {'Size':<6}")
        print("-"*110)
        
        for _, row in centers.iterrows():
            cluster_id = f"Cluster {int(row['cluster'])}"
            
            print(f"{cluster_id:<10} "
                f"{row['pressing_intensity']:<10.3f} "
                f"{row['territorial_dominance']:<10.2f} "
                f"{row['ball_control']:<10.2f} "
                f"{row['possession_efficiency']:<10.4f} "
                f"{row['defensive_positioning']:<10.3f} "
                f"{row['attacking_threat']:<10.3f} "
                f"{row['progression_style']:<10.3f} "
                f"{row['buildup_quality']:<10.4f} "
                f"n={int(row['size']):<4}")
        
        print("="*110)

    def print_archetype_summary(self, characterization, total_teams=None):
        """Display archetype characterization as styled HTML cards"""
        if total_teams is None:
            total_teams = sum(c['size'] for c in characterization.values())
        
        colors = {0: '#4895C4', 1: '#A23B72', 2: '#F18F01', 3: '#06A77D'}

        cards_html = ""
        for cluster_id, info in characterization.items():
            color = colors.get(cluster_id, '#666')
            pct = info['size'] / total_teams * 100
            
            high_rows = ""
            for dim, dev in info['high_characteristics']:
                label = dim.replace('_', ' ').title()
                high_rows += f'<tr><td class="arrow up">↑</td><td class="dim">{label}</td><td class="dev">{dev:+.2f} std</td></tr>'
            
            low_rows = ""
            for dim, dev in info['low_characteristics']:
                if dev < 0:
                    label = dim.replace('_', ' ').title()
                    low_rows += f'<tr><td class="arrow down">↓</td><td class="dim">{label}</td><td class="dev">{dev:+.2f} std</td></tr>'
            
            teams = ", ".join(info['representative_teams'][:5])
            
            cards_html += f"""
            <div class="arc-card">
                <div class="arc-header" style="background:{color};">
                    <span class="arc-name">{info['name']}</span>
                    <span class="arc-count">n={info['size']} ({pct:.1f}%)</span>
                </div>
                <div class="arc-body">
                    <div class="arc-section">
                        <div class="arc-label">Defining Traits</div>
                        <table class="arc-traits">{high_rows}</table>
                    </div>
                    <div class="arc-section">
                        <div class="arc-label">Trade-offs</div>
                        <table class="arc-traits">{low_rows}</table>
                    </div>
                    <div class="arc-section">
                        <div class="arc-label">Representative Teams</div>
                        <div class="arc-teams">{teams}</div>
                    </div>
                </div>
            </div>
            """
        
        html = f"""
        <style>
            .arc-wrap {{ display:flex; gap:16px; flex-wrap:wrap; }}
            .arc-card {{ flex:1; min-width:220px; max-width:340px; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden; font-family:-apple-system,sans-serif; }}
            .arc-header {{ padding:10px 14px; display:flex; justify-content:space-between; align-items:center; }}
            .arc-name {{ color:white; font-weight:700; font-size:13px; }}
            .arc-count {{ color:rgba(255,255,255,0.8); font-size:11px; font-family:'SF Mono',monospace; }}
            .arc-body {{ padding:12px 14px; }}
            .arc-section {{ margin-bottom:10px; }}
            .arc-label {{ font-size:10px; font-weight:700; color:#999; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }}
            .arc-traits {{ border-collapse:collapse; font-size:12px; }}
            .arc-traits td {{ padding:2px 6px 2px 0; }}
            .arc-traits .arrow {{ font-size:13px; font-weight:700; }}
            .arc-traits .up {{ color:#2d6a4f; }}
            .arc-traits .down {{ color:#d62828; }}
            .arc-traits .dim {{ color:#1a1a2e; font-weight:500; }}
            .arc-traits .dev {{ font-family:'SF Mono',monospace; font-size:11px; color:#555; }}
            .arc-teams {{ font-size:11px; color:#555; line-height:1.5; }}
        </style>
        <h4 style="font-family:-apple-system,sans-serif; color:#1a1a2e; margin-bottom:12px;">Archetype Profiles</h4>
        <div class="arc-wrap">
            {cards_html}
        </div>
        """
        
        display(HTML(html))

    def characterize_archetypes(self, profiles_df, cluster_centers, labels, archetype_names=None):
        """Generate detailed archetype characterization."""
        profiles_df = profiles_df.copy()
        profiles_df['cluster'] = labels
        
        characterization = {}
        k = len(np.unique(labels))
        
        for cluster_id in range(k):
            cluster_teams = profiles_df[profiles_df['cluster'] == cluster_id].copy()
            center = cluster_centers.iloc[cluster_id]
            
            # Calculate deviations from global mean
            deviations = {}
            for dim in self.dimensions:
                global_mean = profiles_df[dim].mean()
                global_std = profiles_df[dim].std()
                cluster_mean = center[dim]
                deviation = (cluster_mean - global_mean) / global_std
                deviations[dim] = deviation
            
            # Top characteristics
            top_high = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:3]
            top_low = sorted(deviations.items(), key=lambda x: x[1])[:3]
            
            # Representative teams (closest to center)
            distances = []
            for idx, row in cluster_teams.iterrows():
                team_vector = row[self.dimensions].values.astype(float)
                center_vector = center[self.dimensions].values.astype(float)
                dist = np.sqrt(((team_vector - center_vector)**2).sum())
                distances.append(dist)
            
            cluster_teams['dist_to_center'] = distances
            representative = cluster_teams.nsmallest(5, 'dist_to_center')['team'].tolist()
            
            characterization[cluster_id] = {
                'name': archetype_names[cluster_id] if archetype_names else f'Cluster {cluster_id}',
                'size': len(cluster_teams),
                'high_characteristics': top_high,
                'low_characteristics': top_low,
                'representative_teams': representative,
                'all_teams': cluster_teams['team'].tolist(),
                'center': center[self.dimensions].to_dict()
            }
        
        return characterization
    