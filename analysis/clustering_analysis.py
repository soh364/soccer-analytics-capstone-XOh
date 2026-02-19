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
        """
        Initialize with dimension names.
        
        Args:
            dimensions: List of dimension column names
        """
        self.dimensions = dimensions
        self.scaler = StandardScaler()
        self.kmeans = None
        self.scaled_data = None
        self.pca = None
        
    def prepare_data(self, profiles_df):
        """
        Standardize tactical dimensions.
        
        Args:
            profiles_df: DataFrame with team profiles
            
        Returns:
            scaled_data: Standardized numpy array
        """
        tactical_data = profiles_df[self.dimensions]
        self.scaled_data = self.scaler.fit_transform(tactical_data)
        return self.scaled_data
    
    def run_kmeans(self, k=3, random_state=42, n_init=100):
        """
        Run k-means clustering.
        
        Args:
            k: Number of clusters
            random_state: Random seed
            n_init: Number of initializations
            
        Returns:
            dict with labels, centers, inertia
        """
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = self.kmeans.fit_predict(self.scaled_data)
        
        # Transform centers back to original scale
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
        """
        Find optimal k using elbow and silhouette methods.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            DataFrame with metrics for each k
        """
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
        """
        Calculate comprehensive validation metrics.
        
        Args:
            labels: Cluster labels
            linkage_matrix: Optional hierarchical clustering linkage matrix
            
        Returns:
            dict with all validation metrics
        """
        validation = {}
        
        # Basic metrics
        validation['silhouette_avg'] = silhouette_score(self.scaled_data, labels)
        validation['calinski_harabasz'] = calinski_harabasz_score(self.scaled_data, labels)
        validation['davies_bouldin'] = davies_bouldin_score(self.scaled_data, labels)
        validation['inertia'] = self.kmeans.inertia_ if self.kmeans else None
        
        # Per-cluster silhouette
        silhouette_vals = silhouette_samples(self.scaled_data, labels)
        validation['silhouette_by_cluster'] = {}
        for cluster_id in np.unique(labels):
            cluster_sil = silhouette_vals[labels == cluster_id].mean()
            validation['silhouette_by_cluster'][cluster_id] = cluster_sil
        
        # Variance explained
        total_variance = np.var(self.scaled_data, axis=0).sum()
        within_variance = self.kmeans.inertia_ / len(self.scaled_data) if self.kmeans else 0
        between_variance = total_variance - within_variance
        validation['variance_explained'] = between_variance / total_variance
        
        # GMM comparison
        gmm = GaussianMixture(n_components=len(np.unique(labels)), random_state=random_state)
        gmm_labels = gmm.fit_predict(self.scaled_data)
        validation['kmeans_vs_gmm_ari'] = adjusted_rand_score(labels, gmm_labels)
        
        # Hierarchical comparison if provided
        if linkage_matrix is not None:
            k = len(np.unique(labels))
            hier_labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
            
            # Test all label permutations
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
        """
        Run PCA for visualization.
        
        Args:
            n_components: Number of components
            
        Returns:
            dict with coords and variance explained
        """
        self.pca = PCA(n_components=n_components)
        coords = self.pca.fit_transform(self.scaled_data)
        
        return {
            'coords': coords,
            'variance_explained': self.pca.explained_variance_ratio_,
            'total_variance': self.pca.explained_variance_ratio_.sum()
        }
    
    # In clustering_analysis.py, replace the characterize_archetypes method:

    def characterize_archetypes(self, profiles_df, cluster_centers, labels, archetype_names=None):
        """
        Generate detailed archetype characterization.
        
        Args:
            profiles_df: Original profiles DataFrame (pandas)
            cluster_centers: Cluster centers DataFrame (pandas)
            labels: Cluster labels
            archetype_names: Optional dict mapping cluster_id to name
            
        Returns:
            dict with characterization for each cluster
        """
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
            
            # Representative teams (closest to center) - FIX HERE
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
    
    def print_k_comparison(self, k_range=range(2, 9), random_state=42):
        """Display k-selection metrics with a dedicated GMM Agreement column"""
        from IPython.display import display, HTML
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans
        import pandas as pd
        
        # 1. Get the standard metrics (Inertia, Silhouette, etc.)
        results = self.optimize_k(k_range)
        
        # 2. Calculate GMM ARI for each k
        ari_scores = []
        for k in k_range:
            # FIX: changed n_components to n_clusters
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_labels = km.fit_predict(self.scaled_data)
            
            # GMM still uses n_components
            gmm = GaussianMixture(n_components=k, random_state=random_state)
            gmm_labels = gmm.fit_predict(self.scaled_data)
            
            # Measure agreement
            ari = adjusted_rand_score(km_labels, gmm_labels)
            ari_scores.append(ari)
        
        results['gmm_ari'] = ari_scores
        
        # 3. Identify optimal k for highlighting
        best_sil = results.loc[results['silhouette'].idxmax(), 'k']
        best_ch = results.loc[results['calinski_harabasz'].idxmax(), 'k']
        best_db = results.loc[results['davies_bouldin'].idxmin(), 'k']
        best_ari = results.loc[results['gmm_ari'].idxmax(), 'k']
        
        # 4. Generate Table Rows
        rows_html = ""
        for _, row in results.iterrows():
            k = int(row['k'])
            
            def cell(val, fmt, is_best, color="#2d6a4f"):
                bg = "background:#e8f5e9;" if is_best else ""
                icon = " ✓" if is_best else ""
                return f'<td class="value" style="{bg} color:{color};">{val:{fmt}}{icon}</td>'
            
            rows_html += f"""<tr>
                <td class="metric">{k}</td>
                <td class="value">{row['inertia']:.0f}</td>
                {cell(row['silhouette'], '.3f', k == best_sil)}
                {cell(row['calinski_harabasz'], '.2f', k == best_ch)}
                {cell(row['davies_bouldin'], '.3f', k == best_db)}
                {cell(row['gmm_ari'], '.3f', k == best_ari, color="#1a1a2e")}
            </tr>"""
        
        # 5. Build Final HTML
        html = f"""
        <style>
            .kt-container {{
                max-width: 750px; /* Limits width so it doesn't stretch across the screen */
                margin: 10px 0;   /* Aligns to the left with small vertical spacing */
            }}
            .kt {{ 
                border-collapse:collapse; 
                font-family:-apple-system,sans-serif; 
                font-size:13px; 
                width: 100%; /* Table fills the 750px container */
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .kt th {{ background:#1a1a2e; color:white; padding:10px 14px; text-align:left; font-weight:600; }}
            .kt td {{ padding:8px 14px; border-bottom:1px solid #e0e0e0; }}
            .kt .metric {{ font-weight:700; color:#1a1a2e; text-align:center; background:#f8f9fa; }}
            .kt .value {{ font-family:'SF Mono',monospace; font-weight:600; }}
            .kt .dir {{ font-size:10px; color:#999; font-weight:400; display:block; margin-top:2px; }}
        </style>
        <div class="kt-container">
            <h4 style="font-family:-apple-system,sans-serif; color:#1a1a2e; margin-bottom:10px;">
                K-Selection: Tactical Identity Convergence (k={int(k_range[0])}–{int(k_range[-1])})
            </h4>
            <table class="kt">
                <tr>
                    <th>k</th>
                    <th>Inertia <span class="dir">↓ (Elbow)</span></th>
                    <th>Silhouette <span class="dir">↑ (Separation)</span></th>
                    <th>Calinski <span class="dir">↑ (Variance)</span></th>
                    <th>DB Index <span class="dir">↓ (Similarity)</span></th>
                    <th style="background:#2d3436;">GMM ARI <span class="dir" style="color:#dfe6e9;">↑ (Agreement)</span></th>
                </tr>
                {rows_html}
            </table>
        </div>
        """
        display(HTML(html))
        return results
        
    def print_validation_summary(self, validation):
            """Print formatted validation results as a clean table"""
            from IPython.display import display, HTML
            
            # Main metrics table
            rows = [
                ("Silhouette Score", f"{validation['silhouette_avg']:.3f}", "-1 to +1 (higher = better)"),
                ("Calinski-Harabasz Index", f"{validation['calinski_harabasz']:.2f}", "Higher = better-defined"),
                ("Davies-Bouldin Index", f"{validation['davies_bouldin']:.3f}", "Lower = better (0 is perfect)"),
                ("Variance Explained", f"{validation['variance_explained']:.1%}", "By k clusters"),
                ("K-means vs GMM Agreement", f"{validation['kmeans_vs_gmm_ari']:.3f}", "ARI: 1.0 = perfect agreement"),
            ]
            
            if validation.get('inertia'):
                rows.insert(3, ("Inertia", f"{validation['inertia']:.2f}", "Within-cluster sum of squares"))
            
            if validation.get('kmeans_vs_hierarchical_ari'):
                rows.append(("K-means vs Hierarchical", f"{validation['kmeans_vs_hierarchical_ari']:.3f}", "ARI: 1.0 = perfect agreement"))
            
            html = """
            <style>
                .val-table { border-collapse: collapse; font-family: -apple-system, sans-serif; font-size: 13px; width: 100%; }
                .val-table th { background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; font-weight: 600; }
                .val-table td { padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }
                .val-table tr:hover { background: #f5f5f5; }
                .val-table .metric { font-weight: 600; color: #1a1a2e; }
                .val-table .value { font-family: 'SF Mono', monospace; font-size: 14px; font-weight: 700; color: #2d6a4f; }
                .val-table .note { color: #666; font-size: 11px; }
                .sil-table { border-collapse: collapse; font-family: -apple-system, sans-serif; font-size: 13px; margin-top: 12px; }
                .sil-table th { background: #1a1a2e; color: white; padding: 8px 14px; text-align: left; }
                .sil-table td { padding: 6px 14px; border-bottom: 1px solid #e0e0e0; font-family: 'SF Mono', monospace; }
            </style>
            <h4 style="font-family: -apple-system, sans-serif; color: #1a1a2e; margin-bottom: 8px;">Cluster Validation Summary</h4>
            <table class="val-table">
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            """
            
            for metric, value, note in rows:
                html += f'<tr><td class="metric">{metric}</td><td class="value">{value}</td><td class="note">{note}</td></tr>'
            
            html += "</table>"
            
            # Silhouette by cluster
            html += """
            <h4 style="font-family: -apple-system, sans-serif; color: #1a1a2e; margin-top: 16px; margin-bottom: 8px;">Silhouette Score by Cluster</h4>
            <table class="sil-table">
                <tr><th>Cluster</th><th>Silhouette</th><th>Cohesion</th></tr>
            """
            
            for cluster_id, sil in validation['silhouette_by_cluster'].items():
                bar_width = int(sil * 300)
                color = '#2d6a4f' if sil > 0.25 else '#e76f51' if sil > 0.15 else '#d62828'
                bar = f'<div style="background:{color}; width:{bar_width}px; height:14px; border-radius:3px; display:inline-block;"></div>'
                html += f'<tr><td style="font-weight:600;">Cluster {cluster_id}</td><td>{sil:.3f}</td><td>{bar}</td></tr>'
            
            html += "</table>"
            
            display(HTML(html))

        

    def print_archetype_summary(self, characterization, total_teams=None):
            """Display archetype characterization as styled HTML cards"""
            from IPython.display import display, HTML
            
            if total_teams is None:
                total_teams = sum(c['size'] for c in characterization.values())
            # Archetype colors — use these consistently across all visuals
            colors = {0: '#4895C4', 1: '#A23B72', 2: '#F18F01'}

            cards_html = ""
            for cluster_id, info in characterization.items():
                color = colors.get(cluster_id, '#666')
                pct = info['size'] / total_teams * 100
                
                # Defining traits (top 3 high)
                high_rows = ""
                for dim, dev in info['high_characteristics']:
                    label = dim.replace('_', ' ').title()
                    high_rows += f'<tr><td class="arrow up">↑</td><td class="dim">{label}</td><td class="dev">{dev:+.2f} std</td></tr>'
                
                # Trade-offs (top 3 low)
                low_rows = ""
                for dim, dev in info['low_characteristics']:
                    if dev < 0:
                        label = dim.replace('_', ' ').title()
                        low_rows += f'<tr><td class="arrow down">↓</td><td class="dim">{label}</td><td class="dev">{dev:+.2f} std</td></tr>'
                
                # Representative teams
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