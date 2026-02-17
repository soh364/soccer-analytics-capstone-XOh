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
    
    def validate_clusters(self, labels, linkage_matrix=None):
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
        gmm = GaussianMixture(n_components=len(np.unique(labels)), random_state=42)
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
    
    def print_validation_summary(self, validation):
        """Print formatted validation results"""
        print("="*70)
        print("CLUSTER VALIDATION METRICS")
        print("="*70)
        
        print(f"\n1. Silhouette Score: {validation['silhouette_avg']:.3f}")
        print("   Range: -1 (worst) to +1 (best)")
        if validation['silhouette_avg'] > 0.5:
            print("   → Reasonable structure")
        elif validation['silhouette_avg'] > 0.25:
            print("   → Weak but acceptable structure")
        else:
            print("   → No substantial structure")
        
        print(f"\n2. Calinski-Harabasz Index: {validation['calinski_harabasz']:.2f}")
        print("   Higher = better-defined clusters")
        
        print(f"\n3. Davies-Bouldin Index: {validation['davies_bouldin']:.3f}")
        print("   Lower = better separation (0 is perfect)")
        
        if validation['inertia']:
            print(f"\n4. Inertia: {validation['inertia']:.2f}")
        
        print(f"\n5. Variance Explained: {validation['variance_explained']:.1%}")
        
        print("\n6. Silhouette by Cluster:")
        for cluster_id, sil in validation['silhouette_by_cluster'].items():
            print(f"   Cluster {cluster_id}: {sil:.3f}")
        
        print(f"\n7. K-means vs GMM Agreement (ARI): {validation['kmeans_vs_gmm_ari']:.3f}")
        
        if 'kmeans_vs_hierarchical_ari' in validation:
            print(f"\n8. K-means vs Hierarchical Agreement (ARI): {validation['kmeans_vs_hierarchical_ari']:.3f}")
            if validation['kmeans_vs_hierarchical_ari'] < 0.3:
                print("   → Low agreement reflects continuous tactical space")