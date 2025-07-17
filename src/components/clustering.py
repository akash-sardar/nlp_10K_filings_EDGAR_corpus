from src.entity.artifact_entity import EmbeddingArtifact

import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self, embedding_artifact: EmbeddingArtifact) -> None:
        """
        Performs PCA, T-SNE and K-Means Clustering along with Visualization
        
        Args:
            embedding_artifact: EmbeddingArtifact containing Pandas DataFrame with 
                             embedded vectors as rows
        """
        try:
            self.embedding_artifact = embedding_artifact
            self.results_path = Path("Clustering_Results")          
        except Exception as e:
            raise(e)
        

    def scale_embeddings(self, series: pd.Series):
        """
        Standard scales embeddings using StandardScaler
        
        Args:
            series: Pandas Series containing list of embedding vectors
            
        Returns:
            numpy array with scaled embeddings (mean=0, std=1)
            
        """
        try:
            # Convert list of embeddings to numpy array
            embeddings = np.array(series.tolist())
            # Initialize and fit StandardScaler
            scaler = StandardScaler()
            scaler.fit(embeddings)            
            return scaler.transform(embeddings)
        except Exception as e:
            raise(e)


    def perform_pca(self, arr: np.array, n_components = 75):
        """
        Performs Principal Component Analysis for dimensionality reduction
        
        Args:
            arr: Input array of scaled embeddings
            n_components: Number of principal components to retain (default: 106)
            
        Returns:
            PCA-transformed array with reduced dimensions
        """
        try:
            pca = PCA(n_components)
            pca.fit(arr)
            return pca.transform(arr)
        except Exception as e:
            raise(e)


    def visualize_using_tsne(self, arr:np.array):
        """
        Applies t-SNE for 2D visualization of high-dimensional embeddings
        
        Args:
            arr: Input array of embeddings
            
        Returns:
            2D t-SNE transformed array for visualization
        """
        try:
            tsne = TSNE(random_state=42)
            tsne_data = tsne.fit_transform(arr)
            return tsne_data
        except Exception as e:
            raise(e)
    

    def perform_clustering(self):
        """
        Main method that performs the complete clustering pipeline:
        1. Scale embeddings
        2. Apply t-SNE for visualization
        3. Apply PCA for dimensionality reduction
        4. Perform K-means clustering
        5. Identify outliers using distance threshold
        6. Generate three visualization plots
        7. Create summary statistics
        """
        try:
            df_chunked_pandas = self.embedding_artifact.df
            
            # Step 1: Scale embeddings using StandardScaler
            print(f"Dimension of embeddings before scaling: {df_chunked_pandas['embeddings'].shape}")
            scaled_embeddings = self.scale_embeddings(df_chunked_pandas["embeddings"])
            print(f"Dimension of embeddings after scaling: {scaled_embeddings.shape}")

            # Step 2: Apply t-SNE for 2D visualization
            tsne_data = self.visualize_using_tsne(scaled_embeddings)
            print(f"Dimension of tsne-data : {tsne_data.shape}")
            df_chunked_pandas[["tsne_x", "tsne_y"]] = pd.DataFrame(tsne_data, columns=["tsne_x", "tsne_y"])
            
            # Step 3: Apply PCA for dimensionality reduction before clustering
            pca_reduced_embeddings = self.perform_pca(scaled_embeddings)
            print(f"Dimension of embeddings after PCA: {pca_reduced_embeddings.shape}")

            # Step 4: Perform K-means clustering on PCA-reduced embeddings
            clusterer = KMeans(20)
            preds = clusterer.fit_predict(pca_reduced_embeddings)

            df_chunked_pandas["cluster_group"] = preds

            # Step 5: Calculate distances to cluster centers for outlier detection
            cluster_centers = clusterer.cluster_centers_
            distances = np.min(cdist(pca_reduced_embeddings, cluster_centers, 'euclidean'), axis = 1)
            df_chunked_pandas["distance_to_centroid"] = distances

            # Using mean + 2 STD to calculate outliers (statistical threshold)
            threshold = distances.mean() + 2 * distances.std()
            df_chunked_pandas["outlier_flag"] = distances>threshold

            # Create outlier summary (grouped by company, section, and chunk)
            outlier_df = df_chunked_pandas[df_chunked_pandas["outlier_flag"]]
            outlier_summary = outlier_df.groupby(["cik", "section_name", "chunk_id"]).size().reset_index(name = "count")

            # Create results directory
            os.makedirs(self.results_path, exist_ok=True)

            # Visualization 1: Plot 2D-embeddings colored by Clusters
            plt.figure(figsize = (10,8))
            scatter = plt.scatter(df_chunked_pandas["tsne_x"],
                                df_chunked_pandas["tsne_y"],
                                c = df_chunked_pandas["cluster_group"],
                                cmap = "tab20", s = 15, alpha = 0.7)
            plt.colorbar(scatter)
            plt.title("2D Embeddings in Clusters")
            results_path = self.results_path / "Fig_02_2D_Embeddings_clusters.jpeg"
            os.makedirs(self.results_path, exist_ok=True)            
            plt.savefig(results_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Visualization 2: Plot outliers with company id and section labels
            plt.figure(figsize=(12, 10))

            # Separate normal and outlier data points
            normal = df_chunked_pandas[~df_chunked_pandas["outlier_flag"]]
            outliers = df_chunked_pandas[df_chunked_pandas["outlier_flag"]]

            # Plot normal points in light blue, outliers in red
            plt.scatter(normal["tsne_x"], normal["tsne_y"], c='lightblue', s=15, alpha=0.5, label='Normal')
            plt.scatter(outliers["tsne_x"], outliers["tsne_y"], c='red', s=50, alpha=0.8, label='Outlier')

            # Annotate outliers with CIK, section, and chunk ID
            for _, row in outliers.iterrows():
                x, y = row["tsne_x"], row["tsne_y"]
                label = f"{row['cik'][-4:]}-{row['section_name'][-2:]}-{str(row['chunk_id'])[-6:]}"
                plt.annotate(label, (x, y), fontsize=8, alpha=0.7)
            plt.legend()
            plt.title("Outliers with CIK, Section, and Chunk ID Labels")
            results_path = self.results_path / "Fig_03_2D_Embeddings_outliers.jpeg"
            os.makedirs(self.results_path, exist_ok=True)              
            plt.savefig(results_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Visualization 3: Section name with colors
            plt.figure(figsize=(12,10))
            # Use colorful palette for distinguishing sections
            palette_20 = cc.glasbey[:20]
            sns.scatterplot(
                x='tsne_x', y='tsne_y', 
                hue='section_name', 
                palette=palette_20,
                data=df_chunked_pandas,
                s=40,
                alpha=0.8,
                edgecolor='none'
            )

            plt.title("2D t-SNE Visualization Colored by Section Name")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend(title='Section Name', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            results_dir = self.results_path
            os.makedirs(self.results_path, exist_ok=True)  
            results_path = results_dir / "Fig_01_2D_Embeddings_with Section_name_colors.jpeg"            
            plt.savefig(results_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Generate summary statistics table
            summary_stats = pd.DataFrame({
            'Total Chunks': [len(df_chunked_pandas)],
            'Total Outliers': [df_chunked_pandas['outlier_flag'].sum()],
            'Outlier %': [df_chunked_pandas['outlier_flag'].mean() * 100],
            'Unique Companies': [df_chunked_pandas['cik'].nunique()],
            'Unique Sections': [df_chunked_pandas['section_name'].nunique()],            
            'Avg Chunk Length': [df_chunked_pandas['chunk'].str.len().mean()],
            'K-means Clusters': [df_chunked_pandas['cluster_group'].nunique()],
            })

            # Analyze outliers by company (top companies with most outliers)
            outliers_by_company = df_chunked_pandas[df_chunked_pandas['outlier_flag']].groupby('cik').size()
            outliers_by_company = outliers_by_company.sort_values(ascending=False)

            # Analyze outliers by section (top sections with most outliers)
            outliers_by_section = df_chunked_pandas[df_chunked_pandas['outlier_flag']].groupby('section_name').size()
            outliers_by_section = outliers_by_section.sort_values(ascending=False)

            # Display results to console
            print("Summary Statistics:")
            print(summary_stats.T)
            print("\nTop 5 Companies with Most Outliers:")
            print(outliers_by_company.head())
            print("\nTop 5 Sections with Most Outliers:")
            print(outliers_by_section.head())

            # Save summary statistics to CSV file
            results_dir = self.results_path
            results_path = results_dir / "summary_statistics.csv"              
            summary_stats.to_csv(results_path, index=False)

        except Exception as e:
            raise(e)