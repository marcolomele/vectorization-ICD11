import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import umap
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def get_chapter_from_code(code):
    """Extract chapter number from ICD11 code."""
    code_to_chapter = {'0' : '0',
        '1' : '1', '2' : '2', '3' : '3', '4' : '4',
        '5' : '5', '6' : '6', '7' : '7', '8' : '8',
        '9' : '9', 'A' : '10', 'B' : '11', 'C' : '12',
        'D' : '13', 'E' : '14', 'F' : '15', 'G' : '16',
        'H' : '17', 'J' : '18', 'K' : '19', 'L' : '20',
        'M' : '21', 'N' : '22', 'P' : '23', 'Q' : '24',
        'R' : '25', 'S' : '26', 
        '01' : '1', '02' : '2', '03' : '3', '04' : '4',
        '05' : '5', '06' : '6', '07' : '7', '08' : '8',
        '09' : '9', '10' : '10', '11' : '11', '12' : '12',
        '13' : '13', '14' : '14', '15' : '15', '16' : '16',
        '17' : '17', '18' : '18', '19' : '19', '20' : '20',
        '21' : '21', '22' : '22', '23' : '23', '24' : '24',
        '25' : '25', '26' : '26'
    }
    
    if len(code) >= 2 and code[:2].isdigit():
        chapter_code = code[:2]
    else:
        chapter_code = code[:1]
        
    return code_to_chapter.get(chapter_code, None)

def load_and_process_embeddings(folder_path="/Users/marco/Documents/python_projects/vector-database-ICD/Embeddings/Resulting-embeddings"):
    """
    Function 1: Load CSV files, convert embeddings to numpy arrays, and extract chapter numbers.
    
    Returns:
        dict: Dictionary with model names as keys and dictionaries containing 'embeddings', 'codes', and 'chapters' as values
    """
    embeddings_data = {}
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # Extract model name from filename
        model_name = csv_file.replace('_ICD11_embeddings.csv', '')
        
        # Read CSV file
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Convert embeddings to numpy arrays
        embeddings = []
        codes = []
        chapters = []
        
        for idx, row in df.iterrows():
            try:
                # Convert string representation of vector to numpy array
                embedding_str = row['Vector'].strip('[]')
                embedding_array = np.array([float(x.strip()) for x in embedding_str.split(',')], dtype=np.float64)
                
                # Get chapter from code
                chapter = get_chapter_from_code(row['ICD11_code'])
                
                if chapter is not None:  # Only include if we can determine the chapter
                    embeddings.append(embedding_array)
                    codes.append(row['ICD11_code'])
                    chapters.append(chapter)
                    
            except Exception as e:
                print(f"Error processing row {idx} in {csv_file}: {e}")
                continue
        
        # Store processed data
        embeddings_data[model_name] = {
            'embeddings': np.array(embeddings),
            'codes': codes,
            'chapters': chapters
        }
        
        print(f"Loaded {len(embeddings)} embeddings for {model_name}")
    
    return embeddings_data

def create_tsne_visualization(embeddings_data, save_plots=True):
    """
    Function 2: Create t-SNE visualization of embeddings grouped by chapter.
    
    Args:
        embeddings_data: Output from load_and_process_embeddings function
        save_plots: Whether to save the plots to files
    """
    colors = ['#FF6F61', '#D2B48C', '#AE65D6', '#598A4F', '#4A90E2', '#F39C12', 
              '#E74C3C', '#9B59B6', '#1ABC9C', '#34495E', '#F1C40F', '#E67E22',
              '#95A5A6', '#3498DB', '#2ECC71', '#E91E63', '#FF9800', '#607D8B',
              '#8BC34A', '#FFC107', '#673AB7', '#009688', '#FF5722', '#795548',
              '#9E9E9E', '#CDDC39', '#FFEB3B']
    
    for model_name, data in embeddings_data.items():
        print(f"Creating t-SNE visualization for {model_name}...")
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = tsne.fit_transform(data['embeddings'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Get unique chapters and assign colors
        unique_chapters = sorted(list(set(data['chapters'])))
        chapter_colors = {chapter: colors[i % len(colors)] for i, chapter in enumerate(unique_chapters)}
        
        # Plot each chapter
        for chapter in unique_chapters:
            chapter_mask = np.array(data['chapters']) == chapter
            chapter_embeddings = embeddings_2d[chapter_mask]
            
            plt.scatter(chapter_embeddings[:, 0], chapter_embeddings[:, 1], 
                       c=chapter_colors[chapter], label=f'Chapter {chapter}', 
                       alpha=0.6, s=20)
        
        plt.title(f't-SNE Visualization of {model_name.upper()} Embeddings by ICD11 Chapter', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{model_name}_tsne_visualization.png', dpi=300, bbox_inches='tight')
            print(f"Saved t-SNE plot for {model_name}")
        
        plt.show()

def create_umap_visualization(embeddings_data, save_plots=True):
    """
    Function 3: Create UMAP visualization of embeddings grouped by chapter.
    
    Args:
        embeddings_data: Output from load_and_process_embeddings function
        save_plots: Whether to save the plots to files
    """
    colors = ['#FF6F61', '#D2B48C', '#AE65D6', '#598A4F', '#4A90E2', '#F39C12', 
              '#E74C3C', '#9B59B6', '#1ABC9C', '#34495E', '#F1C40F', '#E67E22',
              '#95A5A6', '#3498DB', '#2ECC71', '#E91E63', '#FF9800', '#607D8B',
              '#8BC34A', '#FFC107', '#673AB7', '#009688', '#FF5722', '#795548',
              '#9E9E9E', '#CDDC39', '#FFEB3B']
    
    for model_name, data in embeddings_data.items():
        print(f"Creating UMAP visualization for {model_name}...")
        
        # Perform UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(data['embeddings'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Get unique chapters and assign colors
        unique_chapters = sorted(list(set(data['chapters'])))
        chapter_colors = {chapter: colors[i % len(colors)] for i, chapter in enumerate(unique_chapters)}
        
        # Plot each chapter
        for chapter in unique_chapters:
            chapter_mask = np.array(data['chapters']) == chapter
            chapter_embeddings = embeddings_2d[chapter_mask]
            
            plt.scatter(chapter_embeddings[:, 0], chapter_embeddings[:, 1], 
                       c=chapter_colors[chapter], label=f'Chapter {chapter}', 
                       alpha=0.6, s=20)
        
        plt.title(f'UMAP Visualization of {model_name.upper()} Embeddings by ICD11 Chapter', fontsize=14)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{model_name}_umap_visualization.png', dpi=300, bbox_inches='tight')
            print(f"Saved UMAP plot for {model_name}")
        
        plt.show()

def calculate_trustworthiness_continuity(X_high, X_low, k=10):
    """
    Calculate trustworthiness and continuity metrics for dimensionality reduction quality.
    
    Args:
        X_high: High-dimensional embeddings
        X_low: Low-dimensional embeddings (t-SNE/UMAP)
        k: Number of nearest neighbors to consider
    
    Returns:
        tuple: (trustworthiness, continuity)
    """
    n_samples = X_high.shape[0]
    
    # Calculate k-nearest neighbors in both spaces
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    
    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    
    # Remove self from neighbors
    indices_high = indices_high[:, 1:]
    indices_low = indices_low[:, 1:]
    
    # Calculate trustworthiness
    trustworthiness = 0
    for i in range(n_samples):
        neighbors_low = set(indices_low[i])
        neighbors_high = set(indices_high[i])
        
        # Points that are neighbors in low-dim but not in high-dim
        false_neighbors = neighbors_low - neighbors_high
        
        for j in false_neighbors:
            # Rank of j in high-dimensional space for point i
            rank_high = np.where(indices_high[i] == j)[0]
            if len(rank_high) == 0:
                # j is not in k-NN of i in high-dim, so rank > k
                rank_high = k + np.where(np.argsort(np.linalg.norm(X_high - X_high[i], axis=1))[k+1:] == j)[0]
                if len(rank_high) > 0:
                    rank_high = k + rank_high[0] + 1
                else:
                    rank_high = n_samples
            else:
                rank_high = rank_high[0] + 1
            
            trustworthiness += max(0, rank_high - k)
    
    trustworthiness = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * trustworthiness
    
    # Calculate continuity
    continuity = 0
    for i in range(n_samples):
        neighbors_high = set(indices_high[i])
        neighbors_low = set(indices_low[i])
        
        # Points that are neighbors in high-dim but not in low-dim
        missing_neighbors = neighbors_high - neighbors_low
        
        for j in missing_neighbors:
            # Rank of j in low-dimensional space for point i
            rank_low = np.where(indices_low[i] == j)[0]
            if len(rank_low) == 0:
                # j is not in k-NN of i in low-dim, so rank > k
                rank_low = k + np.where(np.argsort(np.linalg.norm(X_low - X_low[i], axis=1))[k+1:] == j)[0]
                if len(rank_low) > 0:
                    rank_low = k + rank_low[0] + 1
                else:
                    rank_low = n_samples
            else:
                rank_low = rank_low[0] + 1
            
            continuity += max(0, rank_low - k)
    
    continuity = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * continuity
    
    return trustworthiness, continuity

def evaluate_embedding_quality(embeddings_data, save_results=True):
    """
    Function 5: Evaluate embedding quality using multiple clustering metrics.
    
    Args:
        embeddings_data: Output from load_and_process_embeddings function
        save_results: Whether to save results to CSV files
    
    Returns:
        dict: Comprehensive quality metrics for each model
    """
    print("Evaluating embedding quality...")
    
    quality_metrics = {}
    all_results = []
    
    for model_name, data in embeddings_data.items():
        print(f"Evaluating {model_name}...")
        
        embeddings = data['embeddings']
        chapters = np.array([int(ch) for ch in data['chapters']])
        
        # Skip if not enough samples or chapters
        if len(embeddings) < 10 or len(np.unique(chapters)) < 2:
            print(f"Skipping {model_name}: insufficient data")
            continue
        
        metrics = {}
        
        # 1. Silhouette Score
        try:
            silhouette = silhouette_score(embeddings, chapters)
            metrics['silhouette_score'] = silhouette
        except Exception as e:
            print(f"Error calculating silhouette score for {model_name}: {e}")
            metrics['silhouette_score'] = np.nan
        
        # 2. Calinski-Harabasz Index
        try:
            calinski = calinski_harabasz_score(embeddings, chapters)
            metrics['calinski_harabasz_score'] = calinski
        except Exception as e:
            print(f"Error calculating Calinski-Harabasz score for {model_name}: {e}")
            metrics['calinski_harabasz_score'] = np.nan
        
        # 3. Davies-Bouldin Index
        try:
            davies_bouldin = davies_bouldin_score(embeddings, chapters)
            metrics['davies_bouldin_score'] = davies_bouldin
        except Exception as e:
            print(f"Error calculating Davies-Bouldin score for {model_name}: {e}")
            metrics['davies_bouldin_score'] = np.nan
        
        # 4. Intra-cluster and Inter-cluster distances
        try:
            unique_chapters = np.unique(chapters)
            intra_distances = []
            inter_distances = []
            
            for chapter in unique_chapters:
                chapter_mask = chapters == chapter
                chapter_embeddings = embeddings[chapter_mask]
                
                if len(chapter_embeddings) > 1:
                    # Intra-cluster distances
                    intra_dist = pdist(chapter_embeddings).mean()
                    intra_distances.append(intra_dist)
                
                # Inter-cluster distances
                other_embeddings = embeddings[~chapter_mask]
                if len(other_embeddings) > 0 and len(chapter_embeddings) > 0:
                    inter_dist = np.mean([np.linalg.norm(ce - oe) 
                                        for ce in chapter_embeddings 
                                        for oe in other_embeddings[:100]])  # Sample for efficiency
                    inter_distances.append(inter_dist)
            
            metrics['mean_intra_cluster_distance'] = np.mean(intra_distances) if intra_distances else np.nan
            metrics['mean_inter_cluster_distance'] = np.mean(inter_distances) if inter_distances else np.nan
            metrics['cluster_separation_ratio'] = (np.mean(inter_distances) / np.mean(intra_distances)) if intra_distances and inter_distances else np.nan
            
        except Exception as e:
            print(f"Error calculating cluster distances for {model_name}: {e}")
            metrics['mean_intra_cluster_distance'] = np.nan
            metrics['mean_inter_cluster_distance'] = np.nan
            metrics['cluster_separation_ratio'] = np.nan
        
        # 5. Chapter purity metrics
        try:
            # Calculate how well chapters are separated
            chapter_counts = np.bincount(chapters)
            chapter_entropy = -np.sum((chapter_counts / len(chapters)) * np.log2(chapter_counts / len(chapters) + 1e-10))
            metrics['chapter_entropy'] = chapter_entropy
            
            # Effective number of chapters (based on entropy)
            metrics['effective_num_chapters'] = 2 ** chapter_entropy
            metrics['actual_num_chapters'] = len(unique_chapters)
            
        except Exception as e:
            print(f"Error calculating chapter metrics for {model_name}: {e}")
            metrics['chapter_entropy'] = np.nan
            metrics['effective_num_chapters'] = np.nan
            metrics['actual_num_chapters'] = np.nan
        
        # Store metrics
        quality_metrics[model_name] = metrics
        
        # Prepare for DataFrame
        result_row = {'model': model_name, **metrics}
        all_results.append(result_row)
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate rankings for each metric
    ranking_metrics = ['silhouette_score', 'calinski_harabasz_score', 'cluster_separation_ratio']
    inverse_ranking_metrics = ['davies_bouldin_score', 'mean_intra_cluster_distance', 'chapter_entropy']
    
    rankings_df = results_df.copy()
    
    for metric in ranking_metrics:
        if metric in rankings_df.columns:
            rankings_df[f'{metric}_rank'] = rankings_df[metric].rank(ascending=False, na_option='bottom')
    
    for metric in inverse_ranking_metrics:
        if metric in rankings_df.columns:
            rankings_df[f'{metric}_rank'] = rankings_df[metric].rank(ascending=True, na_option='bottom')
    
    # Calculate overall ranking (average of individual ranks)
    rank_columns = [col for col in rankings_df.columns if col.endswith('_rank')]
    rankings_df['overall_rank'] = rankings_df[rank_columns].mean(axis=1, skipna=True)
    rankings_df['overall_ranking'] = rankings_df['overall_rank'].rank(ascending=True, na_option='bottom')
    
    if save_results:
        results_df.to_csv('embedding_quality_metrics.csv', index=False)
        rankings_df.to_csv('embedding_quality_rankings.csv', index=False)
        print("Saved quality metrics and rankings to CSV files")
    
    return quality_metrics, results_df, rankings_df

def evaluate_dimensionality_reduction_quality(embeddings_data, save_results=True):
    """
    Function 6: Evaluate dimensionality reduction quality for t-SNE and UMAP.
    
    Args:
        embeddings_data: Output from load_and_process_embeddings function
        save_results: Whether to save results to CSV files
    
    Returns:
        dict: Dimensionality reduction quality metrics
    """
    print("Evaluating dimensionality reduction quality...")
    
    dr_metrics = {}
    all_results = []
    
    for model_name, data in embeddings_data.items():
        print(f"Evaluating dimensionality reduction for {model_name}...")
        
        embeddings = data['embeddings']
        chapters = np.array([int(ch) for ch in data['chapters']])
        
        if len(embeddings) < 10:
            print(f"Skipping {model_name}: insufficient data")
            continue
        
        model_metrics = {}
        
        # Subsample for efficiency if dataset is large
        if len(embeddings) > 1000:
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sample = embeddings[indices]
            chapters_sample = chapters[indices]
        else:
            embeddings_sample = embeddings
            chapters_sample = chapters
        
        # t-SNE evaluation
        try:
            print(f"  Computing t-SNE for {model_name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)//4))
            tsne_embeddings = tsne.fit_transform(embeddings_sample)
            
            # Calculate trustworthiness and continuity
            trust, cont = calculate_trustworthiness_continuity(embeddings_sample, tsne_embeddings, k=min(10, len(embeddings_sample)//10))
            
            model_metrics['tsne_trustworthiness'] = trust
            model_metrics['tsne_continuity'] = cont
            model_metrics['tsne_quality_score'] = (trust + cont) / 2
            
            # t-SNE specific: silhouette score in reduced space
            tsne_silhouette = silhouette_score(tsne_embeddings, chapters_sample)
            model_metrics['tsne_silhouette'] = tsne_silhouette
            
        except Exception as e:
            print(f"Error in t-SNE evaluation for {model_name}: {e}")
            model_metrics['tsne_trustworthiness'] = np.nan
            model_metrics['tsne_continuity'] = np.nan
            model_metrics['tsne_quality_score'] = np.nan
            model_metrics['tsne_silhouette'] = np.nan
        
        # UMAP evaluation
        try:
            print(f"  Computing UMAP for {model_name}...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_sample)//5))
            umap_embeddings = reducer.fit_transform(embeddings_sample)
            
            # Calculate trustworthiness and continuity
            trust, cont = calculate_trustworthiness_continuity(embeddings_sample, umap_embeddings, k=min(10, len(embeddings_sample)//10))
            
            model_metrics['umap_trustworthiness'] = trust
            model_metrics['umap_continuity'] = cont
            model_metrics['umap_quality_score'] = (trust + cont) / 2
            
            # UMAP specific: silhouette score in reduced space
            umap_silhouette = silhouette_score(umap_embeddings, chapters_sample)
            model_metrics['umap_silhouette'] = umap_silhouette
            
        except Exception as e:
            print(f"Error in UMAP evaluation for {model_name}: {e}")
            model_metrics['umap_trustworthiness'] = np.nan
            model_metrics['umap_continuity'] = np.nan
            model_metrics['umap_quality_score'] = np.nan
            model_metrics['umap_silhouette'] = np.nan
        
        # Store metrics
        dr_metrics[model_name] = model_metrics
        
        # Prepare for DataFrame
        result_row = {'model': model_name, **model_metrics}
        all_results.append(result_row)
    
    # Create results DataFrame
    dr_results_df = pd.DataFrame(all_results)
    
    # Calculate rankings
    ranking_metrics = ['tsne_trustworthiness', 'tsne_continuity', 'tsne_quality_score', 'tsne_silhouette',
                      'umap_trustworthiness', 'umap_continuity', 'umap_quality_score', 'umap_silhouette']
    
    dr_rankings_df = dr_results_df.copy()
    
    for metric in ranking_metrics:
        if metric in dr_rankings_df.columns:
            dr_rankings_df[f'{metric}_rank'] = dr_rankings_df[metric].rank(ascending=False, na_option='bottom')
    
    # Calculate overall rankings for t-SNE and UMAP separately
    tsne_rank_cols = [col for col in dr_rankings_df.columns if col.startswith('tsne_') and col.endswith('_rank')]
    umap_rank_cols = [col for col in dr_rankings_df.columns if col.startswith('umap_') and col.endswith('_rank')]
    
    if tsne_rank_cols:
        dr_rankings_df['tsne_overall_rank'] = dr_rankings_df[tsne_rank_cols].mean(axis=1, skipna=True)
        dr_rankings_df['tsne_overall_ranking'] = dr_rankings_df['tsne_overall_rank'].rank(ascending=True, na_option='bottom')
    
    if umap_rank_cols:
        dr_rankings_df['umap_overall_rank'] = dr_rankings_df[umap_rank_cols].mean(axis=1, skipna=True)
        dr_rankings_df['umap_overall_ranking'] = dr_rankings_df['umap_overall_rank'].rank(ascending=True, na_option='bottom')
    
    if save_results:
        dr_results_df.to_csv('dimensionality_reduction_metrics.csv', index=False)
        dr_rankings_df.to_csv('dimensionality_reduction_rankings.csv', index=False)
        print("Saved dimensionality reduction metrics and rankings to CSV files")
    
    return dr_metrics, dr_results_df, dr_rankings_df

def create_quality_visualizations(quality_results, dr_results, save_plots=True):
    """
    Create comprehensive visualizations for quality metrics.
    """
    print("Creating quality metric visualizations...")
    
    # 1. Radar chart for embedding quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Embedding quality radar chart
    ax1 = axes[0, 0]
    metrics_to_plot = ['silhouette_score', 'calinski_harabasz_score', 'cluster_separation_ratio']
    
    # Normalize metrics for radar chart
    normalized_data = quality_results[1].copy()
    for metric in metrics_to_plot:
        if metric in normalized_data.columns:
            min_val = normalized_data[metric].min()
            max_val = normalized_data[metric].max()
            if max_val > min_val:
                normalized_data[f'{metric}_norm'] = (normalized_data[metric] - min_val) / (max_val - min_val)
            else:
                normalized_data[f'{metric}_norm'] = 0.5
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#FF6F61', '#D2B48C', '#AE65D6', '#598A4F', '#4A90E2', '#F39C12', '#E74C3C']
    
    for i, (_, row) in enumerate(normalized_data.iterrows()):
        values = [row.get(f'{metric}_norm', 0) for metric in metrics_to_plot]
        values += values[:1]  # Complete the circle
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[i % len(colors)])
        ax1.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot])
    ax1.set_ylim(0, 1)
    ax1.set_title('Embedding Quality Metrics Comparison', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # 2. Ranking heatmap
    ax2 = axes[0, 1]
    ranking_cols = [col for col in quality_results[2].columns if col.endswith('_rank') and not col.startswith('overall')]
    if ranking_cols:
        ranking_data = quality_results[2][['model'] + ranking_cols].set_index('model')
        sns.heatmap(ranking_data.T, annot=True, cmap='RdYlGn_r', ax=ax2, cbar_kws={'label': 'Rank'})
        ax2.set_title('Quality Metrics Rankings', fontsize=14)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Metrics')
    
    # 3. Dimensionality reduction comparison
    ax3 = axes[1, 0]
    if len(dr_results[1]) > 0:
        dr_comparison = dr_results[1][['model', 'tsne_quality_score', 'umap_quality_score']].set_index('model')
        dr_comparison.plot(kind='bar', ax=ax3, color=['#FF6F61', '#AE65D6'])
        ax3.set_title('t-SNE vs UMAP Quality Scores', fontsize=14)
        ax3.set_ylabel('Quality Score')
        ax3.legend(['t-SNE', 'UMAP'])
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Overall rankings summary
    ax4 = axes[1, 1]
    if 'overall_ranking' in quality_results[2].columns:
        overall_rankings = quality_results[2][['model', 'overall_ranking']].sort_values('overall_ranking')
        bars = ax4.barh(overall_rankings['model'], overall_rankings['overall_ranking'], color='#598A4F')
        ax4.set_title('Overall Quality Rankings', fontsize=14)
        ax4.set_xlabel('Ranking (1 = Best)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('embedding_quality_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved quality analysis visualization")
    
    plt.show()

def analyze_embeddings(folder_path="Resulting-embeddings", save_plots=True):
    """
    Function 4: Meta function controlling functions 1 to 6.
    
    Args:
        folder_path: Path to the folder containing CSV files with embeddings
        save_plots: Whether to save visualization plots to files
    """
    print("Starting comprehensive embedding analysis...")
    print("=" * 50)
    
    # Function 1: Load and process embeddings
    print("Step 1: Loading and processing embeddings...")
    embeddings_data = load_and_process_embeddings(folder_path)
    
    if not embeddings_data:
        print("No embeddings data loaded. Please check the folder path and CSV files.")
        return
    
    print(f"Successfully loaded embeddings for {len(embeddings_data)} models")
    print("=" * 50)
    
    # Function 5: Evaluate embedding quality
    print("Step 2: Evaluating embedding quality...")
    quality_metrics, quality_results_df, quality_rankings_df = evaluate_embedding_quality(embeddings_data, save_plots)
    print("=" * 50)
    
    # Function 6: Evaluate dimensionality reduction quality
    print("Step 3: Evaluating dimensionality reduction quality...")
    dr_metrics, dr_results_df, dr_rankings_df = evaluate_dimensionality_reduction_quality(embeddings_data, save_plots)
    print("=" * 50)
    
    # Create quality visualizations
    print("Step 4: Creating quality visualizations...")
    create_quality_visualizations((quality_metrics, quality_results_df, quality_rankings_df), 
                                (dr_metrics, dr_results_df, dr_rankings_df), save_plots)
    print("=" * 50)
    
    # Function 2: Create t-SNE visualizations
    print("Step 5: Creating t-SNE visualizations...")
    create_tsne_visualization(embeddings_data, save_plots)
    print("=" * 50)
    
    # Function 3: Create UMAP visualizations
    print("Step 6: Creating UMAP visualizations...")
    create_umap_visualization(embeddings_data, save_plots)
    print("=" * 50)
    
    print("Analysis complete!")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY FOR REPORT APPENDIX")
    print("="*80)
    
    # Quality metrics summary
    print("\n1. EMBEDDING QUALITY METRICS:")
    print("-" * 40)
    if not quality_results_df.empty:
        print(quality_results_df.round(4).to_string(index=False))
        
        print("\n2. QUALITY RANKINGS:")
        print("-" * 40)
        ranking_summary = quality_rankings_df[['model', 'silhouette_score_rank', 'calinski_harabasz_score_rank', 
                                             'davies_bouldin_score_rank', 'overall_ranking']].round(2)
        print(ranking_summary.to_string(index=False))
    
    # Dimensionality reduction summary
    print("\n3. DIMENSIONALITY REDUCTION QUALITY:")
    print("-" * 40)
    if not dr_results_df.empty:
        print(dr_results_df.round(4).to_string(index=False))
        
        print("\n4. DIMENSIONALITY REDUCTION RANKINGS:")
        print("-" * 40)
        dr_ranking_summary = dr_rankings_df[['model', 'tsne_overall_ranking', 'umap_overall_ranking']].round(2)
        print(dr_ranking_summary.to_string(index=False))
    
    # Best performing models summary
    print("\n5. TOP PERFORMING MODELS:")
    print("-" * 40)
    if not quality_rankings_df.empty:
        best_overall = quality_rankings_df.loc[quality_rankings_df['overall_ranking'].idxmin(), 'model']
        print(f"Best Overall Quality: {best_overall}")
        
        best_silhouette = quality_results_df.loc[quality_results_df['silhouette_score'].idxmax(), 'model']
        print(f"Best Silhouette Score: {best_silhouette}")
        
        if not dr_rankings_df.empty:
            best_tsne = dr_rankings_df.loc[dr_rankings_df['tsne_overall_ranking'].idxmin(), 'model']
            best_umap = dr_rankings_df.loc[dr_rankings_df['umap_overall_ranking'].idxmin(), 'model']
            print(f"Best t-SNE Quality: {best_tsne}")
            print(f"Best UMAP Quality: {best_umap}")
    
    print("\n6. FILES GENERATED FOR APPENDIX:")
    print("-" * 40)
    print("- embedding_quality_metrics.csv")
    print("- embedding_quality_rankings.csv") 
    print("- dimensionality_reduction_metrics.csv")
    print("- dimensionality_reduction_rankings.csv")
    print("- embedding_quality_analysis.png")
    print("- Individual t-SNE and UMAP visualization PNGs")
    
    return {
        'embeddings_data': embeddings_data,
        'quality_metrics': quality_metrics,
        'quality_results': quality_results_df,
        'quality_rankings': quality_rankings_df,
        'dr_metrics': dr_metrics,
        'dr_results': dr_results_df,
        'dr_rankings': dr_rankings_df
    }

if __name__ == "__main__":
    # Run the comprehensive analysis
    results = analyze_embeddings() 