"""
Improved BERTopic Clustering Module

This module implements an enhanced BERTopic clustering approach with:
- Adaptive parameter selection based on dataset size
- Improved preprocessing with embedding normalization
- Enhanced TF-IDF vectorizer with better n-gram ranges
- Optimized UMAP and HDBSCAN parameters
- Comprehensive evaluation metrics
- Enhanced visualizations

Author: Document Categorization Project
"""

import numpy as np
import json
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class ImprovedBERTopicClustering:
    """
    Enhanced BERTopic clustering with improved parameters and evaluation.
    
    This class provides an improved implementation of BERTopic clustering
    with adaptive parameter selection, enhanced preprocessing, and comprehensive
    evaluation metrics.
    """
    
    def __init__(self, embeddings_path, docs_path, output_dir):
        """
        Initialize the improved BERTopic clustering.
        
        Args:
            embeddings_path (str): Path to the embeddings file
            docs_path (str): Path to the document mapping file
            output_dir (str): Directory to save results
        """
        self.embeddings_path = embeddings_path
        self.docs_path = docs_path
        self.output_dir = output_dir
        self.topic_model = None
        self.embeddings = None
        self.documents = None
        self.best_params = None
        
    def load_data(self):
        """
        Load embeddings and documents from files.
        
        Returns:
            tuple: (embeddings, documents)
        """
        # Load embeddings from numpy file
        self.embeddings = np.load(self.embeddings_path)
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        
        # Load document mapping from JSON
        with open(self.docs_path, 'r') as f:
            doc_map = json.load(f)
        
        # Load actual document texts from processed files
        self.documents = []
        for doc_name in doc_map:
            doc_path = os.path.join('data/processed', doc_name)
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    self.documents.append(f.read().strip())
            else:
                self.documents.append("")
        
        print(f"Loaded {len(self.documents)} documents")
        return self.embeddings, self.documents
    
    def preprocess_embeddings(self):
        """
        Preprocess embeddings for better clustering performance.
        
        Returns:
            numpy.ndarray: Preprocessed embeddings
        """
        # Normalize embeddings using StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.embeddings = scaler.fit_transform(self.embeddings)
        
        # Remove any NaN values for numerical stability
        self.embeddings = np.nan_to_num(self.embeddings, nan=0.0)
        
        print(f"Preprocessed embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def find_optimal_parameters(self):
        """
        Find optimal parameters based on dataset characteristics.
        
        Returns:
            dict: Dictionary of optimal parameters
        """
        n_docs = len(self.documents)
        
        # Adaptive parameters based on dataset size
        if n_docs < 10:
            # Very small dataset - use minimal parameters
            return {
                'min_topic_size': 1,
                'nr_topics': min(3, n_docs - 1),
                'umap_n_neighbors': 2,
                'umap_n_components': 2,
                'hdbscan_min_cluster_size': 2,
                'hdbscan_min_samples': 1
            }
        elif n_docs < 50:
            # Small-medium dataset - balanced parameters
            return {
                'min_topic_size': 2,
                'nr_topics': min(8, n_docs // 3),
                'umap_n_neighbors': 5,
                'umap_n_components': 5,
                'hdbscan_min_cluster_size': 3,
                'hdbscan_min_samples': 2
            }
        else:
            # Large dataset - more robust parameters
            return {
                'min_topic_size': 3,
                'nr_topics': min(15, n_docs // 4),
                'umap_n_neighbors': 10,
                'umap_n_components': 10,
                'hdbscan_min_cluster_size': 5,
                'hdbscan_min_samples': 3
            }
    
    def create_improved_vectorizer(self):
        """
        Create an improved TF-IDF vectorizer for better topic representation.
        
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        return TfidfVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            max_features=1000,
            sublinear_tf=True  # Apply sublinear tf scaling
        )
    
    def create_improved_umap(self, params):
        """
        Create an improved UMAP model for dimensionality reduction.
        
        Args:
            params (dict): Parameters for UMAP
            
        Returns:
            UMAP: Configured UMAP model
        """
        return UMAP(
            n_neighbors=params['umap_n_neighbors'],
            n_components=params['umap_n_components'],
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            low_memory=False
        )
    
    def create_improved_hdbscan(self, params):
        """
        Create an improved HDBSCAN model for clustering.
        
        Args:
            params (dict): Parameters for HDBSCAN
            
        Returns:
            HDBSCAN: Configured HDBSCAN model
        """
        return HDBSCAN(
            min_cluster_size=params['hdbscan_min_cluster_size'],
            min_samples=params['hdbscan_min_samples'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
    
    def fit_improved_bertopic(self, n_topics=None):
        """
        Fit BERTopic model with improved parameters and techniques.
        
        Args:
            n_topics (int, optional): Number of topics to extract
            
        Returns:
            tuple: (topics, probabilities)
        """
        # Preprocess embeddings for better clustering
        self.preprocess_embeddings()
        
        # Find optimal parameters based on dataset
        params = self.find_optimal_parameters()
        if n_topics:
            params['nr_topics'] = n_topics
        
        print(f"Using parameters: {params}")
        
        # Create improved components
        vectorizer = self.create_improved_vectorizer()
        umap_model = self.create_improved_umap(params)
        hdbscan_model = self.create_improved_hdbscan(params)
        
        # Initialize BERTopic with improved parameters
        self.topic_model = BERTopic(
            vectorizer_model=vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=params['min_topic_size'],
            nr_topics=params['nr_topics'],
            verbose=True,
            calculate_probabilities=True,
            top_n_words=20,  # More words per topic for better representation
            n_gram_range=(1, 2)
        )
        
        # Fit the model on documents and embeddings
        topics, probs = self.topic_model.fit_transform(
            self.documents, 
            self.embeddings
        )
        
        return topics, probs
    
    def evaluate_clustering_comprehensive(self):
        """
        Comprehensive clustering evaluation with multiple metrics.
        
        Returns:
            dict: Evaluation results
        """
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_improved_bertopic() first.")
        
        topics, probs = self.topic_model.transform(self.documents, self.embeddings)
        unique_topics = set(topics)
        
        print("\n=== COMPREHENSIVE CLUSTERING EVALUATION ===")
        
        # Basic statistics
        print(f"Total documents: {len(topics)}")
        print(f"Number of topics: {len(unique_topics)}")
        print(f"Documents per topic: {len(topics) / len(unique_topics):.1f}")
        
        # Silhouette Score - measures cluster cohesion and separation
        if len(unique_topics) > 1:
            try:
                umap_embeddings = self.topic_model.umap_model.embedding_
                silhouette_avg = silhouette_score(umap_embeddings, topics)
                print(f"Silhouette Score: {silhouette_avg:.3f} (higher is better, >0.3 is good)")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
        
        # Calinski-Harabasz Index - ratio of between-cluster to within-cluster dispersion
        if len(unique_topics) > 1:
            try:
                ch_score = calinski_harabasz_score(self.embeddings, topics)
                print(f"Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
            except Exception as e:
                print(f"Could not calculate Calinski-Harabasz Index: {e}")
        
        # Davies-Bouldin Index - average similarity measure of clusters 

    def save_results(self):
        """
        Save BERTopic model, topic info, document assignments, and topics to output_dir.
        """
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_improved_bertopic() first.")
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Save BERTopic model
        model_dir = os.path.join(self.output_dir, 'improved_bertopic_model')
        self.topic_model.save(model_dir)

        # 2. Save topic info CSV
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv(os.path.join(self.output_dir, 'improved_topic_info.csv'), index=False)

        # 3. Save document assignments CSV
        docs = self.documents
        topics, probs = self.topic_model.transform(docs, self.embeddings)
        assignments = []
        for i, (doc, topic, prob) in enumerate(zip(docs, topics, probs.max(axis=1))):
            assignments.append({
                'document_id': i,
                'topic': topic,
                'confidence': prob,
                'text_preview': doc[:200]
            })
        import pandas as pd
        assignments_df = pd.DataFrame(assignments)
        assignments_df.to_csv(os.path.join(self.output_dir, 'improved_document_assignments.csv'), index=False)

        # 4. Save topics JSON (top words per topic)
        topics_dict = {}
        for topic_id in self.topic_model.get_topics().keys():
            if topic_id == -1:
                continue
            words = self.topic_model.get_topic(topic_id)
            topics_dict[str(topic_id)] = words
        with open(os.path.join(self.output_dir, 'improved_topics.json'), 'w') as f:
            json.dump(topics_dict, f, indent=2)

        # 5. Optionally, save a summary HTML (skip for now)
        print(f"Saved results to {self.output_dir}") 