"""
Improved Document Categorization Pipeline using BERTopic Clustering

This script implements a complete document categorization pipeline that:
1. Processes PDF documents from the raw data directory
2. Extracts and cleans text content
3. Generates embeddings using Mistral API
4. Performs improved BERTopic clustering with enhanced parameters
5. Evaluates clustering quality with comprehensive metrics
6. Creates visualizations and saves results

Author: Document Categorization Project
"""

import os
from dotenv import load_dotenv
import numpy as np
import json
from tqdm import tqdm
from mistral_api.embed import get_embedding
from utils.preprocess import clean_text, extract_text_from_pdf
from clustering.improved_bertopic_clustering import ImprovedBERTopicClustering

# Load API key from environment variables
load_dotenv(dotenv_path='.env')

# Define directory paths for data processing
raw_dir = 'data/raw'
processed_dir = 'data/processed'
embeddings_dir = 'embeddings'
clustering_dir = 'clustering/results'

# Create necessary directories if they don't exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)
os.makedirs(clustering_dir, exist_ok=True)

print("=== IMPROVED DOCUMENT CATEGORIZATION PIPELINE ===")

# Step 1: Process PDF files from raw directory
pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
if pdf_files:
    print(f"Processing {len(pdf_files)} PDF files...")
    for idx, pdf_file in enumerate(tqdm(pdf_files, desc='Processing PDFs')):
        pdf_path = os.path.join(raw_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            cleaned = clean_text(text)
            # Save as processed document with consistent naming
            fname = f'pdf_doc_{idx:04d}.txt'
            with open(os.path.join(processed_dir, fname), 'w', encoding='utf-8') as f:
                f.write(cleaned)
    print(f"Processed {len(pdf_files)} PDF files and saved to {processed_dir}")

# Step 2: Load all cleaned documents and prepare for embedding
processed_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.txt')])
embeddings = []
file_map = []

# Process documents for embedding with limit for testing
max_docs = 50
print(f"Embedding {min(max_docs, len(processed_files))} documents...")
for fname in tqdm(processed_files[:max_docs], desc='Embedding docs'):
    with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if text:
        emb = get_embedding(text)
        embeddings.append(emb)
        file_map.append(fname)
    else:
        embeddings.append([])
        file_map.append(fname)

# Step 3: Save embeddings and document mapping
if embeddings:
    np.save(os.path.join(embeddings_dir, 'mistral_vectors.npy'), np.array(embeddings))
    with open(os.path.join(embeddings_dir, 'doc_map.json'), 'w') as f:
        json.dump(file_map, f)
    print(f"Saved {len(embeddings)} embeddings to {os.path.join(embeddings_dir, 'mistral_vectors.npy')}")
else:
    print("No documents found to embed. Please add PDF files to data/raw/")

# Step 4: Improved BERTopic Clustering
print("\n=== STARTING IMPROVED BERTOPIC CLUSTERING ===")
improved_clusterer = ImprovedBERTopicClustering(
    embeddings_path=os.path.join(embeddings_dir, 'mistral_vectors.npy'),
    docs_path=os.path.join(embeddings_dir, 'doc_map.json'),
    output_dir=clustering_dir
)

# Load data for clustering
embeddings, documents = improved_clusterer.load_data()

# Fit improved BERTopic model with enhanced parameters
print("Fitting improved BERTopic model...")
topics, probs = improved_clusterer.fit_improved_bertopic()

# Step 5: Comprehensive evaluation of clustering quality
print("\n=== COMPREHENSIVE EVALUATION ===")
evaluation_results = improved_clusterer.evaluate_clustering_comprehensive()

# Step 6: Create improved visualizations
print("\n=== CREATING IMPROVED VISUALIZATIONS ===")
improved_clusterer.visualize_improved_topics()

# Step 7: Save comprehensive results
print("\n=== SAVING RESULTS ===")
improved_clusterer.save_results()

print(f"\n=== IMPROVED CLUSTERING COMPLETE ===")
print(f"Results saved to: {clustering_dir}")
print("Files created:")
print("- improved_bertopic_model/ (saved model)")
print("- improved_topic_info.csv (topic statistics)")
print("- improved_document_assignments.csv (document assignments)")
print("- improved_topics.json (topic keywords)")
print("- improved_*.html (enhanced visualizations)")

# Print summary of improvements over baseline
print(f"\n=== IMPROVEMENTS SUMMARY ===")
print("Enhanced preprocessing with embedding normalization")
print("Adaptive parameter selection based on dataset size")
print("Improved TF-IDF vectorizer with better n-gram range")
print("Optimized UMAP and HDBSCAN parameters")
print("Comprehensive evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)")
print("Enhanced visualizations with document plots")
print("Model persistence for reproducible results") 