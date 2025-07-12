# Document Categorization with BERTopic

A document categorization system using BERTopic clustering and Mistral embeddings.

## Overview

This project implements an improved document categorization pipeline that:
- Processes PDF documents and extracts text content
- Generates embeddings using Mistral API
- Performs BERTopic clustering with adaptive parameters
- Provides comprehensive evaluation metrics
- Includes an interactive dashboard for result visualization

## Project Structure

```
doc_categorizer_llm/
├── main_improved.py              # Main pipeline script
├── run_improved_dashboard.py     # Dashboard launcher
├── requirements.txt              # Python dependencies
├── clustering/
│   ├── improved_bertopic_clustering.py  # Enhanced clustering module
│   └── results/                 # Clustering results
├── ui/
│   └── improved_dashboard.py    # Interactive dashboard
├── utils/
│   └── preprocess.py            # Text preprocessing utilities
├── mistral_api/
│   └── embed.py                 # Mistral API integration
├── data/
│   ├── raw/                     # Raw PDF files
│   └── processed/               # Processed text files
└── embeddings/                  # Generated embeddings
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Mistral API key:
```
MISTRAL_API_KEY=your_api_key_here
```

3. Add PDF files to `data/raw/` directory

## Usage

### Run the Pipeline

```bash
python main_improved.py
```

This will:
- Process PDF files from `data/raw/`
- Generate embeddings using Mistral API
- Perform improved BERTopic clustering
- Save results to `clustering/results/`

### Launch Dashboard

```bash
python run_improved_dashboard.py
```

Access the dashboard at `http://localhost:8502`

## Features

### Improved Clustering
- Adaptive parameter selection based on dataset size
- Enhanced preprocessing with embedding normalization
- Improved TF-IDF vectorizer with n-gram ranges
- Optimized UMAP and HDBSCAN parameters

### Comprehensive Evaluation
- Silhouette Score for cluster cohesion
- Calinski-Harabasz Index for cluster separation
- Davies-Bouldin Index for cluster quality
- Topic distribution analysis

### Interactive Dashboard
- Summary statistics and key metrics
- Topic analysis with search functionality
- Document clustering visualizations
- Filterable document assignments
- Export capabilities

## Output Files

- `improved_topic_info.csv` - Topic statistics and counts
- `improved_document_assignments.csv` - Document-to-topic assignments
- `improved_topics.json` - Topic keywords and representations
- `improved_bertopic_model/` - Saved model for reproducibility
- `improved_*.html` - Interactive visualizations

## Configuration

The system automatically adapts parameters based on dataset size:
- Small datasets (< 10 docs): Minimal parameters
- Medium datasets (10-50 docs): Balanced parameters  
- Large datasets (> 50 docs): Robust parameters

## Dependencies

- bertopic
- streamlit
- plotly
- scikit-learn
- umap-learn
- hdbscan
- pandas
- numpy
- requests
- python-dotenv # LLM-Document-Categorizer
