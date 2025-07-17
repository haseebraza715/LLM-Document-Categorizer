# LLM Document Categorizer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) 
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Languages](https://img.shields.io/badge/languages-Python%2C%20Shell-blue)](#)

A pipeline for automated document clustering using BERTopic and Mistral embeddings.

## Features
- PDF extraction & cleaning
- Mistral API embeddings
- BERTopic topic modeling
- Visual and CSV outputs

## Quickstart
```bash
pip install -r requirements.txt
echo "MISTRAL_API_KEY=your_key" > .env
python main.py
```

## Usage
- Place PDF files in `data/raw/`
- Run the pipeline as above
- Outputs are saved in `clustering/results/` and `embeddings/`

## Outputs
- `improved_topic_info.csv`: Topic statistics
- `improved_document_assignments.csv`: Document-topic mapping
- `improved_topics.json`: Topic keywords
- Visualizations: `.html`, `.png` in `clustering/results/` 