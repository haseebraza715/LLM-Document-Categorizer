# LLM-Document-Categorizer

> Document clustering with BERTopic + Mistral.

## Quickstart

```bash
pip install -r requirements.txt
echo "MISTRAL_API_KEY=your_key" > .env
python main_improved.py
python run_improved_dashboard.py
```

## Features

- PDF extraction & cleaning
- Mistral embeddings
- BERTopic clustering
- Streamlit dashboard

## Outputs

- `clustering/results/improved_topic_info.csv`
- `clustering/results/improved_document_assignments.csv`
- `clustering/results/improved_topics.json` 