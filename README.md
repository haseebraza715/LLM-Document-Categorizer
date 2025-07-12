# LLM-Document-Categorizer

Minimal pipeline for document clustering and topic discovery using BERTopic and Mistral embeddings.

---

## Features
- PDF text extraction & cleaning
- Mistral API embeddings
- BERTopic clustering (adaptive)
- Streamlit dashboard
- Exportable results

---

## Quickstart
```bash
pip install -r requirements.txt
# Add your API key
echo "MISTRAL_API_KEY=your_api_key_here" > .env
# Add PDFs to data/raw/
python main_improved.py
python run_improved_dashboard.py  # http://localhost:8502
```

---

## Layout
```
main_improved.py            # Pipeline
run_improved_dashboard.py   # Dashboard
clustering/                 # BERTopic/results
ui/                        # Dashboard UI
utils/                     # Preprocessing
mistral_api/               # Embedding API
data/                      # raw/ processed/
embeddings/                # Vectors
```

---

## Outputs
- `clustering/results/improved_topic_info.csv` — Topic stats
- `clustering/results/improved_document_assignments.csv` — Doc assignments
- `clustering/results/improved_topics.json` — Topic keywords
- `clustering/results/improved_bertopic_model/` — Saved model

---

## Requirements
Python 3.8+
- bertopic, streamlit, plotly, scikit-learn, umap-learn, hdbscan, pandas, numpy, requests, python-dotenv 