import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json

# --- Paths to results ---
RESULTS_DIR = 'clustering/results'
TOPIC_INFO_CSV = os.path.join(RESULTS_DIR, 'improved_topic_info.csv')
DOC_ASSIGN_CSV = os.path.join(RESULTS_DIR, 'improved_document_assignments.csv')
TOPICS_JSON = os.path.join(RESULTS_DIR, 'improved_topics.json')
TSNE_PNG = os.path.join(RESULTS_DIR, 'tsne_topics.png')
CLUSTER_DIST_PNG = os.path.join(RESULTS_DIR, 'cluster_size_distribution.png')
TOPIC_DIST_PNG = os.path.join(RESULTS_DIR, 'topic_distribution.png')

st.set_page_config(
    page_title="LLM Document Categorizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Upload & Process",
    "Cluster Overview",
    "Topic Details",
    "Document Explorer",
    "Download Results"
])

# --- Helper Functions ---
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, use_column_width=True, caption=caption)
    else:
        st.warning(f"Image not found: {path}")

# --- Load Data ---
topic_info = load_csv(TOPIC_INFO_CSV)
doc_assign = load_csv(DOC_ASSIGN_CSV)
topics_dict = load_json(TOPICS_JSON)

# --- Overview Page ---
if page == "Overview":
    st.title("LLM Document Categorizer")
    st.markdown("""
    **Document clustering with BERTopic + Mistral.**
    
    - PDF extraction & cleaning
    - Mistral embeddings
    - BERTopic clustering
    - Interactive visualizations
    """)
    st.subheader("Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Topics", topic_info.shape[0] if not topic_info.empty else "-")
    col2.metric("Documents", doc_assign.shape[0] if not doc_assign.empty else "-")
    col3.metric("Assignments", doc_assign.shape[0] if not doc_assign.empty else "-")
    st.markdown("---")
    st.subheader("Pipeline Steps")
    st.markdown("""
    1. **Upload PDFs** → 2. **Text Extraction & Cleaning** → 3. **Embedding (Mistral)** → 4. **Clustering (BERTopic)** → 5. **Visualization & Analysis**
    """)
    st.markdown("---")
    st.subheader("Sample Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        show_image(CLUSTER_DIST_PNG, "Cluster Size Distribution")
    with col2:
        show_image(TOPIC_DIST_PNG, "Topic Distribution")

# --- Upload & Process Page ---
elif page == "Upload & Process":
    st.title("Upload & Process Documents")
    st.info("Upload PDF files to process and categorize.")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) uploaded.")
        st.warning("Processing pipeline must be run manually after upload (see README).")
        for file in uploaded_files:
            with open(os.path.join('data/raw', file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.success("Files saved to data/raw/. Please run the pipeline script to process.")
    else:
        st.write("No files uploaded yet.")

# --- Cluster Overview Page ---
elif page == "Cluster Overview":
    st.title("Cluster Overview")
    st.markdown("Visualize topic and cluster distributions.")
    col1, col2 = st.columns(2)
    with col1:
        show_image(CLUSTER_DIST_PNG, "Cluster Size Distribution")
    with col2:
        show_image(TOPIC_DIST_PNG, "Topic Distribution")
    st.markdown("---")
    st.subheader("t-SNE Topic Visualization")
    show_image(TSNE_PNG, "t-SNE Topics")

# --- Topic Details Page ---
elif page == "Topic Details":
    st.title("Topic Details")
    if topic_info.empty:
        st.warning("No topic info available.")
    else:
        st.dataframe(topic_info, use_container_width=True)
        st.markdown("---")
        st.subheader("Top Words per Topic")
        for topic_id, words in topics_dict.items():
            with st.expander(f"Topic {topic_id}"):
                st.write(words)

# --- Document Explorer Page ---
elif page == "Document Explorer":
    st.title("Document Explorer")
    if doc_assign.empty:
        st.warning("No document assignments available.")
    else:
        st.dataframe(doc_assign, use_container_width=True)
        st.markdown("---")
        st.subheader("Search & Filter")
        topic_filter = st.selectbox("Filter by Topic", options=["All"] + list(topic_info['Topic'].unique()) if not topic_info.empty else ["All"])
        if topic_filter != "All":
            filtered = doc_assign[doc_assign['topic'] == int(topic_filter)]
        else:
            filtered = doc_assign
        st.dataframe(filtered, use_container_width=True)

# --- Download Results Page ---
elif page == "Download Results":
    st.title("Download Results")
    st.info("Download clustering results and visualizations.")
    if os.path.exists(TOPIC_INFO_CSV):
        with open(TOPIC_INFO_CSV, 'rb') as f:
            st.download_button("Download Topic Info CSV", f, file_name="improved_topic_info.csv")
    if os.path.exists(DOC_ASSIGN_CSV):
        with open(DOC_ASSIGN_CSV, 'rb') as f:
            st.download_button("Download Document Assignments CSV", f, file_name="improved_document_assignments.csv")
    if os.path.exists(TOPICS_JSON):
        with open(TOPICS_JSON, 'rb') as f:
            st.download_button("Download Topics JSON", f, file_name="improved_topics.json")
    st.markdown("---")
    st.subheader("Visualizations")
    for img_path, label in [
        (CLUSTER_DIST_PNG, "Cluster Size Distribution"),
        (TOPIC_DIST_PNG, "Topic Distribution"),
        (TSNE_PNG, "t-SNE Topics")
    ]:
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                st.download_button(f"Download {label}", f, file_name=os.path.basename(img_path)) 