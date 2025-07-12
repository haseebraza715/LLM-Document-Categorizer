"""
Improved Document Categorization Dashboard

This Streamlit application provides an interactive dashboard for visualizing and analyzing
document clustering results from the improved BERTopic pipeline. Features include:

- Summary statistics and key metrics
- Interactive topic analysis with search functionality
- Document clustering visualizations
- Filterable document assignments
- Export capabilities for results

Author: Document Categorization Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import warnings
from pathlib import Path

# Suppress all warnings globally
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Document Categorizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced black and white CSS
st.markdown("""
<style>
    .main {
        background-color: white;
        color: black;
    }
    .metric-card {
        background-color: black;
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        transition: transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.6rem;
        color: black;
        margin: 25px 0 15px 0;
        border-bottom: 3px solid black;
        padding-bottom: 8px;
        font-weight: bold;
    }
    .topic-box {
        background-color: white;
        border: 2px solid black;
        padding: 25px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .topic-box:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .topic-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
        border-bottom: 1px solid black;
        padding-bottom: 5px;
    }
    .topic-count {
        font-size: 1rem;
        color: black;
        margin-bottom: 12px;
        font-weight: 500;
    }
    .topic-keywords {
        font-size: 0.9rem;
        color: black;
        font-style: italic;
        line-height: 1.5;
        background-color: #f8f8f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid black;
    }
    .summary-box {
        background-color: #f8f8f8;
        border: 2px solid black;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
    }
    .export-button {
        background-color: black !important;
        color: white !important;
        border: 2px solid black !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .export-button:hover {
        background-color: white !important;
        color: black !important;
        border-color: black !important;
    }
    .stDataFrame {
        border: 2px solid black !important;
        border-radius: 8px !important;
    }
    .stDataFrame th {
        background-color: black !important;
        color: white !important;
        font-weight: bold !important;
    }
    .stDataFrame td {
        border: 1px solid #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_improved_results():
    """Load improved clustering results."""
    results_dir = Path("clustering/results")
    
    results = {}
    
    # Load improved topic info
    if (results_dir / "improved_topic_info.csv").exists():
        results['topic_info'] = pd.read_csv(results_dir / "improved_topic_info.csv")
    
    # Load improved document assignments
    if (results_dir / "improved_document_assignments.csv").exists():
        results['document_assignments'] = pd.read_csv(results_dir / "improved_document_assignments.csv")
    
    # Load improved topics JSON
    if (results_dir / "improved_topics.json").exists():
        with open(results_dir / "improved_topics.json", 'r') as f:
            results['topics'] = json.load(f)
    
    # Load embeddings with numerical stability
    embeddings_path = Path("embeddings/mistral_vectors.npy")
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        # Clean embeddings for numerical stability
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        results['embeddings'] = embeddings
    
    return results

def create_simple_topic_chart(topic_info):
    """Create simple black and white topic distribution chart with better contrast."""
    topic_data = topic_info[topic_info['Topic'] != -1].copy()
    
    fig = go.Figure(data=[
        go.Bar(
            x=topic_data['Topic'],
            y=topic_data['Count'],
            marker_color='black',
            opacity=0.9,
            marker_line_color='black',
            marker_line_width=1,
            text=topic_data['Count'],
            textposition='outside',
            textfont=dict(color='black', size=12),
            hovertemplate='<b>Topic %{x}</b><br>' +
                         'Documents: %{y}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Topic Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'black', 'weight': 'bold'}
        },
        xaxis_title="Topic ID",
        yaxis_title="Number of Documents",
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', size=12),
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(
        gridcolor='#e0e0e0', 
        zerolinecolor='black',
        zerolinewidth=2,
        showgrid=True,
        gridwidth=1,
        title_font=dict(color='black', size=14, weight='bold'),
        tickfont=dict(color='black', size=12),
        tickcolor='black',
        tickwidth=2
    )
    fig.update_yaxes(
        gridcolor='#e0e0e0', 
        zerolinecolor='black',
        zerolinewidth=2,
        showgrid=True,
        gridwidth=1,
        title_font=dict(color='black', size=14, weight='bold'),
        tickfont=dict(color='black', size=12),
        tickcolor='black',
        tickwidth=2
    )
    
    return fig

def create_simple_clustering_chart(document_assignments, embeddings):
    """Create simple black and white clustering visualization with better contrast."""
    if embeddings is None or len(embeddings) == 0:
        return None
    
    try:
        from sklearn.decomposition import PCA
        import warnings
        warnings.filterwarnings('ignore')
        
        embeddings_clean = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_clean)
        
        plot_data = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Topic': document_assignments['topic'],
            'Confidence': document_assignments['confidence']
        })
        
        # Create custom colors for better contrast
        colors = ['black', '#333333', '#666666', '#999999', '#cccccc', '#e0e0e0']
        
        fig = px.scatter(
            plot_data,
            x='x',
            y='y',
            color='Topic',
            size='Confidence',
            title="Document Clustering",
            color_discrete_sequence=colors,
            size_max=15
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=12),
            title={
                'text': "Document Clustering",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black', 'weight': 'bold'}
            },
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                font=dict(color='black', size=12, weight='bold'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        # Update axes for better contrast
        fig.update_xaxes(
            gridcolor='#e0e0e0',
            zerolinecolor='black',
            zerolinewidth=2,
            showgrid=True,
            gridwidth=1,
            title_font=dict(color='black', size=14, weight='bold'),
            tickfont=dict(color='black', size=12),
            tickcolor='black',
            tickwidth=2
        )
        fig.update_yaxes(
            gridcolor='#e0e0e0',
            zerolinecolor='black',
            zerolinewidth=2,
            showgrid=True,
            gridwidth=1,
            title_font=dict(color='black', size=14, weight='bold'),
            tickfont=dict(color='black', size=12),
            tickcolor='black',
            tickwidth=2
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Could not create clustering visualization: {e}")
        return None

def display_metric(label, value, subtitle=""):
    """Display a simple metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f'<div style="font-size: 0.7rem; color: white; margin-top: 5px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Simple header
    st.title("Document Categorizer")
    st.markdown("---")
    
    # Load results
    with st.spinner("Loading results..."):
        results = load_improved_results()
    
    if not results:
        st.markdown("<p style='color: black; font-weight: bold; background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 3px solid #dc3545;'>No results found. Please run the pipeline first.</p>", unsafe_allow_html=True)
        return
    
    # Success message
    st.markdown("<p style='color: black; font-weight: bold; background-color: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 3px solid #28a745;'>Model loaded successfully</p>", unsafe_allow_html=True)
    
    # Summary Statistics
    st.markdown('<h2 class="section-title">Summary Statistics</h2>', unsafe_allow_html=True)
    
    if 'topic_info' in results and 'document_assignments' in results:
        total_docs = len(results['document_assignments'])
        valid_topics = results['topic_info'][results['topic_info']['Topic'] != -1]
        total_topics = len(valid_topics)
        outliers = results['topic_info'][results['topic_info']['Topic'] == -1]['Count'].iloc[0] if -1 in results['topic_info']['Topic'].values else 0
        avg_confidence = results['document_assignments']['confidence'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="summary-box">
                <h3 style="color: black; margin-bottom: 15px;">Dataset Overview</h3>
                <p style="color: black;"><strong>Total Documents:</strong> {total_docs}</p>
                <p style="color: black;"><strong>Valid Topics:</strong> {total_topics}</p>
                <p style="color: black;"><strong>Outliers:</strong> {outliers} ({outliers/total_docs*100:.1f}%)</p>
                <p style="color: black;"><strong>Average Confidence:</strong> {avg_confidence:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if len(valid_topics) > 0:
                largest_topic = valid_topics['Count'].max()
                smallest_topic = valid_topics['Count'].min()
                topic_balance = valid_topics['Count'].std()
                
                st.markdown(f"""
                <div class="summary-box">
                    <h3 style="color: black; margin-bottom: 15px;">Topic Distribution</h3>
                    <p style="color: black;"><strong>Largest Topic:</strong> {largest_topic} documents</p>
                    <p style="color: black;"><strong>Smallest Topic:</strong> {smallest_topic} documents</p>
                    <p style="color: black;"><strong>Balance (std):</strong> {topic_balance:.1f}</p>
                    <p style="color: black;"><strong>Avg Docs/Topic:</strong> {total_docs/total_topics:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown('<h2 class="section-title">Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'topic_info' in results:
            total_topics = len(results['topic_info'][results['topic_info']['Topic'] != -1])
            display_metric("Topics", total_topics, "excluding outliers")
    
    with col2:
        if 'document_assignments' in results:
            total_docs = len(results['document_assignments'])
            display_metric("Documents", total_docs, "processed")
    
    with col3:
        if 'document_assignments' in results:
            avg_confidence = results['document_assignments']['confidence'].mean()
            display_metric("Confidence", f"{avg_confidence:.3f}", "average")
    
    with col4:
        if 'topic_info' in results and 'document_assignments' in results:
            valid_topics = results['topic_info'][results['topic_info']['Topic'] != -1]
            docs_per_topic = len(results['document_assignments']) / len(valid_topics)
            display_metric("Docs/Topic", f"{docs_per_topic:.1f}", "average")
    
    # Model Quality
    st.markdown('<h2 class="section-title">Model Quality</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'topic_info' in results:
            outliers = results['topic_info'][results['topic_info']['Topic'] == -1]['Count'].iloc[0] if -1 in results['topic_info']['Topic'].values else 0
            outlier_percentage = (outliers / len(results['document_assignments'])) * 100
            display_metric("Outliers", f"{outliers}", f"{outlier_percentage:.1f}%")
    
    with col2:
        if 'topic_info' in results:
            valid_topics = results['topic_info'][results['topic_info']['Topic'] != -1]
            topic_balance = valid_topics['Count'].std()
            display_metric("Balance", f"{topic_balance:.1f}", "std dev")
    
    with col3:
        if 'topic_info' in results:
            largest_topic = results['topic_info'][results['topic_info']['Topic'] != -1]['Count'].max()
            display_metric("Largest", largest_topic, "documents")
    
    # Topic Distribution
    st.markdown('<h2 class="section-title">Topic Distribution</h2>', unsafe_allow_html=True)
    
    if 'topic_info' in results:
        fig = create_simple_topic_chart(results['topic_info'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Topic Analysis
    st.markdown('<h2 class="section-title">Topic Analysis</h2>', unsafe_allow_html=True)
    
    if 'topics' in results:
        topic_ids = [tid for tid in results['topics'].keys() if tid != "-1"]
        
        # Add search functionality
        search_term = st.text_input("Search topics by keywords:", placeholder="Enter keyword to filter topics...")
        
        # Filter topics based on search
        filtered_topics = []
        for topic_id in sorted(topic_ids, key=int):
            topic_words = results['topics'][topic_id]
            keywords_text = ' '.join([word.lower() for word, _ in topic_words[:8]])
            
            if not search_term or search_term.lower() in keywords_text:
                filtered_topics.append(topic_id)
        
        if search_term and len(filtered_topics) == 0:
            st.markdown("<p style='color: black; font-weight: 500; background-color: #f8f8f8; padding: 10px; border-radius: 5px; border-left: 3px solid black;'>No topics found matching your search term.</p>", unsafe_allow_html=True)
        
        # Display topic count
        st.markdown(f"<p style='color: white; font-style: italic; font-weight: 500; background-color: black; padding: 5px 10px; border-radius: 5px; display: inline-block;'>Showing {len(filtered_topics)} of {len(topic_ids)} topics</p>", unsafe_allow_html=True)
        
        for topic_id in filtered_topics:
            topic_words = results['topics'][topic_id]
            topic_count = results['topic_info'][results['topic_info']['Topic'] == int(topic_id)]['Count'].iloc[0]
            
            # Calculate topic percentage
            total_docs = len(results['document_assignments'])
            topic_percentage = (topic_count / total_docs) * 100
            
            st.markdown(f"""
            <div class="topic-box">
                <div class="topic-title">Topic {topic_id}</div>
                <div class="topic-count">{topic_count} documents ({topic_percentage:.1f}% of total)</div>
                <div class="topic-keywords">Keywords: {', '.join([word for word, _ in topic_words[:8]])}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Document Clustering
    st.markdown('<h2 class="section-title">Document Clustering</h2>', unsafe_allow_html=True)
    
    if 'document_assignments' in results and 'embeddings' in results:
        fig = create_simple_clustering_chart(results['document_assignments'], results['embeddings'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Document Assignments
    st.markdown('<h2 class="section-title">Document Assignments</h2>', unsafe_allow_html=True)
    
    if 'document_assignments' in results:
        display_df = results['document_assignments'].copy()
        display_df['text_preview'] = display_df['text_preview'].str[:100] + "..."
        display_df = display_df.rename(columns={
            'document_id': 'Doc ID',
            'topic': 'Topic',
            'confidence': 'Confidence',
            'text_preview': 'Preview'
        })
        
        # Add filtering options
        col1, col2 = st.columns(2)
        
        with col1:
            topic_filter = st.selectbox(
                "Filter by Topic:",
                ["All Topics"] + sorted([str(t) for t in display_df['Topic'].unique() if t != -1])
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Show only documents with confidence above this threshold"
            )
        
        # Apply filters
        filtered_df = display_df.copy()
        if topic_filter != "All Topics":
            filtered_df = filtered_df[filtered_df['Topic'] == int(topic_filter)]
        
        filtered_df = filtered_df[filtered_df['Confidence'] >= confidence_threshold]
        
        st.markdown(f"<p style='color: white; font-style: italic; font-weight: 500; background-color: black; padding: 5px 10px; border-radius: 5px; display: inline-block;'>Showing {len(filtered_df)} of {len(display_df)} documents</p>", unsafe_allow_html=True)
        
        st.dataframe(
            filtered_df, 
            use_container_width=True, 
            height=400,
            column_config={
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    help="Clustering confidence score",
                    format="%.3f"
                )
            }
        )
    
    # Export
    st.markdown('<h2 class="section-title">Export Results</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8f8f8; padding: 20px; border-radius: 8px; border: 2px solid black;">
        <h3 style="color: black; margin-bottom: 15px;">Download Analysis Results</h3>
        <p style="color: black; margin-bottom: 20px;">Export your clustering results for further analysis or reporting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'topic_info' in results:
            csv = results['topic_info'].to_csv(index=False)
            st.download_button(
                label="Topic Information",
                data=csv,
                file_name="improved_topic_info.csv",
                mime="text/csv",
                help="Download detailed topic information including counts and keywords"
            )
    
    with col2:
        if 'document_assignments' in results:
            csv = results['document_assignments'].to_csv(index=False)
            st.download_button(
                label="Document Assignments",
                data=csv,
                file_name="improved_document_assignments.csv",
                mime="text/csv",
                help="Download document-to-topic assignments with confidence scores"
            )
    
    with col3:
        if 'topics' in results:
            # Create a summary report
            summary_data = []
            for topic_id in results['topics'].keys():
                if topic_id != "-1":
                    topic_words = results['topics'][topic_id]
                    topic_count = results['topic_info'][results['topic_info']['Topic'] == int(topic_id)]['Count'].iloc[0]
                    keywords = ', '.join([word for word, _ in topic_words[:5]])
                    summary_data.append({
                        'Topic_ID': topic_id,
                        'Document_Count': topic_count,
                        'Top_Keywords': keywords
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Summary Report",
                    data=csv,
                    file_name="clustering_summary.csv",
                    mime="text/csv",
                    help="Download a summary report with topic counts and keywords"
                )

if __name__ == "__main__":
    main() 