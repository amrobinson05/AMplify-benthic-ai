import streamlit as st

def apply_metrics_styles():
    """Applies shared CSS styles for the metrics cards."""
    st.markdown("""
    <style>
    .metrics-card {
        text-align: center;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        padding: 2rem;
        margin-top: 2rem;
    }
    .metrics-card h2 {
        color: #0D47A1;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .metrics-card p {
        color: #04365C;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)
