"""Minimal Streamlit app for testing"""

import streamlit as st
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.title("GCT Market Sentiment - Test")

try:
    from src.analysis_pipeline import GCTAnalysisPipeline
    from src.database import GCTDatabase
    
    st.success("Imports successful!")
    
    # Initialize
    with st.spinner("Initializing..."):
        pipeline = GCTAnalysisPipeline(use_mock=False)
        db = GCTDatabase()
    
    st.success("Pipeline initialized!")
    
    # Get summary
    summary = pipeline.get_market_summary()
    st.json(summary)
    
except Exception as e:
    st.error(f"Error: {e}")
    st.code(str(e))