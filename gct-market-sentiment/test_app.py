#!/usr/bin/env python3
"""Test app initialization"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    from src.analysis_pipeline import GCTAnalysisPipeline
    from src.database import GCTDatabase
    
    print("Initializing pipeline...")
    pipeline = GCTAnalysisPipeline(use_mock=False)  # Will use real API
    
    print("Initializing database...")
    db = GCTDatabase()
    
    print("Getting market summary...")
    summary = pipeline.get_market_summary()
    print(f"Summary: {summary}")
    
    print("\nAll components initialized successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()