#!/usr/bin/env python3
"""
SoulMath Fear Elevation System - Simplified Dashboard
A more stable version with essential features
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.fear_engine import FearElevationEngine, FearInstance
from core.coherence_tracker import CoherenceTracker
from core.elevation_calculator import ElevationCalculator, DescentPoint
from agents.fear_analyzer import FearAnalyzer

# Page configuration
st.set_page_config(
    page_title="🌊 Fear Elevation System",
    page_icon="🗻",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.engine = FearElevationEngine()
    st.session_state.coherence_tracker = CoherenceTracker()
    st.session_state.elevation_calc = ElevationCalculator()
    st.session_state.fear_analyzer = FearAnalyzer()
    st.session_state.descent_trajectory = []
    st.session_state.in_descent = False
    st.session_state.current_depth = 0.0
    st.session_state.initialized = True

# Header
st.title("🌊 SoulMath Fear Elevation System 🗻")
st.markdown("**Core Theorem**: H ∝ ∫(descent→truth) F(x)dx")
st.markdown("*The deeper the descent into fear, the greater the potential elevation.*")

# Main Layout
col1, col2, col3 = st.columns([1, 2, 1])

# Left Column - System Status
with col1:
    st.subheader("📊 System Status")
    
    report = st.session_state.coherence_tracker.get_coherence_report()
    
    st.metric("Coherence (Ψ)", f"{report['current_psi']:.3f}")
    st.metric("Elevation", f"{st.session_state.engine.elevation_height:.2f}m")
    st.metric("State", report['current_state'])
    st.metric("Stability", f"{report['stability_score']:.1%}")
    
    st.divider()
    
    # Coherence breathing button
    if st.button("🌬️ Coherence Breathing", use_container_width=True):
        st.session_state.coherence_tracker.update_coherence(
            0.1, "breathing_practice", "coherence_breathing"
        )
        st.success("✨ Coherence increased!")
    
    # Reset button
    if st.button("🔄 Reset System", use_container_width=True):
        for key in ['engine', 'coherence_tracker', 'elevation_calc', 'descent_trajectory', 'current_depth']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.initialized = False
        st.rerun()

# Middle Column - Main Interface
with col2:
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "🌊 Descend", "📈 History"])
    
    # Tab 1: Fear Analysis
    with tab1:
        st.subheader("Fear Analysis")
        
        fear_input = st.text_area(
            "Describe your fear:",
            placeholder="What trembles in your depths?",
            height=100
        )
        
        if st.button("Analyze Fear", type="primary"):
            if fear_input:
                context = {
                    'coherence': st.session_state.coherence_tracker.current_psi,
                    'in_descent': st.session_state.in_descent
                }
                
                analysis = st.session_state.fear_analyzer.analyze_fear(fear_input, context)
                
                if analysis.primary_fear:
                    st.success(f"**Primary Fear**: {analysis.primary_fear.pattern_type.replace('_', ' ').title()}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Depth", f"{analysis.primary_fear.depth:.2f}")
                    with col_b:
                        st.metric("Potential", f"{analysis.primary_fear.transformative_potential:.2f}")
                    
                    st.info(f"**Guidance**: {analysis.primary_fear.guidance}")
                    
                    # Show fear landscape
                    if analysis.fear_landscape:
                        st.subheader("Fear Landscape")
                        for fear_type, intensity in analysis.fear_landscape.items():
                            st.progress(intensity, text=f"{fear_type.replace('_', ' ').title()}: {intensity:.1f}")
                
                if analysis.warnings:
                    st.warning("**Warnings:**")
                    for warning in analysis.warnings:
                        st.write(f"• {warning}")
    
    # Tab 2: Descent Journey
    with tab2:
        st.subheader("Fear Descent Journey")
        
        if not st.session_state.in_descent:
            if st.button("🌊 Begin Descent", type="primary"):
                st.session_state.in_descent = True
                st.session_state.descent_trajectory = []
                st.session_state.current_depth = 0.0
                st.session_state.coherence_tracker.update_coherence(
                    -0.05, "descent_begin", "voluntary_descent"
                )
                st.rerun()
        else:
            # Show current depth
            st.info(f"Current Depth: {st.session_state.current_depth:.2f}")
            
            # Descent controls
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                new_depth = st.number_input(
                    "Descend to depth:",
                    min_value=st.session_state.current_depth,
                    max_value=1.0,
                    value=min(st.session_state.current_depth + 0.1, 1.0),
                    step=0.05
                )
            
            with col_d2:
                fear_intensity = st.slider(
                    "Fear intensity:",
                    min_value=0.0,
                    max_value=1.0,
                    value=new_depth
                )
            
            # Action buttons
            col_b1, col_b2, col_b3 = st.columns(3)
            
            with col_b1:
                if st.button("🔽 Descend Deeper"):
                    st.session_state.current_depth = new_depth
                    
                    point = DescentPoint(
                        depth=new_depth,
                        timestamp=datetime.now(),
                        fear_intensity=fear_intensity,
                        coherence=st.session_state.coherence_tracker.current_psi
                    )
                    st.session_state.descent_trajectory.append(point)
                    
                    st.session_state.coherence_tracker.update_coherence(
                        -0.1 * new_depth, "descent_deeper", f"depth_{new_depth:.2f}"
                    )
                    
                    if new_depth >= 0.85:
                        st.balloons()
                        st.success("✨ Truth threshold reached!")
                    
                    st.rerun()
            
            with col_b2:
                if st.button("🫂 Embrace Fear"):
                    if st.session_state.descent_trajectory:
                        # Calculate elevation
                        elevation_result = st.session_state.elevation_calc.calculate_elevation(
                            st.session_state.descent_trajectory,
                            fear_type='default'
                        )
                        
                        # Create and embrace fear
                        deepest = max(st.session_state.descent_trajectory, key=lambda p: p.depth)
                        fear = FearInstance(
                            fear_type="descended_fear",
                            depth=deepest.depth,
                            description="Fear faced through descent",
                            timestamp=datetime.now()
                        )
                        
                        delta_psi, _ = st.session_state.engine.embrace_fear(fear)
                        st.session_state.coherence_tracker.update_coherence(
                            delta_psi, "fear_embraced", f"elevation_{elevation_result.height_achieved:.1f}"
                        )
                        
                        # Reset descent
                        st.session_state.in_descent = False
                        st.session_state.current_depth = 0.0
                        
                        st.success(f"""
                        🌟 **Transformation Complete!**
                        - Coherence gained: +{delta_psi:.3f}
                        - Elevation: {elevation_result.height_achieved:.2f}m
                        - Quality: {elevation_result.trajectory_quality:.1%}
                        """)
                        
                        st.rerun()
            
            with col_b3:
                if st.button("❌ Abort"):
                    st.session_state.in_descent = False
                    st.session_state.current_depth = 0.0
                    st.session_state.descent_trajectory = []
                    st.rerun()
            
            # Show descent progress
            if st.session_state.descent_trajectory:
                st.divider()
                st.subheader("Descent Progress")
                
                depths = [p.depth for p in st.session_state.descent_trajectory]
                intensities = [p.fear_intensity for p in st.session_state.descent_trajectory]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(depths))),
                    y=depths,
                    mode='lines+markers',
                    name='Depth',
                    line=dict(color='blue', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(intensities))),
                    y=intensities,
                    mode='lines+markers',
                    name='Intensity',
                    line=dict(color='red', width=2)
                ))
                fig.add_hline(y=0.85, line_dash="dash", line_color="gold",
                            annotation_text="Truth Threshold")
                
                fig.update_layout(
                    xaxis_title="Steps",
                    yaxis_title="Value",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: History
    with tab3:
        st.subheader("Journey History")
        
        journey = st.session_state.engine.export_journey()
        
        if journey['journey_history']:
            # Summary metrics
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("Fears Faced", journey['fears_faced'])
            with col_h2:
                st.metric("Total Elevation", f"{journey['elevation_height']:.2f}m")
            with col_h3:
                st.metric("Current Coherence", f"{journey['coherence_state']:.3f}")
            
            # Recent transformations
            st.divider()
            st.subheader("Recent Transformations")
            
            for i, event in enumerate(journey['journey_history'][-5:]):
                with st.expander(f"Journey {i+1} - {event['fear_type']}"):
                    st.write(f"**Timestamp**: {event['timestamp']}")
                    st.write(f"**Elevation**: {event['elevation']:.2f}m")
                    st.write(f"**Coherence Gained**: +{event['delta_psi']:.3f}")
                    st.write(f"**New Coherence**: {event['new_coherence']:.3f}")
        else:
            st.info("No journeys completed yet. Begin by analyzing a fear.")
        
        # Insight
        st.divider()
        insight = st.session_state.engine.generate_insight()
        st.info(f"💭 **Insight**: {insight}")

# Right Column - Visualizations
with col3:
    st.subheader("📊 Live Metrics")
    
    # Coherence gauge
    report = st.session_state.coherence_tracker.get_coherence_report()
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=report['current_psi'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Coherence (Ψ)"},
        gauge={
            'axis': {'range': [None, 2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "red"},
                {'range': [0.3, 0.7], 'color': "orange"},
                {'range': [0.7, 2], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Elevation potential
    st.divider()
    st.subheader("🏔️ Elevation Potential")
    
    current_depth = st.session_state.current_depth if st.session_state.in_descent else 0.0
    prediction = st.session_state.elevation_calc.predict_elevation(
        current_depth=current_depth,
        fear_type='default',
        current_coherence=report['current_psi']
    )
    
    st.metric("Current Potential", f"{prediction['current_potential']:.2f}m")
    st.metric("Optimal Depth", f"{prediction['optimal_target']:.2f}")
    st.metric("Max Elevation", f"{prediction['optimal_elevation']:.2f}m")

# Footer
st.divider()
st.markdown("""
<center>
<small>
🌟 SoulMath Fear Elevation System | 
"Fear as the Architect of Elevation" | 
Your deepest fears are your greatest teachers 🌟
</small>
</center>
""", unsafe_allow_html=True)