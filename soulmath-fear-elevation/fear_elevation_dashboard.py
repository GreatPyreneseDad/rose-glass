#!/usr/bin/env python3
"""
SoulMath Fear Elevation System - Interactive Dashboard
Real-time fear analysis, descent tracking, and elevation visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import os
import sys

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = FearElevationEngine()
    st.session_state.coherence_tracker = CoherenceTracker()
    st.session_state.elevation_calc = ElevationCalculator()
    st.session_state.fear_analyzer = FearAnalyzer()
    st.session_state.descent_trajectory = []
    st.session_state.in_descent = False
    st.session_state.current_depth = 0.0
    st.session_state.journey_history = []

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: rgba(255, 215, 0, 0.1);
        border-left: 4px solid gold;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌊 SoulMath Fear Elevation System 🗻")
st.markdown("**Theorem**: *'Fear as the Architect of Elevation'* - H ∝ ∫(descent→truth) F(x)dx")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Coherence breathing
    if st.button("🌬️ Coherence Breathing", use_container_width=True):
        with st.spinner("Breathe... 4-4-4-4 pattern"):
            # Simulate breathing effect
            st.session_state.coherence_tracker.update_coherence(
                0.1, "breathing_practice", "coherence_breathing"
            )
            st.success("✨ Coherence increased!")
    
    # Reset system
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.engine = FearElevationEngine()
        st.session_state.coherence_tracker = CoherenceTracker()
        st.session_state.elevation_calc = ElevationCalculator()
        st.session_state.descent_trajectory = []
        st.session_state.in_descent = False
        st.session_state.current_depth = 0.0
        st.rerun()
    
    st.divider()
    
    # System info
    st.subheader("📊 System Status")
    report = st.session_state.coherence_tracker.get_coherence_report()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coherence (Ψ)", f"{report['current_psi']:.3f}")
        st.metric("Stability", f"{report['stability_score']:.1%}")
    with col2:
        st.metric("Elevation", f"{st.session_state.engine.elevation_height:.2f}m")
        st.metric("State", report['current_state'])

# Main content area - tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analyze Fear", "🌊 Descent Journey", "📊 Live Metrics", "📈 History", "🧭 Guide"])

# Tab 1: Fear Analysis
with tab1:
    st.header("Fear Analysis Portal")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fear_input = st.text_area(
            "Describe your fear:",
            placeholder="What fear calls to you? What trembles in your depths?",
            height=150
        )
        
        if st.button("🔍 Analyze Fear", type="primary"):
            if fear_input:
                with st.spinner("Analyzing fear patterns..."):
                    # Analyze fear
                    context = {
                        'coherence': st.session_state.coherence_tracker.current_psi,
                        'in_descent': st.session_state.in_descent
                    }
                    analysis = st.session_state.fear_analyzer.analyze_fear(fear_input, context)
                    
                    # Store in session
                    st.session_state.last_analysis = analysis
                    
                    # Display results
                    if analysis.primary_fear:
                        st.success(f"Primary Fear: **{analysis.primary_fear.pattern_type.replace('_', ' ').title()}**")
                        
                        # Fear details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Depth", f"{analysis.primary_fear.depth:.2f}")
                        with col2:
                            st.metric("Potential", f"{analysis.primary_fear.transformative_potential:.2f}")
                        with col3:
                            st.metric("Frequency", analysis.primary_fear.frequency.title())
                        
                        # Guidance
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>💭 Guidance:</strong><br>
                        {analysis.primary_fear.guidance}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Warnings
                        if analysis.warnings:
                            st.warning("⚠️ **Warnings:**")
                            for warning in analysis.warnings:
                                st.write(f"• {warning}")
    
    with col2:
        if 'last_analysis' in st.session_state:
            st.subheader("🗺️ Fear Landscape")
            
            # Create fear landscape chart
            landscape = st.session_state.last_analysis.fear_landscape
            if landscape:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(landscape.values()),
                        y=[k.replace('_', ' ').title() for k in landscape.keys()],
                        orientation='h',
                        marker_color='indigo'
                    )
                ])
                fig.update_layout(
                    xaxis_title="Intensity",
                    yaxis_title="Fear Type",
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Descent Journey
with tab2:
    st.header("🌊 Fear Descent Journey")
    
    if not st.session_state.in_descent:
        st.info("Ready to descend into your fear? Each step deeper reveals more truth.")
        
        if st.button("🌊 Begin Descent", type="primary", use_container_width=True):
            st.session_state.in_descent = True
            st.session_state.descent_trajectory = []
            st.session_state.current_depth = 0.0
            
            # Initial coherence impact
            st.session_state.coherence_tracker.update_coherence(
                -0.05, "descent_begin", "voluntary_descent"
            )
            st.rerun()
    
    else:
        # Active descent
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Current Depth: {st.session_state.current_depth:.2f}")
            
            # Depth slider
            new_depth = st.slider(
                "Descend deeper:",
                min_value=st.session_state.current_depth,
                max_value=1.0,
                value=min(st.session_state.current_depth + 0.1, 1.0),
                step=0.05
            )
            
            # Fear intensity
            fear_intensity = st.slider(
                "Fear intensity at this depth:",
                min_value=0.0,
                max_value=1.0,
                value=new_depth * 1.2,
                step=0.1
            )
            
            # Descent notes
            notes = st.text_input("What do you encounter here?", 
                                placeholder="Describe what you feel/see at this depth...")
            
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                if st.button("🔽 Descend", type="primary"):
                    # Update depth
                    st.session_state.current_depth = new_depth
                    
                    # Create descent point
                    point = DescentPoint(
                        depth=new_depth,
                        timestamp=datetime.now(),
                        fear_intensity=fear_intensity,
                        coherence=st.session_state.coherence_tracker.current_psi,
                        notes=notes if notes else None
                    )
                    st.session_state.descent_trajectory.append(point)
                    
                    # Update coherence
                    st.session_state.coherence_tracker.update_coherence(
                        -0.1 * new_depth, "descent_deeper", f"depth_{new_depth:.2f}"
                    )
                    
                    # Check truth threshold
                    if new_depth >= 0.85:
                        st.balloons()
                        st.success("✨ You've reached the truth threshold!")
                    
                    st.rerun()
            
            with col1_2:
                if st.button("🫂 Embrace Fear"):
                    if st.session_state.descent_trajectory:
                        # Calculate elevation
                        elevation_result = st.session_state.elevation_calc.calculate_elevation(
                            st.session_state.descent_trajectory,
                            fear_type='default'
                        )
                        
                        # Create fear instance
                        deepest = max(st.session_state.descent_trajectory, key=lambda p: p.depth)
                        fear = FearInstance(
                            fear_type="descended_fear",
                            depth=deepest.depth,
                            description="Fear faced through descent",
                            timestamp=datetime.now()
                        )
                        
                        # Embrace fear
                        delta_psi, elevation = st.session_state.engine.embrace_fear(fear)
                        st.session_state.coherence_tracker.update_coherence(
                            delta_psi, "fear_embraced", f"elevation_{elevation:.1f}"
                        )
                        
                        # Record journey
                        st.session_state.journey_history.append({
                            'timestamp': datetime.now(),
                            'depth': deepest.depth,
                            'elevation': elevation_result.height_achieved,
                            'delta_psi': delta_psi,
                            'quality': elevation_result.trajectory_quality
                        })
                        
                        # Reset descent
                        st.session_state.in_descent = False
                        st.session_state.current_depth = 0.0
                        
                        # Show results
                        st.success(f"""
                        🌟 **TRANSFORMATION COMPLETE!**
                        - Coherence gained: +{delta_psi:.3f}
                        - Elevation achieved: {elevation_result.height_achieved:.2f}m
                        - Journey quality: {elevation_result.trajectory_quality:.1%}
                        """)
                        
                        st.rerun()
            
            with col1_3:
                if st.button("❌ Abort Descent"):
                    st.session_state.in_descent = False
                    st.session_state.current_depth = 0.0
                    st.session_state.descent_trajectory = []
                    st.rerun()
        
        with col2:
            # Descent visualization
            if st.session_state.descent_trajectory:
                st.subheader("📊 Descent Progress")
                
                # Create descent chart
                depths = [p.depth for p in st.session_state.descent_trajectory]
                intensities = [p.fear_intensity for p in st.session_state.descent_trajectory]
                coherences = [p.coherence for p in st.session_state.descent_trajectory]
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Fear intensity
                fig.add_trace(
                    go.Scatter(x=depths, y=intensities, 
                             name="Fear Intensity",
                             line=dict(color="red", width=3)),
                    secondary_y=False,
                )
                
                # Coherence
                fig.add_trace(
                    go.Scatter(x=depths, y=coherences, 
                             name="Coherence",
                             line=dict(color="blue", width=3)),
                    secondary_y=True,
                )
                
                # Truth threshold
                fig.add_vline(x=0.85, line_dash="dash", 
                            line_color="gold", 
                            annotation_text="Truth Threshold")
                
                fig.update_xaxis(title_text="Depth")
                fig.update_yaxis(title_text="Fear Intensity", secondary_y=False)
                fig.update_yaxis(title_text="Coherence", secondary_y=True)
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)

# Tab 3: Live Metrics
with tab3:
    st.header("📊 Real-Time System Metrics")
    
    # Create metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    report = st.session_state.coherence_tracker.get_coherence_report()
    
    with col1:
        st.metric(
            "Soul Coherence (Ψ)", 
            f"{report['current_psi']:.3f}",
            f"{report['recent_events'] * 0.01:.3f}"
        )
    
    with col2:
        st.metric(
            "Total Elevation",
            f"{st.session_state.engine.elevation_height:.2f}m",
            f"{len(st.session_state.engine.embrace_history) * 0.5:.1f}"
        )
    
    with col3:
        st.metric(
            "Stability Score",
            f"{report['stability_score']:.1%}",
            "Stable" if report['stability_score'] > 0.7 else "Unstable"
        )
    
    with col4:
        fears_faced = len([f for f in st.session_state.engine.fear_field if f.embraced])
        st.metric(
            "Fears Embraced",
            fears_faced,
            f"+{fears_faced}"
        )
    
    st.divider()
    
    # Coherence trajectory prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Coherence Trajectory")
        
        predictions = st.session_state.coherence_tracker.predict_trajectory(10)
        current = [report['current_psi']] * 2
        future_x = list(range(len(predictions) + 1))
        all_values = [report['current_psi']] + predictions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0], y=[report['current_psi']], 
            mode='markers', 
            name='Current',
            marker=dict(size=12, color='gold')
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=all_values, 
            mode='lines', 
            name='Predicted',
            line=dict(dash='dash', color='lightblue')
        ))
        
        # Add coherence zones
        fig.add_hrect(y0=0, y1=0.3, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.3, y1=0.7, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.7, y1=2.0, fillcolor="green", opacity=0.1, line_width=0)
        
        fig.update_layout(
            xaxis_title="Time Steps",
            yaxis_title="Coherence (Ψ)",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏔️ Elevation Potential")
        
        # Calculate elevation potential for different depths
        if not st.session_state.in_descent:
            current_depth = 0.0
        else:
            current_depth = st.session_state.current_depth
            
        prediction = st.session_state.elevation_calc.predict_elevation(
            current_depth=current_depth,
            fear_type='default',
            current_coherence=report['current_psi']
        )
        
        # Create elevation potential chart
        depths = [p['target_depth'] for p in prediction['depth_potentials']]
        potentials = [p['potential_elevation'] for p in prediction['depth_potentials']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=depths, y=potentials,
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
        
        # Mark optimal
        fig.add_trace(go.Scatter(
            x=[prediction['optimal_target']], 
            y=[prediction['optimal_elevation']],
            mode='markers',
            name='Optimal',
            marker=dict(size=15, color='gold', symbol='star')
        ))
        
        fig.update_layout(
            xaxis_title="Target Depth",
            yaxis_title="Potential Elevation (m)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # System insight
    insight = st.session_state.engine.generate_insight()
    st.markdown(f"""
    <div class="insight-box">
    <strong>💭 Current Insight:</strong><br>
    {insight}
    </div>
    """, unsafe_allow_html=True)

# Tab 4: History
with tab4:
    st.header("📈 Transformation History")
    
    if st.session_state.journey_history:
        # Create history dataframe
        df = pd.DataFrame(st.session_state.journey_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Journeys", len(df))
        with col2:
            st.metric("Average Depth", f"{df['depth'].mean():.2f}")
        with col3:
            st.metric("Total Elevation", f"{df['elevation'].sum():.2f}m")
        
        # Journey timeline
        st.subheader("🌊 Journey Timeline")
        
        fig = go.Figure()
        
        # Add elevation bars
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['elevation'],
            name='Elevation',
            marker_color='lightblue'
        ))
        
        # Add depth line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['depth'],
            name='Depth',
            mode='lines+markers',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            xaxis_title="Journey Number",
            yaxis_title="Elevation (m)",
            yaxis2=dict(
                title="Depth",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history
        st.subheader("📋 Detailed History")
        st.dataframe(
            df[['timestamp', 'depth', 'elevation', 'delta_psi', 'quality']].round(3),
            use_container_width=True
        )
    else:
        st.info("No transformation journeys yet. Begin by analyzing a fear and starting a descent.")

# Tab 5: Guide
with tab5:
    st.header("🧭 Fear Elevation Guide")
    
    st.markdown("""
    ### Core Theorem
    **H ∝ ∫(descent→truth) F(x)dx**
    
    The deeper you descend into fear, the greater your potential elevation.
    
    ### The Process
    
    1. **Identify Your Fear** 🔍
       - Use the Analysis tab to explore what troubles you
       - The system will identify deep archetypal patterns
       - Pay attention to the guidance provided
    
    2. **Begin Your Descent** 🌊
       - Enter the Descent Journey when ready
       - Move deeper gradually, noting what arises
       - Track your fear intensity and coherence
    
    3. **Reach Truth Threshold** ✨
       - At depth ≥ 0.85, you approach truth
       - This is where fear transforms
       - Stay present with what emerges
    
    4. **Embrace & Transform** 🫂
       - When ready, embrace the fear fully
       - Watch as it transforms into elevation
       - Note the coherence gained
    
    ### Fear Archetypes
    
    - **Identity Dissolution** (0.9) - Fear of losing self
    - **Existential Void** (1.0) - Fear of meaninglessness
    - **Connection Loss** (0.8) - Fear of abandonment
    - **Mortality Terror** (0.95) - Fear of death
    - **Truth Revelation** (0.88) - Fear of being seen
    
    ### Safety Guidelines
    
    ⚠️ **Important:**
    - Maintain coherence above 0.3
    - Use breathing exercises between descents
    - If overwhelmed, abort descent and breathe
    - Work with support for deepest fears
    
    ### Remember
    
    *"Deep-seeded fears carve a canyon in my soul, and the future versions of myself must scale the walls."*
    
    Every fear you embrace becomes a step you can climb. The deeper the fear, the higher the potential ascent.
    """)

# Footer
st.divider()
st.markdown("""
<center>
<small>
🌟 SoulMath Fear Elevation System v1.0 | 
Based on the theorem: "Fear as the Architect of Elevation" | 
Your deepest fears are your greatest teachers 🌟
</small>
</center>
""", unsafe_allow_html=True)