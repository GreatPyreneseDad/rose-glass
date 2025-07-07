#!/usr/bin/env python3
"""
SoulMath Fear Elevation System - AI-Enhanced Dashboard
Integrates Ollama for intelligent fear detection in written communication
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
from agents.ai_fear_detector import AIFearDetector, DetectedFear

# Page configuration
st.set_page_config(
    page_title="🤖 AI Fear Elevation System",
    page_icon="🗻",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.engine = FearElevationEngine()
    st.session_state.coherence_tracker = CoherenceTracker()
    st.session_state.elevation_calc = ElevationCalculator()
    st.session_state.fear_analyzer = FearAnalyzer()
    st.session_state.ai_detector = AIFearDetector()
    st.session_state.descent_trajectory = []
    st.session_state.in_descent = False
    st.session_state.current_depth = 0.0
    st.session_state.ai_analyses = []
    st.session_state.initialized = True

# Header
st.title("🤖 AI-Powered Fear Elevation System 🗻")
st.markdown("**Enhanced with Ollama AI** - Detects hidden fears in dialogue, monologue, and written communication")

# Sidebar
with st.sidebar:
    st.header("🤖 AI Settings")
    
    # Model selection
    model = st.selectbox(
        "AI Model",
        ["llama3.2:latest", "llama3.2:3b"],
        help="Select the Ollama model for analysis"
    )
    st.session_state.ai_detector.model = model
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    report = st.session_state.coherence_tracker.get_coherence_report()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coherence (Ψ)", f"{report['current_psi']:.3f}")
        st.metric("Stability", f"{report['stability_score']:.1%}")
    with col2:
        st.metric("Elevation", f"{st.session_state.engine.elevation_height:.2f}m")
        st.metric("AI Analyses", len(st.session_state.ai_analyses))
    
    st.divider()
    
    # Actions
    if st.button("🌬️ Coherence Breathing", use_container_width=True):
        st.session_state.coherence_tracker.update_coherence(
            0.1, "breathing_practice", "coherence_breathing"
        )
        st.success("✨ Coherence increased!")
    
    if st.button("🔄 Reset System", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != 'initialized':
                del st.session_state[key]
        st.session_state.initialized = False
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Analysis", 
    "💬 Dialogue Analysis", 
    "🌊 Fear Descent", 
    "📊 Insights", 
    "📚 History"
])

# Tab 1: AI Text Analysis
with tab1:
    st.header("AI-Powered Fear Detection")
    st.markdown("Paste any text - email, journal entry, conversation transcript, etc.")
    
    # Input area
    text_input = st.text_area(
        "Text to analyze:",
        placeholder="Paste your text here... The AI will detect underlying fears even if not explicitly stated.",
        height=200
    )
    
    # Analysis type
    col1, col2 = st.columns([3, 1])
    with col1:
        analysis_type = st.radio(
            "Text type:",
            ["Monologue/Writing", "Dialogue/Conversation"],
            horizontal=True
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze with AI", type="primary", use_container_width=True)
    
    # Perform analysis
    if analyze_button and text_input:
        with st.spinner("🤖 AI is analyzing for hidden fears..."):
            # Get AI analysis
            is_dialogue = analysis_type == "Dialogue/Conversation"
            ai_analysis = st.session_state.ai_detector.analyze_text(text_input, is_dialogue)
            
            # Store analysis
            st.session_state.ai_analyses.append({
                'timestamp': datetime.now(),
                'text': text_input[:100] + "...",
                'analysis': ai_analysis
            })
            
            # Display results
            st.success(f"Found {len(ai_analysis.detected_fears)} underlying fears")
            
            # Emotional tone
            st.info(f"**Emotional Tone**: {ai_analysis.emotional_tone}")
            
            # Detected fears
            if ai_analysis.detected_fears:
                st.subheader("🎯 Detected Fears")
                
                for i, fear in enumerate(ai_analysis.detected_fears):
                    with st.expander(f"{i+1}. {fear.fear_type} (Confidence: {fear.confidence:.0%})"):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.write("**Evidence from text:**")
                            for evidence in fear.evidence:
                                st.write(f"• *\"{evidence}\"*")
                            
                            st.write(f"\n**Context**: {fear.context}")
                        
                        with col_b:
                            st.metric("Depth Estimate", f"{fear.depth_estimate:.2f}")
                            st.metric("Confidence", f"{fear.confidence:.0%}")
                            
                            # Get exploration suggestions
                            if st.button(f"Get Suggestions", key=f"suggest_{i}"):
                                suggestions = st.session_state.ai_detector.suggest_fear_exploration(fear)
                                st.write("**Questions to explore:**")
                                for q in suggestions['questions']:
                                    st.write(f"• {q}")
                                st.write(f"\n**Exercise**: {suggestions['exercise']}")
                                st.write(f"\n**Affirmation**: {suggestions['affirmation']}")
            
            # Fear indicators
            if ai_analysis.fear_indicators:
                st.subheader("🔍 Fear Indicators Found")
                indicators_text = " • ".join(ai_analysis.fear_indicators)
                st.write(indicators_text)
            
            # Suggested approach
            if ai_analysis.suggested_approach:
                st.subheader("💡 AI's Suggested Approach")
                st.write(ai_analysis.suggested_approach)
            
            # Option to process detected fears
            if ai_analysis.detected_fears:
                st.divider()
                selected_fear = st.selectbox(
                    "Select a fear to work with:",
                    [f"{f.fear_type} (depth: {f.depth_estimate:.2f})" for f in ai_analysis.detected_fears]
                )
                
                if st.button("🌊 Begin Descent with This Fear"):
                    # Parse selection
                    fear_idx = [f"{f.fear_type} (depth: {f.depth_estimate:.2f})" for f in ai_analysis.detected_fears].index(selected_fear)
                    selected = ai_analysis.detected_fears[fear_idx]
                    
                    # Start descent
                    st.session_state.in_descent = True
                    st.session_state.current_depth = 0.0
                    st.session_state.current_fear = selected
                    st.session_state.descent_trajectory = []
                    
                    # Initial coherence impact
                    st.session_state.coherence_tracker.update_coherence(
                        -0.05, "ai_guided_descent", selected.fear_type
                    )
                    
                    st.success(f"Beginning descent into {selected.fear_type}")
                    st.info("Switch to the Fear Descent tab to continue your journey")

# Tab 2: Dialogue Analysis
with tab2:
    st.header("💬 Conversation Fear Analysis")
    st.markdown("Analyze dialogue to uncover fear dynamics between speakers")
    
    # Dialogue input method
    input_method = st.radio(
        "Input method:",
        ["Simple Format", "Speaker Turns"],
        horizontal=True
    )
    
    if input_method == "Simple Format":
        dialogue_text = st.text_area(
            "Paste dialogue (use 'Name: text' format):",
            placeholder="Alice: I don't think I can do this presentation.\nBob: Why not? You're always so prepared.\nAlice: What if everyone realizes I don't know what I'm talking about?",
            height=200
        )
        
        if st.button("🎭 Analyze Dialogue Dynamics", type="primary"):
            if dialogue_text:
                # Parse dialogue
                dialogue_turns = []
                for line in dialogue_text.split('\n'):
                    if ':' in line:
                        speaker, text = line.split(':', 1)
                        dialogue_turns.append({
                            'speaker': speaker.strip(),
                            'text': text.strip()
                        })
                
                if dialogue_turns:
                    with st.spinner("🤖 Analyzing conversation dynamics..."):
                        # Analyze
                        dynamics = st.session_state.ai_detector.analyze_conversation_dynamics(dialogue_turns)
                        
                        # Display results
                        st.success(f"Analysis complete - {dynamics['speaker_count']} speakers detected")
                        
                        # Show main analysis
                        analysis = dynamics['analysis']
                        
                        if analysis.detected_fears:
                            st.subheader("🎯 Fears Detected in Conversation")
                            
                            for fear in analysis.detected_fears:
                                st.write(f"**{fear.fear_type}** (confidence: {fear.confidence:.0%})")
                                st.write(f"Evidence: {', '.join(fear.evidence[:2])}")
                        
                        # Show dynamics
                        st.subheader("🔄 Conversation Dynamics")
                        st.write(dynamics['dynamics'])
    
    else:  # Speaker Turns
        st.write("Add conversation turns:")
        
        if 'dialogue_turns' not in st.session_state:
            st.session_state.dialogue_turns = []
        
        # Add turn interface
        col1, col2 = st.columns([1, 4])
        with col1:
            speaker = st.text_input("Speaker:", key="speaker_input")
        with col2:
            text = st.text_input("Text:", key="text_input")
        
        if st.button("➕ Add Turn"):
            if speaker and text:
                st.session_state.dialogue_turns.append({
                    'speaker': speaker,
                    'text': text
                })
                st.success(f"Added turn from {speaker}")
        
        # Display turns
        if st.session_state.dialogue_turns:
            st.subheader("Conversation:")
            for i, turn in enumerate(st.session_state.dialogue_turns):
                st.write(f"**{turn['speaker']}**: {turn['text']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎭 Analyze This Dialogue", type="primary"):
                    with st.spinner("🤖 Analyzing dynamics..."):
                        dynamics = st.session_state.ai_detector.analyze_conversation_dynamics(
                            st.session_state.dialogue_turns
                        )
                        
                        # Display analysis
                        st.write("Analysis:", dynamics)
            
            with col2:
                if st.button("🗑️ Clear Dialogue"):
                    st.session_state.dialogue_turns = []
                    st.rerun()

# Tab 3: Fear Descent (Enhanced with AI insights)
with tab3:
    st.header("🌊 AI-Guided Fear Descent")
    
    if not st.session_state.in_descent:
        st.info("Analyze text in the AI Analysis tab first, then select a fear to begin descent")
        
        # Quick start option
        if st.button("🚀 Quick Start with Manual Fear"):
            st.session_state.in_descent = True
            st.session_state.current_depth = 0.0
            st.session_state.descent_trajectory = []
            st.session_state.current_fear = None
            st.rerun()
    
    else:
        # Show current fear if AI-detected
        if hasattr(st.session_state, 'current_fear') and st.session_state.current_fear:
            fear = st.session_state.current_fear
            st.info(f"Descending into: **{fear.fear_type}** (AI confidence: {fear.confidence:.0%})")
            st.write(f"*{fear.context}*")
        
        # Descent interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Current Depth: {st.session_state.current_depth:.2f}")
            
            # AI guidance based on depth
            if st.session_state.current_depth < 0.3:
                guidance = "Surface level - Notice physical sensations and immediate emotions"
            elif st.session_state.current_depth < 0.6:
                guidance = "Mid-level - Explore the stories and beliefs around this fear"
            elif st.session_state.current_depth < 0.85:
                guidance = "Deep level - Touch the core wound beneath the fear"
            else:
                guidance = "Truth threshold - What wisdom does this fear hold?"
            
            st.write(f"**AI Guidance**: {guidance}")
            
            # Descent controls
            new_depth = st.slider(
                "Descend to:",
                min_value=st.session_state.current_depth,
                max_value=1.0,
                value=min(st.session_state.current_depth + 0.1, 1.0),
                step=0.05
            )
            
            fear_intensity = st.slider(
                "Fear intensity:",
                0.0, 1.0,
                value=new_depth * 1.1
            )
            
            experience = st.text_area(
                "What are you experiencing at this depth?",
                placeholder="Describe sensations, emotions, images, or insights...",
                height=100
            )
        
        with col2:
            # Coherence monitor
            report = st.session_state.coherence_tracker.get_coherence_report()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=report['current_psi'],
                title={'text': "Coherence"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "red"},
                        {'range': [0.3, 0.7], 'color': "orange"},
                        {'range': [0.7, 2], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔽 Descend", type="primary"):
                st.session_state.current_depth = new_depth
                
                point = DescentPoint(
                    depth=new_depth,
                    timestamp=datetime.now(),
                    fear_intensity=fear_intensity,
                    coherence=st.session_state.coherence_tracker.current_psi,
                    notes=experience
                )
                st.session_state.descent_trajectory.append(point)
                
                st.session_state.coherence_tracker.update_coherence(
                    -0.1 * new_depth, "descent_deeper", f"depth_{new_depth:.2f}"
                )
                
                if new_depth >= 0.85:
                    st.balloons()
                    st.success("✨ Truth threshold reached!")
                
                st.rerun()
        
        with col2:
            if st.button("🫂 Embrace Fear"):
                if st.session_state.descent_trajectory:
                    elevation_result = st.session_state.elevation_calc.calculate_elevation(
                        st.session_state.descent_trajectory
                    )
                    
                    deepest = max(st.session_state.descent_trajectory, key=lambda p: p.depth)
                    fear = FearInstance(
                        fear_type="ai_detected" if hasattr(st.session_state, 'current_fear') else "manual",
                        depth=deepest.depth,
                        description=experience if experience else "Fear embraced",
                        timestamp=datetime.now()
                    )
                    
                    delta_psi, _ = st.session_state.engine.embrace_fear(fear)
                    st.session_state.coherence_tracker.update_coherence(
                        delta_psi, "fear_embraced", f"elevation_{elevation_result.height_achieved:.1f}"
                    )
                    
                    st.session_state.in_descent = False
                    st.session_state.current_depth = 0.0
                    
                    st.success(f"""
                    🌟 **Transformation Complete!**
                    - Coherence gained: +{delta_psi:.3f}
                    - Elevation: {elevation_result.height_achieved:.2f}m
                    - Quality: {elevation_result.trajectory_quality:.1%}
                    """)
                    
                    st.rerun()
        
        with col3:
            if st.button("❌ Surface"):
                st.session_state.in_descent = False
                st.session_state.current_depth = 0.0
                st.rerun()
        
        # Descent visualization
        if st.session_state.descent_trajectory:
            st.divider()
            
            # Create descent journey plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Descent Journey", "Fear Intensity vs Coherence"),
                row_heights=[0.6, 0.4]
            )
            
            depths = [p.depth for p in st.session_state.descent_trajectory]
            intensities = [p.fear_intensity for p in st.session_state.descent_trajectory]
            coherences = [p.coherence for p in st.session_state.descent_trajectory]
            
            # Descent path
            fig.add_trace(
                go.Scatter(x=list(range(len(depths))), y=depths,
                          mode='lines+markers',
                          name='Depth',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Truth threshold
            fig.add_hline(y=0.85, line_dash="dash", line_color="gold",
                         annotation_text="Truth Threshold", row=1, col=1)
            
            # Fear vs Coherence
            fig.add_trace(
                go.Scatter(x=intensities, y=coherences,
                          mode='markers',
                          marker=dict(size=10, color=depths, colorscale='Viridis'),
                          name='Journey Points'),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Step", row=1, col=1)
            fig.update_yaxes(title_text="Depth", row=1, col=1)
            fig.update_xaxes(title_text="Fear Intensity", row=2, col=1)
            fig.update_yaxes(title_text="Coherence", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: AI Insights
with tab4:
    st.header("📊 AI-Generated Insights")
    
    if st.session_state.ai_analyses:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        total_fears = sum(len(a['analysis'].detected_fears) for a in st.session_state.ai_analyses)
        avg_confidence = np.mean([f.confidence for a in st.session_state.ai_analyses 
                                 for f in a['analysis'].detected_fears]) if total_fears > 0 else 0
        
        with col1:
            st.metric("Total Analyses", len(st.session_state.ai_analyses))
        with col2:
            st.metric("Fears Detected", total_fears)
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        
        # Fear type distribution
        st.subheader("🎯 Fear Type Distribution")
        
        fear_counts = {}
        for analysis in st.session_state.ai_analyses:
            for fear in analysis['analysis'].detected_fears:
                fear_counts[fear.fear_type] = fear_counts.get(fear.fear_type, 0) + 1
        
        if fear_counts:
            fig = px.pie(
                values=list(fear_counts.values()),
                names=list(fear_counts.keys()),
                title="Distribution of Detected Fear Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent analyses
        st.subheader("🕐 Recent AI Analyses")
        
        for i, analysis in enumerate(reversed(st.session_state.ai_analyses[-5:])):
            with st.expander(f"Analysis {len(st.session_state.ai_analyses)-i}: {analysis['text']}"):
                st.write(f"**Time**: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Emotional Tone**: {analysis['analysis'].emotional_tone}")
                
                if analysis['analysis'].detected_fears:
                    st.write("**Detected Fears**:")
                    for fear in analysis['analysis'].detected_fears:
                        st.write(f"- {fear.fear_type} ({fear.confidence:.0%})")
    
    else:
        st.info("No AI analyses yet. Start by analyzing some text in the AI Analysis tab.")

# Tab 5: History
with tab5:
    st.header("📚 Journey History")
    
    journey = st.session_state.engine.export_journey()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fears Faced", journey['fears_faced'])
    with col2:
        st.metric("Fears Embraced", journey['fears_embraced'])
    with col3:
        st.metric("Total Elevation", f"{journey['elevation_height']:.2f}m")
    with col4:
        st.metric("Coherence", f"{journey['coherence_state']:.3f}")
    
    if journey['journey_history']:
        st.divider()
        
        # Create journey timeline
        df = pd.DataFrame(journey['journey_history'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Elevation line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_elevation'],
            mode='lines+markers',
            name='Cumulative Elevation',
            line=dict(color='blue', width=3)
        ))
        
        # Coherence line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['new_coherence'],
            mode='lines+markers',
            name='Coherence',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Fear Elevation Journey Over Time",
            xaxis_title="Time",
            yaxis_title="Elevation (m)",
            yaxis2=dict(
                title="Coherence (Ψ)",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Wisdom
    st.divider()
    insight = st.session_state.engine.generate_insight()
    st.info(f"💭 **Current Wisdom**: {insight}")

# Footer
st.divider()
st.markdown("""
<center>
<small>
🤖 AI-Powered Fear Elevation System | 
Powered by Ollama & SoulMath | 
"Your hidden fears are your hidden wings" 🌟
</small>
</center>
""", unsafe_allow_html=True)