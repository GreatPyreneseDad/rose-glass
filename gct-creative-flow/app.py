"""
GCT Creative Flow Analysis Dashboard
Real-time monitoring and optimization of creative processes
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import asyncio
import pandas as pd

# Import creative flow components
from src.creative_flow_engine import (
    CreativeFlowEngine, CreativeState, CreativeMetrics, 
    BiometricData, CreativeEnvironmentOptimizer
)
from src.collaborative_creativity import (
    CollaborativeCreativityOptimizer, TeamMember,
    CreativeProjectOrchestrator
)
from src.creative_visualizer import CreativeFlowVisualizer
from src.creative_database import CreativeFlowDatabase

# Import GCT components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../gct-market-sentiment/src'))
from gct_engine import GCTVariables


# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = CreativeFlowEngine()
    st.session_state.visualizer = CreativeFlowVisualizer()
    st.session_state.env_optimizer = CreativeEnvironmentOptimizer()
    st.session_state.collab_optimizer = CollaborativeCreativityOptimizer()
    st.session_state.database = CreativeFlowDatabase()
    st.session_state.session_id = None
    st.session_state.history = []
    st.session_state.is_monitoring = False
    st.session_state.biometric_simulation = True
    st.session_state.team_mode = False


def main():
    st.set_page_config(
        page_title="GCT Creative Flow Analysis",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 GCT Creative Flow Analysis")
    st.markdown("*Applying Grounded Coherence Theory to enhance creative processes*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # User settings
        user_id = st.text_input("User ID", value="creator_001")
        project_name = st.text_input("Project Name", value="Creative Project")
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["Individual Creator", "Team Collaboration", "Analysis & Insights"]
        )
        st.session_state.team_mode = (mode == "Team Collaboration")
        
        # Biometric input method
        if mode == "Individual Creator":
            st.subheader("Biometric Input")
            biometric_method = st.radio(
                "Data Source",
                ["Manual Input", "Simulated Data", "External Device"]
            )
            st.session_state.biometric_simulation = (biometric_method == "Simulated Data")
    
    # Main content based on mode
    if mode == "Individual Creator":
        individual_creator_mode(user_id, project_name)
    elif mode == "Team Collaboration":
        team_collaboration_mode(user_id, project_name)
    else:
        analysis_insights_mode(user_id)


def individual_creator_mode(user_id: str, project_name: str):
    """Individual creator monitoring and optimization"""
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Start Session", disabled=st.session_state.is_monitoring):
            st.session_state.session_id = st.session_state.database.create_session(
                user_id, project_name, "individual"
            )
            st.session_state.is_monitoring = True
            st.session_state.history = []
            st.success(f"Session started! ID: {st.session_state.session_id}")
    
    with col2:
        if st.button("⏹️ End Session", disabled=not st.session_state.is_monitoring):
            if st.session_state.session_id:
                st.session_state.database.end_session(st.session_state.session_id)
            st.session_state.is_monitoring = False
            st.info("Session ended and saved.")
    
    with col3:
        if st.button("💡 Predict Breakthrough"):
            if st.session_state.history:
                prediction = st.session_state.engine.predict_creative_trajectory(
                    st.session_state.history[-1], horizon_minutes=30
                )
                st.write(prediction)
    
    with col4:
        if st.button("🌿 Optimize Environment"):
            if st.session_state.history:
                current_state = st.session_state.history[-1]['creative_state']
                recommendations = st.session_state.env_optimizer.optimize_environment(
                    current_state
                )
                st.write(recommendations)
    
    # Main monitoring dashboard
    if st.session_state.is_monitoring:
        # Create placeholders for real-time updates
        coherence_placeholder = st.empty()
        state_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            mandala_placeholder = st.empty()
        with col2:
            flow_placeholder = st.empty()
        
        breakthrough_placeholder = st.empty()
        biometric_placeholder = st.empty()
        recommendations_placeholder = st.empty()
        
        # Real-time monitoring loop
        while st.session_state.is_monitoring:
            # Get current creative variables
            variables, biometrics = get_current_state()
            
            # Analyze creative state
            analysis = st.session_state.engine.analyze_creative_state(
                variables, biometrics
            )
            
            # Store in history
            st.session_state.history.append(analysis)
            
            # Record in database
            if st.session_state.session_id:
                st.session_state.database.record_creative_state(
                    st.session_state.session_id,
                    analysis['creative_state'],
                    analysis['gct_result'],
                    vars(variables)
                )
                st.session_state.database.record_creative_metrics(
                    st.session_state.session_id,
                    analysis['creative_metrics']
                )
                if biometrics:
                    st.session_state.database.record_biometric_data(
                        st.session_state.session_id,
                        biometrics
                    )
            
            # Update displays
            update_displays(
                analysis, variables, biometrics,
                coherence_placeholder, state_placeholder, metrics_placeholder,
                mandala_placeholder, flow_placeholder, breakthrough_placeholder,
                biometric_placeholder, recommendations_placeholder
            )
            
            # Check for breakthrough
            if analysis['breakthrough_probability'] > 0.5:
                st.balloons()
                if st.session_state.session_id:
                    st.session_state.database.record_breakthrough(
                        st.session_state.session_id,
                        st.session_state.history[-2]['gct_result']['coherence'] if len(st.session_state.history) > 1 else 0,
                        analysis['gct_result']['coherence'],
                        str(st.session_state.history[-2]['creative_state']) if len(st.session_state.history) > 1 else "",
                        str(analysis['creative_state']),
                        "Creative breakthrough detected!",
                        analysis['breakthrough_probability']
                    )
            
            time.sleep(2)  # Update every 2 seconds
    
    else:
        st.info("👆 Click 'Start Session' to begin monitoring your creative flow")
        
        # Show recent sessions
        if st.checkbox("Show Recent Sessions"):
            sessions = st.session_state.database.get_session_history(user_id, days_back=7)
            if not sessions.empty:
                st.dataframe(sessions)


def team_collaboration_mode(user_id: str, project_name: str):
    """Team collaboration optimization and monitoring"""
    
    st.header("👥 Team Creative Collaboration")
    
    # Team setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Composition")
        
        # Add team members
        num_members = st.number_input("Number of team members", 2, 8, 4)
        
        team_members = []
        for i in range(num_members):
            with st.expander(f"Member {i+1}"):
                member = TeamMember(
                    id=f"member_{i}",
                    name=st.text_input(f"Name", f"Member {i+1}", key=f"name_{i}"),
                    creative_profile={
                        'psi': st.slider("Clarity preference", 0.0, 1.0, 0.5, key=f"psi_{i}"),
                        'rho': st.slider("Wisdom level", 0.0, 1.0, 0.5, key=f"rho_{i}"),
                        'q': st.slider("Emotional intensity", 0.0, 1.0, 0.5, key=f"q_{i}"),
                        'f': st.slider("Social orientation", 0.0, 1.0, 0.5, key=f"f_{i}")
                    },
                    coherence_history=[np.random.random() for _ in range(10)],
                    state_history=[np.random.choice(list(CreativeState)) for _ in range(10)],
                    skills=set(st.multiselect(
                        "Skills",
                        ["ideation", "analysis", "synthesis", "implementation", "critique"],
                        key=f"skills_{i}"
                    )),
                    collaboration_preferences={
                        'group_size': st.slider("Preferred group size", 0.0, 1.0, 0.5, key=f"group_{i}"),
                        'interaction_style': st.slider("Interactive vs Independent", 0.0, 1.0, 0.5, key=f"interact_{i}")
                    }
                )
                team_members.append(member)
    
    with col2:
        st.subheader("Project Requirements")
        
        required_skills = st.multiselect(
            "Required Skills",
            ["ideation", "analysis", "synthesis", "implementation", "critique", 
             "visual_design", "technical", "communication"]
        )
        
        project_phase = st.selectbox(
            "Current Phase",
            ["ideation", "development", "refinement", "finalization"]
        )
        
        priority_roles = st.multiselect(
            "Priority Roles",
            ["explorer", "synthesizer", "illuminator", "refiner", "connector"]
        )
    
    # Optimization controls
    if st.button("🔮 Optimize Team Composition"):
        project_requirements = {
            'required_skills': required_skills,
            'priority_roles': priority_roles
        }
        
        team_comp = st.session_state.collab_optimizer.optimize_team_composition(
            team_members,
            project_requirements
        )
        
        # Display results
        st.success(f"Optimal team synergy: {team_comp.predicted_synergy:.2f}")
        
        # Team visualization
        if team_comp.members:
            fig = st.session_state.visualizer.create_collaborative_network(
                vars(team_comp),
                team_comp.interaction_patterns
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Role assignments
        st.subheader("Recommended Role Assignments")
        for member_id, role in team_comp.role_assignments.items():
            member_name = next(m.name for m in team_comp.members if m.id == member_id)
            st.write(f"**{member_name}**: {role}")
        
        # Warnings
        if team_comp.warnings:
            st.warning("⚠️ Potential Issues:")
            for warning in team_comp.warnings:
                st.write(f"- {warning}")
    
    # Real-time collaboration monitoring
    if st.checkbox("Start Real-time Collaboration Monitoring"):
        st.subheader("Live Team Coherence")
        
        # Simulate current states
        current_states = {}
        for member in team_members:
            current_states[member.id] = {
                'coherence': np.random.random(),
                'creative_state': np.random.choice(list(CreativeState)),
                'components': {
                    'psi': np.random.random(),
                    'rho': np.random.random(),
                    'q_raw': np.random.random(),
                    'f': np.random.random()
                }
            }
        
        # Get collaboration guidance
        guidance = st.session_state.collab_optimizer.real_time_collaboration_guide(
            team_members,
            current_states,
            project_phase
        )
        
        # Display metrics
        metrics = guidance['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Team Coherence", f"{metrics.team_coherence:.2f}")
        with col2:
            st.metric("Synchronization", f"{metrics.synchronization_index:.2f}")
        with col3:
            st.metric("Creative Friction", f"{metrics.creative_friction:.2f}")
        with col4:
            st.metric("Emergence Potential", f"{metrics.emergence_potential:.2f}")
        
        # Recommendations
        st.subheader("Real-time Recommendations")
        for rec in guidance['recommendations']:
            urgency_color = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢'
            }
            st.write(f"{urgency_color.get(rec['urgency'], '⚪')} **{rec['action']}**")
            if 'suggestions' in rec:
                for suggestion in rec['suggestions']:
                    st.write(f"  - {suggestion}")


def analysis_insights_mode(user_id: str):
    """Analysis and insights from creative history"""
    
    st.header("📊 Creative Analysis & Insights")
    
    # Time range selection
    days_back = st.slider("Days to analyze", 1, 90, 30)
    
    # Get user statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        flow_stats = st.session_state.database.get_flow_statistics(user_id)
        st.metric("Total Flow Hours", f"{flow_stats['total_flow_hours']:.1f}")
        st.metric("Avg Flow per Session", f"{flow_stats['avg_flow_minutes']:.1f} min")
    
    with col2:
        patterns = st.session_state.database.get_creative_patterns(user_id)
        if patterns['state_distribution']:
            most_common = max(patterns['state_distribution'], 
                            key=lambda x: patterns['state_distribution'][x]['count'])
            st.metric("Most Common State", most_common)
    
    with col3:
        breakthroughs = st.session_state.database.get_breakthrough_history(user_id)
        st.metric("Total Breakthroughs", len(breakthroughs))
    
    # Detailed visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Session History", "Creative Patterns", "Breakthrough Analysis", "Recommendations"
    ])
    
    with tab1:
        sessions = st.session_state.database.get_session_history(user_id, days_back)
        if not sessions.empty:
            # Session timeline
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sessions['start_time'],
                y=sessions['peak_coherence'],
                mode='markers',
                marker=dict(
                    size=sessions['flow_duration_seconds'] / 60,  # Size by flow duration
                    color=sessions['breakthroughs'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Breakthroughs")
                ),
                text=sessions['project_name'],
                hovertemplate='%{text}<br>Peak: %{y:.2f}<br>Flow: %{marker.size:.0f} min'
            ))
            fig.update_layout(
                title="Creative Sessions Overview",
                xaxis_title="Date",
                yaxis_title="Peak Coherence"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if patterns['state_distribution']:
            # State distribution pie chart
            states = list(patterns['state_distribution'].keys())
            counts = [patterns['state_distribution'][s]['count'] for s in states]
            
            fig = go.Figure(data=[go.Pie(
                labels=states,
                values=counts,
                hole=0.3
            )])
            fig.update_layout(title="Creative State Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak creative hours
            if patterns['peak_creative_hours']:
                hours = [h['hour'] for h in patterns['peak_creative_hours']]
                coherences = [h['coherence'] for h in patterns['peak_creative_hours']]
                
                st.subheader("Your Peak Creative Hours")
                for hour, coherence in zip(hours, coherences):
                    st.write(f"**{hour}:00** - Average coherence: {coherence:.2f}")
    
    with tab3:
        if not breakthroughs.empty:
            # Breakthrough timeline
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=breakthroughs['timestamp'],
                y=breakthroughs['impact_score'],
                mode='markers+lines',
                marker=dict(size=15, color='gold'),
                line=dict(color='orange', width=2),
                text=breakthroughs['project_name'],
                hovertemplate='%{text}<br>Impact: %{y:.2f}'
            ))
            fig.update_layout(
                title="Breakthrough Events",
                xaxis_title="Date",
                yaxis_title="Impact Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Breakthrough details
            st.subheader("Recent Breakthroughs")
            for _, breakthrough in breakthroughs.head(5).iterrows():
                st.write(f"**{breakthrough['project_name']}** - {breakthrough['timestamp']}")
                st.write(f"State transition: {breakthrough['state_before']} → {breakthrough['state_after']}")
                st.write(f"Coherence jump: {breakthrough['coherence_before']:.2f} → {breakthrough['coherence_after']:.2f}")
                st.write("---")
    
    with tab4:
        st.subheader("Personalized Recommendations")
        
        # Generate recommendations based on patterns
        recommendations = generate_personalized_recommendations(patterns, flow_stats)
        
        for i, rec in enumerate(recommendations):
            st.write(f"**{i+1}. {rec['title']}**")
            st.write(rec['description'])
            if 'action' in rec:
                st.info(f"💡 {rec['action']}")


def get_current_state() -> Tuple[GCTVariables, Optional[BiometricData]]:
    """Get current creative state (manual or simulated)"""
    
    if st.session_state.biometric_simulation:
        # Simulate realistic creative flow
        t = time.time()
        
        # Oscillating patterns for creative work
        psi = 0.5 + 0.3 * np.sin(t / 30)  # Clarity cycles
        rho = 0.6 + 0.2 * np.sin(t / 45)  # Wisdom accumulation
        q_raw = 0.4 + 0.3 * np.sin(t / 20) + 0.1 * np.random.random()  # Emotional waves
        f = 0.5 + 0.2 * np.sin(t / 60)  # Social connection
        
        # Clamp values
        psi = max(0, min(1, psi))
        rho = max(0, min(1, rho))
        q_raw = max(0, min(1, q_raw))
        f = max(0, min(1, f))
        
        variables = GCTVariables(
            psi=psi,
            rho=rho,
            q_raw=q_raw,
            f=f,
            timestamp=datetime.now()
        )
        
        # Simulate biometrics
        biometrics = BiometricData(
            hrv=0.5 + 0.2 * np.sin(t / 40),
            eeg_alpha=0.6 + 0.2 * np.sin(t / 25),
            eeg_theta=0.5 + 0.3 * np.sin(t / 35),
            eeg_gamma=0.3 + 0.2 * np.random.random(),
            gsr=0.4 + 0.3 * np.sin(t / 15),
            eye_movement_entropy=0.3 + 0.4 * np.random.random(),
            posture_stability=0.7 + 0.2 * np.sin(t / 50)
        )
    else:
        # Manual input
        with st.sidebar:
            st.subheader("Manual Input")
            psi = st.slider("Clarity (ψ)", 0.0, 1.0, 0.5)
            rho = st.slider("Wisdom (ρ)", 0.0, 1.0, 0.6)
            q_raw = st.slider("Emotion (q)", 0.0, 1.0, 0.4)
            f = st.slider("Social (f)", 0.0, 1.0, 0.5)
            
            variables = GCTVariables(
                psi=psi,
                rho=rho,
                q_raw=q_raw,
                f=f,
                timestamp=datetime.now()
            )
            
            if st.checkbox("Include Biometrics"):
                biometrics = BiometricData(
                    hrv=st.slider("HRV", 0.0, 1.0, 0.5),
                    eeg_alpha=st.slider("Alpha waves", 0.0, 1.0, 0.6),
                    eeg_theta=st.slider("Theta waves", 0.0, 1.0, 0.5),
                    eeg_gamma=st.slider("Gamma waves", 0.0, 1.0, 0.3),
                    gsr=st.slider("GSR", 0.0, 1.0, 0.4),
                    eye_movement_entropy=st.slider("Eye movement", 0.0, 1.0, 0.5),
                    posture_stability=st.slider("Posture", 0.0, 1.0, 0.7)
                )
            else:
                biometrics = None
    
    return variables, biometrics


def update_displays(analysis, variables, biometrics, *placeholders):
    """Update all display elements"""
    
    (coherence_ph, state_ph, metrics_ph, mandala_ph, flow_ph, 
     breakthrough_ph, biometric_ph, recommendations_ph) = placeholders
    
    # Coherence and state
    with coherence_ph:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coherence", f"{analysis['gct_result']['coherence']:.3f}")
        with col2:
            st.metric("Creative State", analysis['creative_state'].value)
        with col3:
            flow_score = analysis['flow_analysis']['flow_score']
            st.metric("Flow Score", f"{flow_score:.2f}")
    
    # Creative metrics
    with metrics_ph:
        metrics = analysis['creative_metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Novelty", f"{metrics.novelty_score:.2f}")
        with col2:
            st.metric("Fluency", f"{metrics.fluency_rate:.2f}")
        with col3:
            st.metric("Flexibility", f"{metrics.flexibility_index:.2f}")
        with col4:
            st.metric("Breakthrough", f"{metrics.breakthrough_probability:.1%}")
    
    # Visualizations
    with mandala_ph:
        fig = st.session_state.visualizer.create_coherence_mandala(
            analysis['gct_result'], metrics
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with flow_ph:
        if len(st.session_state.history) > 1:
            fig = st.session_state.visualizer.create_flow_river(
                st.session_state.history[-50:]  # Last 50 measurements
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with breakthrough_ph:
        fig = st.session_state.visualizer.create_breakthrough_radar(
            metrics, analysis['breakthrough_probability']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if biometrics:
        with biometric_ph:
            fig = st.session_state.visualizer.create_biometric_dashboard(biometrics)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    with recommendations_ph:
        st.subheader("Real-time Recommendations")
        for rec in analysis['recommendations'][:3]:  # Top 3 recommendations
            urgency_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(rec.get('urgency', 'low'), '⚪')
            
            st.write(f"{urgency_icon} **{rec['type']}**: {rec['action']}")
            if 'rationale' in rec:
                st.caption(rec['rationale'])


def generate_personalized_recommendations(patterns: Dict, flow_stats: Dict) -> List[Dict]:
    """Generate personalized recommendations based on creative patterns"""
    
    recommendations = []
    
    # Flow optimization
    if flow_stats['avg_flow_minutes'] < 20:
        recommendations.append({
            'title': 'Extend Your Flow States',
            'description': 'Your average flow duration is relatively short. Consider longer uninterrupted work sessions.',
            'action': 'Block out 45-90 minute creative sessions with all notifications disabled.'
        })
    
    # State balance
    if patterns['state_distribution']:
        blocked_percentage = patterns['state_distribution'].get('blocked', {}).get('count', 0)
        total_states = sum(s['count'] for s in patterns['state_distribution'].values())
        
        if blocked_percentage / total_states > 0.3:
            recommendations.append({
                'title': 'Reduce Creative Blocks',
                'description': 'You spend significant time in blocked states. This may indicate environmental or process issues.',
                'action': 'Try changing your physical environment or creative medium when feeling blocked.'
            })
    
    # Peak hour utilization
    if patterns['peak_creative_hours']:
        peak_hour = patterns['peak_creative_hours'][0]['hour']
        recommendations.append({
            'title': 'Leverage Your Peak Hours',
            'description': f'Your most creative time is around {peak_hour}:00.',
            'action': f'Schedule your most important creative work between {peak_hour-1}:00 and {peak_hour+1}:00.'
        })
    
    # Breakthrough cultivation
    recommendations.append({
        'title': 'Cultivate More Breakthroughs',
        'description': 'Breakthroughs often follow periods of incubation and varied exploration.',
        'action': 'Alternate between focused work and relaxed exploration. Take regular breaks for unconscious processing.'
    })
    
    return recommendations


if __name__ == "__main__":
    main()