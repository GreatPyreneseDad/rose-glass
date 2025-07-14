"""
GCT Dream Analysis Application
A novel application of Grounded Coherence Theory to dream analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import numpy as np
from typing import Dict, List, Optional
import json

# Import our dream analysis components
from src.dream_gct_engine import DreamGCTEngine, DreamVariables
from src.dream_database import DreamDatabase
from src.dream_interpreter import DreamInterpreter

# Page configuration
st.set_page_config(
    page_title="GCT Dream Analysis",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dream-like aesthetic
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .dream-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .coherence-high { color: #4ade80; }
    .coherence-medium { color: #fbbf24; }
    .coherence-low { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = DreamGCTEngine()
    st.session_state.db = DreamDatabase()
    st.session_state.interpreter = DreamInterpreter()
    st.session_state.user_id = "default_user"  # In production, implement proper auth

# Title and introduction
st.title("🌙 GCT Dream Analysis Engine")
st.markdown("""
*Applying Grounded Coherence Theory to unlock the wisdom of your dreams*

This revolutionary system analyzes your dreams through four dimensions:
- **ψ (Psi)**: Clarity and vividness of dream imagery
- **ρ (Rho)**: Depth of symbolic meaning and archetypal wisdom  
- **q**: Emotional intensity and charge
- **f**: Social and relational connections
""")

# Sidebar navigation
with st.sidebar:
    st.header("🌟 Navigation")
    page = st.radio(
        "Choose a section:",
        ["📝 Dream Journal", "📊 Analysis Dashboard", "🔮 Insights & Patterns", 
         "😴 Sleep Tracking", "📚 Symbol Library", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.subheader("Quick Stats")
    
    # Get user stats
    dreams_df = st.session_state.db.get_user_dreams(st.session_state.user_id, days=30)
    if not dreams_df.empty:
        st.metric("Dreams Recorded", len(dreams_df))
        avg_coherence = dreams_df['coherence'].mean() if 'coherence' in dreams_df else 0
        st.metric("Avg Coherence", f"{avg_coherence:.3f}")
        
        # Coherence trend
        if len(dreams_df) > 1:
            recent = dreams_df.head(7)['coherence'].mean() if 'coherence' in dreams_df else 0
            older = dreams_df.tail(7)['coherence'].mean() if 'coherence' in dreams_df else 0
            trend = "📈" if recent > older else "📉" if recent < older else "➡️"
            st.metric("7-Day Trend", trend)

# Main content area
if page == "📝 Dream Journal":
    st.header("📝 Dream Journal")
    
    # Dream entry form
    with st.form("dream_entry"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dream_date = st.date_input("Dream Date", value=date.today())
            title = st.text_input("Dream Title (optional)", placeholder="Give your dream a title...")
            narrative = st.text_area(
                "Dream Narrative", 
                placeholder="Describe your dream in detail. Include emotions, symbols, people, and any significant events...",
                height=300
            )
        
        with col2:
            lucid = st.checkbox("Lucid Dream")
            sleep_quality = st.slider("Sleep Quality", 1, 10, 5)
            sleep_hours = st.number_input("Hours of Sleep", 0.0, 12.0, 7.0, 0.5)
            
            st.subheader("Tags")
            common_tags = ["nightmare", "recurring", "prophetic", "healing", "symbolic"]
            selected_tags = st.multiselect("Select tags:", common_tags)
            custom_tag = st.text_input("Add custom tag:")
            if custom_tag:
                selected_tags.append(custom_tag)
        
        submitted = st.form_submit_button("💫 Analyze Dream", use_container_width=True)
        
        if submitted and narrative:
            with st.spinner("Analyzing dream coherence..."):
                # Extract variables
                variables = st.session_state.engine.extract_dream_variables(
                    narrative, sleep_quality/10, lucid
                )
                
                # Calculate coherence
                result = st.session_state.engine.calculate_dream_coherence(variables)
                
                # Save to database
                dream_id = st.session_state.db.save_dream(
                    st.session_state.user_id, narrative, 
                    datetime.combine(dream_date, datetime.min.time()),
                    title, lucid, sleep_quality/10, sleep_hours, selected_tags
                )
                
                # Prepare analysis data for storage
                analysis_data = {
                    'coherence': result.coherence,
                    'variables': {
                        'psi': variables.psi,
                        'rho': variables.rho,
                        'q_raw': variables.q_raw,
                        'f': variables.f,
                        'lucidity': variables.lucidity,
                        'recurring': variables.recurring,
                        'shadow': variables.shadow
                    },
                    'q_opt': result.q_opt,
                    'dc_dt': result.dc_dt,
                    'd2c_dt2': result.d2c_dt2,
                    'dream_state': result.dream_state,
                    'components': result.components,
                    'insights': result.insights
                }
                
                st.session_state.db.save_analysis(dream_id, analysis_data)
                
                # Display results
                st.success("Dream analyzed and saved!")
                
                # Coherence visualization
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dream Coherence", f"{result.coherence:.3f}")
                with col2:
                    st.metric("Dream State", result.dream_state.replace('_', ' ').title())
                with col3:
                    coherence_change = "🔺" if result.dc_dt > 0 else "🔻" if result.dc_dt < 0 else "➖"
                    st.metric("Momentum", coherence_change)
                
                # Component breakdown
                st.subheader("Coherence Components")
                fig = go.Figure(data=[go.Bar(
                    x=list(result.components.keys()),
                    y=list(result.components.values()),
                    marker_color=['#4ade80', '#60a5fa', '#f472b6', '#a78bfa', '#fbbf24', '#f87171']
                )])
                fig.update_layout(
                    title="Dream Coherence Breakdown",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                if result.insights:
                    st.subheader("🔮 Dream Insights")
                    for insight in result.insights:
                        st.info(insight)
    
    # Recent dreams
    st.markdown("---")
    st.subheader("Recent Dreams")
    
    recent_dreams = st.session_state.db.get_user_dreams(st.session_state.user_id, days=7)
    if not recent_dreams.empty:
        for _, dream in recent_dreams.iterrows():
            with st.expander(f"{dream['dream_date']} - {dream.get('title', 'Untitled')}"):
                st.write(dream['narrative'][:200] + "..." if len(dream['narrative']) > 200 else dream['narrative'])
                if pd.notna(dream.get('coherence')):
                    coherence_class = (
                        "coherence-high" if dream['coherence'] > 0.7 
                        else "coherence-medium" if dream['coherence'] > 0.4 
                        else "coherence-low"
                    )
                    st.markdown(f"<span class='{coherence_class}'>Coherence: {dream['coherence']:.3f}</span>", 
                              unsafe_allow_html=True)
                    st.write(f"State: {dream.get('dream_state', 'Unknown')}")
    else:
        st.info("No dreams recorded yet. Start by entering your first dream above!")

elif page == "📊 Analysis Dashboard":
    st.header("📊 Dream Analysis Dashboard")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
    )
    
    days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": 365}
    days = days_map[time_range]
    
    # Get coherence timeline
    timeline_df = st.session_state.db.get_coherence_timeline(st.session_state.user_id, days)
    
    if not timeline_df.empty:
        # Coherence over time
        fig = go.Figure()
        
        # Main coherence line
        fig.add_trace(go.Scatter(
            x=timeline_df['dream_date'],
            y=timeline_df['coherence'],
            mode='lines+markers',
            name='Coherence',
            line=dict(color='#60a5fa', width=3),
            marker=dict(size=8)
        ))
        
        # Add component traces
        components = ['psi', 'rho', 'q_opt', 'f']
        colors = ['#4ade80', '#f472b6', '#fbbf24', '#a78bfa']
        
        for comp, color in zip(components, colors):
            if comp in timeline_df.columns:
                fig.add_trace(go.Scatter(
                    x=timeline_df['dream_date'],
                    y=timeline_df[comp],
                    mode='lines',
                    name=comp.upper(),
                    line=dict(color=color, width=2, dash='dot'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Dream Coherence Timeline",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font_color='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dream state distribution
        col1, col2 = st.columns(2)
        
        with col1:
            state_dist = st.session_state.db.get_dream_states_distribution(st.session_state.user_id)
            if state_dist:
                fig = go.Figure(data=[go.Pie(
                    labels=[s.replace('_', ' ').title() for s in state_dist.keys()],
                    values=list(state_dist.values()),
                    hole=.3
                )])
                fig.update_layout(
                    title="Dream State Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Variable correlations
            if len(timeline_df) > 3:
                corr_matrix = timeline_df[['coherence', 'psi', 'rho', 'q_opt', 'f']].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu"
                )
                fig.update_layout(
                    title="Variable Correlations",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Pattern analysis
        pattern_analysis = st.session_state.engine.get_pattern_analysis()
        if pattern_analysis.get('total_dreams_analyzed', 0) >= 5:
            st.subheader("Pattern Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Coherence", f"{pattern_analysis['avg_coherence']:.3f}")
            with col2:
                st.metric("Coherence Trend", pattern_analysis['coherence_trend'].title())
            with col3:
                st.metric("Dominant State", pattern_analysis['dominant_state'].replace('_', ' ').title())
            
            # Top symbols
            if pattern_analysis.get('top_symbols'):
                st.subheader("Most Frequent Symbols")
                symbols_df = pd.DataFrame(
                    pattern_analysis['top_symbols'],
                    columns=['Symbol', 'Frequency']
                )
                st.dataframe(symbols_df, hide_index=True)
    else:
        st.info("Record more dreams to see analysis dashboard!")

elif page == "🔮 Insights & Patterns":
    st.header("🔮 Deep Insights & Patterns")
    
    # Get all dreams for analysis
    all_dreams = st.session_state.db.get_user_dreams(st.session_state.user_id, days=90)
    
    if len(all_dreams) >= 3:
        # Coherence pattern interpretation
        coherence_history = all_dreams['coherence'].tolist() if 'coherence' in all_dreams else []
        states_history = all_dreams['dream_state'].tolist() if 'dream_state' in all_dreams else []
        
        if coherence_history:
            pattern_insight = st.session_state.interpreter.interpret_coherence_pattern(
                coherence_history, states_history
            )
            
            # Display pattern analysis
            st.subheader("Coherence Pattern Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(pattern_insight['interpretation'])
            with col2:
                st.metric("Pattern", pattern_insight['pattern'].replace('_', ' ').title())
                st.metric("Volatility", pattern_insight['volatility'].title())
            
            # Visualize pattern
            if len(coherence_history) > 5:
                # Phase portrait
                dc_dt = np.diff(coherence_history)
                fig = go.Figure(data=go.Scatter(
                    x=coherence_history[:-1],
                    y=dc_dt,
                    mode='lines+markers',
                    line=dict(color='#60a5fa'),
                    marker=dict(
                        size=8,
                        color=list(range(len(dc_dt))),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Time")
                    )
                ))
                fig.update_layout(
                    title="Coherence Phase Portrait",
                    xaxis_title="Coherence",
                    yaxis_title="dC/dt (Rate of Change)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.1)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Symbol analysis
        st.subheader("Symbol Analysis")
        symbol_freq = st.session_state.db.get_symbol_frequency(st.session_state.user_id, days=90)
        
        if not symbol_freq.empty:
            # Get symbol evolution data (simplified for now)
            symbol_insights = st.session_state.interpreter.generate_symbol_insights(
                dict(zip(symbol_freq['symbol'], symbol_freq['frequency'])),
                {}  # Would need to implement symbol evolution tracking
            )
            
            for insight in symbol_insights[:5]:  # Top 5 symbols
                with st.expander(f"🔮 {insight['symbol']} (appears {insight['frequency']} times)"):
                    st.write(insight['interpretation'])
                    if insight['archetype']:
                        st.info(f"Archetypal Association: {insight['archetype'].replace('_', ' ').title()}")
        
        # Emotional progression
        st.subheader("Emotional Journey")
        
        dreams_data = all_dreams.to_dict('records')
        emotional_analysis = st.session_state.interpreter.analyze_emotional_progression(dreams_data)
        
        if emotional_analysis.get('timeline'):
            # Emotion timeline visualization
            emotion_df = pd.DataFrame(emotional_analysis['timeline'])
            
            fig = go.Figure()
            
            # Create emotion color map
            emotion_colors = {
                'fear': '#ef4444', 'joy': '#10b981', 'anger': '#f59e0b',
                'sadness': '#3b82f6', 'love': '#ec4899', 'shame': '#8b5cf6',
                'neutral': '#6b7280'
            }
            
            for emotion in emotion_colors:
                emotion_data = emotion_df[emotion_df['emotion'] == emotion]
                if not emotion_data.empty:
                    fig.add_trace(go.Scatter(
                        x=emotion_data['date'],
                        y=emotion_data['intensity'],
                        mode='markers',
                        name=emotion.title(),
                        marker=dict(
                            size=12,
                            color=emotion_colors[emotion],
                            symbol='circle'
                        )
                    ))
            
            fig.update_layout(
                title="Emotional Journey Through Dreams",
                xaxis_title="Date",
                yaxis_title="Emotional Intensity",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(emotional_analysis['interpretation'])
        
        # Psychological themes
        st.subheader("Psychological Themes")
        
        narratives = all_dreams['narrative'].tolist()
        themes = st.session_state.interpreter.detect_psychological_themes(narratives)
        
        if themes:
            # Create theme radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=list(themes.values()),
                theta=[t.replace('_', ' ').title() for t in themes.keys()],
                fill='toself',
                line=dict(color='#60a5fa')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(themes.values()) * 1.2]
                    )
                ),
                showlegend=False,
                title="Psychological Theme Profile",
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Personalized recommendations
        st.subheader("🌟 Personalized Recommendations")
        
        analysis_summary = {
            'avg_coherence': all_dreams['coherence'].mean() if 'coherence' in all_dreams else 0.5,
            'dominant_state': all_dreams['dream_state'].mode()[0] if 'dream_state' in all_dreams and not all_dreams['dream_state'].empty else 'unknown',
            'coherence_trend': pattern_insight.get('trend', 'stable'),
            'top_symbols': symbol_freq.head(5).to_dict('records') if not symbol_freq.empty else []
        }
        
        recommendations = st.session_state.interpreter.generate_personalized_recommendations(analysis_summary)
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    else:
        st.info("Record at least 3 dreams to unlock deep insights and pattern analysis!")

elif page == "😴 Sleep Tracking":
    st.header("😴 Sleep & Dream Correlation")
    
    # Sleep entry form
    with st.form("sleep_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            sleep_date = st.date_input("Date", value=date.today())
            bedtime = st.time_input("Bedtime")
            wake_time = st.time_input("Wake Time")
        
        with col2:
            quality = st.slider("Sleep Quality", 1, 10, 5)
            dream_count = st.number_input("Dreams Recalled", 0, 10, 1)
            notes = st.text_area("Notes", placeholder="Any observations about your sleep...")
        
        submitted = st.form_submit_button("Save Sleep Data")
        
        if submitted:
            st.session_state.db.save_sleep_pattern(
                st.session_state.user_id,
                datetime.combine(sleep_date, datetime.min.time()),
                datetime.combine(sleep_date, bedtime),
                datetime.combine(sleep_date, wake_time),
                quality / 10,
                dream_count,
                notes
            )
            st.success("Sleep data saved!")
    
    # Sleep-coherence correlation
    st.subheader("Sleep Quality vs Dream Coherence")
    
    correlation_df = st.session_state.db.get_sleep_coherence_correlation(st.session_state.user_id)
    
    if not correlation_df.empty and 'sleep_quality' in correlation_df and 'avg_coherence' in correlation_df:
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=correlation_df['sleep_quality'],
            y=correlation_df['avg_coherence'],
            mode='markers',
            marker=dict(
                size=correlation_df['dream_count'] * 10,
                color=correlation_df['dream_recall_count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Dreams Recalled")
            ),
            text=[f"Date: {d}<br>Dreams: {c}" for d, c in 
                  zip(correlation_df['date'], correlation_df['dream_count'])],
            hovertemplate='%{text}<br>Sleep Quality: %{x}<br>Avg Coherence: %{y}<extra></extra>'
        ))
        
        # Add trend line
        if len(correlation_df) > 3:
            z = np.polyfit(correlation_df['sleep_quality'].dropna(), 
                          correlation_df['avg_coherence'].dropna(), 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[p(0), p(1)],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend'
            ))
        
        fig.update_layout(
            title="Sleep Quality vs Dream Coherence Correlation",
            xaxis_title="Sleep Quality",
            yaxis_title="Average Dream Coherence",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        if len(correlation_df) > 5:
            corr = correlation_df[['sleep_quality', 'avg_coherence']].corr().iloc[0, 1]
            if abs(corr) > 0.5:
                st.info(f"Strong correlation detected (r={corr:.2f}): {'Better' if corr > 0 else 'Worse'} sleep quality is associated with {'higher' if corr > 0 else 'lower'} dream coherence.")
            elif abs(corr) > 0.3:
                st.info(f"Moderate correlation detected (r={corr:.2f}) between sleep quality and dream coherence.")
            else:
                st.info("Weak correlation between sleep quality and dream coherence. Other factors may be more influential.")

elif page == "📚 Symbol Library":
    st.header("📚 Dream Symbol Library")
    
    st.markdown("""
    Explore the symbols that appear in your dreams and their potential meanings.
    This library combines your personal symbol patterns with universal archetypal interpretations.
    """)
    
    # Get user's symbol frequency
    symbol_freq = st.session_state.db.get_symbol_frequency(st.session_state.user_id, days=365)
    
    if not symbol_freq.empty:
        # Symbol search
        search_term = st.text_input("Search symbols:", placeholder="e.g., water, flying, shadow...")
        
        if search_term:
            filtered_symbols = symbol_freq[symbol_freq['symbol'].str.contains(search_term, case=False)]
        else:
            filtered_symbols = symbol_freq
        
        # Display symbols
        for _, symbol_data in filtered_symbols.head(20).iterrows():
            symbol = symbol_data['symbol']
            frequency = symbol_data['frequency']
            avg_charge = symbol_data['avg_charge']
            
            with st.expander(f"🔮 {symbol} (appears {frequency} times)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Generate interpretation
                    interpretation = st.session_state.interpreter._interpret_symbol(
                        symbol, frequency, None, "stable"
                    )
                    st.write(interpretation)
                
                with col2:
                    st.metric("Frequency", frequency)
                    st.metric("Avg Emotional Charge", f"{avg_charge:.2f}")
                    
                    # Emotion indicator
                    if avg_charge > 0.7:
                        st.write("🔥 High emotional significance")
                    elif avg_charge > 0.4:
                        st.write("💫 Moderate emotional charge")
                    else:
                        st.write("💤 Low emotional charge")
        
        # Symbol cloud visualization
        if len(symbol_freq) > 5:
            st.subheader("Symbol Cloud")
            
            # Create word cloud data
            words = []
            for _, row in symbol_freq.head(30).iterrows():
                words.extend([row['symbol']] * int(row['frequency']))
            
            if words:
                # Simple frequency visualization
                word_freq = pd.DataFrame(symbol_freq.head(20))
                fig = px.treemap(
                    word_freq,
                    path=['symbol'],
                    values='frequency',
                    color='avg_charge',
                    color_continuous_scale='RdBu'
                )
                fig.update_layout(
                    title="Dream Symbol Map (size = frequency, color = emotional charge)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start recording dreams to build your personal symbol library!")
    
    # Universal symbols reference
    st.subheader("Universal Dream Symbols")
    
    universal_symbols = {
        "🌊 Water": "Emotions, unconscious, purification, life force",
        "🔥 Fire": "Transformation, passion, destruction, enlightenment",
        "🏠 House": "Self, psyche, different rooms represent aspects of consciousness",
        "🐍 Snake": "Transformation, healing, wisdom, hidden fears",
        "✈️ Flying": "Freedom, transcendence, escaping limitations, spiritual elevation",
        "💀 Death": "Transformation, endings, rebirth, letting go of the old",
        "👶 Baby": "New beginnings, innocence, vulnerability, potential",
        "🌳 Tree": "Growth, life, connection between earth and sky, family roots",
        "🪞 Mirror": "Self-reflection, truth, illusion, how you see yourself",
        "🌉 Bridge": "Transition, connection, moving between states of being"
    }
    
    cols = st.columns(2)
    for i, (symbol, meaning) in enumerate(universal_symbols.items()):
        with cols[i % 2]:
            st.write(f"**{symbol}**: {meaning}")

elif page == "ℹ️ About":
    st.header("ℹ️ About GCT Dream Analysis")
    
    st.markdown("""
    ## The Science Behind Dream Coherence
    
    This application represents a groundbreaking fusion of Grounded Coherence Theory (GCT) with dream analysis,
    creating a unique framework for understanding the patterns and meanings within our dreams.
    
    ### What is Grounded Coherence Theory?
    
    Originally developed for analyzing human communication and social dynamics, GCT measures coherence through
    four fundamental dimensions:
    
    1. **ψ (Psi) - Clarity**: In dreams, this represents the vividness and clarity of dream imagery
    2. **ρ (Rho) - Wisdom**: The depth of symbolic meaning and archetypal content
    3. **q - Emotion**: The intensity and quality of emotional experience
    4. **f - Social**: The relational and interpersonal aspects of dreams
    
    ### Dream-Specific Enhancements
    
    This adaptation extends GCT with dream-specific variables:
    
    - **Lucidity Factor**: Measures conscious awareness within dreams
    - **Shadow Integration**: Tracks the integration of rejected aspects of self
    - **Recurrence Strength**: Identifies and weights recurring patterns
    
    ### The Coherence Formula
    
    Dream coherence is calculated using an enhanced version of the GCT formula:
    
    ```
    C = (ψ + ρ×ψ + q_opt + f×ψ + lucidity_bonus + shadow_bonus) × recurring_factor
    ```
    
    This produces a single coherence score that reflects the overall integration and psychological
    significance of each dream.
    
    ### Dream States
    
    The system classifies dreams into various states based on coherence patterns:
    
    - **Lucid Integration**: High conscious awareness and control
    - **Shadow Work**: Active integration of rejected aspects
    - **Breakthrough**: Psychological breakthroughs occurring
    - **Processing**: Active processing of experiences
    - **Chaotic**: Reorganization of psychological structures
    
    ### Why Track Dream Coherence?
    
    1. **Psychological Health**: Coherence patterns reveal psychological integration
    2. **Pattern Recognition**: Identify recurring themes needing attention
    3. **Growth Tracking**: Monitor psychological development over time
    4. **Insight Generation**: AI-powered interpretations provide actionable insights
    
    ### Privacy & Data
    
    All dream data is stored locally and never shared. Your dreams remain completely private.
    
    ### Credits
    
    Built on the foundation of Grounded Coherence Theory, originally developed for market sentiment
    analysis. This novel adaptation demonstrates the universal applicability of GCT principles to
    understanding human consciousness and experience.
    
    ---
    
    *"Dreams are the royal road to the unconscious." - Sigmund Freud*
    
    *"Who looks outside, dreams; who looks inside, awakes." - Carl Jung*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>🌙 GCT Dream Analysis Engine | Powered by Grounded Coherence Theory</center>",
    unsafe_allow_html=True
)