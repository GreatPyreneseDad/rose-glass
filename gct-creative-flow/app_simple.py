"""
GCT Creative Flow - Simple & User-Friendly Version
Easy tracking and analysis of your creative states
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time

# Import only essential components
from src.creative_flow_engine import CreativeFlowEngine, CreativeState, BiometricData
from src.creative_database import CreativeFlowDatabase

# Import GCT components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../gct-market-sentiment/src'))
from gct_engine import GCTVariables

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = CreativeFlowEngine()
    st.session_state.database = CreativeFlowDatabase()
    st.session_state.current_session_id = None
    st.session_state.quick_entries = []
    
# Constants for easy presets
CREATIVE_MOODS = {
    "🔥 On Fire": {"psi": 0.8, "rho": 0.7, "q": 0.6, "f": 0.5},
    "💡 Inspired": {"psi": 0.7, "rho": 0.8, "q": 0.7, "f": 0.6},
    "🌊 Flowing": {"psi": 0.9, "rho": 0.6, "q": 0.4, "f": 0.5},
    "🤔 Exploring": {"psi": 0.4, "rho": 0.5, "q": 0.7, "f": 0.6},
    "😴 Incubating": {"psi": 0.3, "rho": 0.8, "q": 0.3, "f": 0.4},
    "😤 Blocked": {"psi": 0.2, "rho": 0.3, "q": 0.8, "f": 0.2},
    "🎯 Focused": {"psi": 0.8, "rho": 0.6, "q": 0.5, "f": 0.3},
    "🎨 Playful": {"psi": 0.5, "rho": 0.4, "q": 0.8, "f": 0.7}
}

def main():
    st.set_page_config(
        page_title="Creative Flow Tracker",
        page_icon="🎨",
        layout="wide"
    )
    
    # Header
    st.title("🎨 Creative Flow Tracker")
    st.markdown("*Track your creative states and discover your patterns*")
    
    # User identification (simple)
    col1, col2 = st.columns([3, 1])
    with col1:
        user_name = st.text_input("Your Name", value=st.session_state.get('user_name', 'Creative Soul'))
        st.session_state.user_name = user_name
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Quick Entry", "📊 My Patterns", "📈 Progress", "⚙️ Settings"])
    
    with tab1:
        quick_entry_tab()
    
    with tab2:
        patterns_tab(user_name)
    
    with tab3:
        progress_tab(user_name)
        
    with tab4:
        settings_tab()

def quick_entry_tab():
    """Quick and easy creative state entry"""
    st.header("How are you feeling creatively?")
    
    # Method 1: Mood buttons (easiest)
    st.subheader("🎯 Quick Mood Selection")
    
    # Create button grid
    cols = st.columns(4)
    selected_mood = None
    
    for idx, (mood, values) in enumerate(CREATIVE_MOODS.items()):
        with cols[idx % 4]:
            if st.button(mood, key=f"mood_{idx}", use_container_width=True):
                selected_mood = (mood, values)
    
    # Method 2: Sliders (more control)
    with st.expander("🎚️ Custom Input (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            clarity = st.slider("Mental Clarity", 0.0, 1.0, 0.5, 
                              help="How clear and focused is your thinking?")
            wisdom = st.slider("Depth/Insight", 0.0, 1.0, 0.5,
                             help="How deep are your insights?")
        
        with col2:
            emotion = st.slider("Emotional Energy", 0.0, 1.0, 0.5,
                              help="How emotionally engaged are you?")
            social = st.slider("Connection", 0.0, 1.0, 0.5,
                             help="How connected do you feel?")
        
        custom_values = {"psi": clarity, "rho": wisdom, "q": emotion, "f": social}
    
    # Project/Activity tracking
    st.subheader("📌 What are you working on?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Get recent projects for dropdown
        recent_projects = get_recent_projects(st.session_state.user_name)
        project_options = ["New Project..."] + recent_projects
        
        project_select = st.selectbox("Project", project_options)
        
        if project_select == "New Project...":
            project_name = st.text_input("Project Name")
        else:
            project_name = project_select
    
    with col2:
        activity_type = st.selectbox("Activity", 
            ["Creating", "Planning", "Learning", "Reviewing", "Collaborating"])
    
    # Notes (optional)
    notes = st.text_area("Notes (optional)", placeholder="Any thoughts or breakthroughs?")
    
    # Save button
    if st.button("💾 Save Entry", type="primary", use_container_width=True):
        # Determine values to use
        if selected_mood:
            values = selected_mood[1]
            mood_name = selected_mood[0]
        else:
            values = custom_values
            mood_name = "Custom"
        
        # Create entry
        save_quick_entry(
            user_name=st.session_state.user_name,
            values=values,
            project=project_name,
            activity=activity_type,
            notes=notes,
            mood=mood_name
        )
        
        st.success("✅ Entry saved! Keep creating! 🎨")
        st.balloons()
        
        # Show instant feedback
        with st.container():
            coherence = calculate_simple_coherence(values)
            state = detect_creative_state(values, coherence)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coherence", f"{coherence:.2f}")
            with col2:
                st.metric("State", state)
            with col3:
                st.metric("Flow Score", f"{calculate_flow_score(values, coherence):.2f}")

def patterns_tab(user_name: str):
    """View creative patterns and insights"""
    st.header("📊 Your Creative Patterns")
    
    # Get user data
    df = load_user_data(user_name)
    
    if df.empty:
        st.info("No data yet! Start tracking your creative states in the Quick Entry tab.")
        return
    
    # Time range selector
    col1, col2 = st.columns([2, 1])
    with col1:
        time_range = st.selectbox("Time Period", 
            ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"])
    
    # Filter data
    df_filtered = filter_by_time_range(df, time_range)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_coherence = df_filtered['coherence'].mean()
        st.metric("Avg Coherence", f"{avg_coherence:.2f}")
    
    with col2:
        flow_percentage = (df_filtered['state'] == 'Flow').mean() * 100
        st.metric("Flow Time", f"{flow_percentage:.0f}%")
    
    with col3:
        most_common_state = df_filtered['state'].mode()[0] if not df_filtered.empty else "N/A"
        st.metric("Most Common State", most_common_state)
    
    with col4:
        total_entries = len(df_filtered)
        st.metric("Total Entries", total_entries)
    
    # Visualizations
    st.subheader("📈 Coherence Over Time")
    fig_timeline = create_coherence_timeline(df_filtered)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 State Distribution")
        fig_states = create_state_distribution(df_filtered)
        st.plotly_chart(fig_states, use_container_width=True)
    
    with col2:
        st.subheader("⏰ Best Creative Times")
        fig_times = create_time_heatmap(df_filtered)
        st.plotly_chart(fig_times, use_container_width=True)
    
    # Project breakdown
    if 'project' in df_filtered.columns:
        st.subheader("📁 Project Performance")
        project_stats = df_filtered.groupby('project').agg({
            'coherence': 'mean',
            'state': lambda x: (x == 'Flow').mean()
        }).round(2)
        project_stats.columns = ['Avg Coherence', 'Flow Rate']
        st.dataframe(project_stats.sort_values('Avg Coherence', ascending=False))
    
    # Insights
    st.subheader("💡 Insights")
    insights = generate_insights(df_filtered)
    for insight in insights:
        st.info(f"💡 {insight}")

def progress_tab(user_name: str):
    """Track progress over time"""
    st.header("📈 Your Creative Progress")
    
    df = load_user_data(user_name)
    
    if df.empty:
        st.info("Start tracking to see your progress!")
        return
    
    # Progress metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Coherence trend
        st.subheader("📊 Coherence Trend")
        fig_trend = create_progress_chart(df, 'coherence')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Streak tracking
        st.subheader("🔥 Tracking Streak")
        streak = calculate_streak(df)
        st.metric("Current Streak", f"{streak} days")
    
    with col2:
        # Flow achievement
        st.subheader("🌊 Flow Achievement")
        fig_flow = create_flow_progress(df)
        st.plotly_chart(fig_flow, use_container_width=True)
        
        # Milestones
        st.subheader("🏆 Milestones")
        milestones = check_milestones(df)
        for milestone in milestones:
            st.success(f"🏆 {milestone}")
    
    # Weekly summary
    st.subheader("📅 Weekly Summary")
    weekly_summary = create_weekly_summary(df)
    st.plotly_chart(weekly_summary, use_container_width=True)
    
    # Export data
    st.subheader("💾 Export Your Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"creative_flow_{user_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Generate Report"):
            report = generate_report(df, user_name)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"creative_report_{user_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

def settings_tab():
    """User settings and preferences"""
    st.header("⚙️ Settings")
    
    # Reminder settings
    st.subheader("⏰ Reminders")
    enable_reminders = st.checkbox("Enable daily check-in reminders")
    if enable_reminders:
        reminder_time = st.time_input("Reminder time", value=datetime.strptime("21:00", "%H:%M").time())
        st.info(f"You'll be reminded to track your creative state at {reminder_time}")
    
    # Goals
    st.subheader("🎯 Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        coherence_goal = st.slider("Target Average Coherence", 0.0, 1.0, 0.7)
        st.caption("Your goal for average coherence")
    
    with col2:
        flow_goal = st.slider("Target Flow Percentage", 0, 100, 30)
        st.caption("Percentage of time in flow state")
    
    # Data management
    st.subheader("📊 Data Management")
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("I understand this will delete all my data"):
            clear_user_data(st.session_state.user_name)
            st.success("Data cleared!")
    
    # About
    st.subheader("ℹ️ About")
    st.info("""
    **Creative Flow Tracker** uses Grounded Coherence Theory (GCT) to help you understand and optimize your creative states.
    
    **Components:**
    - 🧠 **Clarity (ψ)**: Mental clarity and focus
    - 📚 **Wisdom (ρ)**: Depth of insight and understanding  
    - ❤️ **Emotion (q)**: Emotional engagement and energy
    - 🤝 **Connection (f)**: Social connection and collaboration
    
    **Creative States:**
    - 🌊 **Flow**: Optimal creative performance
    - 💡 **Illumination**: Breakthrough moments
    - 🔍 **Exploration**: Discovering new ideas
    - 🌙 **Incubation**: Subconscious processing
    - 🚧 **Blocked**: Creative obstacles
    """)

# Helper functions
def save_quick_entry(user_name, values, project, activity, notes, mood):
    """Save a quick entry to the database"""
    # Create GCT variables
    variables = GCTVariables(
        psi=values['psi'],
        rho=values['rho'],
        q_raw=values['q'],
        f=values['f'],
        timestamp=datetime.now()
    )
    
    # Calculate coherence
    coherence = calculate_simple_coherence(values)
    state = detect_creative_state(values, coherence)
    
    # Save to simple JSON file for easy access
    entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_name,
        'project': project,
        'activity': activity,
        'mood': mood,
        'psi': values['psi'],
        'rho': values['rho'],
        'q': values['q'],
        'f': values['f'],
        'coherence': coherence,
        'state': state,
        'notes': notes
    }
    
    # Save to file
    save_to_file(entry)

def save_to_file(entry):
    """Save entry to JSON file"""
    file_path = Path("creative_flow_data.json")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(entry)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_user_data(user_name):
    """Load user data from file"""
    file_path = Path("creative_flow_data.json")
    
    if not file_path.exists():
        return pd.DataFrame()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    if df.empty:
        return df
    
    # Filter by user
    df = df[df['user'] == user_name].copy()
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def calculate_simple_coherence(values):
    """Simple coherence calculation"""
    psi = values['psi']
    rho = values['rho']
    q = values['q']
    f = values['f']
    
    # Optimize q
    q_opt = q / (1 + 0.5 * q)
    
    # Simple coherence formula
    coherence = (psi + rho * psi + q_opt + f * psi) / (1 + rho + f)
    
    return min(1.0, max(0.0, coherence))

def detect_creative_state(values, coherence):
    """Detect creative state from values"""
    psi = values['psi']
    q = values['q']
    
    if coherence > 0.75 and psi > 0.7:
        return "Flow"
    elif coherence > 0.6 and psi > 0.8:
        return "Illumination"
    elif psi < 0.4 and q > 0.6:
        return "Exploration"
    elif coherence < 0.3:
        return "Blocked"
    elif psi < 0.4 and q < 0.4:
        return "Incubation"
    else:
        return "Transition"

def calculate_flow_score(values, coherence):
    """Calculate flow score"""
    if coherence > 0.75:
        return coherence
    return coherence * 0.5

def get_recent_projects(user_name):
    """Get list of recent projects"""
    df = load_user_data(user_name)
    if df.empty or 'project' not in df.columns:
        return []
    
    return df['project'].dropna().unique().tolist()[-10:]  # Last 10 projects

def filter_by_time_range(df, time_range):
    """Filter dataframe by time range"""
    if df.empty:
        return df
    
    now = datetime.now()
    
    if time_range == "Last 7 Days":
        start_date = now - timedelta(days=7)
    elif time_range == "Last 30 Days":
        start_date = now - timedelta(days=30)
    elif time_range == "Last 3 Months":
        start_date = now - timedelta(days=90)
    else:  # All Time
        return df
    
    return df[df['timestamp'] >= start_date]

def create_coherence_timeline(df):
    """Create coherence timeline chart"""
    fig = go.Figure()
    
    # Add coherence line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['coherence'],
        mode='lines+markers',
        name='Coherence',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add flow zone
    fig.add_hrect(y0=0.75, y1=1.0, 
                  fillcolor="green", opacity=0.1,
                  annotation_text="Flow Zone")
    
    fig.update_layout(
        title="Coherence Over Time",
        xaxis_title="Date",
        yaxis_title="Coherence",
        yaxis_range=[0, 1],
        hovermode='x unified'
    )
    
    return fig

def create_state_distribution(df):
    """Create state distribution pie chart"""
    state_counts = df['state'].value_counts()
    
    colors = {
        'Flow': '#4CAF50',
        'Illumination': '#FFD700',
        'Exploration': '#FF6B6B',
        'Incubation': '#4ECDC4',
        'Blocked': '#95A5A6',
        'Transition': '#9B59B6'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=state_counts.index,
        values=state_counts.values,
        hole=0.3,
        marker_colors=[colors.get(state, '#333') for state in state_counts.index]
    )])
    
    fig.update_layout(title="Creative State Distribution")
    
    return fig

def create_time_heatmap(df):
    """Create time-based heatmap"""
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    
    # Create pivot table
    heatmap_data = df.pivot_table(
        values='coherence',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Best Creative Times",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week"
    )
    
    return fig

def generate_insights(df):
    """Generate insights from data"""
    insights = []
    
    if df.empty:
        return insights
    
    # Best time
    df['hour'] = df['timestamp'].dt.hour
    best_hour = df.groupby('hour')['coherence'].mean().idxmax()
    insights.append(f"Your most creative time is around {best_hour}:00")
    
    # Flow percentage
    flow_pct = (df['state'] == 'Flow').mean() * 100
    if flow_pct > 30:
        insights.append(f"Great job! You're in flow {flow_pct:.0f}% of the time")
    else:
        insights.append(f"Try to increase your flow time (currently {flow_pct:.0f}%)")
    
    # Consistency
    daily_entries = df.groupby(df['timestamp'].dt.date).size()
    if len(daily_entries) > 7:
        insights.append(f"You've been tracking for {len(daily_entries)} days. Keep it up!")
    
    return insights

def create_progress_chart(df, metric):
    """Create progress chart"""
    daily_avg = df.groupby(df['timestamp'].dt.date)[metric].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_avg.index,
        y=daily_avg.values,
        mode='lines+markers',
        name=metric.capitalize(),
        line=dict(width=2)
    ))
    
    # Add trend line only if we have enough data
    if len(daily_avg) > 1:
        try:
            z = np.polyfit(range(len(daily_avg)), daily_avg.values, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=daily_avg.index,
                y=p(range(len(daily_avg))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        except:
            pass  # Skip trend line if polyfit fails
    
    fig.update_layout(
        title=f"{metric.capitalize()} Progress",
        showlegend=True
    )
    
    return fig

def create_flow_progress(df):
    """Create flow achievement chart"""
    df['week'] = df['timestamp'].dt.to_period('W')
    weekly_flow = df.groupby('week').apply(lambda x: (x['state'] == 'Flow').mean() * 100)
    
    fig = go.Figure(data=[
        go.Bar(x=weekly_flow.index.astype(str), y=weekly_flow.values)
    ])
    
    fig.update_layout(
        title="Weekly Flow Achievement (%)",
        xaxis_title="Week",
        yaxis_title="Flow %"
    )
    
    return fig

def calculate_streak(df):
    """Calculate current tracking streak"""
    if df.empty:
        return 0
    
    dates = df['timestamp'].dt.date.unique()
    dates = sorted(dates, reverse=True)
    
    streak = 0
    today = datetime.now().date()
    
    for i, date in enumerate(dates):
        expected_date = today - timedelta(days=i)
        if date == expected_date:
            streak += 1
        else:
            break
    
    return streak

def check_milestones(df):
    """Check for achieved milestones"""
    milestones = []
    
    total_entries = len(df)
    if total_entries >= 100:
        milestones.append("100 entries logged!")
    elif total_entries >= 50:
        milestones.append("50 entries logged!")
    elif total_entries >= 10:
        milestones.append("10 entries logged!")
    
    flow_entries = (df['state'] == 'Flow').sum()
    if flow_entries >= 50:
        milestones.append("50 flow states achieved!")
    elif flow_entries >= 20:
        milestones.append("20 flow states achieved!")
    
    return milestones

def create_weekly_summary(df):
    """Create weekly summary chart"""
    # Last 4 weeks
    four_weeks_ago = datetime.now() - timedelta(weeks=4)
    recent_df = df[df['timestamp'] >= four_weeks_ago].copy()
    
    recent_df['week'] = recent_df['timestamp'].dt.to_period('W')
    
    weekly_stats = recent_df.groupby('week').agg({
        'coherence': 'mean',
        'state': 'count'
    }).round(2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weekly_stats.index.astype(str),
        y=weekly_stats['state'],
        name='Entries',
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=weekly_stats.index.astype(str),
        y=weekly_stats['coherence'],
        name='Avg Coherence',
        yaxis='y',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Weekly Summary",
        yaxis=dict(title="Coherence", side="left", range=[0, 1]),
        yaxis2=dict(title="Entries", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    return fig

def generate_report(df, user_name):
    """Generate text report"""
    report = f"""
CREATIVE FLOW REPORT
User: {user_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

SUMMARY
-------
Total Entries: {len(df)}
Date Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}
Average Coherence: {df['coherence'].mean():.2f}
Flow Percentage: {(df['state'] == 'Flow').mean() * 100:.1f}%

STATE DISTRIBUTION
-----------------
"""
    
    state_dist = df['state'].value_counts()
    for state, count in state_dist.items():
        report += f"{state}: {count} ({count/len(df)*100:.1f}%)\n"
    
    report += f"""
TOP PROJECTS
-----------
"""
    
    if 'project' in df.columns:
        project_stats = df.groupby('project')['coherence'].agg(['mean', 'count']).round(2)
        project_stats = project_stats.sort_values('mean', ascending=False).head(5)
        
        for project, stats in project_stats.iterrows():
            report += f"{project}: {stats['mean']:.2f} avg coherence ({int(stats['count'])} entries)\n"
    
    return report

def clear_user_data(user_name):
    """Clear user data"""
    file_path = Path("creative_flow_data.json")
    
    if not file_path.exists():
        return
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Filter out user's data
    data = [entry for entry in data if entry.get('user') != user_name]
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()