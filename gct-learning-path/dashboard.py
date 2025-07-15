"""
Streamlit Dashboard for GCT Learning Path Visualization
Interactive exploration of learning paths and coherence
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys
import json

sys.path.append('src')

from repository import CourseRepository
from metadata_extractor import MetadataExtractor
from gct_engine import LearningGCTEngine
from path_optimizer import LearningPathOptimizer


# Page config
st.set_page_config(
    page_title="GCT Learning Path Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'repo' not in st.session_state:
    st.session_state.repo = CourseRepository({'type': 'json', 'path': 'data/demo_modules.json'})
    st.session_state.extractor = MetadataExtractor()
    st.session_state.gct_engine = LearningGCTEngine()
    st.session_state.current_path = None


def main():
    st.title("🎓 GCT Learning Path Dashboard")
    st.markdown("*Visualize and explore adaptive learning paths powered by Grounded Coherence Theory*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Learner profile
        st.subheader("Learner Profile")
        learner_name = st.text_input("Name", "Explorer")
        skill_level = st.slider("Skill Level", 0.0, 1.0, 0.5)
        learning_style = st.selectbox(
            "Learning Style",
            ["visual", "reading", "kinesthetic", "balanced"]
        )
        
        max_daily_minutes = st.number_input(
            "Max Daily Minutes",
            min_value=15,
            max_value=240,
            value=60,
            step=15
        )
        
        # GCT weights
        st.subheader("GCT Weights")
        psi = st.slider("ψ (Topic Coherence)", 0.0, 1.0, 0.3)
        rho = st.slider("ρ (Knowledge Building)", 0.0, 1.0, 0.2)
        q_opt = st.slider("q (Difficulty Progress)", 0.0, 1.0, 0.2)
        flow = st.slider("f (Flow State)", 0.0, 1.0, 0.2)
        alpha = st.slider("α (Personalization)", 0.0, 1.0, 0.1)
        
        # Update engine weights
        from gct_engine import GCTWeights
        st.session_state.gct_engine.weights = GCTWeights(
            psi=psi, rho=rho, q_opt=q_opt, flow=flow, alpha=alpha
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Module Explorer",
        "🛤️ Path Generator", 
        "📊 Coherence Analysis",
        "📈 Learning Analytics"
    ])
    
    with tab1:
        module_explorer()
    
    with tab2:
        path_generator(learner_name, skill_level, learning_style, max_daily_minutes)
    
    with tab3:
        coherence_analysis()
    
    with tab4:
        learning_analytics()


def module_explorer():
    """Explore available learning modules"""
    st.header("📚 Module Explorer")
    
    # Load modules
    modules = st.session_state.repo.list_modules()
    
    if not modules:
        st.info("No modules found. Run example_usage.py to create sample modules.")
        return
    
    # Module selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        module_ids = [m['module_id'] for m in modules]
        selected_module = st.selectbox("Select Module", module_ids)
    
    with col2:
        # Topic filter
        all_topics = set()
        for m in modules:
            all_topics.update(m.get('topic_tags', []))
        
        selected_topics = st.multiselect("Filter by Topics", sorted(all_topics))
    
    # Display module details
    if selected_module:
        module = next(m for m in modules if m['module_id'] == selected_module)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Difficulty", f"{module['difficulty']:.2f}")
            st.metric("Duration", f"{module['duration_minutes']} min")
        
        with col2:
            st.metric("Cognitive Load", f"{module.get('cognitive_load', 0.5):.2f}")
            st.metric("Engagement", f"{module.get('engagement_score', 0.7):.2f}")
        
        with col3:
            st.metric("Prerequisites", len(module.get('prerequisites', [])))
            st.metric("Objectives", len(module.get('learning_objectives', [])))
        
        # Module information
        st.subheader("Module Information")
        st.write(f"**Title:** {module['title']}")
        st.write(f"**Description:** {module['description']}")
        st.write(f"**Topics:** {', '.join(module['topic_tags'])}")
        
        if module.get('prerequisites'):
            st.write(f"**Prerequisites:** {', '.join(module['prerequisites'])}")
        
        if module.get('learning_objectives'):
            st.write("**Learning Objectives:**")
            for obj in module['learning_objectives']:
                st.write(f"- {obj}")
    
    # Module network visualization
    st.subheader("Module Relationship Network")
    create_module_network(modules)


def path_generator(learner_name, skill_level, learning_style, max_daily_minutes):
    """Generate and visualize learning paths"""
    st.header("🛤️ Path Generator")
    
    # Path configuration
    col1, col2, col3 = st.columns(3)
    
    modules = st.session_state.repo.list_modules()
    module_ids = [m['module_id'] for m in modules]
    
    with col1:
        start_module = st.selectbox("Start Module", module_ids)
    
    with col2:
        target_module = st.selectbox("Target Module", module_ids)
    
    with col3:
        max_steps = st.number_input("Max Steps", 3, 20, 10)
    
    if st.button("🚀 Generate Learning Path", type="primary"):
        with st.spinner("Calculating optimal path..."):
            # Create learner profile
            learner_profile = {
                'learner_id': learner_name.lower().replace(' ', '_'),
                'skill_level': skill_level,
                'learning_style': learning_style,
                'goals': [f"Master {target_module}"],
                'constraints': {'max_daily_minutes': max_daily_minutes}
            }
            
            # Extract metadata
            topic_vectors_df = st.session_state.extractor.extract_topic_vectors(modules)
            difficulty_df = st.session_state.extractor.assign_difficulty_scores(modules)
            metadata_df = pd.merge(topic_vectors_df, difficulty_df, on='module_id')
            
            # Calculate coherence
            score_matrix = st.session_state.gct_engine.score_transitions(metadata_df)
            module_graph = st.session_state.gct_engine.create_transition_graph(score_matrix)
            
            # Generate path
            optimizer = LearningPathOptimizer(st.session_state.gct_engine, learner_profile)
            path = optimizer.build_path(start_module, target_module, module_graph, metadata_df, max_steps)
            
            st.session_state.current_path = path
            
            # Display results
            st.success(f"✨ Path generated with coherence score: {path.total_coherence:.2f}")
            
            # Path summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Modules", len(path.modules))
            with col2:
                st.metric("Total Duration", f"{path.estimated_duration} min")
            with col3:
                st.metric("Avg Coherence", f"{path.total_coherence:.2f}")
            
            # Path visualization
            st.subheader("Learning Path Visualization")
            visualize_learning_path(path, modules, score_matrix)
            
            # Detailed path steps
            st.subheader("Detailed Path Steps")
            for i, module_id in enumerate(path.modules):
                module = next(m for m in modules if m['module_id'] == module_id)
                
                with st.expander(f"Step {i+1}: {module['title']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {module['description']}")
                        st.write(f"**Topics:** {', '.join(module['topic_tags'])}")
                        st.write(f"**Duration:** {module['duration_minutes']} minutes")
                    
                    with col2:
                        st.metric("Difficulty", f"{path.difficulty_curve[i]:.2f}")
                        if i < len(path.modules) - 1:
                            next_module = path.modules[i + 1]
                            coherence = score_matrix.loc[module_id, next_module]
                            st.metric("Next Transition", f"{coherence:.2f}")
            
            # Alternative paths
            st.subheader("Alternative Paths")
            alternatives = optimizer.generate_alternative_paths(
                start_module, target_module, module_graph, metadata_df, n_alternatives=3
            )
            
            for i, alt_path in enumerate(alternatives):
                st.write(f"**Option {i+1}:** {' → '.join(alt_path.modules[:5])}{'...' if len(alt_path.modules) > 5 else ''}")
                st.write(f"Coherence: {alt_path.total_coherence:.2f}, Duration: {alt_path.estimated_duration} min")


def coherence_analysis():
    """Analyze coherence patterns"""
    st.header("📊 Coherence Analysis")
    
    modules = st.session_state.repo.list_modules()
    if not modules:
        st.info("No modules available for analysis")
        return
    
    # Extract metadata and calculate coherence
    with st.spinner("Calculating coherence matrix..."):
        topic_vectors_df = st.session_state.extractor.extract_topic_vectors(modules)
        difficulty_df = st.session_state.extractor.assign_difficulty_scores(modules)
        metadata_df = pd.merge(topic_vectors_df, difficulty_df, on='module_id')
        
        score_matrix = st.session_state.gct_engine.score_transitions(metadata_df)
    
    # Coherence heatmap
    st.subheader("Module Transition Coherence Heatmap")
    
    fig = go.Figure(data=go.Heatmap(
        z=score_matrix.values,
        x=score_matrix.columns,
        y=score_matrix.index,
        colorscale='Viridis',
        text=np.round(score_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Coherence Scores Between Modules",
        xaxis_title="To Module",
        yaxis_title="From Module",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Component analysis
    st.subheader("Coherence Component Analysis")
    
    # Select transition to analyze
    col1, col2 = st.columns(2)
    module_ids = score_matrix.index.tolist()
    
    with col1:
        from_module = st.selectbox("From Module", module_ids, key="comp_from")
    
    with col2:
        to_module = st.selectbox("To Module", module_ids, key="comp_to")
    
    if from_module and to_module and from_module != to_module:
        # Get component scores
        from_data = metadata_df[metadata_df['module_id'] == from_module].iloc[0].to_dict()
        to_data = metadata_df[metadata_df['module_id'] == to_module].iloc[0].to_dict()
        
        result = st.session_state.gct_engine.score_transition(from_data, to_data)
        
        # Display components
        components = result['components']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                text=[f"{v:.3f}" for v in components.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Component Breakdown: {from_module} → {to_module}",
            xaxis_title="Component",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.write(f"**Overall Coherence:** {result['coherence']:.3f}")
        st.write(f"**Transition Quality:** {result['transition_quality']}")


def learning_analytics():
    """Display learning analytics"""
    st.header("📈 Learning Analytics")
    
    # Simulated analytics data
    st.subheader("Module Difficulty Distribution")
    
    modules = st.session_state.repo.list_modules()
    difficulties = [m['difficulty'] for m in modules]
    
    fig = px.histogram(
        x=difficulties,
        nbins=20,
        title="Distribution of Module Difficulties",
        labels={'x': 'Difficulty', 'y': 'Count'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic coverage
    st.subheader("Topic Coverage")
    
    topic_counts = {}
    for module in modules:
        for topic in module.get('topic_tags', []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    fig = px.bar(
        x=list(topic_counts.values()),
        y=list(topic_counts.keys()),
        orientation='h',
        title="Module Count by Topic",
        labels={'x': 'Number of Modules', 'y': 'Topic'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Learning path statistics
    if st.session_state.current_path:
        st.subheader("Current Path Analysis")
        
        path = st.session_state.current_path
        
        # Difficulty progression
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(path.difficulty_curve))),
            y=path.difficulty_curve,
            mode='lines+markers',
            name='Difficulty',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Difficulty Progression",
            xaxis_title="Module Index",
            yaxis_title="Difficulty",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_module_network(modules):
    """Create network visualization of module relationships"""
    G = nx.DiGraph()
    
    # Add nodes
    for module in modules:
        G.add_node(module['module_id'], **module)
    
    # Add edges based on prerequisites
    for module in modules:
        for prereq in module.get('prerequisites', []):
            if prereq in G:
                G.add_edge(prereq, module['module_id'])
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=20,
            colorbar=dict(
                thickness=15,
                title='Difficulty',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        
    # Color by difficulty
    node_trace['marker']['color'] = [G.nodes[node]['difficulty'] for node in G.nodes()]
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=400
                   ))
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_learning_path(path, modules, score_matrix):
    """Visualize a specific learning path"""
    if not path or not path.modules:
        return
    
    # Create path visualization
    fig = go.Figure()
    
    # Module nodes
    x = list(range(len(path.modules)))
    y = path.difficulty_curve
    
    # Add path line
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='lightblue', width=3),
        name='Difficulty Path'
    ))
    
    # Add module markers
    colors = []
    sizes = []
    texts = []
    
    for i, module_id in enumerate(path.modules):
        module = next(m for m in modules if m['module_id'] == module_id)
        
        # Color by topic
        if 'python' in module['topic_tags']:
            colors.append('blue')
        elif 'data_science' in module['topic_tags']:
            colors.append('green')
        elif 'machine_learning' in module['topic_tags']:
            colors.append('red')
        else:
            colors.append('gray')
        
        sizes.append(20 + module['duration_minutes'] / 5)
        texts.append(f"{module['title']}<br>Duration: {module['duration_minutes']} min")
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[m.split('_')[-1] for m in path.modules],
        textposition="top center",
        hovertext=texts,
        hoverinfo='text',
        name='Modules'
    ))
    
    # Add coherence annotations
    for i in range(len(path.modules) - 1):
        coherence = score_matrix.loc[path.modules[i], path.modules[i + 1]]
        
        fig.add_annotation(
            x=i + 0.5,
            y=(y[i] + y[i + 1]) / 2,
            text=f"{coherence:.2f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    fig.update_layout(
        title="Learning Path Visualization",
        xaxis_title="Step",
        yaxis_title="Difficulty Level",
        yaxis_range=[0, 1],
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()