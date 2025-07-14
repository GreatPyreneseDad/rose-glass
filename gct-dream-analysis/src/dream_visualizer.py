"""
Advanced Visualizations for Dream Analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import colorsys


class DreamVisualizer:
    """Create beautiful visualizations for dream analysis"""
    
    def __init__(self):
        # Dream-themed color palette
        self.colors = {
            'primary': '#60a5fa',      # Soft blue
            'secondary': '#a78bfa',    # Lavender
            'accent': '#f472b6',       # Pink
            'success': '#4ade80',      # Green
            'warning': '#fbbf24',      # Amber
            'danger': '#f87171',       # Red
            'dark': '#1e293b',         # Dark blue
            'light': '#e0e7ff'         # Light purple
        }
        
        # Emotion colors
        self.emotion_colors = {
            'fear': '#ef4444',
            'joy': '#10b981',
            'anger': '#f59e0b',
            'sadness': '#3b82f6',
            'love': '#ec4899',
            'shame': '#8b5cf6',
            'neutral': '#6b7280'
        }
        
        # State colors
        self.state_colors = {
            'lucid_integration': '#10b981',
            'shadow_work': '#8b5cf6',
            'breakthrough': '#f59e0b',
            'integrated': '#4ade80',
            'processing': '#3b82f6',
            'fragmenting': '#ef4444',
            'stable': '#60a5fa',
            'recurring_pattern': '#a78bfa',
            'exploring': '#06b6d4',
            'emotional_processing': '#ec4899',
            'chaotic': '#dc2626'
        }
    
    def create_coherence_mandala(self, components: Dict[str, float]) -> go.Figure:
        """Create a mandala visualization of coherence components"""
        
        # Prepare data for radial plot
        categories = list(components.keys())
        values = list(components.values())
        
        # Create multiple rings for depth
        fig = go.Figure()
        
        # Outer ring - actual values
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(96, 165, 250, 0.3)',
            line=dict(color=self.colors['primary'], width=3),
            name='Current'
        ))
        
        # Inner ring - 70% of values for visual depth
        fig.add_trace(go.Scatterpolar(
            r=[v * 0.7 for v in values],
            theta=categories,
            fill='toself',
            fillcolor='rgba(167, 139, 250, 0.2)',
            line=dict(color=self.colors['secondary'], width=2),
            name='Core'
        ))
        
        # Center point
        fig.add_trace(go.Scatterpolar(
            r=[0.1],
            theta=[categories[0]],
            mode='markers',
            marker=dict(size=20, color=self.colors['accent']),
            name='Center',
            showlegend=False
        ))
        
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(30, 41, 59, 0.5)',
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2],
                    showticklabels=False,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    linecolor='rgba(255, 255, 255, 0.2)'
                )
            ),
            showlegend=False,
            title={
                'text': 'Dream Coherence Mandala',
                'font': {'color': 'white', 'size': 20}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_emotional_constellation(self, emotion_timeline: List[Dict]) -> go.Figure:
        """Create a constellation view of emotional patterns"""
        
        if not emotion_timeline:
            return go.Figure()
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(emotion_timeline)
        
        # Create figure
        fig = go.Figure()
        
        # Plot each emotion as a star/node
        for i, row in df.iterrows():
            emotion = row['emotion']
            intensity = row.get('intensity', 0.5)
            
            # Position based on emotion type and time
            angle = list(self.emotion_colors.keys()).index(emotion) * (2 * np.pi / len(self.emotion_colors))
            x = intensity * np.cos(angle) + i * 0.1
            y = intensity * np.sin(angle)
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=20 * intensity,
                    color=self.emotion_colors.get(emotion, '#6b7280'),
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                name=emotion,
                text=f"{emotion}<br>Intensity: {intensity:.2f}",
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add connections to previous emotion
            if i > 0:
                prev_row = df.iloc[i-1]
                prev_emotion = prev_row['emotion']
                prev_intensity = prev_row.get('intensity', 0.5)
                prev_angle = list(self.emotion_colors.keys()).index(prev_emotion) * (2 * np.pi / len(self.emotion_colors))
                prev_x = prev_intensity * np.cos(prev_angle) + (i-1) * 0.1
                prev_y = prev_intensity * np.sin(prev_angle)
                
                fig.add_trace(go.Scatter(
                    x=[prev_x, x],
                    y=[prev_y, y],
                    mode='lines',
                    line=dict(
                        color='rgba(255, 255, 255, 0.3)',
                        width=1
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title='Emotional Constellation',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig
    
    def create_dream_spiral(self, coherence_history: List[float], 
                          dates: List[datetime]) -> go.Figure:
        """Create a spiral visualization of coherence over time"""
        
        if len(coherence_history) < 3:
            return go.Figure()
        
        # Generate spiral coordinates
        n_points = len(coherence_history)
        theta = np.linspace(0, 4 * np.pi, n_points)
        
        # Radius based on coherence
        r = np.array(coherence_history) * 5 + 1
        
        # Convert to cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create color gradient based on time
        colors = [f'hsl({int(i * 360 / n_points)}, 70%, 50%)' for i in range(n_points)]
        
        fig = go.Figure()
        
        # Add spiral line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(
                color=colors,
                width=3
            ),
            marker=dict(
                size=[5 + c * 10 for c in coherence_history],
                color=coherence_history,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Coherence')
            ),
            text=[f"Date: {d.strftime('%Y-%m-%d')}<br>Coherence: {c:.3f}" 
                  for d, c in zip(dates, coherence_history)],
            hoverinfo='text'
        ))
        
        # Add center point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=10, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='Dream Coherence Spiral',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_symbol_network(self, symbol_connections: Dict[str, List[str]],
                            symbol_frequencies: Dict[str, int]) -> go.Figure:
        """Create a network graph of symbol relationships"""
        
        # Create nodes
        nodes = list(symbol_connections.keys())
        node_sizes = [10 + symbol_frequencies.get(node, 1) * 5 for node in nodes]
        
        # Create edges
        edge_x = []
        edge_y = []
        
        # Position nodes in a circle
        n_nodes = len(nodes)
        node_positions = {}
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n_nodes
            x = np.cos(angle)
            y = np.sin(angle)
            node_positions[node] = (x, y)
        
        # Create edges
        for node, connections in symbol_connections.items():
            if node in node_positions:
                x0, y0 = node_positions[node]
                for connected_node in connections:
                    if connected_node in node_positions:
                        x1, y1 = node_positions[connected_node]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(255, 255, 255, 0.3)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        node_x = [pos[0] for pos in node_positions.values()]
        node_y = [pos[1] for pos in node_positions.values()]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=[symbol_frequencies.get(node, 1) for node in nodes],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Frequency')
            ),
            text=nodes,
            textposition='top center',
            hovertext=[f"{node}<br>Frequency: {symbol_frequencies.get(node, 0)}" 
                      for node in nodes],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Dream Symbol Network',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_sleep_coherence_heatmap(self, sleep_data: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing sleep quality vs dream coherence patterns"""
        
        # Prepare data for heatmap
        # Group by sleep quality (rounded) and time of week
        if 'date' in sleep_data.columns:
            sleep_data['weekday'] = pd.to_datetime(sleep_data['date']).dt.day_name()
            sleep_data['sleep_quality_rounded'] = (sleep_data['sleep_quality'] * 10).round()
            
            # Create pivot table
            heatmap_data = sleep_data.pivot_table(
                values='avg_coherence',
                index='sleep_quality_rounded',
                columns='weekday',
                aggfunc='mean'
            )
            
            # Reorder weekdays
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(columns=weekday_order)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                text=np.round(heatmap_data.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Sleep Quality × Dream Coherence Patterns',
                xaxis_title='Day of Week',
                yaxis_title='Sleep Quality',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='white')
            )
            
            return fig
        
        return go.Figure()  # Return empty figure if data not suitable
    
    def create_coherence_flow_river(self, timeline_data: pd.DataFrame) -> go.Figure:
        """Create a river/stream plot showing component flow over time"""
        
        components = ['psi', 'rho', 'q_opt', 'f']
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['warning']]
        
        fig = go.Figure()
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            if comp in timeline_data.columns:
                fig.add_trace(go.Scatter(
                    x=timeline_data['dream_date'],
                    y=timeline_data[comp],
                    mode='lines',
                    name=comp.upper(),
                    line=dict(width=0.5, color=color),
                    stackgroup='one',
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.4)'),
                    hovertemplate='%{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Coherence Components Flow',
            xaxis_title='Date',
            yaxis_title='Component Value',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            font=dict(color='white')
        )
        
        return fig