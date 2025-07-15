"""
Creative Flow Visualization System
Real-time visual feedback for creative states and processes
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import colorsys

from .creative_flow_engine import CreativeState, CreativeMetrics, BiometricData


class CreativeFlowVisualizer:
    """Specialized visualizations for creative flow states"""
    
    def __init__(self):
        # Color schemes for different creative states
        self.state_colors = {
            CreativeState.EXPLORATION: '#FF6B6B',     # Vibrant red
            CreativeState.INCUBATION: '#4ECDC4',      # Calm teal
            CreativeState.ILLUMINATION: '#FFE66D',    # Bright yellow
            CreativeState.VERIFICATION: '#95E1D3',    # Mint green
            CreativeState.FLOW: '#A8E6CF',           # Serene green
            CreativeState.BLOCKED: '#2D3436',        # Dark gray
            CreativeState.TRANSITION: '#DDA0DD',     # Plum
            CreativeState.PREPARATION: '#87CEEB'     # Sky blue
        }
        
        # Creative component colors
        self.component_colors = {
            'psi': '#7B68EE',     # Medium slate blue (clarity)
            'rho': '#FFD700',     # Gold (wisdom)
            'q': '#FF1493',       # Deep pink (emotion)
            'f': '#32CD32'        # Lime green (social)
        }
        
    def create_coherence_mandala(self, gct_result: Dict, 
                                creative_metrics: CreativeMetrics) -> go.Figure:
        """Create a mandala visualization of creative coherence"""
        
        # Extract component values
        components = gct_result.get('components', {})
        psi = components.get('psi', 0)
        rho = components.get('rho', 0)
        q = components.get('q_raw', 0)
        f = components.get('f', 0)
        
        # Create polar coordinates for mandala
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Create layers of the mandala
        fig = go.Figure()
        
        # Core coherence circle
        coherence = gct_result.get('coherence', 0)
        r_core = coherence * 0.3
        fig.add_trace(go.Scatterpolar(
            r=[r_core] * len(theta),
            theta=np.degrees(theta),
            fill='toself',
            fillcolor=f'rgba(255, 255, 255, {coherence})',
            line=dict(color='white', width=2),
            name='Core Coherence',
            showlegend=False
        ))
        
        # Component petals
        components_data = [
            ('Clarity', psi, self.component_colors['psi']),
            ('Wisdom', rho, self.component_colors['rho']),
            ('Emotion', q, self.component_colors['q']),
            ('Social', f, self.component_colors['f'])
        ]
        
        for i, (name, value, color) in enumerate(components_data):
            angle = i * 90  # 4 components at 90-degree intervals
            
            # Create petal shape
            petal_theta = np.linspace(angle - 30, angle + 30, 50)
            petal_r = value * 0.5 * (1 + 0.3 * np.cos(6 * np.radians(petal_theta - angle)))
            
            fig.add_trace(go.Scatterpolar(
                r=petal_r,
                theta=petal_theta,
                fill='toself',
                fillcolor=f'rgba{(*self._hex_to_rgb(color), 0.6)}',
                line=dict(color=color, width=1),
                name=f'{name}: {value:.2f}',
                showlegend=True
            ))
        
        # Creative metrics as outer ring segments
        metrics_data = [
            ('Novelty', creative_metrics.novelty_score),
            ('Fluency', creative_metrics.fluency_rate),
            ('Flexibility', creative_metrics.flexibility_index),
            ('Elaboration', creative_metrics.elaboration_depth),
            ('Flow', creative_metrics.flow_intensity)
        ]
        
        segment_size = 360 / len(metrics_data)
        for i, (metric_name, value) in enumerate(metrics_data):
            start_angle = i * segment_size
            end_angle = (i + 1) * segment_size
            
            segment_theta = np.linspace(start_angle, end_angle, 20)
            inner_r = 0.6
            outer_r = 0.6 + value * 0.3
            
            # Create ring segment
            r_values = [inner_r] * len(segment_theta) + [outer_r] * len(segment_theta)
            theta_values = list(segment_theta) + list(segment_theta[::-1])
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                fillcolor=f'rgba(128, 128, 128, {value * 0.4})',
                line=dict(color='gray', width=0.5),
                name=f'{metric_name}: {value:.2f}',
                showlegend=False
            ))
        
        # Update layout for mandala appearance
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(visible=False)
            ),
            title={
                'text': f"Creative Coherence Mandala<br><sub>Overall Coherence: {coherence:.2f}</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=True,
            legend=dict(x=1.1, y=0.5),
            height=600,
            width=800
        )
        
        return fig
    
    def create_flow_river(self, history: List[Dict], 
                         time_window: timedelta = timedelta(hours=1)) -> go.Figure:
        """Create a flowing river visualization of creative states over time"""
        
        if not history:
            return go.Figure()
        
        # Extract time series data
        times = [h['timestamp'] for h in history]
        states = [h['creative_state'] for h in history]
        coherences = [h['gct_result']['coherence'] for h in history]
        
        # Create figure
        fig = go.Figure()
        
        # Add coherence river
        fig.add_trace(go.Scatter(
            x=times,
            y=coherences,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(135, 206, 235, 0.3)',
            line=dict(color='skyblue', width=3, shape='spline'),
            name='Coherence Flow'
        ))
        
        # Add state markers
        for i, (time, state, coherence) in enumerate(zip(times, states, coherences)):
            color = self.state_colors.get(state, '#888888')
            
            fig.add_trace(go.Scatter(
                x=[time],
                y=[coherence],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(color='white', width=2)
                ),
                name=state.value,
                showlegend=i == 0,  # Only show first occurrence in legend
                hovertext=f"{state.value}<br>Coherence: {coherence:.2f}"
            ))
        
        # Add flow zones
        fig.add_hrect(
            y0=0.75, y1=1.0,
            fillcolor="green", opacity=0.1,
            annotation_text="Flow Zone", annotation_position="top right"
        )
        
        fig.add_hrect(
            y0=0, y1=0.3,
            fillcolor="red", opacity=0.1,
            annotation_text="Blocked Zone", annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title="Creative Flow River",
            xaxis_title="Time",
            yaxis_title="Coherence",
            yaxis_range=[0, 1],
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_breakthrough_radar(self, creative_metrics: CreativeMetrics,
                                 breakthrough_prob: float) -> go.Figure:
        """Create radar chart showing breakthrough indicators"""
        
        categories = [
            'Novelty',
            'Fluency', 
            'Flexibility',
            'Elaboration',
            'Convergence',
            'Flow Intensity'
        ]
        
        values = [
            creative_metrics.novelty_score,
            creative_metrics.fluency_rate,
            creative_metrics.flexibility_index,
            creative_metrics.elaboration_depth,
            creative_metrics.convergence_ratio,
            creative_metrics.flow_intensity
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        # Add current metrics
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f'rgba(255, 99, 71, {breakthrough_prob})',
            line=dict(color='tomato', width=2),
            name='Current State'
        ))
        
        # Add breakthrough threshold
        threshold_values = [0.7] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=threshold_values,
            theta=categories,
            line=dict(color='gold', width=2, dash='dash'),
            name='Breakthrough Threshold'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title={
                'text': f"Breakthrough Potential: {breakthrough_prob:.1%}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_biometric_dashboard(self, biometrics: Optional[BiometricData]) -> go.Figure:
        """Create a comprehensive biometric dashboard"""
        
        if not biometrics:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Heart Rate Variability', 'Brain Waves', 'Arousal',
                'Eye Movement', 'Physical State', 'Overall Balance'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'scatter'}]
            ]
        )
        
        # HRV gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=biometrics.hrv,
            title={'text': "HRV"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': self._hrv_color(biometrics.hrv)},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "gray"},
                    {'range': [0.7, 1], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ), row=1, col=1)
        
        # Brain waves bar chart
        brain_waves = {
            'Theta': biometrics.eeg_theta,
            'Alpha': biometrics.eeg_alpha,
            'Gamma': biometrics.eeg_gamma
        }
        
        fig.add_trace(go.Bar(
            x=list(brain_waves.keys()),
            y=list(brain_waves.values()),
            marker_color=['purple', 'blue', 'orange'],
            showlegend=False
        ), row=1, col=2)
        
        # GSR gauge (arousal)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=biometrics.gsr,
            title={'text': "Arousal"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': 'orange'},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightblue"},
                    {'range': [0.3, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}
                ]
            }
        ), row=1, col=3)
        
        # Eye movement entropy
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=biometrics.eye_movement_entropy,
            title={'text': "Visual Exploration"},
            delta={'reference': 0.5, 'relative': True}
        ), row=2, col=1)
        
        # Posture stability
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=biometrics.posture_stability,
            title={'text': "Physical Stability"},
            delta={'reference': 0.7, 'relative': True}
        ), row=2, col=2)
        
        # Overall balance scatter
        balance_score = self._calculate_biometric_balance(biometrics)
        fig.add_trace(go.Scatter(
            x=['Relaxation', 'Focus', 'Energy'],
            y=[
                biometrics.hrv,
                (biometrics.eeg_alpha + biometrics.eeg_theta) / 2,
                biometrics.gsr
            ],
            mode='markers+lines',
            marker=dict(size=20, color='green'),
            line=dict(width=2),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            showlegend=False
        ), row=2, col=3)
        
        # Update layout
        fig.update_layout(
            title="Biometric Creative State Dashboard",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_collaborative_network(self, team_composition: Dict,
                                   interaction_patterns: Dict[Tuple[str, str], float]) -> go.Figure:
        """Create network visualization of team collaboration"""
        
        # Extract team members
        members = team_composition.get('members', [])
        if not members:
            return go.Figure()
        
        # Create network layout
        n_members = len(members)
        angles = np.linspace(0, 2 * np.pi, n_members, endpoint=False)
        
        # Position nodes in a circle
        x_nodes = np.cos(angles)
        y_nodes = np.sin(angles)
        
        # Create figure
        fig = go.Figure()
        
        # Add interaction edges
        for (member1_id, member2_id), strength in interaction_patterns.items():
            # Find member positions
            idx1 = next((i for i, m in enumerate(members) if m.id == member1_id), None)
            idx2 = next((i for i, m in enumerate(members) if m.id == member2_id), None)
            
            if idx1 is not None and idx2 is not None:
                # Edge color based on strength
                edge_color = f'rgba(0, 0, 255, {strength})'
                
                fig.add_trace(go.Scatter(
                    x=[x_nodes[idx1], x_nodes[idx2]],
                    y=[y_nodes[idx1], y_nodes[idx2]],
                    mode='lines',
                    line=dict(width=strength * 5, color=edge_color),
                    showlegend=False,
                    hoverinfo='none'
                ))
        
        # Add member nodes
        role_assignments = team_composition.get('role_assignments', {})
        
        for i, member in enumerate(members):
            role = role_assignments.get(member.id, 'unknown')
            role_color = self._get_role_color(role)
            
            fig.add_trace(go.Scatter(
                x=[x_nodes[i]],
                y=[y_nodes[i]],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color=role_color,
                    line=dict(color='white', width=2)
                ),
                text=member.name,
                textposition='top center',
                name=f"{member.name} ({role})",
                hovertext=f"Role: {role}<br>Coherence: {member.coherence_history[-1] if member.coherence_history else 0:.2f}"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Team Collaboration Network<br><sub>Predicted Synergy: {team_composition.get('predicted_synergy', 0):.2f}</sub>",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800,
            showlegend=True
        )
        
        return fig
    
    def create_environment_settings_visual(self, recommendations: Dict) -> go.Figure:
        """Visualize recommended environment settings"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Lighting', 'Sound', 'Visual Complexity', 'Workspace'),
            specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # Extract settings
        immediate = recommendations.get('immediate_changes', [])
        
        # Lighting settings
        lighting = next((c for c in immediate if c['type'] == 'lighting'), None)
        if lighting:
            settings = lighting['settings']
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=settings['temperature'],
                title={'text': "Color Temperature (K)"},
                gauge={
                    'axis': {'range': [2000, 6500]},
                    'bar': {'color': self._kelvin_to_rgb(settings['temperature'])},
                    'steps': [
                        {'range': [2000, 3000], 'color': "lightyellow"},
                        {'range': [3000, 4500], 'color': "white"},
                        {'range': [4500, 6500], 'color': "lightblue"}
                    ]
                }
            ), row=1, col=1)
        
        # Sound settings
        sound = next((c for c in immediate if c['type'] == 'sound'), None)
        if sound:
            settings = sound['settings']
            sound_types = ['ambient', 'binaural', 'nature', 'silence']
            sound_values = [1 if settings['type'] == t else 0.2 for t in sound_types]
            
            fig.add_trace(go.Bar(
                x=sound_types,
                y=sound_values,
                marker_color=['skyblue', 'purple', 'green', 'gray'],
                showlegend=False
            ), row=1, col=2)
        
        # Visual complexity pie
        fig.add_trace(go.Pie(
            labels=['High Complexity', 'Medium Complexity', 'Low Complexity'],
            values=[1, 2, 3],  # Example values
            hole=0.3,
            marker_colors=['red', 'yellow', 'green'],
            showlegend=False
        ), row=2, col=1)
        
        # Workspace elements scatter
        workspace_elements = ['Tools', 'Distractions', 'Comfort', 'Inspiration']
        element_scores = [0.8, 0.2, 0.9, 0.7]  # Example scores
        
        fig.add_trace(go.Scatter(
            x=workspace_elements,
            y=element_scores,
            mode='markers',
            marker=dict(size=30, color=element_scores, colorscale='Viridis'),
            showlegend=False
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Optimal Creative Environment Settings",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _hrv_color(self, hrv: float) -> str:
        """Get color for HRV value"""
        if hrv < 0.3:
            return "red"
        elif hrv < 0.7:
            return "green"
        else:
            return "orange"
    
    def _calculate_biometric_balance(self, biometrics: BiometricData) -> float:
        """Calculate overall biometric balance score"""
        relaxation = biometrics.hrv
        focus = (biometrics.eeg_alpha + biometrics.eeg_theta) / 2
        energy = biometrics.gsr
        
        # Optimal balance has moderate values for all
        balance = 1 - np.std([relaxation, focus, energy])
        return balance
    
    def _get_role_color(self, role: str) -> str:
        """Get color for creative role"""
        role_colors = {
            'explorer': '#FF6B6B',
            'synthesizer': '#4ECDC4',
            'illuminator': '#FFE66D',
            'refiner': '#95E1D3',
            'connector': '#A8E6CF'
        }
        return role_colors.get(role, '#888888')
    
    def _kelvin_to_rgb(self, kelvin: int) -> str:
        """Convert color temperature to RGB"""
        # Simplified conversion
        if kelvin < 3000:
            return "rgb(255, 200, 150)"  # Warm
        elif kelvin < 4500:
            return "rgb(255, 255, 255)"  # Neutral
        else:
            return "rgb(200, 220, 255)"  # Cool