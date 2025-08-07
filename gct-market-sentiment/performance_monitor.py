"""
Real-time Performance Monitoring Dashboard for TraderAI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import psutil
import time
from typing import Dict, List
import numpy as np

# Page config
st.set_page_config(
    page_title="TraderAI Performance Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class PerformanceMonitor:
    """Real-time performance monitoring for TraderAI"""
    
    def __init__(self, db_path: str = "data/gct_market.db"):
        self.db_path = db_path
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = []
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_query_performance(self, hours: int = 1) -> pd.DataFrame:
        """Get query performance metrics"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    query_type,
                    AVG(execution_time_ms) as avg_time,
                    MAX(execution_time_ms) as max_time,
                    MIN(execution_time_ms) as min_time,
                    COUNT(*) as query_count,
                    SUM(rows_returned) as total_rows,
                    strftime('%Y-%m-%d %H:%M', timestamp) as time_bucket
                FROM QueryPerformance
                WHERE timestamp > datetime('now', '-{} hours')
                GROUP BY query_type, strftime('%Y-%m-%d %H:%M', timestamp)
                ORDER BY timestamp DESC
            """.format(hours)
            
            df = pd.read_sql_query(query, conn)
            return df
    
    def get_cache_performance(self) -> Dict:
        """Get cache hit rates and statistics"""
        with self.get_connection() as conn:
            # Cache statistics from summary table
            cache_query = """
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN last_updated > datetime('now', '-5 minutes') THEN 1 END) as fresh_entries,
                    AVG(total_articles) as avg_articles_cached
                FROM TickerSummaryCache
            """
            
            cache_stats = pd.read_sql_query(cache_query, conn).iloc[0].to_dict()
            
            # Calculate hit rate (simplified - in real system would track actual hits)
            cache_stats['hit_rate'] = (
                cache_stats['fresh_entries'] / max(cache_stats['total_entries'], 1)
            ) * 100
            
            return cache_stats
    
    def get_gct_performance(self, hours: int = 1) -> pd.DataFrame:
        """Get GCT processing performance"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    strftime('%Y-%m-%d %H:%M', created_at) as time_bucket,
                    COUNT(*) as articles_processed,
                    AVG(coherence) as avg_coherence,
                    AVG(ABS(dc_dt)) as avg_derivative,
                    SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END) as bearish_count,
                    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                FROM GCTScores
                WHERE created_at > datetime('now', '-{} hours')
                GROUP BY strftime('%Y-%m-%d %H:%M', created_at)
                ORDER BY time_bucket DESC
            """.format(hours)
            
            df = pd.read_sql_query(query, conn)
            return df
    
    def get_system_metrics(self) -> Dict:
        """Get system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'process_memory_mb': process_memory.rss / (1024**2),
            'thread_count': process.num_threads()
        }
    
    def plot_query_performance(self, df: pd.DataFrame):
        """Plot query performance metrics"""
        if df.empty:
            st.warning("No query performance data available")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Query Time', 'Query Volume', 
                          'Query Time Distribution', 'Performance by Type')
        )
        
        # Average query time over time
        for query_type in df['query_type'].unique():
            type_data = df[df['query_type'] == query_type]
            fig.add_trace(
                go.Scatter(
                    x=type_data['time_bucket'],
                    y=type_data['avg_time'],
                    mode='lines+markers',
                    name=query_type
                ),
                row=1, col=1
            )
        
        # Query volume
        volume_data = df.groupby('time_bucket')['query_count'].sum().reset_index()
        fig.add_trace(
            go.Bar(
                x=volume_data['time_bucket'],
                y=volume_data['query_count'],
                name='Query Count'
            ),
            row=1, col=2
        )
        
        # Query time distribution
        fig.add_trace(
            go.Box(
                y=df['avg_time'],
                x=df['query_type'],
                name='Query Times'
            ),
            row=2, col=1
        )
        
        # Performance by type
        type_summary = df.groupby('query_type').agg({
            'avg_time': 'mean',
            'query_count': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=type_summary['query_type'],
                y=type_summary['avg_time'],
                name='Avg Time (ms)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_gct_performance(self, df: pd.DataFrame):
        """Plot GCT processing performance"""
        if df.empty:
            st.warning("No GCT performance data available")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Articles Processed', 'Average Coherence', 
                          'Sentiment Distribution', 'Processing Rate')
        )
        
        # Articles processed over time
        fig.add_trace(
            go.Scatter(
                x=df['time_bucket'],
                y=df['articles_processed'],
                mode='lines+markers',
                name='Articles',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Average coherence
        fig.add_trace(
            go.Scatter(
                x=df['time_bucket'],
                y=df['avg_coherence'],
                mode='lines+markers',
                name='Coherence',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # Sentiment distribution
        sentiment_cols = ['bullish_count', 'bearish_count', 'neutral_count']
        colors = ['green', 'red', 'gray']
        
        for col, color in zip(sentiment_cols, colors):
            fig.add_trace(
                go.Bar(
                    x=df['time_bucket'],
                    y=df[col],
                    name=col.replace('_count', ''),
                    marker_color=color
                ),
                row=2, col=1
            )
        
        # Processing rate (articles per minute)
        if len(df) > 1:
            time_diffs = pd.to_datetime(df['time_bucket']).diff().dt.total_seconds() / 60
            processing_rate = df['articles_processed'] / time_diffs.fillna(1)
            
            fig.add_trace(
                go.Scatter(
                    x=df['time_bucket'],
                    y=processing_rate,
                    mode='lines+markers',
                    name='Articles/min',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_system_metrics(self, metrics: Dict):
        """Plot system resource metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{metrics['cpu_percent']:.1f}%",
                delta=f"{metrics['cpu_percent'] - 50:.1f}%"
            )
        
        with col2:
            st.metric(
                "Memory Usage",
                f"{metrics['memory_percent']:.1f}%",
                delta=f"{metrics['memory_used_gb']:.1f} GB"
            )
        
        with col3:
            st.metric(
                "Process Memory",
                f"{metrics['process_memory_mb']:.1f} MB",
                delta=f"{metrics['thread_count']} threads"
            )
        
        with col4:
            st.metric(
                "Disk Usage",
                f"{metrics['disk_percent']:.1f}%",
                delta=f"{100 - metrics['disk_percent']:.1f}% free"
            )
    
    def run(self):
        """Run the performance monitoring dashboard"""
        st.title("🚀 TraderAI Performance Monitor")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            
            time_range = st.selectbox(
                "Time Range",
                options=[1, 6, 12, 24],
                format_func=lambda x: f"Last {x} hours"
            )
            
            refresh_rate = st.slider(
                "Refresh Rate (seconds)",
                min_value=5,
                max_value=60,
                value=10
            )
            
            if st.button("🔄 Refresh Now"):
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        # Auto-refresh
        if (datetime.now() - st.session_state.last_update).seconds > refresh_rate:
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Query Performance",
            "🧠 GCT Processing",
            "💾 Cache Statistics",
            "💻 System Resources"
        ])
        
        with tab1:
            st.header("Database Query Performance")
            query_df = self.get_query_performance(time_range)
            self.plot_query_performance(query_df)
            
            # Show raw data
            with st.expander("View Raw Data"):
                st.dataframe(query_df)
        
        with tab2:
            st.header("GCT Processing Performance")
            gct_df = self.get_gct_performance(time_range)
            self.plot_gct_performance(gct_df)
            
            # Calculate and show summary stats
            if not gct_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_processed = gct_df['articles_processed'].sum()
                    st.metric("Total Articles", f"{total_processed:,}")
                
                with col2:
                    avg_rate = total_processed / max(time_range, 1)
                    st.metric("Avg Rate", f"{avg_rate:.1f}/hour")
                
                with col3:
                    avg_coherence = gct_df['avg_coherence'].mean()
                    st.metric("Avg Coherence", f"{avg_coherence:.3f}")
        
        with tab3:
            st.header("Cache Performance")
            cache_stats = self.get_cache_performance()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Cache Entries",
                    f"{cache_stats['total_entries']:,}",
                    delta=f"{cache_stats['fresh_entries']} fresh"
                )
            
            with col2:
                st.metric(
                    "Cache Hit Rate",
                    f"{cache_stats['hit_rate']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Avg Cached Articles",
                    f"{cache_stats['avg_articles_cached']:.0f}"
                )
            
            # Cache efficiency chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cache_stats['hit_rate'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cache Efficiency"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("System Resources")
            system_metrics = self.get_system_metrics()
            self.plot_system_metrics(system_metrics)
            
            # Historical system metrics
            st.session_state.metrics_history.append({
                'timestamp': datetime.now(),
                **system_metrics
            })
            
            # Keep only last 100 measurements
            if len(st.session_state.metrics_history) > 100:
                st.session_state.metrics_history = st.session_state.metrics_history[-100:]
            
            # Plot historical metrics
            if len(st.session_state.metrics_history) > 1:
                history_df = pd.DataFrame(st.session_state.metrics_history)
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('CPU & Memory Usage', 'Process Memory')
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['cpu_percent'],
                        mode='lines',
                        name='CPU %',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['memory_percent'],
                        mode='lines',
                        name='Memory %',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['process_memory_mb'],
                        mode='lines',
                        name='Process MB',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Footer with last update time
        st.markdown("---")
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run()