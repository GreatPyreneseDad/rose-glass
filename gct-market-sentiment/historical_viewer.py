#!/usr/bin/env python3
"""
Historical Market Data Viewer
Analyzes and visualizes historical market monitoring data
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

class HistoricalAnalyzer:
    def __init__(self):
        self.master_file = 'historical_snapshots/master_history.jsonl'
        self.snapshots_dir = 'historical_snapshots'
        
    def load_master_history(self):
        """Load the master historical data"""
        if not os.path.exists(self.master_file):
            print("No historical data found yet.")
            return pd.DataFrame()
        
        data = []
        with open(self.master_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    
    def generate_historical_report(self):
        """Generate a comprehensive historical analysis report"""
        df = self.load_master_history()
        
        if df.empty:
            return
        
        # Create plots directory
        os.makedirs('historical_plots', exist_ok=True)
        
        # Set style
        plt.style.use('dark_background')
        
        # 1. Market Health Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        health_colors = {'HEALTHY': 'green', 'STRESSED': 'red'}
        
        for health in df['market_health'].unique():
            mask = df['market_health'] == health
            ax.scatter(df[mask]['timestamp'], [1]*sum(mask), 
                      label=health, color=health_colors.get(health, 'yellow'), 
                      alpha=0.6, s=100)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Market Health')
        ax.set_title('Market Health Status Over Time')
        ax.legend()
        plt.tight_layout()
        plt.savefig('historical_plots/market_health_timeline.png')
        plt.close()
        
        # 2. Key Metrics Over Time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average Coherence
        axes[0, 0].plot(df['timestamp'], df['avg_coherence'], color='cyan', linewidth=2)
        axes[0, 0].fill_between(df['timestamp'], df['avg_coherence'], alpha=0.3, color='cyan')
        axes[0, 0].axhline(y=0.6, color='green', linestyle='--', label='Optimal')
        axes[0, 0].set_title('Average Market Coherence')
        axes[0, 0].set_ylabel('Coherence')
        axes[0, 0].legend()
        
        # Average Truth Cost
        axes[0, 1].plot(df['timestamp'], df['avg_truth_cost'], color='orange', linewidth=2)
        axes[0, 1].fill_between(df['timestamp'], df['avg_truth_cost'], alpha=0.3, color='orange')
        axes[0, 1].axhline(y=0.3, color='red', linestyle='--', label='Warning Level')
        axes[0, 1].set_title('Average Truth Cost')
        axes[0, 1].set_ylabel('Truth Cost')
        axes[0, 1].legend()
        
        # Fear & Greed Index
        axes[1, 0].plot(df['timestamp'], df['avg_fear_greed'], color='yellow', linewidth=2)
        axes[1, 0].fill_between(df['timestamp'], df['avg_fear_greed'], alpha=0.3, color='yellow')
        axes[1, 0].axhline(y=0.5, color='white', linestyle='--', label='Neutral')
        axes[1, 0].axhline(y=0.2, color='blue', linestyle='--', label='Fear')
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', label='Greed')
        axes[1, 0].set_title('Fear & Greed Index')
        axes[1, 0].set_ylabel('Index Value')
        axes[1, 0].legend()
        
        # Alert Count
        axes[1, 1].bar(df['timestamp'], df['alert_count'], color='red', alpha=0.7)
        axes[1, 1].set_title('Critical Alerts Count')
        axes[1, 1].set_ylabel('Number of Alerts')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('historical_plots/metrics_overview.png')
        plt.close()
        
        # 3. Correlation Matrix
        if len(df) > 10:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_data = df[['avg_coherence', 'avg_truth_cost', 'avg_fear_greed', 'extreme_emotions_count', 'alert_count']].corr()
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            ax.set_title('Metrics Correlation Matrix')
            plt.tight_layout()
            plt.savefig('historical_plots/correlation_matrix.png')
            plt.close()
        
        # Generate HTML Report
        self.generate_html_report(df)
        
        print(f"Historical analysis complete. Generated {len(df)} data points.")
        print("Reports saved to: historical_plots/ and historical_report.html")
    
    def generate_html_report(self, df):
        """Generate an HTML report of historical data"""
        
        # Calculate statistics
        latest = df.iloc[-1] if not df.empty else None
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GCT Historical Market Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #0a0a0a;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2 {{
            color: #ffffff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #888;
            font-size: 14px;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-card {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .data-table th {{
            background: #2a2a2a;
            font-weight: bold;
        }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        .neutral {{ color: #ff9800; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 GCT Historical Market Analysis</h1>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Data Points: {len(df)} | Period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}</p>
        </div>
"""
        
        if latest is not None:
            # Add statistics
            html += """
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Current Market Health</div>
                <div class="stat-value" style="color: {};">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Coherence</div>
                <div class="stat-value">{:.3f}</div>
                <div class="stat-label">Last 24h: {:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Truth Cost</div>
                <div class="stat-value">{:.3f}</div>
                <div class="stat-label">Last 24h: {:.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Alerts</div>
                <div class="stat-value">{}</div>
                <div class="stat-label">Last 24h: {}</div>
            </div>
        </div>
""".format(
                'green' if latest['market_health'] == 'HEALTHY' else 'red',
                latest['market_health'],
                df['avg_coherence'].mean(),
                df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]['avg_coherence'].mean() if len(df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]) > 0 else 0,
                df['avg_truth_cost'].mean(),
                df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]['avg_truth_cost'].mean() if len(df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]) > 0 else 0,
                df['alert_count'].sum(),
                df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]['alert_count'].sum() if len(df[df['timestamp'] > df['timestamp'].max() - timedelta(days=1)]) > 0 else 0
            )
        
        # Add plots
        html += """
        <h2>📊 Visualizations</h2>
        <div class="plot-grid">
            <div class="plot-card">
                <h3>Market Health Timeline</h3>
                <img src="historical_plots/market_health_timeline.png" alt="Market Health Timeline">
            </div>
            <div class="plot-card">
                <h3>Key Metrics Over Time</h3>
                <img src="historical_plots/metrics_overview.png" alt="Metrics Overview">
            </div>
"""
        
        if os.path.exists('historical_plots/correlation_matrix.png'):
            html += """
            <div class="plot-card">
                <h3>Metrics Correlation</h3>
                <img src="historical_plots/correlation_matrix.png" alt="Correlation Matrix">
            </div>
"""
        
        html += """
        </div>
        
        <h2>📋 Recent Data Points</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Market Health</th>
                    <th>Avg Coherence</th>
                    <th>Avg Truth Cost</th>
                    <th>Fear/Greed</th>
                    <th>Alerts</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add last 20 data points
        for _, row in df.tail(20).iterrows():
            coherence_class = 'positive' if row['avg_coherence'] > 0.6 else 'negative' if row['avg_coherence'] < 0.3 else 'neutral'
            truth_class = 'negative' if row['avg_truth_cost'] > 0.3 else 'positive' if row['avg_truth_cost'] < 0.2 else 'neutral'
            fear_greed_class = 'negative' if row['avg_fear_greed'] > 0.8 or row['avg_fear_greed'] < 0.2 else 'neutral'
            
            html += f"""
                <tr>
                    <td>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td class="{'positive' if row['market_health'] == 'HEALTHY' else 'negative'}">{row['market_health']}</td>
                    <td class="{coherence_class}">{row['avg_coherence']:.3f}</td>
                    <td class="{truth_class}">{row['avg_truth_cost']:.3f}</td>
                    <td class="{fear_greed_class}">{row['avg_fear_greed']:.3f}</td>
                    <td>{row['alert_count']}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        with open('historical_report.html', 'w') as f:
            f.write(html)

def main():
    analyzer = HistoricalAnalyzer()
    analyzer.generate_historical_report()

if __name__ == "__main__":
    main()