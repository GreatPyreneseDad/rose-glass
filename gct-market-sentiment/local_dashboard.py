#!/usr/bin/env python3
"""
Local-only dashboard using matplotlib - no web server required
Generates static HTML with embedded charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import glob
import webbrowser
import numpy as np

def load_latest_coherence_data():
    """Load most recent coherence calculation"""
    pattern = "data/raw/coherence_calculated_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    latest = sorted(files)[-1]
    return pd.read_csv(latest)

def load_ticker_history(ticker):
    """Load historical data for a ticker"""
    ticker_dir = f"data/tickers/{ticker}"
    if not os.path.exists(ticker_dir):
        return None
    
    files = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
    if not files:
        return None
    
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(ticker_dir, latest_file))
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_coherence_gauge(ax, value, title):
    """Create a gauge chart for coherence value"""
    # Create gauge segments
    theta = np.linspace(0, np.pi, 100)
    r_inner = 0.7
    r_outer = 1.0
    
    # Color segments
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    boundaries = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i in range(len(colors)):
        theta_seg = theta[(theta >= boundaries[i] * np.pi) & (theta < boundaries[i+1] * np.pi)]
        if len(theta_seg) > 0:
            ax.fill_between(theta_seg, r_inner, r_outer, color=colors[i], alpha=0.6)
    
    # Add needle
    angle = value * np.pi
    ax.plot([angle, angle], [0, r_outer], 'k-', linewidth=3)
    ax.plot(angle, 0, 'ko', markersize=10)
    
    # Formatting
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, np.pi)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(np.pi/2, -0.2, title, ha='center', fontsize=14, fontweight='bold')
    ax.text(np.pi/2, -0.35, f'{value:.3f}', ha='center', fontsize=18)
    
    # Add labels
    ax.text(0, -0.1, 'Low', ha='center', fontsize=10)
    ax.text(np.pi, -0.1, 'High', ha='center', fontsize=10)

def generate_dashboard():
    """Generate static HTML dashboard with embedded charts"""
    print("Generating local dashboard...")
    
    # Load data
    coherence_df = load_latest_coherence_data()
    if coherence_df is None:
        print("No coherence data found. Run calculate_coherence_offline.py first.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('GCT Market Coherence Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Top coherence scores (bar chart)
    ax1 = plt.subplot(3, 3, 1)
    top_5 = coherence_df.nlargest(5, 'coherence')
    ax1.barh(top_5['ticker'], top_5['coherence'], color='steelblue')
    ax1.set_xlabel('Coherence Score')
    ax1.set_title('Top 5 Coherent Stocks')
    ax1.set_xlim(0, 1)
    for i, (idx, row) in enumerate(top_5.iterrows()):
        ax1.text(row['coherence'] + 0.01, i, f"{row['coherence']:.3f}", va='center')
    
    # 2. Coherence components scatter
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.scatter(coherence_df['psi'], coherence_df['rho'], 
                         c=coherence_df['coherence'], s=100, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('ψ (Clarity)')
    ax2.set_ylabel('ρ (Wisdom/Reflection)')
    ax2.set_title('Coherence Components')
    for idx, row in coherence_df.iterrows():
        ax2.annotate(row['ticker'], (row['psi'], row['rho']), fontsize=8)
    plt.colorbar(scatter, ax=ax2, label='Coherence')
    
    # 3. Volatility vs Trend
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(coherence_df['volatility'], coherence_df['trend_strength'], 
                s=coherence_df['coherence']*200, alpha=0.6, c='coral')
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Trend Strength')
    ax3.set_title('Risk vs Momentum')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    for idx, row in coherence_df.iterrows():
        ax3.annotate(row['ticker'], (row['volatility'], row['trend_strength']), fontsize=8)
    
    # 4-6. Individual stock price charts with coherence
    for i, ticker in enumerate(['NVDA', 'AAPL', 'TSLA']):
        ax = plt.subplot(3, 3, 4 + i)
        df = load_ticker_history(ticker)
        if df is not None and len(df) > 0:
            ax.plot(df['date'], df['close'], 'b-', linewidth=2)
            ax.set_title(f'{ticker} Price Chart')
            ax.set_ylabel('Price ($)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.xticks(rotation=45)
            
            # Add coherence info
            ticker_coherence = coherence_df[coherence_df['ticker'] == ticker]
            if not ticker_coherence.empty:
                coh = ticker_coherence.iloc[0]['coherence']
                ax.text(0.02, 0.98, f'Coherence: {coh:.3f}', 
                       transform=ax.transAxes, va='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7-9. Coherence gauges for market indices
    for i, ticker in enumerate(['SPY', 'QQQ', 'NVDA']):
        ax = plt.subplot(3, 3, 7 + i, projection='polar')
        ticker_data = coherence_df[coherence_df['ticker'] == ticker]
        if not ticker_data.empty:
            create_coherence_gauge(ax, ticker_data.iloc[0]['coherence'], ticker)
    
    plt.tight_layout()
    
    # Save as PNG
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    png_file = f'dashboard_{timestamp}.png'
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"Saved dashboard image: {png_file}")
    
    # Create HTML with embedded image and data table
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GCT Market Coherence Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .timestamp {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
            }}
            img {{
                width: 100%;
                height: auto;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .high-coherence {{
                background-color: #c8e6c9;
            }}
            .low-coherence {{
                background-color: #ffcdd2;
            }}
            .alerts {{
                margin-top: 30px;
                padding: 15px;
                background-color: #fff3cd;
                border-left: 5px solid #ffc107;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GCT Market Coherence Dashboard</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            
            <img src="{png_file}" alt="Dashboard Charts">
            
            <h2>Coherence Data Table</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Coherence</th>
                    <th>ψ (Clarity)</th>
                    <th>ρ (Wisdom)</th>
                    <th>q (Emotion)</th>
                    <th>f (Social)</th>
                    <th>Trend</th>
                    <th>Volatility</th>
                    <th>Price</th>
                </tr>
    """
    
    # Add data rows
    for idx, row in coherence_df.iterrows():
        row_class = ''
        if row['coherence'] > 0.5:
            row_class = 'high-coherence'
        elif row['coherence'] < 0.3:
            row_class = 'low-coherence'
            
        html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{row['ticker']}</strong></td>
                    <td>{row['coherence']:.3f}</td>
                    <td>{row['psi']:.3f}</td>
                    <td>{row['rho']:.3f}</td>
                    <td>{row['q_raw']:.3f}</td>
                    <td>{row['f']:.3f}</td>
                    <td>{row['trend_strength']:.2%}</td>
                    <td>{row['volatility']:.1f}%</td>
                    <td>${row['last_price']:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <div class="alerts">
                <h3>Pattern Alerts</h3>
                <ul>
    """
    
    # Add alerts
    for idx, row in coherence_df.iterrows():
        if row['coherence'] > 0.5 and row['trend_strength'] > 0.02:
            html_content += f"<li>🟢 <strong>{row['ticker']}</strong>: Building coherence pattern detected (Score: {row['coherence']:.3f})</li>"
        elif row['coherence'] < 0.3 and row['volatility'] > 40:
            html_content += f"<li>🔴 <strong>{row['ticker']}</strong>: Falling coherence with high volatility (Score: {row['coherence']:.3f})</li>"
    
    html_content += """
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    html_file = f'dashboard_{timestamp}.html'
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"Saved HTML dashboard: {html_file}")
    
    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    
    plt.close()

def main():
    generate_dashboard()

if __name__ == "__main__":
    main()