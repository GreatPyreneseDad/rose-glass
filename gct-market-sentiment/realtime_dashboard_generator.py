#!/usr/bin/env python3
"""
Real-time Dashboard Generator
Creates auto-updating HTML dashboard from monitoring data
"""

import json
import os
from datetime import datetime
import pandas as pd

def load_realtime_data():
    """Load the latest monitoring data"""
    try:
        with open('realtime_data.json', 'r') as f:
            return json.load(f)
    except:
        return None

def generate_html_dashboard(data):
    """Generate the HTML dashboard"""
    if not data:
        return "<html><body><h1>No data available</h1></body></html>"
    
    timestamp = data.get('timestamp', datetime.now().isoformat())
    dt = datetime.fromisoformat(timestamp)
    formatted_date = dt.strftime('%B %d, %Y')
    formatted_time = dt.strftime('%I:%M %p %Z')
    market_summary = data.get('market_summary', {})
    alerts = data.get('alerts', [])
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GCT Real-Time Market Monitor</title>
    <meta http-equiv="refresh" content="120"> <!-- Auto-refresh every 2 minutes -->
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #0a0a0a;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #ffffff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 10px;
        }}
        .timestamp {{
            color: #888;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #222;
        }}
        .metric-label {{
            color: #999;
        }}
        .metric-value {{
            font-weight: bold;
        }}
        .positive {{
            color: #4caf50;
        }}
        .negative {{
            color: #f44336;
        }}
        .neutral {{
            color: #ff9800;
        }}
        .alert {{
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            border-left: 4px solid;
        }}
        .alert-critical {{
            background: rgba(244, 67, 54, 0.1);
            border-color: #f44336;
        }}
        .alert-warning {{
            background: rgba(255, 152, 0, 0.1);
            border-color: #ff9800;
        }}
        .alert-info {{
            background: rgba(33, 150, 243, 0.1);
            border-color: #2196f3;
        }}
        .ticker-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .ticker-card {{
            background: #222;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .ticker-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border-color: #666;
        }}
        .ticker-symbol {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .coherence-bar {{
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin: 5px 0;
            overflow: hidden;
        }}
        .coherence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
            transition: width 0.3s;
        }}
        .market-health {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .health-healthy {{
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
            border: 2px solid #4caf50;
        }}
        .health-stressed {{
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
            border: 2px solid #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #2a2a2a;
            font-weight: bold;
            color: #fff;
        }}
        tr:hover {{
            background: #1a1a1a;
        }}
        .fear-greed-meter {{
            width: 100%;
            height: 30px;
            background: linear-gradient(90deg, #4169E1 0%, #87CEEB 25%, #90EE90 50%, #FFD700 75%, #FF4500 100%);
            border-radius: 15px;
            position: relative;
            margin: 20px 0;
        }}
        .fear-greed-indicator {{
            position: absolute;
            top: -5px;
            width: 40px;
            height: 40px;
            background: white;
            border: 3px solid #333;
            border-radius: 50%;
            transform: translateX(-50%);
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .live-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 GCT Real-Time Market Monitor</h1>
            <h2 style="color: #ddd; font-weight: normal; margin: 10px 0;">{formatted_date}</h2>
            <p class="timestamp">
                <span class="live-indicator"></span>
                Last Update: {formatted_time}
            </p>
        </div>
        
        <!-- Market Summary -->
        <div class="card">
            <h2>Market Summary</h2>
            <div class="market-health health-{market_summary.get('market_health', 'HEALTHY').lower()}">
                Market Health: {market_summary.get('market_health', 'UNKNOWN')}
            </div>
            <div class="grid">
                <div class="metric">
                    <span class="metric-label">Average Coherence</span>
                    <span class="metric-value">{market_summary.get('avg_coherence', 0):.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Truth Cost</span>
                    <span class="metric-value">{market_summary.get('avg_truth_cost', 0):.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">High Coherence Stocks</span>
                    <span class="metric-value positive">{market_summary.get('high_coherence_count', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">High Truth Cost Warnings</span>
                    <span class="metric-value negative">{market_summary.get('high_truth_cost_count', 0)}</span>
                </div>
            </div>
            
            <!-- Fear & Greed Meter -->
            <h3>Market Fear & Greed Index</h3>
            <div class="fear-greed-meter">
                <div class="fear-greed-indicator" style="left: {market_summary.get('avg_fear_greed', 0.5) * 100}%"></div>
            </div>
            <p style="text-align: center; color: #888;">
                Fear ← {market_summary.get('avg_fear_greed', 0.5):.2f} → Greed
            </p>
        </div>
"""
    
    # Add indices section
    html += generate_ticker_section("Market Indices", data.get('indices', {}))
    
    # Add Mag 7 section
    html += generate_ticker_section("Magnificent 7", data.get('mag7', {}))
    
    # Add volatile stocks section
    html += generate_ticker_section("Most Volatile Stocks", data.get('volatile', {}))
    
    # Add truth cost explanation
    html += """
        <div class="card">
            <h2>📊 Understanding Truth Cost</h2>
            <p style="color: #ccc; line-height: 1.6;">
                <strong>Truth Cost</strong> measures the energy required to maintain market patterns that deviate from natural coherence. 
                Think of it as the "effort" needed to keep an unsustainable trend going.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="padding: 10px; background: #1a1a1a; border-radius: 6px; border-left: 3px solid #4caf50;">
                    <strong style="color: #4caf50;">Low Cost (< 0.2)</strong><br>
                    Natural, sustainable movement. Market flows with minimal resistance.
                </div>
                <div style="padding: 10px; background: #1a1a1a; border-radius: 6px; border-left: 3px solid #ff9800;">
                    <strong style="color: #ff9800;">Moderate (0.2-0.4)</strong><br>
                    Some strain detected. Watch for potential reversals.
                </div>
                <div style="padding: 10px; background: #1a1a1a; border-radius: 6px; border-left: 3px solid #f44336;">
                    <strong style="color: #f44336;">High Cost (> 0.4)</strong><br>
                    Unsustainable pattern. Expect significant corrections.
                </div>
            </div>
        </div>
    """
    
    # Add alerts section
    html += """
        <div class="card">
            <h2>🚨 Recent Alerts</h2>
"""
    
    if alerts:
        for alert in alerts[-10:]:  # Show last 10 alerts
            severity_class = f"alert-{alert.get('severity', 'info')}"
            html += f"""
            <div class="alert {severity_class}">
                <strong>{alert.get('ticker', 'MARKET')}</strong> - {alert.get('message', '')}
                <span style="float: right; color: #666; font-size: 12px;">
                    {alert.get('timestamp', '')}
                </span>
            </div>
"""
    else:
        html += "<p style='color: #666;'>No recent alerts</p>"
    
    html += """
        </div>
    </div>
    
    <script>
        // Auto-reload page every 2 minutes
        setTimeout(function() {
            location.reload();
        }, 120000);
        
        // Update relative timestamps
        function updateTimestamps() {
            const now = new Date();
            document.querySelectorAll('.timestamp').forEach(el => {
                const timestamp = new Date(el.dataset.time);
                const diff = Math.floor((now - timestamp) / 1000);
                if (diff < 60) {
                    el.textContent = diff + ' seconds ago';
                } else if (diff < 3600) {
                    el.textContent = Math.floor(diff / 60) + ' minutes ago';
                } else {
                    el.textContent = Math.floor(diff / 3600) + ' hours ago';
                }
            });
        }
        
        // Update every 10 seconds
        setInterval(updateTimestamps, 10000);
    </script>
</body>
</html>
"""
    
    return html

def generate_ticker_section(title, tickers_data):
    """Generate HTML section for a group of tickers"""
    if not tickers_data:
        return ""
    
    html = f"""
        <div class="card">
            <h2>{title}</h2>
            <div class="ticker-grid">
"""
    
    for ticker, data in tickers_data.items():
        coherence_data = data.get('coherence', {})
        coherence = coherence_data.get('coherence', 0)
        price = coherence_data.get('price', 0)
        daily_change = coherence_data.get('daily_change', 0)
        monthly_change = coherence_data.get('monthly_change', 0)
        
        truth_cost = data.get('truth_cost', {}).get('truth_cost', 0)
        emotions = data.get('emotions', {})
        fear_greed = emotions.get('fear_greed_index', 0.5)
        emotional_state = emotions.get('emotional_state', 'NEUTRAL')
        
        # Determine colors
        coherence_color = 'positive' if coherence > 0.6 else 'negative' if coherence < 0.3 else 'neutral'
        truth_cost_color = 'negative' if truth_cost > 0.5 else 'positive' if truth_cost < 0.2 else 'neutral'
        daily_color = 'positive' if daily_change > 0 else 'negative'
        monthly_color = 'positive' if monthly_change > 0 else 'negative'
        
        html += f"""
            <div class="ticker-card">
                <div class="ticker-symbol">{ticker}</div>
                <div style="font-size: 18px; font-weight: bold;">${price:.2f}</div>
                <div class="{daily_color}" style="font-size: 14px;">
                    Day: {daily_change:+.2f}%
                </div>
                <div class="{monthly_color}" style="font-size: 12px;">
                    Month: {monthly_change:+.2f}%
                </div>
                
                <div style="font-size: 12px; color: #888; margin-top: 10px;">Coherence</div>
                <div class="coherence-bar">
                    <div class="coherence-fill" style="width: {coherence * 100}%"></div>
                </div>
                <div class="{coherence_color}" style="font-size: 14px;">{coherence:.3f}</div>
                
                <div style="font-size: 12px; color: #888; margin-top: 10px;">Truth Cost</div>
                <div class="{truth_cost_color}" style="font-size: 14px;">{truth_cost:.3f}</div>
                
                <div style="font-size: 12px; color: #888; margin-top: 10px;">Emotion</div>
                <div style="font-size: 14px;">{emotional_state}</div>
            </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    return html

def main():
    # Load data
    data = load_realtime_data()
    
    # Generate HTML
    html = generate_html_dashboard(data)
    
    # Save dashboard
    output_file = 'realtime_dashboard.html'
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Dashboard updated: {output_file}")
    
    # Also update the main dashboard location
    import shutil
    try:
        shutil.copy(output_file, 'dashboard_realtime.html')
    except:
        pass

if __name__ == "__main__":
    main()