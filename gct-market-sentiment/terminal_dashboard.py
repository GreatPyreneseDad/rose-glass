#!/usr/bin/env python3
"""
Terminal-based dashboard using Rich library
No web server, runs entirely in console with auto-refresh
"""

import pandas as pd
import os
import glob
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

def load_latest_coherence_data():
    """Load most recent coherence calculation"""
    pattern = "data/raw/coherence_calculated_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    latest = sorted(files)[-1]
    return pd.read_csv(latest)

def create_coherence_bar(value, max_width=20):
    """Create text-based progress bar"""
    filled = int(value * max_width)
    empty = max_width - filled
    
    if value > 0.7:
        color = "green"
    elif value > 0.4:
        color = "yellow"
    else:
        color = "red"
    
    bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {value:.3f}"
    return bar

def create_trend_indicator(trend):
    """Create trend arrow indicator"""
    if trend > 0.02:
        return f"[green]↑ +{trend:.2%}[/green]"
    elif trend < -0.02:
        return f"[red]↓ {trend:.2%}[/red]"
    else:
        return f"[yellow]→ {trend:.2%}[/yellow]"

def generate_dashboard():
    """Generate terminal dashboard"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=20),
        Layout(name="alerts", size=10)
    )
    
    # Load data
    df = load_latest_coherence_data()
    if df is None:
        console.print("[red]No coherence data found. Run calculate_coherence_offline.py first.[/red]")
        return layout
    
    # Header
    header = Panel(
        Text("GCT MARKET COHERENCE DASHBOARD", justify="center", style="bold white on blue"),
        subtitle=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    layout["header"].update(header)
    
    # Main section - split into two columns
    main_layout = Layout()
    main_layout.split_row(
        Layout(name="top_stocks", ratio=1),
        Layout(name="metrics", ratio=1)
    )
    
    # Top coherent stocks table
    top_table = Table(title="Top Coherent Stocks", show_header=True, header_style="bold magenta")
    top_table.add_column("Ticker", style="cyan", width=8)
    top_table.add_column("Coherence", width=25)
    top_table.add_column("Trend", width=12)
    top_table.add_column("Price", justify="right", width=10)
    
    top_10 = df.nlargest(10, 'coherence')
    for _, row in top_10.iterrows():
        top_table.add_row(
            row['ticker'],
            create_coherence_bar(row['coherence']),
            create_trend_indicator(row['trend_strength']),
            f"${row['last_price']:.2f}"
        )
    
    # Metrics summary
    metrics_table = Table(title="Market Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan", width=20)
    metrics_table.add_column("Value", width=20)
    
    avg_coherence = df['coherence'].mean()
    high_coherence = len(df[df['coherence'] > 0.5])
    low_coherence = len(df[df['coherence'] < 0.3])
    avg_volatility = df['volatility'].mean()
    bullish = len(df[df['trend_strength'] > 0.02])
    bearish = len(df[df['trend_strength'] < -0.02])
    
    metrics_table.add_row("Avg Coherence", create_coherence_bar(avg_coherence))
    metrics_table.add_row("High Coherence (>0.5)", f"[green]{high_coherence} stocks[/green]")
    metrics_table.add_row("Low Coherence (<0.3)", f"[red]{low_coherence} stocks[/red]")
    metrics_table.add_row("Avg Volatility", f"{avg_volatility:.1f}%")
    metrics_table.add_row("Bullish Trends", f"[green]{bullish} stocks[/green]")
    metrics_table.add_row("Bearish Trends", f"[red]{bearish} stocks[/red]")
    
    main_layout["top_stocks"].update(Panel(top_table))
    main_layout["metrics"].update(Panel(metrics_table))
    layout["main"].update(main_layout)
    
    # Alerts section
    alerts_text = Text("PATTERN ALERTS\n", style="bold yellow")
    alerts_found = False
    
    for _, row in df.iterrows():
        if row['coherence'] > 0.5 and row['trend_strength'] > 0.02:
            alerts_text.append(f"\n🟢 BUILDING: {row['ticker']} - High coherence ({row['coherence']:.3f}) with positive trend", style="green")
            alerts_found = True
        elif row['coherence'] < 0.3 and row['volatility'] > 40:
            alerts_text.append(f"\n🔴 FALLING: {row['ticker']} - Low coherence ({row['coherence']:.3f}) with high volatility", style="red")
            alerts_found = True
        elif row['momentum'] > 0.01 and row['volume_ratio'] > 1.5:
            alerts_text.append(f"\n⚡ MOMENTUM: {row['ticker']} - Strong momentum with high volume", style="yellow")
            alerts_found = True
    
    if not alerts_found:
        alerts_text.append("\nNo significant patterns detected", style="dim")
    
    layout["alerts"].update(Panel(alerts_text))
    
    return layout

def main():
    """Run terminal dashboard with auto-refresh"""
    console.print("[bold green]Starting Terminal Dashboard[/bold green]")
    console.print("Press Ctrl+C to exit\n")
    
    try:
        with Live(generate_dashboard(), refresh_per_second=0.5, console=console) as live:
            while True:
                time.sleep(10)  # Refresh every 10 seconds
                live.update(generate_dashboard())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")

if __name__ == "__main__":
    # Check if Rich is installed
    try:
        import rich
    except ImportError:
        print("Installing Rich library for terminal UI...")
        os.system("pip install rich")
        print("Please run the script again.")
        exit(1)
    
    main()