#!/usr/bin/env python3
"""
PDF Report Generator - Creates a static PDF report of coherence data
No servers, no web dependencies
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import glob
import numpy as np

def load_latest_coherence_data():
    """Load most recent coherence calculation"""
    pattern = "data/raw/coherence_calculated_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    latest = sorted(files)[-1]
    return pd.read_csv(latest)

def generate_pdf_report(output_file='coherence_report.pdf'):
    """Generate comprehensive PDF report"""
    print(f"Generating PDF report: {output_file}")
    
    # Load data
    df = load_latest_coherence_data()
    if df is None:
        print("No coherence data found.")
        return
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('GCT Market Coherence Report', fontsize=24, fontweight='bold')
        
        # Add timestamp
        plt.text(0.5, 0.9, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', fontsize=12, transform=fig.transFigure)
        
        # Summary statistics
        summary_text = f"""
Market Summary:
• Average Coherence: {df['coherence'].mean():.3f}
• High Coherence Stocks (>0.5): {len(df[df['coherence'] > 0.5])}
• Low Coherence Stocks (<0.3): {len(df[df['coherence'] < 0.3])}
• Average Volatility: {df['volatility'].mean():.1f}%
• Bullish Trends: {len(df[df['trend_strength'] > 0.02])} stocks
• Bearish Trends: {len(df[df['trend_strength'] < -0.02])} stocks

Top 3 Coherent Stocks:
"""
        
        top_3 = df.nlargest(3, 'coherence')
        for idx, row in top_3.iterrows():
            summary_text += f"\n{row['ticker']}: {row['coherence']:.3f} (Trend: {row['trend_strength']:.2%})"
        
        plt.text(0.1, 0.7, summary_text, fontsize=12, transform=fig.transFigure, 
                verticalalignment='top', family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Coherence Analysis Charts', fontsize=16, fontweight='bold')
        
        # Chart 1: Top coherence bar chart
        top_10 = df.nlargest(10, 'coherence')
        ax1.barh(top_10['ticker'], top_10['coherence'], color='steelblue')
        ax1.set_xlabel('Coherence Score')
        ax1.set_title('Top 10 Coherent Stocks')
        ax1.set_xlim(0, 1)
        
        # Chart 2: Coherence components scatter
        scatter = ax2.scatter(df['psi'], df['rho'], c=df['coherence'], 
                            s=100, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('ψ (Clarity)')
        ax2.set_ylabel('ρ (Wisdom)')
        ax2.set_title('Coherence Components')
        plt.colorbar(scatter, ax=ax2)
        
        # Chart 3: Volatility vs Trend
        ax3.scatter(df['volatility'], df['trend_strength'], 
                   s=df['coherence']*200, alpha=0.6, c='coral')
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Trend Strength')
        ax3.set_title('Risk vs Momentum')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axvline(x=30, color='r', linestyle='--', alpha=0.3)
        
        # Chart 4: Coherence distribution
        ax4.hist(df['coherence'], bins=20, color='lightblue', edgecolor='black')
        ax4.set_xlabel('Coherence Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Coherence Distribution')
        ax4.axvline(x=df['coherence'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {df["coherence"].mean():.3f}')
        ax4.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Data Table
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Ticker', 'Coherence', 'ψ', 'ρ', 'Trend', 'Vol%', 'Price']
        
        for idx, row in df.iterrows():
            table_data.append([
                row['ticker'],
                f"{row['coherence']:.3f}",
                f"{row['psi']:.3f}",
                f"{row['rho']:.3f}",
                f"{row['trend_strength']:.2%}",
                f"{row['volatility']:.1f}",
                f"${row['last_price']:.2f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code by coherence
        for i in range(len(table_data)):
            coherence_val = df.iloc[i]['coherence']
            if coherence_val > 0.5:
                table[(i+1, 1)].set_facecolor('#90EE90')  # Light green
            elif coherence_val < 0.3:
                table[(i+1, 1)].set_facecolor('#FFB6C1')  # Light red
        
        ax.set_title('Detailed Coherence Data', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Alerts
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Pattern Alerts', fontsize=16, fontweight='bold')
        
        alerts_text = ""
        alert_count = 0
        
        for idx, row in df.iterrows():
            if row['coherence'] > 0.5 and row['trend_strength'] > 0.02:
                alerts_text += f"🟢 BUILDING: {row['ticker']} - Coherence: {row['coherence']:.3f}, Trend: {row['trend_strength']:.2%}\n"
                alert_count += 1
            elif row['coherence'] < 0.3 and row['volatility'] > 40:
                alerts_text += f"🔴 FALLING: {row['ticker']} - Coherence: {row['coherence']:.3f}, Volatility: {row['volatility']:.1f}%\n"
                alert_count += 1
            elif row['momentum'] > 0.01 and row['volume_ratio'] > 1.5:
                alerts_text += f"⚡ MOMENTUM: {row['ticker']} - Momentum: {row['momentum']:.3f}, Volume Ratio: {row['volume_ratio']:.2f}\n"
                alert_count += 1
        
        if alert_count == 0:
            alerts_text = "No significant patterns detected in current market conditions."
        
        plt.text(0.1, 0.8, alerts_text, fontsize=11, transform=fig.transFigure,
                verticalalignment='top', family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report generated: {output_file}")
    
    # Open the PDF
    if os.path.exists(output_file):
        os.system(f"open {output_file}")

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'coherence_report_{timestamp}.pdf'
    generate_pdf_report(output_file)

if __name__ == "__main__":
    main()