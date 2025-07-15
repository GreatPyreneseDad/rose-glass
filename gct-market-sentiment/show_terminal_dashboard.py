#!/usr/bin/env python3
"""Show terminal dashboard output once"""
import terminal_dashboard

# Generate and print dashboard once
layout = terminal_dashboard.generate_dashboard()
if layout:
    terminal_dashboard.console.print(layout)