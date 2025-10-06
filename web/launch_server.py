#!/usr/bin/env python3
"""Launch the Rose Glass Dream server and open browser"""

import subprocess
import time
import webbrowser
import os

# Change to the web directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Activate virtual environment and start server
print("🌹 Starting Rose Glass Dream Analysis Server...")
server_process = subprocess.Popen(
    [os.path.join("venv", "bin", "python"), "dream_app_simple.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("Waiting for server to start...")
time.sleep(3)

# Open browser
url = "http://localhost:8002"
print(f"Opening {url} in your browser...")
webbrowser.open(url)

print("\nServer is running! Press Ctrl+C to stop.")
print("If the browser didn't open, manually visit: http://localhost:8002")

try:
    # Keep the script running
    server_process.wait()
except KeyboardInterrupt:
    print("\nStopping server...")
    server_process.terminate()