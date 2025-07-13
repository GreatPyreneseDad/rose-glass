#!/bin/bash
# Setup script for GCT Market Sentiment Collector Service

echo "Setting up GCT Market Sentiment Collector Service..."

# Create logs directory
mkdir -p logs

# For macOS, use launchd instead of systemd
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Creating launchd configuration..."
    
    # Create launchd plist file
    cat > ~/Library/LaunchAgents/com.gct.market-collector.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gct.market-collector</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$PWD/continuous_collector.py</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$PWD</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>$PWD/logs/collector.log</string>
    
    <key>StandardErrorPath</key>
    <string>$PWD/logs/collector_error.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$PWD</string>
    </dict>
</dict>
</plist>
EOF

    echo "Loading service..."
    launchctl load ~/Library/LaunchAgents/com.gct.market-collector.plist
    
    echo "Service setup complete for macOS!"
    echo ""
    echo "To manage the service:"
    echo "  Start:   launchctl start com.gct.market-collector"
    echo "  Stop:    launchctl stop com.gct.market-collector"
    echo "  Status:  launchctl list | grep gct"
    echo "  Logs:    tail -f logs/collector.log"
    echo ""
    echo "To uninstall:"
    echo "  launchctl unload ~/Library/LaunchAgents/com.gct.market-collector.plist"
    echo "  rm ~/Library/LaunchAgents/com.gct.market-collector.plist"
    
else
    # Linux systemd setup
    echo "Detected Linux. Setting up systemd service..."
    
    # Replace $USER with actual username in service file
    sed -i "s/\$USER/$USER/g" gct-collector.service
    
    # Copy service file to systemd directory
    sudo cp gct-collector.service /etc/systemd/system/
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable gct-collector.service
    sudo systemctl start gct-collector.service
    
    echo "Service setup complete for Linux!"
    echo ""
    echo "To manage the service:"
    echo "  Start:   sudo systemctl start gct-collector"
    echo "  Stop:    sudo systemctl stop gct-collector"
    echo "  Status:  sudo systemctl status gct-collector"
    echo "  Logs:    sudo journalctl -u gct-collector -f"
fi

# Create simple management script
cat > manage_collector.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            launchctl start com.gct.market-collector
        else
            sudo systemctl start gct-collector
        fi
        echo "Collector started"
        ;;
    stop)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            launchctl stop com.gct.market-collector
        else
            sudo systemctl stop gct-collector
        fi
        echo "Collector stopped"
        ;;
    restart)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            launchctl stop com.gct.market-collector
            launchctl start com.gct.market-collector
        else
            sudo systemctl restart gct-collector
        fi
        echo "Collector restarted"
        ;;
    status)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            launchctl list | grep gct
        else
            sudo systemctl status gct-collector
        fi
        ;;
    logs)
        tail -f logs/collector.log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF

chmod +x manage_collector.sh

echo ""
echo "Setup complete! Use ./manage_collector.sh to control the service."