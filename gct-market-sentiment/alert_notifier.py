#!/usr/bin/env python3
"""
Alert Notification System
Sends push notifications for critical market alerts
"""

import json
import os
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertNotifier:
    def __init__(self, config_file='alert_config.json'):
        """Initialize with configuration"""
        self.config = self.load_config(config_file)
        self.sent_alerts = set()  # Track sent alerts to avoid duplicates
        
    def load_config(self, config_file):
        """Load notification configuration"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipient': ''
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            },
            'pushover': {
                'enabled': False,
                'user_key': '',
                'api_token': ''
            },
            'webhook': {
                'enabled': False,
                'url': ''
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config file: {config_file}")
            print("Please edit this file to enable notifications")
        
        return default_config
    
    def send_email(self, alert):
        """Send email notification"""
        if not self.config['email']['enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['username']
            msg['To'] = self.config['email']['recipient']
            msg['Subject'] = f"GCT Alert: {alert['type']} - {alert['ticker']}"
            
            body = f"""
Market Alert from GCT Monitor

Type: {alert['type']}
Ticker: {alert['ticker']}
Message: {alert['message']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

This is an automated alert from your GCT Market Sentiment Monitor.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], 
                                 self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], 
                        self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Email sent for {alert['ticker']}")
            
        except Exception as e:
            print(f"Email error: {e}")
    
    def send_telegram(self, alert):
        """Send Telegram notification"""
        if not self.config['telegram']['enabled']:
            return
        
        try:
            message = f"""
🚨 *GCT Market Alert*

*Type:* {alert['type']}
*Ticker:* {alert['ticker']}
*Message:* {alert['message']}
*Severity:* {alert['severity']}
            """
            
            url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
            data = {
                'chat_id': self.config['telegram']['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print(f"Telegram sent for {alert['ticker']}")
            
        except Exception as e:
            print(f"Telegram error: {e}")
    
    def send_pushover(self, alert):
        """Send Pushover notification"""
        if not self.config['pushover']['enabled']:
            return
        
        try:
            priority = 1 if alert['severity'] == 'critical' else 0
            
            data = {
                'token': self.config['pushover']['api_token'],
                'user': self.config['pushover']['user_key'],
                'title': f"GCT: {alert['type']}",
                'message': f"{alert['ticker']}: {alert['message']}",
                'priority': priority,
                'timestamp': int(datetime.now().timestamp())
            }
            
            response = requests.post('https://api.pushover.net/1/messages.json', 
                                   data=data)
            if response.status_code == 200:
                print(f"Pushover sent for {alert['ticker']}")
                
        except Exception as e:
            print(f"Pushover error: {e}")
    
    def send_webhook(self, alert):
        """Send webhook notification"""
        if not self.config['webhook']['enabled']:
            return
        
        try:
            response = requests.post(self.config['webhook']['url'], 
                                   json=alert,
                                   timeout=10)
            if response.status_code in [200, 201, 204]:
                print(f"Webhook sent for {alert['ticker']}")
                
        except Exception as e:
            print(f"Webhook error: {e}")
    
    def send_alert(self, alert):
        """Send alert through all enabled channels"""
        # Create unique alert ID to avoid duplicates
        alert_id = f"{alert['ticker']}_{alert['type']}_{alert['timestamp']}"
        
        if alert_id in self.sent_alerts:
            return  # Already sent
        
        # Send through all channels
        self.send_email(alert)
        self.send_telegram(alert)
        self.send_pushover(alert)
        self.send_webhook(alert)
        
        # Mark as sent
        self.sent_alerts.add(alert_id)
        
        # Keep only recent alerts (last 1000)
        if len(self.sent_alerts) > 1000:
            self.sent_alerts = set(list(self.sent_alerts)[-500:])

# Example usage
if __name__ == "__main__":
    notifier = AlertNotifier()
    
    # Test alert
    test_alert = {
        'type': 'EXTREME_GREED',
        'ticker': 'TEST',
        'message': 'This is a test alert from GCT Monitor',
        'severity': 'critical',
        'timestamp': datetime.now().isoformat()
    }
    
    print("Sending test alert...")
    notifier.send_alert(test_alert)