# notification_service.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import os
from datetime import datetime
import json

class NotificationService:
    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
        self.sms_api_key = os.getenv('SMS_API_KEY')
        self.notification_history = []
        
    def send_email_alert(self, to_email, subject, message):
        """Email uyarısı gönderir"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            body = self._create_email_template(message)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)
                
            self._log_notification('email', to_email, subject, True)
            return True
            
        except Exception as e:
            self._log_notification('email', to_email, subject, False, str(e))
            return False
    
    def send_sms_alert(self, phone_number, message):
        """SMS uyarısı gönderir"""
        try:
            # SMS API entegrasyonu (örnek)
            url = "https://api.smsservice.com/send"
            payload = {
                'apiKey': self.sms_api_key,
                'to': phone_number,
                'message': message
            }
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                self._log_notification('sms', phone_number, message, True)
                return True
            else:
                raise Exception(f"SMS gönderilemedi: {response.text}")
                
        except Exception as e:
            self._log_notification('sms', phone_number, message, False, str(e))
            return False
    
    def _create_email_template(self, message):
        """HTML email şablonu oluşturur"""
        return f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="padding: 20px; background-color: #f8f9fa;">
                    <h2 style="color: #333;">Borsa İstanbul Uyarısı</h2>
                    <div style="padding: 15px; background-color: white; border-radius: 5px;">
                        {message}
                    </div>
                    <p style="color: #666; font-size: 12px; margin-top: 20px;">
                        Bu otomatik bir bildirimdir.
                    </p>
                </div>
            </body>
        </html>
        """
    
    def _log_notification(self, type, recipient, content, success, error=None):
        """Bildirim geçmişini kaydeder"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': type,
            'recipient': recipient,
            'content': content,
            'success': success,
            'error': error
        }
        
        self.notification_history.append(log_entry)
        self._save_notification_history()
    
    def _save_notification_history(self):
        """Bildirim geçmişini dosyaya kaydeder"""
        with open('notification_history.json', 'w') as f:
            json.dump(self.notification_history, f, indent=4)
    
    def get_notification_history(self, filter_type=None):
        """Bildirim geçmişini getirir"""
        if filter_type:
            return [n for n in self.notification_history if n['type'] == filter_type]
        return self.notification_history