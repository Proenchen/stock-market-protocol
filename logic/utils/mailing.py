import os
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() 

class Mail:

    @staticmethod
    def send_email_with_attachment(
        to_email: str,
        subject: str,
        body: str,
        attachment_path: str = None
    ):
        """
        Sends an email using SMTP credentials stored in .env
        """
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        username = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASSWORD")

        if not all([smtp_server, smtp_port, username, password]):
            raise RuntimeError("Missing SMTP configuration in .env file")

        msg = MIMEMultipart()
        msg["From"] = username
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if attachment_path and Path(attachment_path).exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{Path(attachment_path).name}"'
            )
            msg.attach(part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)