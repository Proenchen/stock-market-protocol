import io
import pandas as pd
import threading
from flask import Flask, render_template, redirect, url_for, request
from logic.mailing import Mail
from logic.strategy_analysis import run_analysis

app: Flask = Flask(__name__)

@app.route("/")
def index() -> redirect:
    return redirect(url_for('home'))

@app.route('/home')
def home() -> str:
    return render_template('home.html')

@app.route('/upload')
def upload() -> str:
    return render_template('upload.html')

@app.route('/overview')
def overview() -> str:
    return render_template('overview.html')

@app.route('/setup')
def setup() -> str:
    return render_template('setup.html')

@app.route('/usage')
def usage() -> str:
    return render_template('usage.html')

@app.route('/contact')
def contact() -> str:
    return render_template('contact.html')

@app.route('/navbar')
def navbar() -> str:
    return render_template('navbar.html')

@app.route('/upload_excel', methods=['POST'])
def upload_excel() -> str:
    uploaded_file = request.files.get("excel-file")

    if uploaded_file:
        # Read file bytes and filename inside request context
        file_bytes = uploaded_file.read()
        filename = uploaded_file.filename.lower()

        def analyze_and_send_results(file_bytes: bytes, filename: str):

            try:
                # Recreate file-like object from bytes
                file_obj = io.BytesIO(file_bytes)

                # Load dataframe depending on file extension
                if filename.endswith('.xlsx') or filename.endswith('.xls'):
                    df = pd.read_excel(file_obj, engine='openpyxl')
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_obj)
                else:
                    raise ValueError("Error: Only CSV- or Excel-files are allowed.")

                zip_path = run_analysis(df)
                Mail.send_email_with_attachment(
                    to_email="julianshen2002@yahoo.de",
                    subject="Factor Model Analysis Results",
                    body="Attached are the results of your analysis.",
                    attachment_path=zip_path
                )

            except Exception as e:
                Mail.send_email_with_attachment(
                    to_email="julianshen2002@yahoo.de",
                    subject="Factor Model Analysis - Error",
                    body=f"An error occurred during processing:\n\n{str(e)}"
                )

        threading.Thread(target=analyze_and_send_results, args=(file_bytes, filename)).start()
        return render_template('upload.html', result="Analysis started. Results will be sent via Email.")

    return render_template('upload.html', result="No file uploaded.")


if __name__ == "__main__": 
    app.run(debug=True)