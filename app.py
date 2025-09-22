import io
import os
import pandas as pd
import shutil
import threading
from queue import Queue
from flask import Flask, render_template, redirect, url_for, request
from logic.utils.mailing import Mail
from logic.analysis import run_complete_analysis

import traceback

app: Flask = Flask(__name__)

# --- Task queue for sequential execution ---
task_queue = Queue()

def worker():
    while True:
        file_bytes, filename, email, signal_name = task_queue.get()
        try:
            file_obj = io.BytesIO(file_bytes)

            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file_obj, engine='openpyxl')
            elif filename.endswith('.csv'):
                df = pd.read_csv(file_obj)
            else:
                raise ValueError("Error: Only CSV- or Excel-files are allowed.")

            zip_path = run_complete_analysis(df, signal_name)
            Mail.send_email_with_attachment(
                to_email=email,
                subject=f"Global Stock Market Protocol Analysis RESULTS - {signal_name}",
                body="Attached are the results of your analysis.",
                attachment_path=zip_path
            )

        except Exception as e:
            Mail.send_email_with_attachment(
                to_email=email,
                subject=f"Global Stock Market Protocol Analysis ERROR - {signal_name}",
                body=f"An error occurred during processing:\n\n{str(e)}"
            )

            tb = traceback.extract_tb(e.__traceback__)
            filename, lineno, func, text = tb[-1]   # letzte Stelle im Traceback
            msg = (
                f"An error occurred during processing:\n\n"
                f"{str(e)}\n\n"
                f"(File {filename}, line {lineno}, in {func})\n"
                f"--> {text}"
            )
            print(msg)

        finally:
            task_queue.task_done()
            print("finished")
            # Clean downloads directory after mail has been sent
            downloads_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "static", "downloads"
            )
            downloads_dir = os.path.abspath(downloads_dir) 

            if os.path.exists(downloads_dir):
                for file in os.listdir(downloads_dir):
                    file_path = os.path.join(downloads_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as cleanup_err:
                        print(f"Cleanup error: {cleanup_err}")

# Start one background worker thread
threading.Thread(target=worker, daemon=True).start()

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
    email = request.form.get("user-email")
    signal_name = request.form.get("signal-name")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.filename.lower()

        # Add task to queue (processed sequentially by worker)
        task_queue.put((file_bytes, filename, email, signal_name))

        return render_template('success.html')

    return render_template('upload.html', result="No file uploaded.")


if __name__ == "__main__":
    app.run(debug=True)
