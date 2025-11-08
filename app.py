import io
import os
import pandas as pd
import shutil
import threading
from queue import Queue
from flask import Flask, render_template, redirect, url_for, request
from logic.utils.abbreviation_mapping import ISO_TO_NAME, FF12_LABELS
from logic.utils.mailing import Mail
from logic.analysis import Analysis
from logic.compose.registry import Registry

import traceback
from functools import lru_cache 

app: Flask = Flask(__name__)

# --- Task queue for sequential execution ---
task_queue = Queue()

def worker():
    while True:
        file_bytes, filename, email, signal_name, selected_analyzers, country_filter, ff12_filter, min_pct, max_pct = task_queue.get()
        try:
            file_obj = io.BytesIO(file_bytes)

            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file_obj, engine='openpyxl')
            elif filename.endswith('.csv'):
                df = pd.read_csv(file_obj)
            else:
                raise ValueError("Error: Only CSV- or Excel-files are allowed.")

            zip_path = Analysis.run_complete_analysis(
                df, 
                signal_name, selected_analyzers=selected_analyzers,                 
                country_filter=country_filter,
                ff12_filter=ff12_filter,
                min_mcap_pct=min_pct,   
                max_mcap_pct=max_pct 
            )
            
            Mail.send_email_with_attachment(
                to_email=email,
                subject=f"Global Stock Market Protocol Analysis RESULTS - {signal_name}",
                body="Attached are the results of your analysis.",
                attachment_path=zip_path,
            )

        except Exception as e:
            Mail.send_email_with_attachment(
                to_email=email,
                subject=f"Global Stock Market Protocol Analysis ERROR - {signal_name}",
                body=f"An error occurred during processing:\n\n{str(e)}"
            )

            tb = traceback.extract_tb(e.__traceback__)
            filename, lineno, func, text = tb[-1]  
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


@lru_cache(maxsize=1)
def _list_analyzers_cached():
    analyzers = []
    for cls in Registry.list_all_analyzers():
        analyzers.append({
            "fullname": f"{cls.__module__}.{cls.__name__}",
            "cls_name": cls.__name__,
            "title": getattr(cls, "TITLE", None),
            "order": getattr(cls, "ORDER", 100),
            "enabled": bool(getattr(cls, "ENABLED", False)),
        })
    analyzers.sort(key=lambda a: a["order"])
    return analyzers


@lru_cache(maxsize=1)
def _read_crsp_uniques_cached():
    crsp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dsws_crsp.csv")
    df = pd.read_csv(crsp_path, usecols=["country", "ff12"])
    df["country"] = df["country"].astype(str).str.strip()
    country_codes = sorted({c.lower() for c in df["country"].dropna().unique()})
    countries = []
    for code in country_codes:
        name = ISO_TO_NAME.get(code, code)
        countries.append({"code": code, "name": name})
    countries.sort(key=lambda x: x["name"])
    ff = pd.to_numeric(df["ff12"], errors="coerce").dropna().astype(int).unique().tolist()
    ff = sorted([f for f in ff if 1 <= f <= 12])
    industries = [{"code": f, "name": FF12_LABELS.get(f, f"FF12 {f}")} for f in ff]
    return countries, industries


@app.route('/upload')
def upload() -> str:
    analyzers = _list_analyzers_cached()
    countries, industries = _read_crsp_uniques_cached()
    return render_template('upload.html', analyzers=analyzers, countries=countries, industries=industries)


@app.route('/upload_excel', methods=['POST'])
def upload_excel() -> str:
    uploaded_file = request.files.get("excel-file")
    email = request.form.get("user-email")
    signal_name = request.form.get("signal-name")

    selected_analyzers = request.form.getlist("enabled_tests")
    country_filter = request.form.getlist("countries")    
    ff12_filter_raw = request.form.getlist("industries")  
    ff12_filter = [int(x) for x in ff12_filter_raw if x.isdigit()]
                
    min_pct_raw = request.form.get("mc-min")
    max_pct_raw = request.form.get("mc-max")
    min_pct = float(min_pct_raw) if (min_pct_raw not in (None, "")) else None
    max_pct = float(max_pct_raw) if (max_pct_raw not in (None, "")) else None

    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.filename.lower()
        task_queue.put((file_bytes, filename, email, signal_name, 
                        selected_analyzers, country_filter, ff12_filter,
                        min_pct, max_pct))
        return render_template('success.html')

    analyzers = _list_analyzers_cached()
    countries, industries = _read_crsp_uniques_cached()
    return render_template('upload.html', analyzers=analyzers, countries=countries, industries=industries, result="No file uploaded.")


if __name__ == "__main__":
    app.run(debug=True)
