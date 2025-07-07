import os
import pandas as pd
import zipfile
import uuid
from flask import Flask, render_template, redirect, url_for, request
from logic.strategy_analysis import EqualWeightedFactorModelAnalyzer, ValueWeightedFactorModelAnalyzer, FamaMacBethAnalyzer

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
        try:
            filename = uploaded_file.filename.lower()

            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                return render_template('upload.html', result="Fehler: Nur CSV- oder Excel-Dateien sind erlaubt.")

            equal_factor_model_analyzer = EqualWeightedFactorModelAnalyzer(df)
            value_factor_model_analyzer = ValueWeightedFactorModelAnalyzer(df)
            fama_macbeth_analyzer = FamaMacBethAnalyzer(df)

            ff3_equal, ff5_equal, q_equal, long_short_equal = equal_factor_model_analyzer.analyze()
            ff3_value, ff5_value, q_value, long_short_value = value_factor_model_analyzer.analyze()
            fama_macbeth_res = fama_macbeth_analyzer.analyze()

            basedir = os.path.abspath(os.path.dirname(__file__))
            result_dir = os.path.join(basedir, "static", "downloads")
            os.makedirs(result_dir, exist_ok=True)

            session_id = str(uuid.uuid4())
            zip_filename = f"{session_id}_results.zip"
            zip_path = os.path.join(result_dir, zip_filename)

            results = {
                "ff3_equal.txt": ff3_equal,
                "ff5_equal.txt": ff5_equal,
                "q_equal.txt": q_equal,
                "long_short_equal.txt": long_short_equal,
                "ff3_value.txt": ff3_value,
                "ff5_value.txt": ff5_value,
                "q_value.txt": q_value,
                "long_short_value.txt": long_short_value,
                "fama_macbeth.txt": fama_macbeth_res
            }

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename, content in results.items():
                    temp_file = os.path.join(result_dir, f"{session_id}_{filename}")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(str(content))
                    zipf.write(temp_file, arcname=filename)
                    os.remove(temp_file)

            download_url = url_for('static', filename=f'downloads/{zip_filename}')
            result_msg = f'Analyse abgeschlossen. <a href="{download_url}" class="btn btn-success mt-2">Download Results</a>'
            return render_template('upload.html', result=result_msg)

        except Exception as e:
            return render_template('upload.html', result=f"Error: {str(e)}")

    return render_template('upload.html', result="No file uploaded.")


if __name__ == "__main__": 
    app.run(debug=True)