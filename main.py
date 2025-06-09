import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
from werkzeug.datastructures import FileStorage
from typing import Optional
from logic.strategy_analysis import SimpleAnalyzer

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

@app.route('/upload_excel', methods=['GET', 'POST'])
def upload_excel() -> str:
    analysis_result: Optional[str] = None
    
    if request.method == 'POST':
        uploaded_file: FileStorage = request.files["excel-file"]
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                analyzer = SimpleAnalyzer(df)
                analysis_result, monthly_avg_str, permno_mapping = analyzer.analyze()
            
            except Exception as e:
                analysis_result = f"Error: {str(e)}"
    
    return render_template('upload.html', 
        result=analysis_result, 
        monthly_avg=monthly_avg_str,
        mapping=permno_mapping
    )


if __name__ == "__main__": 
    app.run(debug=True)