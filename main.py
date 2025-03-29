import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
from werkzeug.datastructures import FileStorage
from typing import Optional


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
    a1_value: Optional[str] = None
    
    if request.method == 'POST':
        uploaded_file: FileStorage = request.files["excel-file"]
        
        if uploaded_file:
            try:
                df: pd.DataFrame = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
                
                if not df.empty:
                    a1_value = str(df.iloc[0, 0])  
                else:
                    a1_value = None 
            
            except Exception as e:
                a1_value = f"Error: {str(e)}"
    
    return render_template('upload.html', cell_value=a1_value)


if __name__ == "__main__": 
    app.run(debug=True)