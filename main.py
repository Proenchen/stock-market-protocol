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
                return render_template('upload.html', result="Error: Only CSV- or Excel-files are allowed.")

            equal_factor_model_analyzer = EqualWeightedFactorModelAnalyzer(df)
            value_factor_model_analyzer = ValueWeightedFactorModelAnalyzer(df)
            fama_macbeth_analyzer = FamaMacBethAnalyzer(df)

            results_equal, long_short_results_equal = equal_factor_model_analyzer.analyze()
            ff3_equal, ff5_equal, q_equal = equal_factor_model_analyzer.results_to_strings(results_equal)
            long_short_equal = equal_factor_model_analyzer.long_short_res_to_string(long_short_results_equal)
            latex_ff3_equal = equal_factor_model_analyzer.generate_latex_table(results_equal, "FF3")
            latex_ff5_equal = equal_factor_model_analyzer.generate_latex_table(results_equal, "FF5")
            latex_q_equal = equal_factor_model_analyzer.generate_latex_table(results_equal, "Q")
            latex_long_short_equal = equal_factor_model_analyzer.generate_long_short_latex_table(long_short_results_equal)

            results_value, long_short_results_value = value_factor_model_analyzer.analyze()
            ff3_value, ff5_value, q_value = value_factor_model_analyzer.results_to_strings(results_value)
            long_short_value = value_factor_model_analyzer.long_short_res_to_string(long_short_results_value)
            latex_ff3_value = value_factor_model_analyzer.generate_latex_table(results_value, "FF3")
            latex_ff5_value = value_factor_model_analyzer.generate_latex_table(results_value, "FF5")
            latex_q_value = value_factor_model_analyzer.generate_latex_table(results_value, "Q")
            latex_long_short_value = value_factor_model_analyzer.generate_long_short_latex_table(long_short_results_value)

            beta_mean, beta_std, t_stat, n = fama_macbeth_analyzer.analyze()
            fama_macbeth_res = fama_macbeth_analyzer.fama_macbeth_res_to_string(beta_mean, beta_std, t_stat, n)
            latex_fama_macbeth = fama_macbeth_analyzer.generate_fama_macbeth_latex_table(beta_mean, beta_std, t_stat, n)

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
                "fama_macbeth.txt": fama_macbeth_res,
                "latex_output.tex": create_complete_latex_document(
                    latex_ff3_equal, 
                    latex_ff5_equal, 
                    latex_q_equal, 
                    latex_long_short_equal,
                    latex_ff3_value,
                    latex_ff5_value,
                    latex_q_value,
                    latex_long_short_value,
                    latex_fama_macbeth
                )
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


def create_complete_latex_document(
        ff3_equal_tex: str, 
        ff5_equal_tex: str, 
        q_equal_tex: str, 
        long_short_equal_tex: str, 
        ff3_value_tex: str, 
        ff5_value_tex: str, 
        q_value_tex: str, 
        long_short_value_tex: str,
        fama_macbeth_tex: str,
        title: str = "Factor Model Regression Results"
    ) -> str:
    """
    Wraps given LaTeX tables into a complete .tex document.

    Args:
        ff3_tex (str): LaTeX table for FF3 model.
        ff5_tex (str): LaTeX table for FF5 model.
        q_tex (str): LaTeX table for Q-Factor model.
        title (str): Title of the LaTeX document.

    Returns:
        str: Full LaTeX document as a string.
    """
    document = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{amsmath}

\title{""" + title + r"""}
\date{\today}

\begin{document}

\maketitle

""" + r"""
\section{Fama-French Analysis with Equal Weighting}
\subsection{Fama-French 3-Factor Model}
""" + "\n" + ff3_equal_tex + "\n\n" + r"""
\pagebreak
\subsection{Fama-French 5-Factor Model}
""" + "\n" + ff5_equal_tex + "\n\n" + r"""
\pagebreak
\subsection{Q-Factor Model}
""" + "\n" + q_equal_tex + "\n" + r"""

\subsection{Long-Short Analysis}
""" + "\n" + long_short_equal_tex + "\n" + r"""
\pagebreak
\section{Fama-French Analysis with Value Weighting}
\subsection{Fama-French 3-Factor Model}
""" + "\n" + ff3_value_tex + "\n\n" + r"""
\pagebreak
\subsection{Fama-French 5-Factor Model}
""" + "\n" + ff5_value_tex + "\n\n" + r"""
\pagebreak
\subsection{Q-Factor Model}
""" + "\n" + q_value_tex + "\n" + r"""
\pagebreak
\subsection{Long-Short Analysis}
""" + "\n" + long_short_value_tex + "\n" + r"""

\section{Fama-MacBeth Regression Result}
""" + "\n" + fama_macbeth_tex + "\n" + r"""
\end{document}
"""

    return document

if __name__ == "__main__": 
    app.run(debug=True)