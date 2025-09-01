import pandas as pd
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

class Formatter:

    @staticmethod
    def results_to_strings(results: dict) -> Tuple[str, str, str]:
        ff3_str, ff5_str, q_str = "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Slice {q} ---\n{res['FF3'].summary().as_text()}\n"
            ff5_str += f"\n--- Slice {q} ---\n{res['FF5'].summary().as_text()}\n"
            q_str += f"\n--- Slice {q} ---\n{res['Q'].summary().as_text()}\n"

        return ff3_str, ff5_str, q_str

    @staticmethod
    def long_short_res_to_string(result: dict) -> str:
        output = ""
        output = "\n======= Long-Short Portfolio (Slice 10 - Slice 1) =======\n"
        output += "\n--- FF3 Regression ---\n" + result["FF3"].summary().as_text()
        output += "\n\n--- FF5 Regression ---\n" + result["FF5"].summary().as_text()
        output += "\n\n--- Q-Factor Regression ---\n" + result["Q"].summary().as_text()

        return output
    
    @staticmethod
    def fama_macbeth_res_to_string(means: pd.Series, tstats: pd.Series, n_months: int, signal_name: str) -> str:
        lines = [f"\n======= Fama-MacBeth Results (T={n_months} months) =======\n"] 
        for coef in means.index:
            coef_name = signal_name if coef == "Signal" else coef
            lines.append(f"{coef_name:10s}: {means[coef]:.4f}  (t={tstats[coef]:.2f})")
        return "\n".join(lines)
    
    @staticmethod
    def generate_fama_macbeth_latex_table(means: pd.Series, tstats: pd.Series, signal_name: str):
        rows = []
        for coef in means.index:
            coef_name = Formatter._latex_escape(signal_name) if coef == "Signal" else coef
            rows.append(
                f"{coef_name} & {means[coef]:.4f} & {tstats[coef]:.2f} \\\\"
            )
        body = "\n".join(rows)
        return (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\begin{tabular}{lrr}\n"
            "\\toprule\n"
            "Variable & Mean Estimate & T-Stat \\\\\n"
            "\\midrule\n" +
            body + "\n" +
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\caption{Fama-MacBeth Results for}\n"
            "\\end{table}\n"
        )
                
    @staticmethod
    def generate_latex_table(results_dict: dict, model_name: str) -> str:
        """
        Generates a LaTeX table from regression results.

        Args:
            results_dict (dict): Dictionary mapping slice to statsmodels regression results.
            model_name (str): One of 'FF3', 'FF5', or 'Q'.

        Returns:
            str: LaTeX code for the table.
        """

        factor_order = {
            "FF3": ["const", "MKTRF_usd", "SMB_usd", "HML_usd"],
            "FF5": ["const", "MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
            "Q"  : ["const", "MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"]
        }

        factor_labels = {
            'const': r'$\alpha$',
            'MKTRF_usd': r'$\beta_{MKT}$',
            'SMB_usd': r'$\beta_{SMB}$',
            'HML_usd': r'$\beta_{HML}$',
            'RMW_A_usd': r'$\beta_{RMW}$',
            'CMA_usd': r'$\beta_{CMA}$',
            'ME_usd': r'$\beta_{SIZE}$',
            'IA_usd': r'$\beta_{IA}$',
            'ROE_usd': r'$\beta_{ROE}$'
        }

        # Table header
        headers = [factor_labels[p] for p in factor_order[model_name]]
        table_header = " & " + " & ".join(headers) + r" \\ \toprule" + "\n"

        # Table body
        body = ""
        for slice_id in sorted(results_dict.keys()):
            model = results_dict[slice_id][model_name]
            coef = model.params
            tvals = model.tvalues

            row = [str(slice_id)]
            for factor in factor_order[model_name]:
                if factor in coef:
                    val = f"{coef[factor]:.4f}"
                    tstat = f"{tvals[factor]:.2f}"
                    cell = r"\begin{tabular}{@{}c@{}}" + val +  r"\\\relax [" +  tstat + r"]\end{tabular}"
                else:
                    cell = ""
                row.append(cell)

            body += " & ".join(row) + r" \\[12pt]" + "\n"

        # Final LaTeX table
        table = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{l" + "c" * len(factor_order[model_name]) + "}" + "\n"
            r"\toprule" + "\n"
            "Slice" + table_header + body + r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            rf"\caption{{Regression Results: {model_name}. T-statistics are in brackets.}}" + "\n"
            r"\end{table}"
        )

        return table
    
    @staticmethod
    def generate_long_short_latex_table(results_dict: dict) -> str:
        """
        Generates a LaTeX table for the long-short regression results across FF3, FF5, and Q models.

        Args:
            res_ff3, res_ff5, res_q (RegressionResultsWrapper): Regression results for each model.

        Returns:
            str: LaTeX code for the table.
        """
        models = {
            'FF3': results_dict["FF3"],
            'FF5': results_dict["FF5"],
            'Q': results_dict["Q"]
        }

        # Desired fixed order of all potential factors
        all_factors = ["const", "MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd", "ME_usd", "IA_usd", "ROE_usd"]

        # LaTeX-friendly labels
        factor_labels = {
            'const': r'$\alpha$',
            'MKTRF_usd': r'$\beta_{MKT}$',
            'SMB_usd': r'$\beta_{SMB}$',
            'HML_usd': r'$\beta_{HML}$',
            'RMW_A_usd': r'$\beta_{RMW}$',
            'CMA_usd': r'$\beta_{CMA}$',
            'ME_usd': r'$\beta_{SIZE}$',
            'IA_usd': r'$\beta_{IA}$',
            'ROE_usd': r'$\beta_{ROE}$'
        }

        # Table header
        headers = ["Model"] + [factor_labels.get(f, f) for f in all_factors]
        table_header = " & ".join(headers) + r" \\ \toprule" + "\n"

        # Table body with t-stats
        body = ""
        for model_name, res in models.items():
            row = [model_name]
            coef = res.params
            tvals = res.tvalues

            for factor in all_factors:
                if factor in coef:
                    val = f"{coef[factor]:.4f}"
                    tstat = f"{tvals[factor]:.2f}"
                    cell = r"\begin{tabular}{@{}c@{}}" + val +  r"\\\relax[" +  tstat + r"]\end{tabular}"
                else:
                    cell = ""
                row.append(cell)
            body += " & ".join(row) + r" \\[12pt]" + "\n"

        # Full table
        table = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{l" + "c" * len(all_factors) + "}" + "\n"
            r"\toprule" + "\n"
            + table_header + body +
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{Long-Short Regression Results (Slice 10 - Slice 1). T-statistics are in brackets.}" + "\n"
            r"\end{table}"
        )

        return table
    
    @staticmethod
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
        title: str = "Global Stock Market Protocol Analysis Results"
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
        document = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{float}

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
    
    @staticmethod
    def tex_file_to_pdf(tex_path: str, output_path: str) -> str:
        """
        Compiles a LaTeX .tex file into a PDF.

        Args:
            tex_path (str): Path to the input .tex file.
            output_path (str): Path where the output PDF should be saved.

        Returns:
            str: The absolute path to the generated PDF.
        """
        tex_path = Path(tex_path).resolve()
        output_path = Path(output_path).resolve()

        if not tex_path.exists():
            raise FileNotFoundError(f"Input .tex file not found: {tex_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tmp_tex_path = tmpdir_path / tex_path.name

            tmp_tex_path.write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")

            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tmp_tex_path.name],
                    cwd=tmpdir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"LaTeX compilation failed:\n{e.stderr.decode()}")

            generated_pdf = tmpdir_path / tex_path.with_suffix(".pdf").name
            if not generated_pdf.exists():
                raise FileNotFoundError("PDF was not generated. Check your LaTeX code.")

            shutil.move(str(generated_pdf), str(output_path))

        return str(output_path)
    
    @staticmethod
    def _inject_title_fallback(doc: str, title: str) -> str:
        # If Formatter can't take a doc_title kwarg, inject it into the LaTeX string.
        # 1) Replace existing \title{...} if present; else
        # 2) Insert \title{...}\maketitle before \begin{document}.
        if r'\title{' in doc:
            return re.sub(r'\\title\{.*?\}', f'\\title{{{title}}}', doc, count=1, flags=re.S)
        return doc.replace(r'\begin{document}', f'\\title{{{title}}}\n\\maketitle\n\\begin{document}', 1)

    @staticmethod
    def _latex_escape(s: str) -> str:
        """
        Escape LaTeX special chars in strings for safe use inside LaTeX documents.
        """
        replacements = {
            "\\": r"\textbackslash{}",
            "_": r"\_",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)
        return s
