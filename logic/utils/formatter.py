import pandas as pd
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

class Formatter:
    """Helpers to format regression outputs and LaTeX documents.

    This class provides static utilities to:
    - Convert regression results into plain-text blocks.
    - Build LaTeX tables for FF3/FF5/Q decile regressions and long–short models.
    - Assemble a full LaTeX document from given table fragments.
    - Compile a LaTeX .tex file into a PDF in a temporary directory.
    - Escape LaTeX special characters and inject a title when needed.
    """

    @staticmethod
    def results_to_strings(results: dict) -> Tuple[str, str, str]:
        """Create plain-text summaries for FF3, FF5, and Q per decile.

        Parameters
        ----------
        results : dict
            Mapping of decile/slice to a dict with keys 'FF3', 'FF5', 'Q', each
            containing a statsmodels regression results wrapper.

        Returns
        -------
        Tuple[str, str, str]
            The concatenated text summaries for FF3, FF5, and Q respectively.
        """
        ff3_str, ff5_str, q_str = "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Slice {q} ---\n{res['FF3'].summary().as_text()}\n"
            ff5_str += f"\n--- Slice {q} ---\n{res['FF5'].summary().as_text()}\n"
            q_str += f"\n--- Slice {q} ---\n{res['Q'].summary().as_text()}\n"

        return ff3_str, ff5_str, q_str

    @staticmethod
    def long_short_res_to_string(result: dict) -> str:
        """Create a plain-text summary for the long–short (D10–D1) portfolio.

        Parameters
        ----------
        result : dict
            Dict with keys 'FF3', 'FF5', 'Q' mapping to statsmodels results.

        Returns
        -------
        str
            Combined multi-section text with model summaries.
        """
        output = ""
        output = "\n======= Long-Short Portfolio (Slice 10 - Slice 1) =======\n"
        output += "\n--- FF3 Regression ---\n" + result["FF3"].summary().as_text()
        output += "\n\n--- FF5 Regression ---\n" + result["FF5"].summary().as_text()
        output += "\n\n--- Q-Factor Regression ---\n" + result["Q"].summary().as_text()

        return output
    
    @staticmethod
    def fama_macbeth_res_to_string(means: pd.Series, tstats: pd.Series, n_months: int, signal_name: str) -> str:
        """Render Fama–MacBeth average coefficients and t-stats as text.

        Parameters
        ----------
        means : pd.Series
            Time-series averages of monthly cross-sectional coefficients.
        tstats : pd.Series
            FM t-statistics aligned to the same coefficient names.
        n_months : int
            Number of months used in the FM aggregation.
        signal_name : str
            Pretty name for the signal to display instead of 'Signal'.

        Returns
        -------
        str
            A multi-line block with coefficient names, means, and t-stats.
        """
        lines = [f"\n======= Fama-MacBeth Results (T={n_months} months) =======\n"] 
        for coef in means.index:
            coef_name = signal_name if coef == "Signal" else coef
            lines.append(f"{coef_name:10s}: {means[coef]:.4f}  (t={tstats[coef]:.2f})")
        return "\n".join(lines)
    
    @staticmethod
    def generate_fama_macbeth_latex_table(means: pd.Series, tstats: pd.Series, signal_name: str):
        """Build a LaTeX table for Fama–MacBeth mean coefficients and t-stats."""
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
            "\\caption{Fama-MacBeth Results}\n"
            "\\end{table}\n"
        )

    @staticmethod
    def alpha_table_to_latex(df: pd.DataFrame, group_name: str, caption: str) -> str:
        """Render a compact per-group alpha table (single page).

        Parameters
        ----------
        df : pd.DataFrame
            Rows are groups/schemes; columns in {'FF3','FF5','Q'}; cells are
            LaTeX strings like 'alpha [t]'.
        group_name : str
            Column header for the group identifier.
        caption : str
            LaTeX caption for the table.
        """
        cols = ["FF3","FF5","Q"]
        cols = [c for c in cols if c in df.columns]
        body = "\n".join(
            f"{Formatter._latex_escape(str(idx))} & " +
            " & ".join(df.loc[idx, c] if isinstance(df.loc[idx, c], str) else "" for c in cols) +
            r" \\[12pt]"
            for idx in df.index
        )
        header = group_name + " & " + " & ".join(cols) + r" \\ \midrule"
        table_out = (
            "\\begin{table}[H]\n\\centering\n"
            "\\begin{tabular}{l" + "c"*len(cols) + "}\n\\toprule\n" +
            header + "\n" + body + "\n\\bottomrule\n\\end{tabular}\n" +
            f"\\caption{{{Formatter._latex_escape(caption)}}}\n\\end" + "{table}\n"
        )
        return table_out 
    
    @staticmethod
    def alpha_table_to_latex_four_quarters_two_pages(df: pd.DataFrame, group_name: str, caption: str) -> str:
        """Render a large per-group alpha table across two pages (4 blocks)."""
        # Select columns present
        cols = [c for c in ["FF3", "FF5", "Q"] if c in df.columns]

        # Split into four nearly equal parts
        n = len(df)
        q, r = divmod(n, 4)
        sizes = [(q + 1 if i < r else q) for i in range(4)]
        parts = []
        start = 0
        for s in sizes:
            parts.append(df.iloc[start:start+s])
            start += s

        # Build a tabular inside a minipage for each part
        tabulars = []
        for part in parts:
            # Body rows
            rows = []
            for idx in part.index:
                row_cells = [Formatter._latex_escape(str(idx))]
                for c in cols:
                    v = part.loc[idx, c]
                    row_cells.append(v if isinstance(v, str) else "")
                rows.append(" & ".join(row_cells) + r" \\[12pt]")
            body = "\n".join(rows)

            # Header + full tabular
            header = f"{group_name} & " + " & ".join(cols) + r" \\ \midrule"
            tabular = (
                "\\begin{minipage}{0.48\\linewidth}\n" +
                "\\centering\n" +
                "\\begin{tabular}{l" + "c"*len(cols) + "}\n" +
                "\\toprule\n" +
                header + "\n" +
                body + ("\n" if body else "") +  # if empty, avoid extra newline
                "\\bottomrule\n" +
                "\\end{tabular}\n" +
                "\\end{minipage}\n"
                )
            tabulars.append(tabular)

        # Ensure we have 4 entries (may be empty)
        while len(tabulars) < 4:
            tabulars.append(
                "\\begin{minipage}{0.48\\linewidth}\n\\centering\n" +
                "\\begin{tabular}{l" + "c"*len(cols) + "}\n\\toprule\n" +
                (f"{group_name} & " + " & ".join(cols) + r" \\ \midrule\n" if cols else "") +
                "\\bottomrule\n\\end{tabular}\n\\end{minipage}\n"
            )

        # Page 1: left Q1, right Q2
        page1 = (
            "\\begin{table}[H]\n\\centering\n"
            + tabulars[0]
            + "\\hfill\n"
            + tabulars[1]
            + "\\end{table}\n"
        )

        # Page 2: left Q3, right Q4
        page2 = (
            "\\begin{table}[H]\n\\centering\n"
            + tabulars[2]
            + "\\hfill\n"
            + tabulars[3]
            + f"\\caption{{{Formatter._latex_escape(caption)}}}\n"
            + "\\end{table}\n"
        )

        # Page break between the two
        return page1 + "\\newpage\n" + page2
                    
    @staticmethod
    def generate_latex_table(results_dict: dict, model_name: str) -> str:
        """Generate a LaTeX table for one factor model across deciles.

        Parameters
        ----------
        results_dict : dict
            Mapping from decile/slice to regression results by model.
        model_name : str
            One of 'FF3', 'FF5', or 'Q'.

        Returns
        -------
        str
            LaTeX code for a tabular with coefficients and t-stats.
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
            r"\begin{table}[H]" + "\n"
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
        """Generate a LaTeX table for long–short regression results across models.

        Parameters
        ----------
        results_dict : dict
            Dict mapping 'FF3', 'FF5', 'Q' to statsmodels results wrappers.

        Returns
        -------
        str
            LaTeX code with one row per model and columns for all factors.
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
            r"\begin{table}[H]" + "\n"
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
        industry_tex: str,
        industry_aggregated_tex: str,
        country_tex: str,
        country_aggregated_tex: str,
        title: str = "Global Stock Market Protocol Analysis Results"
    ) -> str:
        """Wrap provided LaTeX tables into a complete article-class document.

        Returns the LaTeX source as a single string.
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
\section{Factor Model Analysis with Equal Weighting}
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
\section{Factor Model Analysis with Value Weighting}
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

\section{Factor Model Analysis by Industries}
""" + "\n" + industry_tex + "\n" + r"""
""" + "\n" + industry_aggregated_tex + "\n" + r"""

\section{Factor Model Analysis by Countries}
""" + "\n" + country_tex + "\n" + r"""
\pagebreak
""" + "\n" + country_aggregated_tex + "\n" + r"""

\end{document}
        """

        return document
    
    @staticmethod
    def tex_file_to_pdf(tex_path: str, output_path: str) -> str:
        """Compile a .tex file with pdflatex and write the resulting PDF.

        Parameters
        ----------
        tex_path : str
            Path to the .tex source file to compile.
        output_path : str
            Destination path for the compiled PDF.

        Returns
        -------
        str
            Absolute path to the generated PDF.
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
        """Inject a LaTeX \title if missing, otherwise replace the existing one.

        Behavior
        --------
        1) If a "\\title{" exists, replace its contents.
        2) Otherwise, insert a title and \maketitle before \begin{document}.
        """
        # If Formatter can't take a doc_title kwarg, inject it into the LaTeX string.
        # 1) Replace existing \title{...} if present; else
        # 2) Insert \title{...}\maketitle before \begin{document}.
        if r'\title{' in doc:
            return re.sub(r'\\title\{[^\}]*\}', f'\\title{{{title}}}', doc, count=1, flags=re.S)
        return doc.replace(r'\begin{document}', f'\\title{{{title}}}\n\\maketitle\n' +'\\begin{document}', 1)

    @staticmethod
    def _latex_escape(s: str) -> str:
        """Escape LaTeX special characters in arbitrary strings."""
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
