from __future__ import annotations

import os
import re
import zipfile
from typing import Dict, List

import pandas as pd

from logic.analyzers.base import SharedContext
from logic.compose.registry import Registry
from logic.compose.composer import Composer
from logic.utils.formatter import Formatter


class Analysis:
    @staticmethod
    def run_complete_analysis(
        df: pd.DataFrame, 
        signal_name: str, 
        selected_analyzers: List[str] = [], 
        country_filter: List[str] | None = None,
        ff12_filter: List[int] | None = None,
        min_mcap_pct: float | None = None,    
        max_mcap_pct: float | None = None
    ) -> str:
        """Run all analyzers, compose a LaTeX report, and bundle results as ZIP.

        Workflow
        --------
        1. Load shared datasets (CRSP-like panel, factor returns, FM variables).
        2. Auto-discover enabled analyzers and run them on the provided signal.
        3. Compose a LaTeX document from all analyzers' LaTeX blocks.
        4. Render a PDF from the LaTeX source and package everything into a ZIP
        alongside any plugin-provided raw text artifacts.

        Parameters
        ----------
        df : pd.DataFrame
            Input data where the first three columns are interpreted as
            ``[DSCD, date, signal]``.
        signal_name : str
            Human-readable name of the signal used in titles and filenames.

        Returns
        -------
        str
            Absolute path to the created ZIP archive containing:
            - ``output.tex`` and ``output.pdf`` (rendered report)
            - All plugin ``raw_texts`` under a ``raw/`` subfolder in the archive.
        """
        # Load shared data (adjust paths as needed for your environment)
        crsp_full = pd.read_csv("./data/dsws_crsp.csv")
        factors_full = pd.read_csv("./data/Factors.csv")
        fm_full = pd.read_csv("./data/Fama_Macbeth.csv")
        corr_full = pd.read_parquet("./data/Predictor_different.parquet", engine='fastparquet')

        if country_filter:
            crsp_full["country"] = crsp_full["country"].astype(str).str.strip()
            fm_full["country"] = fm_full["country"].astype(str).str.strip()
            country_whitelist = {c.strip().lower() for c in country_filter}
            crsp_full = crsp_full[crsp_full["country"].str.lower().isin(country_whitelist)]
            fm_full = fm_full[fm_full["country"].str.lower().isin(country_whitelist)]

        if ff12_filter:
            crsp_full = crsp_full.copy()
            crsp_full["ff12"] = pd.to_numeric(crsp_full["ff12"], errors="coerce").astype("Int64")
            crsp_full = crsp_full[crsp_full["ff12"].isin(pd.Series(ff12_filter, dtype="Int64"))]

        if (min_mcap_pct is not None) or (max_mcap_pct is not None):
            min_pct = 0.0 if min_mcap_pct is None else float(min_mcap_pct)
            max_pct = 100.0 if max_mcap_pct is None else float(max_mcap_pct)
            min_pct = max(0.0, min(100.0, min_pct))
            max_pct = max(0.0, min(100.0, max_pct))
            q_lo_crsp = crsp_full["size_lag"].astype(float).quantile(min_pct/100) if "size_lag" in crsp_full.columns else None
            q_hi_crsp = crsp_full["size_lag"].astype(float).quantile(max_pct/100) if "size_lag" in crsp_full.columns else None
            q_lo_fm   = fm_full["size_lag"].astype(float).quantile(min_pct/100) if "size_lag" in fm_full.columns else None
            q_hi_fm   = fm_full["size_lag"].astype(float).quantile(max_pct/100) if "size_lag" in fm_full.columns else None
            q_lo_corr   = corr_full["size_lag"].astype(float).quantile(min_pct/100) if "size_lag" in corr_full.columns else None
            q_hi_corr   = corr_full["size_lag"].astype(float).quantile(max_pct/100) if "size_lag" in corr_full.columns else None

            def _between(df_, col, lo, hi):
                if lo is None or hi is None or col not in df_.columns:
                    return df_
                s = pd.to_numeric(df_[col], errors="coerce")
                return df_[s.between(lo, hi, inclusive="both")]

            crsp_full = _between(crsp_full, "size_lag", q_lo_crsp, q_hi_crsp)
            fm_full   = _between(fm_full,   "size_lag", q_lo_fm,   q_hi_fm)
            corr_full = _between(corr_full,   "size_lag", q_lo_corr,   q_hi_corr)

        ctx = SharedContext(crsp=crsp_full, factors=factors_full, fm=fm_full, correlation_data=corr_full)
        plugins = Registry.discover_selected_analyzers(selected_analyzers)

        # --- Analyzer auto-discovery & execution ---
        outputs = []
        for Plugin in plugins:
            plugin = Plugin(ctx, df, signal_name)
            out = plugin.generate_output()
            outputs.append(out)

        # --- Compose LaTeX document ---
        escaped_signal = Formatter._latex_escape(signal_name)
        baseline_title = (
            f"Global Stock Market Protocol Analysis Results for {escaped_signal}"
        )
        latex_output = Composer.compose_document(baseline_title, outputs)

        # --- Collect artifacts for packaging ---
        results: Dict[str, str] = {"output.tex": latex_output}
        # Include any plugin-provided raw text files
        for out in outputs:
            for fname, content in out.raw_texts.items():
                results[fname] = content

        # --- Prepare output directories ---
        basedir = os.path.abspath(os.path.dirname(__file__))
        static_dir = os.path.join(os.path.dirname(basedir), "static")
        result_dir = os.path.join(static_dir, "downloads")
        os.makedirs(result_dir, exist_ok=True)

        safe_signal_name = re.sub(r"[^A-Za-z0-9_-]", "_", signal_name)
        zip_filename = f"{safe_signal_name}.zip"
        zip_path = os.path.join(result_dir, zip_filename)

        # --- Write ZIP (and render PDF) ---
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for filename, content in results.items():
                # Write content to a temporary file on disk first
                temp_file = os.path.join(
                    result_dir, f"{safe_signal_name}_{filename.replace('/', '_')}"
                )
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(str(content))

                # Keep folder structure in archive for the primary output
                if filename.endswith("output.tex"):
                    arcname = filename
                else:
                    # Place plugin raw texts under a 'raw/' folder next to their logical path
                    arcname = os.path.join(os.path.dirname(filename), "raw", os.path.basename(filename))

                zipf.write(temp_file, arcname=arcname)

                # For the LaTeX document, also render & include a PDF
                if filename.endswith("output.tex"):
                    pdf_temp_path = os.path.join(
                        result_dir,
                        f"{safe_signal_name}_{os.path.dirname(filename).replace('/', '_')}_output.pdf",
                    )
                    Formatter.tex_file_to_pdf(temp_file, pdf_temp_path)
                    zipf.write(
                        pdf_temp_path,
                        arcname=os.path.join(os.path.dirname(filename), "output.pdf"),
                    )
                    os.remove(pdf_temp_path)

                # Clean up temporary file
                os.remove(temp_file)

        return zip_path
