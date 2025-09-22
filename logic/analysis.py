from __future__ import annotations 

import re 
import os 
import pandas as pd 
import zipfile 

from logic.analyzers.base import SharedContext
from logic.compose.registry import discover_analyzers
from logic.compose.composer import compose_document
from logic.utils.formatter import Formatter


def run_complete_analysis(df: pd.DataFrame, signal_name: str):
    crsp_full = pd.read_csv("./data/dsws_crsp.csv")
    factors_full = pd.read_csv("./data/Factors.csv")
    fm_full = pd.read_csv("./data/Fama_Macbeth.csv")
    ctx = SharedContext(crsp=crsp_full, factors=factors_full, fm=fm_full)

    # --- Auto-Discovery der Analyzer ---
    outputs = []
    for Plugin in discover_analyzers():
        plugin = Plugin(ctx, df, signal_name)
        out = plugin.generate_output()
        outputs.append(out)

    # --- LaTeX-Dokument zusammensetzen (Titel wie bisher) ---
    escaped_signal = Formatter._latex_escape(signal_name)
    baseline_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Baseline (All Industries)"
    latex_output = compose_document(baseline_title, outputs)

    # --- ZIP-Dateien/RAW-Texte wie bisher zusammenstellen ---
    results = {"output.tex": latex_output}
    # Kopple die bekannten filenames auf Basis der Keys (Rückwärtskompatibilität)
    # Falls ein Plugin bestimmte raw_texts liefert, nimm sie rein:
    for out in outputs:
        for fname, content in out.raw_texts.items():
            results[fname] = content

    basedir = os.path.abspath(os.path.dirname(__file__))
    static_dir = os.path.join(os.path.dirname(basedir), "static")
    result_dir = os.path.join(static_dir, "downloads")
    os.makedirs(result_dir, exist_ok=True)

    safe_signal_name = re.sub(r'[^A-Za-z0-9_-]', '_', signal_name)
    zip_filename = f"{safe_signal_name}.zip"
    zip_path = os.path.join(result_dir, zip_filename)

    # ---------- Write ZIP ----------
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename, content in results.items():
            temp_file = os.path.join(result_dir, f"{safe_signal_name}_{filename.replace('/', '_')}")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(str(content))

            # Keep folder structure in archive
            if filename.endswith("output.tex"):
                arcname = filename
            else:
                arcname = os.path.join(os.path.dirname(filename), "raw", os.path.basename(filename))

            zipf.write(temp_file, arcname=arcname)

            if filename.endswith("output.tex"):
                pdf_temp_path = os.path.join(
                    result_dir, f"{safe_signal_name}_{os.path.dirname(filename).replace('/', '_')}_output.pdf"
                )
                Formatter.tex_file_to_pdf(temp_file, pdf_temp_path)
                zipf.write(pdf_temp_path, arcname=os.path.join(os.path.dirname(filename), "output.pdf"))
                os.remove(pdf_temp_path)

            os.remove(temp_file)

    return zip_path
