from __future__ import annotations 

import re 
import os 
import numpy as np 
import pandas as pd 
import zipfile 
import statsmodels.api as sm 

from abc import ABC, abstractmethod 
from typing import Tuple, Any, Dict
from logic.formatter import Formatter


NUM_OF_SLICES = 10


class BaseAnalyzer(ABC):
    """Abstract analyzer for stock return predictors."""

    def __init__(self, input_file: pd.DataFrame, crsp: pd.DataFrame | None = None, factors: pd.DataFrame | None = None, fm: pd.DataFrame | None = None) -> None:
        self.data = input_file
        self.crsp = crsp if crsp is not None else pd.read_csv("./data/dsws_crsp.csv")
        self.factors = factors if factors is not None else pd.read_csv("./data/Factors.csv")
        self.fm = fm if fm is not None else pd.read_csv("./data/Fama_Macbeth.csv")

    @abstractmethod
    def analyze(self) -> Any:
        pass


class EqualWeightedFactorModelAnalyzer(BaseAnalyzer):
    """Fama–French (3/5/Q) Regressions auf equal-weighted Portfolios."""
    def analyze(self) -> Tuple[dict, dict]:
        # --- Input & Monat ---
        df = self.data.copy()
        # Eingabedatei-Spalten umbenennen: 1=DSCD, 2=dates, 3=signal
        df = df.rename(columns={df.columns[0]: "DSCD",
                                df.columns[1]: "dates",
                                df.columns[2]: "signal"})
        df["date"] = pd.to_datetime(df["dates"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

        # --- CRSP joinen ---
        cr = self.crsp.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")
        cr_small = cr[["DSCD", "month", "RET_USD", "size_lag"]]
        df = pd.merge(df, cr_small, on=["DSCD", "month"], how="inner")

        # --- Returns & Signal ---
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

        # --- 1) Dezile ---
        def assign_deciles(g: pd.DataFrame) -> pd.DataFrame:
            try:
                g["decile"] = pd.qcut(g["signal"], NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1))
            except ValueError:
                r = g["signal"].rank(method="first")
                g["decile"] = pd.qcut(r, NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1))
            g["decile"] = g["decile"].astype(int)
            return g
        df = df.groupby("month").apply(assign_deciles, include_groups=False)

        # --- 2) Equal-weighted Returns ---
        def ewret(g: pd.DataFrame) -> float:
            v = g["ret"].dropna()
            return float(v.mean()) if len(v) else np.nan
        decile_rets = (
            df.groupby(["month", "decile"])
              .apply(ewret, include_groups=False)
              .rename("ewret")
              .reset_index()
        )
        decile_wide = decile_rets.pivot(index="month", columns="decile", values="ewret").sort_index()

        # --- 3) Faktoren ---
        fac = self.factors.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        factor_cols = [c for c in fac.columns if c not in ["DATE"]]
        fac[factor_cols] = fac[factor_cols] * 100

        model_specs = {
            "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
            "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
            "Q"  : ["MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"],
        }

        # --- 4) Excess-Returns & LS ---
        decile_wide_ex = decile_wide.sub(fac["rf_ff"], axis=0)
        ls_ex = decile_wide_ex[NUM_OF_SLICES] - decile_wide_ex[1]

        # --- 5) Helper ---
        def fit_ols(y: pd.Series, X: pd.DataFrame):
            Xc = sm.add_constant(X)
            mask = y.notna() & Xc.notna().all(axis=1)
            return sm.OLS(y[mask], Xc[mask]).fit()

        # --- 6) Regression ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls  = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_equal = {i: {} for i in dec_nums}
        long_short_results_equal = {}

        for label, fac_cols in model_specs.items():
            long_short_results_equal[label] = fit_ols(ts_ls["LS_ex"], ts_ls[fac_cols])
            for i in dec_nums:
                results_equal[i][label] = fit_ols(ts_dec[i], ts_dec[fac_cols])

        return results_equal, long_short_results_equal


class ValueWeightedFactorModelAnalyzer(BaseAnalyzer):
    """Fama–French (3/5/Q) Regressions auf value-weighted Portfolios."""
    def analyze(self) -> Tuple[dict, dict]:
        # --- Input & Monat ---
        df = self.data.copy()
        df = df.rename(columns={df.columns[0]: "DSCD",
                                df.columns[1]: "dates",
                                df.columns[2]: "signal"})
        df["date"] = pd.to_datetime(df["dates"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

        # --- CRSP joinen ---
        cr = self.crsp.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")
        cr_small = cr[["DSCD", "month", "RET_USD", "size_lag"]]
        df = pd.merge(df, cr_small, on=["DSCD", "month"], how="inner")

        # --- Returns, Signal, Gewichte ---
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df["weight"] = pd.to_numeric(df["size_lag"], errors="coerce").clip(lower=0).fillna(0.0)

        # --- 1) Dezile ---
        def assign_deciles(g: pd.DataFrame) -> pd.DataFrame:
            try:
                g["decile"] = pd.qcut(g["signal"], NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1))
            except ValueError:
                r = g["signal"].rank(method="first")
                g["decile"] = pd.qcut(r, NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1))
            g["decile"] = g["decile"].astype(int)
            return g
        df = df.groupby("month").apply(assign_deciles, include_groups=False)

        # --- 2) Value-weighted Returns ---
        def vwret(g: pd.DataFrame) -> float:
            w = pd.to_numeric(g["weight"], errors="coerce")
            r = pd.to_numeric(g["ret"], errors="coerce")
            m = w.notna() & r.notna() & (w > 0)
            if not m.any():
                return np.nan
            return float(np.average(r[m], weights=w[m]))

        decile_rets = (
            df.groupby(["month", "decile"])
              .apply(vwret, include_groups=False)
              .rename("vwret")
              .reset_index()
        )
        decile_wide = decile_rets.pivot(index="month", columns="decile", values="vwret").sort_index()

        # --- 3) Faktoren ---
        fac = self.factors.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        factor_cols = [c for c in fac.columns if c not in ["DATE"]]
        fac[factor_cols] = fac[factor_cols] * 100

        model_specs = {
            "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
            "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
            "Q"  : ["MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"],
        }

        # --- 4) Excess-Returns & LS ---
        decile_wide_ex = decile_wide.sub(fac["rf_ff"], axis=0)
        ls_ex = decile_wide_ex[NUM_OF_SLICES] - decile_wide_ex[1]

        # --- 5) Helper ---
        def fit_ols(y: pd.Series, X: pd.DataFrame):
            Xc = sm.add_constant(X)
            mask = y.notna() & Xc.notna().all(axis=1)
            return sm.OLS(y[mask], Xc[mask]).fit()

        # --- 6) Regression ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls  = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_value = {i: {} for i in dec_nums}
        long_short_results_value = {}

        for label, fac_cols in model_specs.items():
            long_short_results_value[label] = fit_ols(ts_ls["LS_ex"], ts_ls[fac_cols])
            for i in dec_nums:
                results_value[i][label] = fit_ols(ts_dec[i], ts_dec[fac_cols])

        return results_value, long_short_results_value


class FamaMacBethAnalyzer(BaseAnalyzer):
    """
    Fama–MacBeth (1973) Cross-Sectional Regressions

    Spezifikation (monatlich, cross-sectional):
        r_{i,t} = alpha_t + b1_t * ENSEMBLE_raw_{i,t} + b2_t * log(Size_{i,t}) + e_{i,t}

    Ausgabe:
        - mean_params: Zeitreihenmittel der Koeffizienten (alpha, ENSEMBLE_raw, logSize)
        - tstats: klassische FM-t-Statistiken = mean / (std / sqrt(T))
        - monthly_params: DataFrame mit den monatlichen Koeffizienten (für Diagnostics)
        - meta: n_months, Periode
    """

    def _prep(self) -> pd.DataFrame:
        # --- Input-Signal vorbereiten ---
        # Kopie der Inputdaten (Ensemble CSV), Spaltennamen harmonisieren
        sig = self.data.copy()
        sig = sig.rename(columns={sig.columns[0]: "DSCD",      # eindeutiger Firmencode
                                  sig.columns[1]: "dates",     # Datumsspalte
                                  sig.columns[2]: "ENSEMBLE_raw"})  # Signal (Ensemble)
        # Monat extrahieren (Periodenende)
        sig["month"] = pd.to_datetime(sig["dates"]).dt.to_period("M").dt.to_timestamp("M")

        # --- CRSP-Daten vorbereiten ---
        cr = self.crsp.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")

        # Nur benötigte Variablen behalten: Rendite & Size (zeitgleich, nicht gelagged)
        cr_small = cr[["DSCD", "month", "RET_USD", "SIZE"]].copy()

        # --- Merge von Signal & CRSP ---
        # Inner Join auf (DSCD, month), damit pro Aktie und Monat Signal + Rendite + Size da sind
        df = pd.merge(sig, cr_small, on=["DSCD", "month"], how="inner")

        # --- Variablen bereinigen & berechnen ---
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")          # Zielvariable: Rendite
        df["ensemble"] = pd.to_numeric(df["ENSEMBLE_raw"], errors="coerce") # Regressor 1: Ensemble
        df["SIZE"] = pd.to_numeric(df["SIZE"], errors="coerce")             # Rohgröße für logSize
        # log(Size), aber nur wenn Size > 0 (sonst NaN)
        df["logSize"] = np.nan
        mask = df["SIZE"] > 0
        df.loc[mask, "logSize"] = np.log(df.loc[mask, "SIZE"])

        # Endgültiges DataFrame zurückgeben
        return df[["DSCD", "month", "ret", "ensemble", "logSize"]]

    @staticmethod
    def _cs_ols(g: pd.DataFrame) -> pd.Series:
        # --- Cross-Sectional OLS pro Monat ---
        y = g["ret"]                               # abhängige Variable: Rendite r_{i,t}
        X = g[["ensemble", "logSize"]]             # Regressoren: Ensemble + log(Size)
        X = sm.add_constant(X, has_constant="add") # Konstante hinzufügen (alpha_t)

        # Nur vollständige Beobachtungen behalten
        mask = y.notna() & X.notna().all(axis=1)
        y, X = y[mask], X[mask]

        # Mindestbedingung: ≥3 Beobachtungen (2 Regressoren + 1 Konstante)
        if len(y) < 3:
            return pd.Series({"alpha": np.nan, "ENSEMBLE_raw": np.nan, "logSize": np.nan})

        # OLS schätzen (Querschnitt für einen Monat)
        res = sm.OLS(y, X).fit()

        # Koeffizienten extrahieren und zurückgeben
        return pd.Series({
            "alpha": res.params.get("const", np.nan),      # Achsenabschnitt
            "ENSEMBLE_raw": res.params.get("ensemble", np.nan),  # Effekt des Signals
            "logSize": res.params.get("logSize", np.nan),  # Effekt der Firmengröße
        })

    @staticmethod
    def _fm_tstat(series: pd.Series) -> float:
        # --- Fama–MacBeth t-Statistik ---
        # Klassisch: Mittelwert der Monats-Koeffizienten geteilt durch deren Std/sqrt(T)
        s = series.dropna()
        T = len(s)
        if T == 0:
            return np.nan
        return float(s.mean() / (s.std(ddof=1) / np.sqrt(T)))

    def analyze(self):
        # --- Daten vorbereiten ---
        df = self._prep()

        # --- Schritt 1: Monatsweise Querschnittsregressionen ---
        monthly_params = (
            df.groupby("month")
              .apply(self._cs_ols, include_groups=False)   # für jeden Monat: _cs_ols
              .rename(columns={"ENSEMBLE_raw": "ensemble"})
        )
        # Spaltennamen harmonisieren
        monthly_params = monthly_params.rename(columns={"ensemble": "ENSEMBLE_raw"})

        # --- Schritt 2: Zeitreihen-Mittelwerte & FM-t-Statistiken ---
        mean_params = monthly_params.mean(skipna=True).to_dict()  # Durchschnitt über Monate
        tstats = {k: self._fm_tstat(monthly_params[k]) for k in monthly_params.columns} # t-Werte

        # --- Schritt 3: Meta-Infos speichern ---
        meta = {
            "n_months": int(monthly_params.dropna(how="all").shape[0]),
            "period_start": monthly_params.index.min(),
            "period_end": monthly_params.index.max(),
        }

        # --- Schritt 4: Ergebnisse ins Terminal drucken ---
        print("\n=== Fama-MacBeth Regression Results ===")
        print(f"Periode: {meta['period_start'].date()} bis {meta['period_end'].date()} "
              f"({meta['n_months']} Monate)\n")

        header = f"{'Variable':<12} {'MeanCoeff':>12} {'t-Stat':>12}"
        print(header)
        print("-" * len(header))
        for var in ["alpha", "ENSEMBLE_raw", "logSize"]:
            mean_val = mean_params.get(var, np.nan)
            t_val = tstats.get(var, np.nan)
            print(f"{var:<12} {mean_val:12.4f} {t_val:12.2f}")
        print("=" * len(header) + "\n")

        # --- Rückgabe: Ergebnisse + Diagnosedaten ---
        return mean_params, tstats, monthly_params, meta


def _inject_title_fallback(doc: str, title: str) -> str:
    # If Formatter can't take a doc_title kwarg, inject it into the LaTeX string.
    # 1) Replace existing \title{...} if present; else
    # 2) Insert \title{...}\maketitle before \begin{document}.
    import re as _re
    if r'\title{' in doc:
        return _re.sub(r'\\title\{.*?\}', f'\\title{{{title}}}', doc, count=1, flags=_re.S)
    return doc.replace(r'\begin{document}', f'\\title{{{title}}}\n\\maketitle\n\\begin{document}', 1)

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

def run_analysis(df: pd.DataFrame, signal_name: str):
    # Load once & share
    crsp_full = pd.read_csv("./data/dsws_crsp.csv")
    factors_full = pd.read_csv("./data/Factors.csv")

    equal_factor_model_analyzer = EqualWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full)
    value_factor_model_analyzer = ValueWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full)
    #fama_macbeth_analyzer = FamaMacBethAnalyzer(df, crsp=crsp_full, factors=factors_full)
    print(0)
    # ---------- Baseline (all industries) ----------
    results_equal, long_short_results_equal = equal_factor_model_analyzer.analyze()
    ff3_equal, ff5_equal, q_equal = Formatter.results_to_strings(results_equal)
    long_short_equal = Formatter.long_short_res_to_string(long_short_results_equal)
    latex_ff3_equal = Formatter.generate_latex_table(results_equal, "FF3")
    latex_ff5_equal = Formatter.generate_latex_table(results_equal, "FF5")
    latex_q_equal = Formatter.generate_latex_table(results_equal, "Q")
    latex_long_short_equal = Formatter.generate_long_short_latex_table(long_short_results_equal)
    print(1)
    results_value, long_short_results_value = value_factor_model_analyzer.analyze()
    ff3_value, ff5_value, q_value = Formatter.results_to_strings(results_value)
    long_short_value = Formatter.long_short_res_to_string(long_short_results_value)
    latex_ff3_value = Formatter.generate_latex_table(results_value, "FF3")
    latex_ff5_value = Formatter.generate_latex_table(results_value, "FF5")
    latex_q_value = Formatter.generate_latex_table(results_value, "Q")
    latex_long_short_value = Formatter.generate_long_short_latex_table(long_short_results_value)
    print(2)
    """ fmb_all = fama_macbeth_analyzer.analyze()
    fmb_text_parts = []
    for mdl in ["FF3", "FF5", "Q"]:
        fmb_text_parts.append(
            Formatter.fama_macbeth_res_to_string(
                mdl,
                fmb_all[mdl]["means"],
                fmb_all[mdl]["tstats"],
                fmb_all[mdl]["n_months"]
            )
        )
    fama_macbeth_res = "\n\n".join(fmb_text_parts) """

    # LaTeX: je Modell eine Tabelle (in die vorhandene FMB-Sektion einfügen)
    """ latex_fmb_parts = []
    for mdl in ["FF3", "FF5", "Q"]:
        latex_fmb_parts.append(rf"\subsection{{Fama-MacBeth: {mdl}}}")
        latex_fmb_parts.append(
            Formatter.generate_fama_macbeth_two_pass_latex_table(
                mdl,
                fmb_all[mdl]["means"],
                fmb_all[mdl]["tstats"],
                fmb_all[mdl]["n_months"]
            )
        )
    latex_fama_macbeth = "\n".join(latex_fmb_parts) """
    print(3)
    escaped_signal = _latex_escape(signal_name)

    # Title-aware LaTeX for baseline
    baseline_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Baseline (All Industries)"
    try:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            "",
            title=baseline_title,         
        )
    except TypeError:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            ""
        )
        latex_output = _inject_title_fallback(latex_output, baseline_title)
    print(4)
    basedir = os.path.abspath(os.path.dirname(__file__))
    static_dir = os.path.join(os.path.dirname(basedir), "static")
    result_dir = os.path.join(static_dir, "downloads")
    os.makedirs(result_dir, exist_ok=True)

    safe_signal_name = re.sub(r'[^A-Za-z0-9_-]', '_', signal_name)
    zip_filename = f"{safe_signal_name}.zip"
    zip_path = os.path.join(result_dir, zip_filename)

    # Collect outputs (now under baseline/)
    results = {
        "baseline/ff3_equal.txt": ff3_equal,
        "baseline/ff5_equal.txt": ff5_equal,
        "baseline/q_equal.txt": q_equal,
        "baseline/long_short_equal.txt": long_short_equal,
        "baseline/ff3_value.txt": ff3_value,
        "baseline/ff5_value.txt": ff5_value,
        "baseline/q_value.txt": q_value,
        "baseline/long_short_value.txt": long_short_value,
        #"baseline/fama_macbeth.txt": fama_macbeth_res,
        "baseline/output.tex": latex_output,
    }
    print(5)
    """     # ---------- Per-industry runs ----------
    industry_codes = sorted(pd.unique(crsp_full['ff12'].dropna().astype(int)))
    for code in industry_codes:
        code_int = int(code)
        print(code)
        res_eq_ind, ls_eq_ind = equal_factor_model_analyzer.analyze(industry_code=code_int)
        ff3_eq_i, ff5_eq_i, q_eq_i = Formatter.results_to_strings(res_eq_ind)
        ls_eq_i = Formatter.long_short_res_to_string(ls_eq_ind)

        res_vw_ind, ls_vw_ind = value_factor_model_analyzer.analyze(industry_code=code_int)
        ff3_vw_i, ff5_vw_i, q_vw_i = Formatter.results_to_strings(res_vw_ind)
        ls_vw_i = Formatter.long_short_res_to_string(ls_vw_ind)

        fmb_i = fama_macbeth_analyzer.analyze(industry_code=code_int)
        fm_i_text_parts, fm_i_latex_parts = [], []
        for mdl in ["FF3", "FF5", "Q"]:
            fm_i_text_parts.append(
                Formatter.fama_macbeth_res_to_string(
                    mdl, fmb_i[mdl]["means"], fmb_i[mdl]["tstats"], fmb_i[mdl]["n_months"]
                )
            )
            fm_i_latex_parts.append(rf"\subsection{{Fama-MacBeth: {mdl}}}")
            fm_i_latex_parts.append(
                Formatter.generate_fama_macbeth_two_pass_latex_table(
                    mdl, fmb_i[mdl]["means"], fmb_i[mdl]["tstats"], fmb_i[mdl]["n_months"]
                )
            )

        fm_i = "\n\n".join(fm_i_text_parts)

        # Title-aware LaTeX per industry
        industry_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Industry {code_int} (FF12)"
        try:
            latex_output_i = Formatter.create_complete_latex_document(
                Formatter.generate_latex_table(res_eq_ind, "FF3"),
                Formatter.generate_latex_table(res_eq_ind, "FF5"),
                Formatter.generate_latex_table(res_eq_ind, "Q"),
                Formatter.generate_long_short_latex_table(ls_eq_ind),

                Formatter.generate_latex_table(res_vw_ind, "FF3"),
                Formatter.generate_latex_table(res_vw_ind, "FF5"),
                Formatter.generate_latex_table(res_vw_ind, "Q"),
                Formatter.generate_long_short_latex_table(ls_vw_ind),

                "\n".join(fm_i_latex_parts),
                title=industry_title
            )
        except TypeError:
            latex_output_i = Formatter.create_complete_latex_document(
                Formatter.generate_latex_table(res_eq_ind, "FF3"),
                Formatter.generate_latex_table(res_eq_ind, "FF5"),
                Formatter.generate_latex_table(res_eq_ind, "Q"),
                Formatter.generate_long_short_latex_table(ls_eq_ind),

                Formatter.generate_latex_table(res_vw_ind, "FF3"),
                Formatter.generate_latex_table(res_vw_ind, "FF5"),
                Formatter.generate_latex_table(res_vw_ind, "Q"),
                Formatter.generate_long_short_latex_table(ls_vw_ind),

                "\n".join(fm_i_latex_parts)
            )
            latex_output_i = _inject_title_fallback(latex_output_i, industry_title)

        prefix = f"industries/industry_ff12_{code_int}"
        results.update({
            f"{prefix}/ff3_equal.txt": ff3_eq_i,
            f"{prefix}/ff5_equal.txt": ff5_eq_i,
            f"{prefix}/q_equal.txt":   q_eq_i,
            f"{prefix}/long_short_equal.txt": ls_eq_i,
            f"{prefix}/ff3_value.txt": ff3_vw_i,
            f"{prefix}/ff5_value.txt": ff5_vw_i,
            f"{prefix}/q_value.txt":   q_vw_i,
            f"{prefix}/long_short_value.txt": ls_vw_i,
            f"{prefix}/fama_macbeth.txt": fm_i,
            f"{prefix}/output.tex": latex_output_i,
        })
    print(7)

    # ---------- Per-country runs ----------
    countries = sorted(pd.unique(crsp_full['country'].dropna().astype(str)))
    for ctry in countries:
        print(f"country={ctry}")
        res_eq_cty, ls_eq_cty = equal_factor_model_analyzer.analyze(country=ctry)
        ff3_eq_c, ff5_eq_c, q_eq_c = Formatter.results_to_strings(res_eq_cty)
        ls_eq_c = Formatter.long_short_res_to_string(ls_eq_cty)

        res_vw_cty, ls_vw_cty = value_factor_model_analyzer.analyze(country=ctry)
        ff3_vw_c, ff5_vw_c, q_vw_c = Formatter.results_to_strings(res_vw_cty)
        ls_vw_c = Formatter.long_short_res_to_string(ls_vw_cty)

        fmb_c = fama_macbeth_analyzer.analyze(country=ctry)
        fm_c_text_parts, fm_c_latex_parts = [], []
        for mdl in ["FF3", "FF5", "Q"]:
            fm_c_text_parts.append(
                Formatter.fama_macbeth_res_to_string(
                    mdl, fmb_c[mdl]["means"], fmb_c[mdl]["tstats"], fmb_c[mdl]["n_months"]
                )
            )
            fm_c_latex_parts.append(rf"\subsection{{Fama-MacBeth: {mdl}}}")
            fm_c_latex_parts.append(
                Formatter.generate_fama_macbeth_two_pass_latex_table(
                    mdl, fmb_c[mdl]["means"], fmb_c[mdl]["tstats"], fmb_c[mdl]["n_months"]
                )
            )

        fm_c = "\n\n".join(fm_c_text_parts)

        # Sauberer Ländername für Pfade/Titel
        safe_country = re.sub(r'[^A-Za-z0-9_-]+', '_', str(ctry))
        country_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Country {ctry}"

        try:
            latex_output_c = Formatter.create_complete_latex_document(
                Formatter.generate_latex_table(res_eq_cty, "FF3"),
                Formatter.generate_latex_table(res_eq_cty, "FF5"),
                Formatter.generate_latex_table(res_eq_cty, "Q"),
                Formatter.generate_long_short_latex_table(ls_eq_cty),

                Formatter.generate_latex_table(res_vw_cty, "FF3"),
                Formatter.generate_latex_table(res_vw_cty, "FF5"),
                Formatter.generate_latex_table(res_vw_cty, "Q"),
                Formatter.generate_long_short_latex_table(ls_vw_cty),

                "\n".join(fm_c_latex_parts),
                title=country_title
            )
        except TypeError:
            latex_output_c = Formatter.create_complete_latex_document(
                Formatter.generate_latex_table(res_eq_cty, "FF3"),
                Formatter.generate_latex_table(res_eq_cty, "FF5"),
                Formatter.generate_latex_table(res_eq_cty, "Q"),
                Formatter.generate_long_short_latex_table(ls_eq_cty),

                Formatter.generate_latex_table(res_vw_cty, "FF3"),
                Formatter.generate_latex_table(res_vw_cty, "FF5"),
                Formatter.generate_latex_table(res_vw_cty, "Q"),
                Formatter.generate_long_short_latex_table(ls_vw_cty),

                "\n".join(fm_c_latex_parts)
            )
            latex_output_c = _inject_title_fallback(latex_output_c, country_title)

        prefix = f"countries/country_{safe_country}"
        results.update({
            f"{prefix}/ff3_equal.txt": ff3_eq_c,
            f"{prefix}/ff5_equal.txt": ff5_eq_c,
            f"{prefix}/q_equal.txt":   q_eq_c,
            f"{prefix}/long_short_equal.txt": ls_eq_c,
            f"{prefix}/ff3_value.txt": ff3_vw_c,
            f"{prefix}/ff5_value.txt": ff5_vw_c,
            f"{prefix}/q_value.txt":   q_vw_c,
            f"{prefix}/long_short_value.txt": ls_vw_c,
            f"{prefix}/fama_macbeth.txt": fm_c,
            f"{prefix}/output.tex": latex_output_c,
        }) """

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
