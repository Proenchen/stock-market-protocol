from __future__ import annotations 

import re 
import os 
import numpy as np 
import pandas as pd 
import zipfile 
import statsmodels.api as sm 

from abc import ABC, abstractmethod 
from typing import Tuple, Any 
from logic.formatter import Formatter


NUM_OF_SLICES = 10


class BaseAnalyzer(ABC):
    """Abstract analyzer for stock return predictors."""

    def __init__(self, input_file: pd.DataFrame, crsp: pd.DataFrame | None = None, factors: pd.DataFrame | None = None) -> None:
        self.data = input_file
        self.crsp = crsp if crsp is not None else pd.read_csv("./data/dsws_crsp.csv")
        self.factors = factors if factors is not None else pd.read_csv("./data/Factors.csv")

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
    def _fast_rolling_betas_per_slice(self, df_slice: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
        """
        Schnelles Rolling-OLS für Betas je Slice mit O(1)-Fenster-Updates.
        Beta(t) verwendet Daten aus [t-window, ..., t-1] (kein Look-ahead).
        """
        # Nur vollständige Beobachtungen
        df_slice = df_slice.sort_values("year_month").dropna(subset=["excess_ret"] + cols).reset_index(drop=True)
        if df_slice.empty:
            return pd.DataFrame(columns=["year_month"] + cols)

        X = df_slice[cols].to_numpy(dtype=float)        # (T, K)
        y = df_slice["excess_ret"].to_numpy(dtype=float) # (T,)
        T, K = X.shape
        if T < window or K == 0:
            out = pd.DataFrame(columns=cols)
            out["year_month"] = df_slice["year_month"].to_numpy()
            return out

        # Kumulative Summen
        xnz = np.nan_to_num(X)
        ynz = np.nan_to_num(y)
        cs_xy = np.cumsum(xnz * ynz[:, None], axis=0)  # (T, K)

        # Für X'X alle Paare (i,j)
        XX = np.empty((T, K, K), dtype=float)
        for i in range(K):
            for j in range(i, K):
                s = np.cumsum(xnz[:, i] * xnz[:, j])
                XX[:, i, j] = s
                if i != j:
                    XX[:, j, i] = s

        betas = np.full((T, K), np.nan, dtype=float)

        # Beta für t nutzt Beobachtungen t-window .. t-1
        for t in range(window, T):
            lo, hi = t - window, t - 1

            # Summen via cumsum-Differenz
            s_xy = cs_xy[hi] - (cs_xy[lo - 1] if lo > 0 else 0.0)
            s_xx = XX[hi]   - (XX[lo - 1]   if lo > 0 else 0.0)

            # Numerische Stabilität
            try:
                b = np.linalg.solve(s_xx, s_xy)
            except np.linalg.LinAlgError:
                b = np.linalg.lstsq(s_xx + 1e-8 * np.eye(K), s_xy, rcond=None)[0]
            betas[t] = b

        out = pd.DataFrame(betas, columns=cols)
        out["year_month"] = df_slice["year_month"].to_numpy()
        return out

    def analyze(self, industry_code: int | None = None, country: str | None = None, window: int = 60) -> dict:
        # ---------- Equal-Weighted Slices & Returns (wie bei dir) ----------
        df_signal = self.prepare_signal_df(self.data)
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        df_signal['slice'] = (
            df_signal.groupby('year_month')['signal']
                     .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )
        df_signal['ret_month'] = df_signal['year_month'] + 1  # t+1

        df_crsp = self.crsp
        if industry_code is not None and 'ff12' in df_crsp.columns:
            df_crsp = df_crsp.loc[df_crsp['ff12'] == industry_code]
        if country is not None and 'country' in df_crsp.columns:
            df_crsp = df_crsp.loc[df_crsp['country'] == country]

        df_crsp = df_crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'}).copy()
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')
        df_crsp['RET_USD'] = pd.to_numeric(df_crsp['RET_USD'], errors='coerce') / 100.0

        # Map Slices t -> Returns t+1
        ret_t1 = df_crsp[['permno', 'year_month', 'RET_USD']].rename(columns={'year_month': 'ret_month'})
        merged = (df_signal[['permno', 'year_month', 'ret_month', 'slice']]
                  .merge(ret_t1, on=['permno', 'ret_month'], how='inner'))

        # Equal-weighted Portfolio-Returns je Monat x Slice
        port_returns = (merged.groupby(['ret_month', 'slice'])['RET_USD']
                        .mean()
                        .reset_index()
                        .rename(columns={'ret_month': 'year_month', 'RET_USD': 'port_ret'}))
        port_returns['year_month'] = port_returns['year_month'].astype('period[M]')

        # Faktoren + Excess
        df_factors = self._prepare_factors()
        model_data = pd.merge(port_returns, df_factors, on='year_month', how='inner')
        model_data['excess_ret'] = model_data['port_ret'] - model_data['RF']

        # ---------- First Pass: Rolling-Betas bis t-1 (schnell) ----------
        specs = {
            "FF3": ['MKT', 'SMB', 'HML'],
            "FF5": ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],
            "Q":   ['MKT', 'SIZE', 'IA', 'ROE'],
        }

        results = {}
        months = sorted(model_data['year_month'].dropna().unique())

        for mdl_name, cols in specs.items():
            # Betas je Slice via Rolling (kein Look-ahead)
            betas_all = []
            for q in sorted(model_data['slice'].unique()):
                sub = model_data.loc[model_data['slice'] == q, ['year_month', 'excess_ret'] + cols].copy()
                betas_q = self._fast_rolling_betas_per_slice(sub, cols, window=window)
                betas_q['slice'] = q
                betas_all.append(betas_q)

            betas_df = pd.concat(betas_all, ignore_index=True) if betas_all else pd.DataFrame()

            # ---------- Second Pass: Cross-Section je Monat ----------
            monthly_coefs = []
            for t in months:
                # Excess-Returns der Slices in Monat t
                sub_t = model_data.loc[model_data['year_month'] == t, ['slice', 'excess_ret']].dropna()
                if sub_t.empty:
                    continue

                # Betas(t) (die intern aus [t-window, t-1] geschätzt sind) mergen
                bt = betas_df.loc[betas_df['year_month'] == t, ['slice'] + cols].dropna()
                if bt.empty:
                    continue
                sub_t = sub_t.merge(bt, on='slice', how='inner')
                if sub_t.shape[0] < len(cols) + 1:  # mind. K+1 Beobachtungen
                    continue

                x_cs = sm.add_constant(sub_t[cols], has_constant='add')
                y_cs = sub_t['excess_ret']
                try:
                    res_cs = sm.OLS(y_cs, x_cs).fit()
                except Exception:
                    continue

                monthly_coefs.append(res_cs.params)

            if len(monthly_coefs) == 0:
                idx = ['const'] + cols
                results[mdl_name] = {
                    "means": pd.Series(index=idx, dtype=float),
                    "tstats": pd.Series(index=idx, dtype=float),
                    "n_months": 0
                }
                continue

            coef_df = pd.DataFrame(monthly_coefs)  # Zeilen sind Monate
            tm = coef_df.shape[0]
            means = coef_df.mean(axis=0)
            stds  = coef_df.std(axis=0, ddof=1)
            tstats = means / (stds / np.sqrt(tm))

            results[mdl_name] = {
                "means": means,
                "tstats": tstats,
                "n_months": int(tm)
            }

        return results



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
