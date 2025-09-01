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
    def analyze(self, industry_code: int | None = None, country: str | None = None) -> Tuple[dict, dict]:
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

        if industry_code is not None and 'ff12' in cr.columns:
            cr = cr.loc[cr['ff12'] == industry_code]

        if country is not None and 'country' in cr.columns:
            cr = cr.loc[cr['country'] == country]

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
                g["decile"] = pd.qcut(g["signal"], NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1), duplicates="drop")
            except ValueError:
                r = g["signal"].rank(method="first")
                g["decile"] = pd.qcut(r, NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1), duplicates="drop")
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
            y2, X2 = y[mask], Xc[mask]

            # zu wenig Daten? (mind. > Anzahl Parameter)
            if len(y2) == 0 or X2.shape[0] <= X2.shape[1]:
                return None
            
            return sm.OLS(y2, X2).fit()

        # --- 6) Regression ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls  = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_equal = {i: {} for i in dec_nums}
        long_short_results_equal = {}

        for label, fac_cols in model_specs.items():
            res_ls = fit_ols(ts_ls["LS_ex"], ts_ls[fac_cols])
            if res_ls is not None:
                long_short_results_equal[label] = res_ls
            for i in dec_nums:
                res_i = fit_ols(ts_dec[i], ts_dec[fac_cols])
                if res_i is not None:
                    results_equal[i][label] = res_i 

        return results_equal, long_short_results_equal


class ValueWeightedFactorModelAnalyzer(BaseAnalyzer):
    """Fama–French (3/5/Q) Regressions auf value-weighted Portfolios."""
    def analyze(self, industry_code: int | None = None, country: str | None = None) -> Tuple[dict, dict]:
        # --- Input & Monat ---
        df = self.data.copy()
        df = df.rename(columns={df.columns[0]: "DSCD",
                                df.columns[1]: "dates",
                                df.columns[2]: "signal"})
        df["date"] = pd.to_datetime(df["dates"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

        # --- CRSP joinen ---
        cr = self.crsp.copy()

        if industry_code is not None and 'ff12' in cr.columns:
            cr = cr.loc[cr['ff12'] == industry_code]

        if country is not None and 'country' in cr.columns:
            cr = cr.loc[cr['country'] == country]

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
                g["decile"] = pd.qcut(g["signal"], NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1), duplicates="drop")
            except ValueError:
                r = g["signal"].rank(method="first")
                g["decile"] = pd.qcut(r, NUM_OF_SLICES, labels=range(1, NUM_OF_SLICES + 1), duplicates="drop")
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
            y2, X2 = y[mask], Xc[mask]

            # zu wenig Daten? (mind. > Anzahl Parameter)
            if len(y2) == 0 or X2.shape[0] <= X2.shape[1]:
                return None
            
            return sm.OLS(y2, X2).fit()

        # --- 6) Regression ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls  = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_value = {i: {} for i in dec_nums}
        long_short_results_value = {}

        for label, fac_cols in model_specs.items():
            res_ls = fit_ols(ts_ls["LS_ex"], ts_ls[fac_cols])
            if res_ls is not None:
                long_short_results_value[label] = res_ls
            for i in dec_nums:
                res_i = fit_ols(ts_dec[i], ts_dec[fac_cols])
                if res_i is not None:
                    results_value[i][label] = res_i 

        return results_value, long_short_results_value


class FamaMacBethAnalyzer(BaseAnalyzer):
    """
    Fama–MacBeth (1973) Cross-Sectional Regressions

    Spezifikation (monatlich, cross-sectional):
        r_{i,t} = alpha_t
                  + b1_t * signal_{i,t}
                  + b2_t * size_lag_{i,t}
                  + b3_t * mom_2_12_{i,t}
                  + b4_t * bm_{i,t}
                  + b5_t * ag_{i,t}
                  + b6_t * rd_sale_{i,t}
                  + e_{i,t}

    Skalierung:
        Alle Regressoren (nicht die Zielvariable r_{i,t}) werden
        pro Monat auf das Intervall [-1, 1] skaliert.
    """
    def _prep(self, industry_code: int | None = None, country: str | None = None) -> pd.DataFrame:
        sig = self.data.copy()
        sig = sig.rename(columns={
            sig.columns[0]: "DSCD",
            sig.columns[1]: "dates",
            sig.columns[2]: "signal"
        })
        sig["month"] = pd.to_datetime(sig["dates"]).dt.to_period("M").dt.to_timestamp("M")

        fm = self.fm.copy()
        fm["DATE"] = pd.to_datetime(fm["DATE"])
        fm["month"] = fm["DATE"].dt.to_period("M").dt.to_timestamp("M")

        if country is not None and 'country' in fm.columns:
            fm = fm.loc[fm['country'] == country]

        crsp = self.crsp.copy()
        crsp["DATE"] = pd.to_datetime(crsp["DATE"])
        crsp["month"] = crsp["DATE"].dt.to_period("M").dt.to_timestamp("M")

        ff12_map = (
            crsp[["DSCD", "month", "ff12"]]
            .sort_values(["DSCD", "month"])
            .groupby(["DSCD", "month"], as_index=False)
            .first()
        )

        fm = fm.merge(ff12_map, on=["DSCD", "month"], how="left")

        if industry_code is not None:
            fm = fm.loc[fm["ff12"] == industry_code]

        keep = ["DSCD", "month", "RET_USD", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]
        fm_small = fm[keep].copy()

        df = pd.merge(sig, fm_small, on=["DSCD", "month"], how="inner")

        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df["size_lag"] = pd.to_numeric(df["size_lag"], errors="coerce")
        df["mom_2_12"] = pd.to_numeric(df["mom_2_12"], errors="coerce")
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
        df["ag"] = pd.to_numeric(df["ag"], errors="coerce")
        df["rd_sale"] = pd.to_numeric(df["rd_sale"], errors="coerce")

        # Skalierung aller Regressoren auf [-1, 1] pro Monat
        regressors = ["signal", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]
        def scale_monthly(s: pd.Series):
            m = s.mean()
            sd = s.std(ddof=0)  
            if pd.notna(sd) and sd != 0:
                return (s - m) / sd
            return pd.Series(np.nan, index=s.index)
        
        for col in regressors:
            df[col] = df.groupby("month")[col].transform(scale_monthly)

        return df[["DSCD", "month", "ret"] + regressors]

    @staticmethod
    def _cs_ols(g: pd.DataFrame) -> pd.Series:
        # Zielspalten-Layout (immer gleich zurückgeben)
        out_cols = ["Intercept", "Signal", "Size", "Momentum", "BM", "AG", "RD/Sales"]
        out_na = pd.Series({c: np.nan for c in out_cols})

        # y/X sauber und numerisch
        y = pd.to_numeric(g.get("ret"), errors="coerce")
        X = g[["signal", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]].apply(pd.to_numeric, errors="coerce")
        X = sm.add_constant(X, has_constant="add")

        # vollständige Fälle
        mask = y.notna() & X.notna().all(axis=1)
        y, X = y[mask], X[mask]

        # genug Beobachtungen? (mind. > Anzahl Parameter inkl. Konstante)
        if y.shape[0] == 0 or X.shape[0] <= X.shape[1]:
            return out_na

        # Fit versuchen; bei Singulärität/anderen Problemen: NaNs zurückgeben
        try:
            res = sm.OLS(y, X).fit()
            p = res.params
            return pd.Series({
                "Intercept": p.get("const",   np.nan),
                "Signal":    p.get("signal",  np.nan),
                "Size":      p.get("size_lag",np.nan),
                "Momentum":  p.get("mom_2_12",np.nan),
                "BM":        p.get("bm",      np.nan),
                "AG":        p.get("ag",      np.nan),
                "RD/Sales":  p.get("rd_sale", np.nan),
            })
        except Exception:
            return out_na

    @staticmethod
    def _fm_tstat(series: pd.Series) -> float:
        s = series.dropna()
        T = len(s)
        if T == 0:
            return np.nan
        return float(s.mean() / (s.std(ddof=1) / np.sqrt(T)))

    def analyze(self, industry_code: int | None = None, country: str | None = None):
        # --- Daten vorbereiten ---
        df = self._prep(industry_code, country)

        # --- Schritt 1: Monatsweise Querschnittsregressionen ---
        monthly_params = (
            df.groupby("month")
              .apply(self._cs_ols, include_groups=False)
        )

        # --- Schritt 2: Zeitreihen-Mittelwerte & FM-t-Statistiken ---
        mean_params = monthly_params.mean(skipna=True)
        tstats = {k: self._fm_tstat(monthly_params[k]) for k in monthly_params.columns}

        # --- Schritt 3: Meta-Infos ---
        valid_idx = monthly_params.dropna(how="all").index
        n_months = int(len(valid_idx))

        return mean_params, tstats, n_months

def run_analysis(df: pd.DataFrame, signal_name: str):
    # Load once & share
    crsp_full = pd.read_csv("./data/dsws_crsp.csv")
    factors_full = pd.read_csv("./data/Factors.csv")
    fm_full = pd.read_csv("./data/Fama_Macbeth.csv")

    equal_factor_model_analyzer = EqualWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full, fm=fm_full)
    value_factor_model_analyzer = ValueWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full, fm=fm_full)
    fama_macbeth_analyzer = FamaMacBethAnalyzer(df, crsp=crsp_full, factors=factors_full, fm=fm_full)

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
    fmb_result_means, fmb_result_t_stats, n_months = fama_macbeth_analyzer.analyze()
    fmb_res_string = Formatter.fama_macbeth_res_to_string(fmb_result_means, fmb_result_t_stats, n_months, signal_name)
    latex_fmb_res = Formatter.generate_fama_macbeth_latex_table(fmb_result_means, fmb_result_t_stats, signal_name)

    print(3)
    escaped_signal = Formatter._latex_escape(signal_name)

    # Title-aware LaTeX for baseline
    baseline_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Baseline (All Industries)"
    try:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fmb_res,
            title=baseline_title,         
        )
    except TypeError:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fmb_res
        )
        latex_output = Formatter._inject_title_fallback(latex_output, baseline_title)


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
        "baseline/fama_macbeth.txt": fmb_res_string,
        "baseline/output.tex": latex_output,
    }
    print(5)
    # ---------- Per-industry runs ----------
    industry_codes = sorted(pd.unique(crsp_full['ff12'].dropna().astype(int)))
    for code in industry_codes:
        code_int = int(code)
        print(code_int)

        # EW / VW
        res_eq_ind, ls_eq_ind = equal_factor_model_analyzer.analyze(industry_code=code_int)
        res_vw_ind, ls_vw_ind = value_factor_model_analyzer.analyze(industry_code=code_int)
        fmb_means_i, fmb_tstats_i, fmb_n_i = fama_macbeth_analyzer.analyze(industry_code=code_int)

        # --- Defensive Skip: keine Daten ---
        if (all(len(v) == 0 for v in res_eq_ind.values()) or
            all(len(v) == 0 for v in res_vw_ind.values()) or
            (fmb_means_i.isna().all() or fmb_n_i == 0)):
            print(f"skip industry {code_int} (no data)")
            continue

        # Falls Daten da → Formatter benutzen
        ff3_eq_i, ff5_eq_i, q_eq_i = Formatter.results_to_strings(res_eq_ind)
        ls_eq_i_txt = Formatter.long_short_res_to_string(ls_eq_ind)

        ff3_vw_i, ff5_vw_i, q_vw_i = Formatter.results_to_strings(res_vw_ind)
        ls_vw_i_txt = Formatter.long_short_res_to_string(ls_vw_ind)

        fm_i_text   = Formatter.fama_macbeth_res_to_string(fmb_means_i, fmb_tstats_i, fmb_n_i, signal_name)
        latex_fmb_i = Formatter.generate_fama_macbeth_latex_table(fmb_means_i, fmb_tstats_i, signal_name)

        # LaTeX Dokument pro Industry
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

                latex_fmb_i,
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

                latex_fmb_i
            )
            latex_output_i = Formatter._inject_title_fallback(latex_output_i, industry_title)

        prefix = f"industries/industry_ff12_{code_int}"
        results.update({
            f"{prefix}/ff3_equal.txt": ff3_eq_i,
            f"{prefix}/ff5_equal.txt": ff5_eq_i,
            f"{prefix}/q_equal.txt":   q_eq_i,
            f"{prefix}/long_short_equal.txt": ls_eq_i_txt,
            f"{prefix}/ff3_value.txt": ff3_vw_i,
            f"{prefix}/ff5_value.txt": ff5_vw_i,
            f"{prefix}/q_value.txt":   q_vw_i,
            f"{prefix}/long_short_value.txt": ls_vw_i_txt,
            f"{prefix}/fama_macbeth.txt": fm_i_text,
            f"{prefix}/output.tex": latex_output_i,
        })

    # ---------- Per-country runs ----------
    countries = sorted(pd.unique(crsp_full['country'].dropna().astype(str)))
    for ctry in countries:
        print(ctry)
        try:
            res_eq_cty, ls_eq_cty = equal_factor_model_analyzer.analyze(country=ctry)
            res_vw_cty, ls_vw_cty = value_factor_model_analyzer.analyze(country=ctry)
            fmb_means_c, fmb_tstats_c, fmb_n_c = fama_macbeth_analyzer.analyze(country=ctry)

            if (all(len(v) == 0 for v in res_eq_cty.values()) or
                all(len(v) == 0 for v in res_vw_cty.values()) or
                (fmb_means_c.isna().all() or fmb_n_c == 0)):
                print(f"skip country {ctry} (no data)")
                continue

            ff3_eq_c, ff5_eq_c, q_eq_c = Formatter.results_to_strings(res_eq_cty)
            ls_eq_c_txt = Formatter.long_short_res_to_string(ls_eq_cty)

            ff3_vw_c, ff5_vw_c, q_vw_c = Formatter.results_to_strings(res_vw_cty)
            ls_vw_c_txt = Formatter.long_short_res_to_string(ls_vw_cty)

            fm_c_text   = Formatter.fama_macbeth_res_to_string(fmb_means_c, fmb_tstats_c, fmb_n_c, signal_name)
            latex_fmb_c = Formatter.generate_fama_macbeth_latex_table(fmb_means_c, fmb_tstats_c, signal_name)
        
        except ValueError:
            continue

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

                latex_fmb_c,
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

                latex_fmb_c
            )
            latex_output_c = Formatter._inject_title_fallback(latex_output_c, country_title)

        prefix = f"countries/country_{safe_country}"
        results.update({
            f"{prefix}/ff3_equal.txt": ff3_eq_c,
            f"{prefix}/ff5_equal.txt": ff5_eq_c,
            f"{prefix}/q_equal.txt":   q_eq_c,
            f"{prefix}/long_short_equal.txt": ls_eq_c_txt,
            f"{prefix}/ff3_value.txt": ff3_vw_c,
            f"{prefix}/ff5_value.txt": ff5_vw_c,
            f"{prefix}/q_value.txt":   q_vw_c,
            f"{prefix}/long_short_value.txt": ls_vw_c_txt,
            f"{prefix}/fama_macbeth.txt": fm_c_text,
            f"{prefix}/output.tex": latex_output_c,
        })

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
