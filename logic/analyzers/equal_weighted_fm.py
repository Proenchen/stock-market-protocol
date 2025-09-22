from __future__ import annotations 

import numpy as np 
import pandas as pd 
import statsmodels.api as sm 

from typing import Tuple

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter

NUM_OF_SLICES = 10

class EqualWeightedFactorModelAnalyzer(BaseAnalyzer, AutoRegistered):

    ENABLED = True
    ORDER = 10
    TITLE = "Factor Model Analysis with Equal Weighting"

    def __init__(self, ctx, df_input, signal_name):
        super().__init__(ctx, df_input, signal_name)

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
    
    def generate_output(self) -> AnalyzerOutput:
        results_equal, long_short_results_equal = self.analyze()

        ff3_equal, ff5_equal, q_equal = Formatter.results_to_strings(results_equal)
        long_short_equal = Formatter.long_short_res_to_string(long_short_results_equal)

        latex_ff3_equal = Formatter.generate_latex_table(results_equal, "FF3")
        latex_ff5_equal = Formatter.generate_latex_table(results_equal, "FF5")
        latex_q_equal   = Formatter.generate_latex_table(results_equal, "Q")
        latex_ls_equal  = Formatter.generate_long_short_latex_table(long_short_results_equal)

        raw = {
            "ff3_equal.txt": ff3_equal,
            "ff5_equal.txt": ff5_equal,
            "q_equal.txt":   q_equal,
            "long_short_equal.txt": long_short_equal,
        }
        blocks = [
            "\\subsection{Fama-French 3-Factor Model}\n" + latex_ff3_equal,
            "\\subsection{Fama-French 5-Factor Model}\n" + latex_ff5_equal,
            "\\subsection{Q-Factor Model}\n"            + latex_q_equal,
            "\\subsection{Long-Short Analysis}\n"       + latex_ls_equal,
        ]
        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts=raw,
            latex_blocks=blocks
        )