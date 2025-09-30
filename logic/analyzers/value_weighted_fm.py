from __future__ import annotations
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter

NUM_OF_SLICES = 10


class ValueWeightedFactorModelAnalyzer(BaseAnalyzer, AutoRegistered):
    """Analyze factor exposures using value-weighted signal deciles.

    Pipeline
    --------
    1. Align the user-provided signal with CRSP-like returns at month-end.
    2. Within each month, sort into deciles by the signal.
    3. Compute value-weighted decile returns using ``size_lag`` as weights.
    4. Subtract the risk-free rate to obtain excess returns.
    5. Run time-series regressions against FF3, FF5, and Q-factor models for
       each decile and the D10–D1 long–short portfolio.
    """

    ENABLED = True
    ORDER = 20
    TITLE = "Factor Model Analysis with Value Weighting"

    def __init__(self, ctx, df_input: pd.DataFrame, signal_name: str) -> None:
        """Initialize analyzer with shared context, data, and a label.

        Parameters
        ----------
        ctx
            Shared context containing CRSP, factor, and FM dataframes.
        df_input : pd.DataFrame
            Input dataframe whose first three columns are interpreted as
            ``[dscd, date, signal]``.
        signal_name : str
            Human-readable name of the signal used for logging and output.
        """
        super().__init__(ctx, df_input, signal_name)

    # Fama–French (3/5/Q) regressions on value-weighted portfolios.
    def analyze(self) -> Tuple[
        Dict[int, Dict[str, sm.regression.linear_model.RegressionResultsWrapper]],
        Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    ]:
        """Run VW decile formation and factor regressions.

        Returns
        -------
        Tuple[dict, dict]
            ``(results_value, long_short_results_value)`` where
            ``results_value`` maps decile -> model label -> fitted results, and
            ``long_short_results_value`` maps model label -> fitted results for
            the D10–D1 long–short portfolio. Missing entries indicate
            insufficient data for a given regression.
        """
        # --- Input & month alignment ---
        df = self.data.copy()
        df = df.rename(columns={df.columns[0]: "DSCD",
                                df.columns[1]: "dates",
                                df.columns[2]: "signal"})
        df["date"] = pd.to_datetime(df["dates"])  # original trading dates
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")  # month-end

        # --- Join CRSP-like data ---
        cr = self.crsp.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")
        cr_small = cr[["DSCD", "month", "RET_USD", "size_lag"]]
        df = pd.merge(df, cr_small, on=["DSCD", "month"], how="inner")

        # --- Returns, signal, and weights ---
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df["weight"] = pd.to_numeric(df["size_lag"], errors="coerce").clip(lower=0).fillna(0.0)

        # --- 1) Deciles ---
        def assign_deciles(g: pd.DataFrame) -> pd.DataFrame:
            """Assign equal-count deciles within each month based on the signal."""
            try:
                g["decile"] = pd.qcut(
                    g["signal"], NUM_OF_SLICES,
                    labels=range(1, NUM_OF_SLICES + 1),
                    duplicates="drop",
                )
            except ValueError:
                r = g["signal"].rank(method="first")
                g["decile"] = pd.qcut(
                    r, NUM_OF_SLICES,
                    labels=range(1, NUM_OF_SLICES + 1),
                    duplicates="drop",
                )
            g["decile"] = g["decile"].astype(int)
            return g

        df = df.groupby("month").apply(assign_deciles, include_groups=False)

        # --- 2) Value-weighted returns ---
        def vwret(g: pd.DataFrame) -> float:
            """Compute value-weighted mean of monthly returns for a decile group. """
            w = pd.to_numeric(g["weight"], errors="coerce")
            r = pd.to_numeric(g["ret"], errors="coerce")
            m = w.notna() & r.notna() & (w > 0)
            if not m.any():
                return float("nan")
            return float(np.average(r[m], weights=w[m]))

        decile_rets = (
            df.groupby(["month", "decile"]).apply(vwret, include_groups=False)
              .rename("vwret").reset_index()
        )
        decile_wide = decile_rets.pivot(index="month", columns="decile", values="vwret").sort_index()

        # --- 3) Factors ---
        fac = self.factors.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        factor_cols = [c for c in fac.columns if c not in ["DATE"]]
        fac[factor_cols] = fac[factor_cols] * 100  # scale to percent

        model_specs = {
            "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
            "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
            "Q":   ["MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"],
        }

        # --- 4) Excess returns & long–short ---
        decile_wide_ex = decile_wide.sub(fac["rf_ff"], axis=0)
        ls_ex = decile_wide_ex[NUM_OF_SLICES] - decile_wide_ex[1]

        # --- 5) Helper: OLS fit with basic sanity checks ---
        def fit_ols(y: pd.Series, X: pd.DataFrame):
            """Fit OLS with an intercept; return ``None`` if data are insufficient."""
            Xc = sm.add_constant(X)
            mask = y.notna() & Xc.notna().all(axis=1)
            y2, X2 = y[mask], Xc[mask]

            # Need more observations than parameters
            if len(y2) == 0 or X2.shape[0] <= X2.shape[1]:
                return None

            return sm.OLS(y2, X2).fit()

        # --- 6) Regressions ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_value: Dict[int, Dict[str, sm.regression.linear_model.RegressionResultsWrapper]] = {i: {} for i in dec_nums}
        long_short_results_value: Dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}

        for label, fac_cols in model_specs.items():
            res_ls = fit_ols(ts_ls["LS_ex"], ts_ls[fac_cols])
            if res_ls is not None:
                long_short_results_value[label] = res_ls
            for i in dec_nums:
                res_i = fit_ols(ts_dec[i], ts_dec[fac_cols])
                if res_i is not None:
                    results_value[i][label] = res_i

        return results_value, long_short_results_value

    def generate_output(self) -> AnalyzerOutput:
        """Produce LaTeX blocks and raw text artifacts for the report.

        Returns
        -------
        AnalyzerOutput
            Object containing section name, raw text exports, and LaTeX blocks
            for FF3, FF5, Q models and the long–short analysis.
        """
        results_value, long_short_results_value = self.analyze()

        ff3_v, ff5_v, q_v = Formatter.results_to_strings(results_value)
        ls_v = Formatter.long_short_res_to_string(long_short_results_value)

        latex_ff3 = Formatter.generate_latex_table(results_value, "FF3")
        latex_ff5 = Formatter.generate_latex_table(results_value, "FF5")
        latex_q = Formatter.generate_latex_table(results_value, "Q")
        latex_ls = Formatter.generate_long_short_latex_table(long_short_results_value)

        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts={
                "ff3_value.txt": ff3_v,
                "ff5_value.txt": ff5_v,
                "q_value.txt": q_v,
                "long_short_value.txt": ls_v,
            },
            latex_blocks=[
                "\\subsection{Fama-French 3-Factor Model}\n" + latex_ff3,
                "\\subsection{Fama-French 5-Factor Model}\n" + latex_ff5,
                "\\subsection{Q-Factor Model}\n" + latex_q,
                "\\subsection{Long-Short Analysis}\n" + latex_ls,
            ],
        )
