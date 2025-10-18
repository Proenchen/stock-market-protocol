from __future__ import annotations
from typing import Tuple, Dict

import pandas as pd
import statsmodels.api as sm

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter

NUM_OF_SLICES = 10


class EqualWeightedFactorModelAnalyzer(BaseAnalyzer, AutoRegistered):
    """Analyze factor exposures using equal-weighted signal deciles.

    This analyzer:
      1. Aligns the user-provided signal with CRSP-like returns at month-end.
      2. Forms equal-weighted decile portfolios by month based on the signal.
      3. Computes decile excess returns by subtracting the risk-free rate.
      4. Runs time-series regressions against FF3, FF5, and Q-factor models.
      5. Returns a standardized :class:`AnalyzerOutput` with LaTeX tables and
         plain-text summaries, suitable for a downstream "Composer".
    """

    ENABLED = True
    ORDER = 10
    TITLE = "Factor Model Analysis with Equal Weighting"

    def __init__(self, ctx, df_input: pd.DataFrame, signal_name: str) -> None:
        """Initialize analyzer with shared context, data, and a label.

        Parameters
        ----------
        ctx
            Shared context containing CRSP, factor, and FM dataframes.
        df_input : pd.DataFrame
            Input dataframe with (at least) three columns interpreted as
            ``[dscd, date, signal]`` in that order.
        signal_name : str
            Human-readable name of the signal used for logging and output.
        """
        super().__init__(ctx, df_input, signal_name)

    # Fama–French (3/5/Q) regressions on equal-weighted portfolios.
    def analyze(self) -> Tuple[Dict[int, Dict[str, sm.regression.linear_model.RegressionResultsWrapper]],
                               Dict[str, sm.regression.linear_model.RegressionResultsWrapper]]:
        """Run the full EW decile construction and factor regressions.

        Returns
        -------
        Tuple[dict, dict]
            ``(results_equal, long_short_results_equal)`` where
            ``results_equal`` maps decile -> model label -> fitted results, and
            ``long_short_results_equal`` maps model label -> fitted results for
            the D10–D1 long–short portfolio. If data are insufficient for a
            given regression, that entry is omitted.
        """
        # --- Input & month alignment ---
        df = self.data.copy()
        # Rename first three columns to standard names: 1=DSCD, 2=dates, 3=signal
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

        # --- Parse returns & signal ---
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

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

        # --- 2) Equal-weighted returns ---
        def ewret(g: pd.DataFrame) -> float:
            """Compute equal-weighted mean of monthly returns for a decile group."""
            v = g["ret"].dropna()
            return float(v.mean()) if len(v) else float("nan")

        decile_rets = (
            df.groupby(["month", "decile"]).apply(ewret, include_groups=False)
              .rename("ewret").reset_index()
        )
        decile_wide = (
            decile_rets.pivot(index="month", columns="decile", values="ewret")
                        .sort_index()
        )

        # --- 3) Factors ---
        fac = self.factors.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        factor_cols = [c for c in fac.columns if c not in ["DATE"]]
        fac[factor_cols] = fac[factor_cols] * 100  # scale to percent

        model_specs = {
            "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
            "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
            "Q":   ["EIGA_usd", "ME_usd", "IA_usd", "ROE_usd"],
        }

        # --- 4) Excess returns & long–short ---
        decile_wide_ex = decile_wide.sub(fac["rf_ff"], axis=0)
        ls_ex = decile_wide_ex[NUM_OF_SLICES] - decile_wide_ex[1]

        # --- 5) Helper: OLS fit with sanity checks ---
        def fit_ols(y: pd.Series, X: pd.DataFrame):
            """Fit OLS with an intercept; return ``None`` if data are insufficient."""
            Xc = sm.add_constant(X)
            mask = y.notna() & Xc.notna().all(axis=1)
            y2, X2 = y[mask], Xc[mask]

            # Require more observations than parameters
            if len(y2) == 0 or X2.shape[0] <= X2.shape[1]:
                return None

            return sm.OLS(y2, X2).fit()

        # --- 6) Regressions ---
        dec_nums = list(range(1, NUM_OF_SLICES + 1))
        ts_dec = decile_wide_ex.join(fac, how="inner").sort_index()
        ts_ls = pd.DataFrame({"LS_ex": ls_ex}).join(fac, how="inner").sort_index()

        results_equal: Dict[int, Dict[str, sm.regression.linear_model.RegressionResultsWrapper]] = {i: {} for i in dec_nums}
        long_short_results_equal: Dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}

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
        """Produce LaTeX blocks and raw text artifacts for the report.

        Returns
        -------
        AnalyzerOutput
            Object containing section name, raw text exports, and LaTeX blocks
            (FF3, FF5, Q models and long–short analysis tables).
        """
        results_equal, long_short_results_equal = self.analyze()

        ff3_equal, ff5_equal, q_equal = Formatter.results_to_strings(results_equal)
        long_short_equal = Formatter.long_short_res_to_string(long_short_results_equal)

        latex_ff3_equal = Formatter.generate_latex_table(results_equal, "FF3")
        latex_ff5_equal = Formatter.generate_latex_table(results_equal, "FF5")
        latex_q_equal = Formatter.generate_latex_table(results_equal, "Q")
        latex_ls_equal = Formatter.generate_long_short_latex_table(long_short_results_equal)

        raw = {
            "ff3_equal.txt": ff3_equal,
            "ff5_equal.txt": ff5_equal,
            "q_equal.txt": q_equal,
            "long_short_equal.txt": long_short_equal,
        }
        blocks = [
            "\\subsection{Fama-French 3-Factor Model}\n" + latex_ff3_equal,
            "\\subsection{Fama-French 5-Factor Model}\n" + latex_ff5_equal,
            "\\subsection{Q-Factor Model}\n" + latex_q_equal,
            "\\subsection{Long-Short Analysis}\n" + latex_ls_equal,
        ]
        return AnalyzerOutput(name=self.TITLE, raw_texts=raw, latex_blocks=blocks)
