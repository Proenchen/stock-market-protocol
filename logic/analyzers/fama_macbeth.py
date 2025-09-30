from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter


class FamaMacBethAnalyzer(BaseAnalyzer, AutoRegistered):
    """Run Fama–MacBeth cross-sectional regressions and summarize results.

    Specification (monthly, cross-sectional)::

        r_{i,t} = alpha_t
                  + b1_t * signal_{i,t}
                  + b2_t * size_lag_{i,t}
                  + b3_t * mom_2_12_{i,t}
                  + b4_t * bm_{i,t}
                  + b5_t * ag_{i,t}
                  + b6_t * rd_sale_{i,t} + e_{i,t}

    Scaling
    -------
    All regressors are scaled to the interval [-1, 1] via rank transformation. 
    This is robust to outliers and unit choices.
    """

    ENABLED = True
    ORDER = 30
    TITLE = "Fama–MacBeth Regression Result"

    def __init__(self, ctx, df_input: pd.DataFrame, signal_name: str) -> None:
        """Initialize analyzer with shared context, data, and a signal label.

        Parameters
        ----------
        ctx
            Shared context with CRSP-like returns and firm-month variables.
        df_input : pd.DataFrame
            Input with three leading columns interpreted as ``[DSCD, date, signal]``.
        signal_name : str
            Human-readable name of the signal for display in outputs.
        """
        super().__init__(ctx, df_input, signal_name)

    def analyze(self):
        """Execute monthly cross-sectional OLS and compute FM means and t-stats.

        Returns
        -------
        tuple[pd.Series, dict[str, float], int]
            ``(mean_params, tstats, n_months)`` where
            - ``mean_params`` are time-series means of the monthly coefficients,
            - ``tstats`` are Fama–MacBeth t-statistics for each coefficient name,
            - ``n_months`` is the number of months with valid CS regressions.
        """
        # --- Prepare merged and scaled panel ---
        df = self._prep()

        # --- Step 1: Monthly cross-sectional regressions ---
        monthly_params = df.groupby("month").apply(self._cs_ols, include_groups=False)

        # --- Step 2: Time-series means and FM t-stats ---
        mean_params = monthly_params.mean(skipna=True)
        tstats = {k: self._fm_tstat(monthly_params[k]) for k in monthly_params.columns}

        # --- Step 3: Meta info ---
        valid_idx = monthly_params.dropna(how="all").index
        n_months = int(len(valid_idx))

        return mean_params, tstats, n_months

    def generate_output(self) -> AnalyzerOutput:
        """Build :class:`AnalyzerOutput` with plain text and LaTeX table."""
        means, tstats, n_months = self.analyze()
        txt = Formatter.fama_macbeth_res_to_string(means, tstats, n_months, self.signal_name)
        latex = Formatter.generate_fama_macbeth_latex_table(means, tstats, self.signal_name)
        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts={"fama_macbeth.txt": txt},
            latex_blocks=[latex],
        )

    # Helper Methods
    # ------------------------------------------------------------------

    def _prep(self) -> pd.DataFrame:
        """Prepare merged firm-month panel and scale regressors to [-1, 1].

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['DSCD','month','ret','signal','size_lag',
            'mom_2_12','bm','ag','rd_sale']`` where regressors are monthly
            rank-scaled to [-1, 1].
        """
        sig = self.data.copy()
        sig = sig.rename(columns={sig.columns[0]: "DSCD", sig.columns[1]: "dates", sig.columns[2]: "signal"})
        sig["month"] = pd.to_datetime(sig["dates"]).dt.to_period("M").dt.to_timestamp("M")

        fm = self.fm.copy()
        fm["DATE"] = pd.to_datetime(fm["DATE"])
        fm["month"] = fm["DATE"].dt.to_period("M").dt.to_timestamp("M")

        crsp = self.crsp.copy()
        crsp["DATE"] = pd.to_datetime(crsp["DATE"])
        crsp["month"] = crsp["DATE"].dt.to_period("M").dt.to_timestamp("M")

        # Map ff12 classification into fm at the (DSCD, month) level if needed
        ff12_map = (
            crsp[["DSCD", "month", "ff12"]]
            .sort_values(["DSCD", "month"]).groupby(["DSCD", "month"], as_index=False).first()
        )
        fm = fm.merge(ff12_map, on=["DSCD", "month"], how="left")

        keep = ["DSCD", "month", "RET_USD", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]
        fm_small = fm[keep].copy()

        df = pd.merge(sig, fm_small, on=["DSCD", "month"], how="inner")

        # Ensure numeric types
        df["ret"] = pd.to_numeric(df["RET_USD"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df["size_lag"] = pd.to_numeric(df["size_lag"], errors="coerce")
        df["mom_2_12"] = pd.to_numeric(df["mom_2_12"], errors="coerce")
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
        df["ag"] = pd.to_numeric(df["ag"], errors="coerce")
        df["rd_sale"] = pd.to_numeric(df["rd_sale"], errors="coerce")

        # Scale regressors to [-1, 1] by within-month ranks
        regressors = ["signal", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]

        def scale_monthly(s: pd.Series) -> pd.Series:
            """Rank-scale a series to [-1, 1] within a month."""
            r = s.rank(method="average")
            n = r.size
            if n <= 1:
                return pd.Series(np.nan, index=s.index)
            u = (r - 1) / (n - 1)
            return 2.0 * u - 1.0

        for col in regressors:
            df[col] = df.groupby("month")[col].transform(scale_monthly)

        return df[["DSCD", "month", "ret"] + regressors]

    def _cs_ols(self, g: pd.DataFrame) -> pd.Series:
        """Run a single-month cross-sectional OLS and return coefficients.

        Returns a fixed set of coefficient names, filling missing values with
        NaNs when data are insufficient or the fit fails.
        """
        out_cols = ["Intercept", "Signal", "Size", "Momentum", "BM", "AG", "RD/Sales"]
        out_na = pd.Series({c: np.nan for c in out_cols})

        # Build y and X as numeric
        y = pd.to_numeric(g.get("ret"), errors="coerce")
        X = g[["signal", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]].apply(pd.to_numeric, errors="coerce")
        X = sm.add_constant(X, has_constant="add")

        # Complete cases only
        mask = y.notna() & X.notna().all(axis=1)
        y, X = y[mask], X[mask]

        # Need more observations than parameters (including constant)
        if y.shape[0] == 0 or X.shape[0] <= X.shape[1]:
            return out_na

        try:
            res = sm.OLS(y, X).fit()
            p = res.params
            return pd.Series({
                "Intercept": p.get("const", np.nan),
                "Signal": p.get("signal", np.nan),
                "Size": p.get("size_lag", np.nan),
                "Momentum": p.get("mom_2_12", np.nan),
                "BM": p.get("bm", np.nan),
                "AG": p.get("ag", np.nan),
                "RD/Sales": p.get("rd_sale", np.nan),
            })
        except Exception:
            return out_na

    def _fm_tstat(self, series: pd.Series) -> float:
        """Compute a Fama–MacBeth t-statistic for a coefficient time series."""
        s = series.dropna()
        T = len(s)
        if T == 0:
            return float("nan")
        return float(s.mean() / (s.std(ddof=1) / np.sqrt(T)))
