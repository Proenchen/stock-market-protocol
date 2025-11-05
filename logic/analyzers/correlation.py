from __future__ import annotations
import numpy as np
import pandas as pd

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter


class CorrelationAnalyzer(BaseAnalyzer, AutoRegistered):
    ENABLED = True
    ORDER = 50
    TITLE = "Correlation Analysis"

    FACTOR_COLS = [
        "stock_vola_60m",
        "idiomom_2_12",
        "mom_2_12",
        "ltr_5y",
        "op_ff",
        "acc_slo",
        "ia",
        "rdca",
        "invest",
        "qmj",
        "bm",
        "eps_disp",
    ]

    def __init__(self, ctx, df_input: pd.DataFrame, signal_name: str) -> None:
        """
        Parameters
        ----------
        ctx
            Shared analysis context that includes CRSP/factor/FM data and
            correlation data (self.corr).
        df_input : pd.DataFrame
            Expected to have at least three columns in the order:
            [DSCD/permno, dates, signal].
        signal_name : str
            Display name of the signal.
        """
        super().__init__(ctx, df_input, signal_name)

    def analyze(self) -> pd.DataFrame:
        """
        For each DSCD, compute the Pearson correlation between `signal` and
        each factor (over all months), then take the unweighted mean of
        correlations across DSCDs per factor.

        Returns
        -------
        pd.DataFrame with columns: ['factor', 'corr'],
        sorted by absolute correlation (|corr|).
        """
        df = self._prep_merged()
        out = []

        for f in self.FACTOR_COLS:
            if f not in df.columns:
                out.append((f, np.nan))
                continue

            rhos = []
            for _, g in df[['DSCD', 'signal', f]].dropna().groupby('DSCD', sort=False):
                if len(g) >= 2:
                    sig_std = g['signal'].std()
                    fac_std = g[f].std()
                    if sig_std > 0 and fac_std > 0:
                        r = float(g['signal'].corr(g[f]))
                        if not np.isnan(r):
                            rhos.append(r)

            mean_rho = float(np.mean(rhos)) if len(rhos) > 0 else np.nan
            out.append((f, mean_rho))

        res = pd.DataFrame(out, columns=['factor', 'corr'])
        res = res.iloc[res['corr'].abs().sort_values(ascending=False).index].reset_index(drop=True)
        return res

    def generate_output(self) -> AnalyzerOutput:
        """
        Generates both text and LaTeX representations of the correlation
        analysis results.
        """
        res = self.analyze()

        # Plain text / CSV-style summary
        lines = [f"Signal: {self.signal_name}", "Factor,Correlation"]
        for _, row in res.iterrows():
            corr_str = "nan" if pd.isna(row["corr"]) else f"{row['corr']:.3f}"
            lines.append(f"{row['factor']},{corr_str}")
        txt = "\n".join(lines)

        # LaTeX table (2 columns)
        latex_rows = []
        for _, row in res.iterrows():
            factor_escaped = Formatter._latex_escape(str(row["factor"]))
            corr_str = "---" if pd.isna(row["corr"]) else f"{row['corr']:.3f}"
            latex_rows.append(f"{factor_escaped} & {corr_str} \\\\")

        caption = f"Correlation between {Formatter._latex_escape(self.signal_name)} and factor proxies"
        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "Factor & Correlation \\\\\n"
            "\\midrule\n"
            + "\n".join(latex_rows) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            f"\\caption{{{caption}}}\n"
            "\\end{table}\n"
        )

        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts={"correlation_summary.txt": txt},
            latex_blocks=[latex],
            meta={"signal": self.signal_name, "results": res.to_dict(orient="records")},
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _prep_merged(self) -> pd.DataFrame:
        """
        Normalize date information to month-end and merge the input signal
        with `self.corr` on (DSCD, month).
        """
        sig = self.data.copy()
        sig = sig.rename(
            columns={
                sig.columns[0]: "DSCD",
                sig.columns[1]: "dates",
                sig.columns[2]: "signal",
            }
        )

        sig["dates"] = pd.to_datetime(sig["dates"], errors="coerce")
        sig["month"] = sig["dates"].dt.to_period("M").dt.to_timestamp("M")

        corr = self.corr.copy()
        keep_cols = ["DSCD", "DATE"] + [c for c in self.FACTOR_COLS if c in corr.columns]
        corr = corr[keep_cols].copy()
        corr["DATE"] = pd.to_datetime(corr["DATE"], errors="coerce")
        corr["month"] = corr["DATE"].dt.to_period("M").dt.to_timestamp("M")

        merged = pd.merge(
            sig[["DSCD", "month", "signal"]],
            corr.drop(columns=["DATE"]),
            on=["DSCD", "month"],
            how="inner",
        )

        merged["signal"] = pd.to_numeric(merged["signal"], errors="coerce")
        for f in self.FACTOR_COLS:
            if f in merged.columns:
                merged[f] = pd.to_numeric(merged[f], errors="coerce")

        return merged
