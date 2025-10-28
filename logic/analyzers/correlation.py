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
            Shared context with CRSP/factors/FM plus correlation_data (self.corr).
        df_input : pd.DataFrame
            Erwartet mind. drei Spalten in der Reihenfolge [DSCD/permno, dates, signal].
        signal_name : str
            Anzeigename des Signals.
        """
        super().__init__(ctx, df_input, signal_name)

    def analyze(self) -> pd.DataFrame:
        """
        Merged Panel auf (DSCD, Monat) erstellen und Pearson-Korrelationen
        zwischen `signal` und jedem Faktor in FACTOR_COLS berechnen.

        Returns
        -------
        pd.DataFrame mit Spalten: ['factor', 'corr', 'n_obs'] sortiert nach |corr|.
        """
        df = self._prep_merged()

        out = []
        for f in self.FACTOR_COLS:
            if f not in df.columns:
                out.append((f, np.nan, 0))
                continue

            sub = df[["signal", f]].dropna()
            n = int(len(sub))
            if n == 0:
                rho = np.nan
            else:
                rho = float(sub["signal"].corr(sub[f]))
            out.append((f, rho, n))

        res = pd.DataFrame(out, columns=["factor", "corr", "n_obs"])
        res = res.iloc[res["corr"].abs().sort_values(ascending=False).index].reset_index(drop=True)
        return res

    def generate_output(self) -> AnalyzerOutput:
        res = self.analyze()

        lines = [f"Signal: {self.signal_name}", "Factor,Correlation,N"]
        for _, row in res.iterrows():
            corr_str = "nan" if pd.isna(row["corr"]) else f"{row['corr']:.3f}"
            lines.append(f"{row['factor']},{corr_str},{int(row['n_obs'])}")
        txt = "\n".join(lines)

        latex_rows = []
        for _, row in res.iterrows():
            factor_escaped = Formatter._latex_escape(str(row["factor"]))
            corr_str = "---" if pd.isna(row["corr"]) else f"{row['corr']:.3f}"
            latex_rows.append(f"{factor_escaped} & {corr_str} & {int(row['n_obs'])} \\\\")

        caption = f"Correlation between {Formatter._latex_escape(self.signal_name)} and factor proxies"
        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            "\\begin{tabular}{lrr}\n"
            "\\toprule\n"
            "Factor & Correlation & N \\\\\n"
            "\\midrule\n"
            + "\n".join(latex_rows) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
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
        Normalisiert Datumsangaben auf Monatsende und merged Input-Signal
        mit self.corr auf (DSCD, month).
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
