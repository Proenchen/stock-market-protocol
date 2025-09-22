from __future__ import annotations 

import numpy as np 
import pandas as pd 
import statsmodels.api as sm 

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter


GROUP_NUM_SLICES = 5 
MAX_ROWS_SINGLE = 20

class IndustryCountryVWLSAnalyzer(BaseAnalyzer, AutoRegistered):
    """
    Hilfsfunktionen für value-weighted Long-Short Alpha-Tabellen.
    Nutzt dieselbe Faktorskalierung und Regressionsroutine wie deine Analyzer.
    """
    ENABLED = True
    ORDER = 40
    TITLE = "Factor Model Analysis by Industries and Countries"

    MODEL_SPECS = {
        "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
        "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
        "Q"  : ["MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"],
    }

    def __init__(self, ctx, df_input, signal_name):
        super().__init__(ctx, df_input, signal_name)

    def analyze(self):
        ind = self.long_short_alphas_by_group(self.data, self.crsp, self.factors, "ff12")
        ind_neu = self.long_short_alphas_neutral_aggregations(self.data, self.crsp, self.factors, "ff12")
        country = self.long_short_alphas_by_group(self.data, self.crsp, self.factors, "country")
        c_neu   = self.long_short_alphas_neutral_aggregations(self.data, self.crsp, self.factors, "country")

        return ind, ind_neu, country, c_neu

    def generate_output(self) -> AnalyzerOutput:
        ind, ind_neu, country, c_neu = self.analyze()
        cap_ind = ("Intercept estimates from times-series regressions of long-short portfolios for each industry using different factor models. "
                   "Each cell displays the monthly intercept with t-statistics in brackets. The portfolios are constructed using value weighting for each industry.")
        cap_ind_neu = "Intercept values for the aggregated industry portfolios based on the market cap and equal weighting."
        cap_cty = ("Intercept estimates from times-series regressions of long-short portfolios for each country using different factor models. "
                   "Each cell displays the monthly intercept with t-statistics in brackets. The portfolios are constructed using value weighting for each country.")
        cap_cty_neu = "Intercept values for the aggregated country portfolios based on the market cap and equal weighting."

        latex_ind = Formatter.alpha_table_to_latex(ind, "Industry", cap_ind)
        latex_ind_neu = Formatter.alpha_table_to_latex(ind_neu, "", cap_ind_neu)
        if len(country) > MAX_ROWS_SINGLE:
            latex_cty = Formatter.alpha_table_to_latex_four_quarters_two_pages(country, "Country", cap_cty)
        else:
            latex_cty = Formatter.alpha_table_to_latex(country, "Country", cap_cty)
        latex_cty_neu = Formatter.alpha_table_to_latex(c_neu, "", cap_cty_neu)

        blocks = [
            "\\subsection{Analysis by Industries}\n" + latex_ind + "\n" + latex_ind_neu,
            "\\subsection{Analysis by Countries}\n" + latex_cty + "\n" + latex_cty_neu,
        ]
        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts={}, 
            latex_blocks=blocks
        )


    # Private Helpers
    # -------------------------------------
    def _fit_alpha(self, y: pd.Series, X: pd.DataFrame):
        Xc = sm.add_constant(X)
        m = y.notna() & Xc.notna().all(axis=1)
        y2, X2 = y[m], Xc[m]
        if len(y2)==0 or X2.shape[0] <= X2.shape[1]:
            return np.nan, np.nan
        res = sm.OLS(y2, X2).fit()
        a = res.params.get("const", np.nan)
        t = res.tvalues.get("const", np.nan)
        return float(a), float(t)

    def _prepare_signal_crsp(self, signal_df: pd.DataFrame, crsp_full: pd.DataFrame) -> pd.DataFrame:
        df = signal_df.copy()
        df = df.rename(columns={df.columns[0]:"DSCD", df.columns[1]:"dates", df.columns[2]:"signal"})
        df["month"] = pd.to_datetime(df["dates"]).dt.to_period("M").dt.to_timestamp("M")

        cr = crsp_full.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")
        keep = ["DSCD","month","RET_USD","size_lag","ff12","country"]
        cr = cr[keep]
        return pd.merge(df, cr, on=["DSCD","month"], how="inner")

    def _assign_deciles_within_groups(self, df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
        def _safe_quantil_labels(s: pd.Series, q: int) -> pd.Series:
            # 1) Direkte Quantil-Schnitte ohne feste Labels → Pandas vergibt 0..k-1
            try:
                cat = pd.qcut(s, q, labels=False, duplicates="drop")
            except Exception:
                # 2) Fallback: Rank → gleichverteiltes Raster
                r = s.rank(method="first")
                if r.nunique() < 2:
                    return pd.Series([np.nan] * len(s), index=s.index)
                try:
                    cat = pd.qcut(r, min(q, int(r.nunique())), labels=False, duplicates="drop")
                except Exception:
                    return pd.Series([np.nan] * len(s), index=s.index)
            # nach 1..k mappen
            return (cat.astype("float") + 1).astype("Int64")

        def _assign(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            g["decile"] = _safe_quantil_labels(g["signal"], GROUP_NUM_SLICES)
            return g

        return df.groupby(by_cols, group_keys=False).apply(_assign)

    def _vwret(self, g: pd.DataFrame) -> float:
        w = pd.to_numeric(g["size_lag"], errors="coerce").clip(lower=0)
        r = pd.to_numeric(g["RET_USD"], errors="coerce")
        m = r.notna() & w.notna() & (w>0)
        if not m.any():
            return np.nan
        return float(np.average(r[m], weights=w[m]))

    def _build_group_decile_returns(self, df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # value-weighted Decile-Returns je (month, group, decile)
        by = ["month", group_col, "decile"]
        ret = (df.groupby(by).apply(self._vwret)
                 .rename("vwret").reset_index())
        # zu wide per Gruppe
        wide = ret.pivot_table(index=["month",group_col], columns="decile", values="vwret")
        # Group Market Cap je Monat (für Aggregation)
        mcap = df.groupby(["month", group_col])["size_lag"].sum().rename("mcap")
        mcap = mcap.reset_index()
        return wide, mcap

    def _aggregate_over_groups(self, wide: pd.DataFrame, mcap: pd.DataFrame,
                               scheme: str) -> pd.DataFrame:
        """
        scheme: 'mcap' (Marktcap-Gewicht) oder 'equal' (1/n über Gruppen)
        returns: decile returns je month (über alle Gruppen aggregiert)
        """
        # bring mcap in wide-index
        wide2 = wide.copy()
        wide2 = wide2.reset_index()  # month, group_col, decile cols
        df = wide2.merge(mcap, on=list(wide2.columns[:2]), how="left")
        # Gewicht je month, group
        if scheme == "mcap":
            df["agg_w"] = df["mcap"].clip(lower=0)
        elif scheme == "equal":
            df["agg_w"] = 1.0
        else:
            raise ValueError("scheme must be 'mcap' or 'equal'.")

        # normiere Gewichte je Monat
        def _wavg(g: pd.DataFrame) -> pd.Series:
            w = g["agg_w"].fillna(0.0).values
            out = {}
            for d in range(1, GROUP_NUM_SLICES+1):
                col = d
                v = g[col].astype(float)
                m = v.notna() & np.isfinite(w)
                if m.any():
                    out[d] = float(np.average(v[m], weights=w[m]))
                else:
                    out[d] = np.nan
            return pd.Series(out)

        agg = df.groupby("month").apply(_wavg).sort_index()
        return agg  

    def long_short_alphas_by_group(self, signal_df: pd.DataFrame,
                                   crsp_full: pd.DataFrame,
                                   factors_full: pd.DataFrame,
                                   group_col: str) -> pd.DataFrame:
        """
        Liefert eine Tabelle (Zeilen=Gruppe, Spalten=FF3/FF5/Q) mit 'alpha (t)' Strings.
        Innerhalb jeder Gruppe werden value-weighted Decile gebildet; LS=10-1; Regression auf Faktoren.
        """
        df = self._prepare_signal_crsp(signal_df, crsp_full)
        # Deciles innerhalb (month, group)
        df = self._assign_deciles_within_groups(df, ["month", group_col])

        # LS pro Gruppe
        grp_cols = ["month", group_col, "decile"]
        ret = (df.groupby(grp_cols).apply(self._vwret)
                 .rename("vwret").reset_index())
        wide = ret.pivot_table(index=["month", group_col], columns="decile", values="vwret").sort_index()
        ls = wide[GROUP_NUM_SLICES] - wide[1]      # Series mit Multiindex (month, group)

        # Faktoren (wie im Rest: *100 und rf_ff ziehen)
        fac = factors_full.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        fac[[c for c in fac.columns if c!="DATE"]] = fac[[c for c in fac.columns if c!="DATE"]] * 100

        rows = []
        for grp, s in ls.groupby(level=1):  # für jede Gruppe Zeitreihe
            ts = s.droplevel(1)
            if ts.size < 8:   # zu wenige Monate für stabile Regression → skip
                continue
            df_ts = pd.DataFrame({"LS": ts}).join(fac, how="inner")
            y = df_ts["LS"] - df_ts["rf_ff"]  # Excess
            row = {"group": grp}
            for name, cols in self.MODEL_SPECS.items():
                a, t = self._fit_alpha(y, df_ts[cols])
                row[name] = (
                    r"\begin{tabular}{@{}c@{}}" 
                    + f"{a:.2f}" 
                    + r"\\\relax [" 
                    + f"{t:.2f}" 
                    + r"]\end{tabular}"
                    if np.isfinite(a) and np.isfinite(t) else ""
                )
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("group").set_index("group")
        return out

    def long_short_alphas_neutral_aggregations(self,
                                               signal_df: pd.DataFrame,
                                               crsp_full: pd.DataFrame,
                                               factors_full: pd.DataFrame,
                                               group_col: str) -> pd.DataFrame:
        """
        Baut die 'neutralen' Aggregationen:
          - erst Deciles value-weighted *innerhalb* jeder Gruppe,
          - dann Aggregation über Gruppen: 'mcap' und 'equal',
          - anschließend LS-Alpha (FF3/FF5/Q) je Aggregationsschema.
        Ergebnis: Zeilen = ['MarketCap-Weighted', 'Equal-Weighted'].
        """
        df = self._prepare_signal_crsp(signal_df, crsp_full)
        df = self._assign_deciles_within_groups(df, ["month", group_col])

        wide, mcap = self._build_group_decile_returns(df, group_col)

        fac = factors_full.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        fac[[c for c in fac.columns if c!="DATE"]] = fac[[c for c in fac.columns if c!="DATE"]] * 100

        rows = []
        for scheme_name, scheme in [("MarketCap-Weighted", "mcap"), ("Equal-Weighted", "equal")]:
            dec = self._aggregate_over_groups(wide, mcap, scheme)
            ls = dec[GROUP_NUM_SLICES] - dec[1]
            df_ts = pd.DataFrame({"LS": ls}).join(fac, how="inner")
            y = df_ts["LS"] - df_ts["rf_ff"]

            row = {"scheme": scheme_name}
            for name, cols in self.MODEL_SPECS.items():
                a, t = self._fit_alpha(y, df_ts[cols])
                row[name] = (
                    r"\begin{tabular}{@{}c@{}}" 
                    + f"{a:.2f}" 
                    + r"\\\relax [" 
                    + f"{t:.2f}" 
                    + r"]\end{tabular}"
                    if np.isfinite(a) and np.isfinite(t) else ""
                )
            rows.append(row)

        return pd.DataFrame(rows).set_index("scheme")
