from __future__ import annotations 

import numpy as np 
import pandas as pd 
import statsmodels.api as sm 

from logic.analyzers.base import BaseAnalyzer
from logic.analyzers.base import AutoRegistered, AnalyzerOutput
from logic.utils.formatter import Formatter

class FamaMacBethAnalyzer(BaseAnalyzer, AutoRegistered):
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
        Alle Variablen werden pro Monat auf das Intervall [-1, 1] skaliert.
    """

    ENABLED = True
    ORDER = 30
    TITLE = "Fama–MacBeth Regression Result"
    
    def __init__(self, ctx, df_input, signal_name):
        super().__init__(ctx, df_input, signal_name)
    
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
    
    def generate_output(self) -> AnalyzerOutput:
        means, tstats, n_months = self.analyze()
        txt  = Formatter.fama_macbeth_res_to_string(means, tstats, n_months, self.signal_name)
        latex = Formatter.generate_fama_macbeth_latex_table(means, tstats, self.signal_name)
        return AnalyzerOutput(
            name=self.TITLE,
            raw_texts={"fama_macbeth.txt": txt},
            latex_blocks=[latex]
        )
    

    # Helper Methods
    # -------------------------------

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

        # Skalierung aller Variablen auf [-1, 1] pro Monat
        regressors = ["signal", "size_lag", "mom_2_12", "bm", "ag", "rd_sale"]

        def scale_monthly(s: pd.Series):
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

    def _fm_tstat(self, series: pd.Series) -> float:
        s = series.dropna()
        T = len(s)
        if T == 0:
            return np.nan
        return float(s.mean() / (s.std(ddof=1) / np.sqrt(T)))
