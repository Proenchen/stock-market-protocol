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
GROUP_NUM_SLICES = 5  # für Industrie-/Länder-Tabellen

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
        Alle Variablen werden pro Monat auf das Intervall [-1, 1] skaliert.
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


class _VWLongShortTables:
    """
    Hilfsfunktionen für value-weighted Long-Short Alpha-Tabellen.
    Nutzt dieselbe Faktorskalierung und Regressionsroutine wie deine Analyzer.
    """

    MODEL_SPECS = {
        "FF3": ["MKTRF_usd", "SMB_usd", "HML_usd"],
        "FF5": ["MKTRF_usd", "SMB_usd", "HML_usd", "RMW_A_usd", "CMA_usd"],
        "Q"  : ["MKTRF_usd", "ME_usd", "IA_usd", "ROE_usd"],
    }

    @staticmethod
    def _fit_alpha(y: pd.Series, X: pd.DataFrame):
        Xc = sm.add_constant(X)
        m = y.notna() & Xc.notna().all(axis=1)
        y2, X2 = y[m], Xc[m]
        if len(y2)==0 or X2.shape[0] <= X2.shape[1]:
            return np.nan, np.nan
        res = sm.OLS(y2, X2).fit()
        a = res.params.get("const", np.nan)
        t = res.tvalues.get("const", np.nan)
        return float(a), float(t)

    @staticmethod
    def _prepare_signal_crsp(signal_df: pd.DataFrame, crsp_full: pd.DataFrame) -> pd.DataFrame:
        df = signal_df.copy()
        df = df.rename(columns={df.columns[0]:"DSCD", df.columns[1]:"dates", df.columns[2]:"signal"})
        df["month"] = pd.to_datetime(df["dates"]).dt.to_period("M").dt.to_timestamp("M")

        cr = crsp_full.copy()
        cr["DATE"] = pd.to_datetime(cr["DATE"])
        cr["month"] = cr["DATE"].dt.to_period("M").dt.to_timestamp("M")
        keep = ["DSCD","month","RET_USD","size_lag","ff12","country"]
        cr = cr[keep]
        return pd.merge(df, cr, on=["DSCD","month"], how="inner")

    @staticmethod
    def _assign_deciles_within_groups(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
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

    @staticmethod
    def _vwret(g: pd.DataFrame) -> float:
        w = pd.to_numeric(g["size_lag"], errors="coerce").clip(lower=0)
        r = pd.to_numeric(g["RET_USD"], errors="coerce")
        m = r.notna() & w.notna() & (w>0)
        if not m.any():
            return np.nan
        return float(np.average(r[m], weights=w[m]))

    @staticmethod
    def _build_group_decile_returns(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # value-weighted Decile-Returns je (month, group, decile)
        by = ["month", group_col, "decile"]
        ret = (df.groupby(by).apply(_VWLongShortTables._vwret)
                 .rename("vwret").reset_index())
        # zu wide per Gruppe
        wide = ret.pivot_table(index=["month",group_col], columns="decile", values="vwret")
        # Group Market Cap je Monat (für Aggregation)
        mcap = df.groupby(["month", group_col])["size_lag"].sum().rename("mcap")
        mcap = mcap.reset_index()
        return wide, mcap

    @staticmethod
    def _aggregate_over_groups(wide: pd.DataFrame, mcap: pd.DataFrame,
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

    @staticmethod
    def long_short_alphas_by_group(signal_df: pd.DataFrame,
                                   crsp_full: pd.DataFrame,
                                   factors_full: pd.DataFrame,
                                   group_col: str) -> pd.DataFrame:
        """
        Liefert eine Tabelle (Zeilen=Gruppe, Spalten=FF3/FF5/Q) mit 'alpha (t)' Strings.
        Innerhalb jeder Gruppe werden value-weighted Decile gebildet; LS=10-1; Regression auf Faktoren.
        """
        df = _VWLongShortTables._prepare_signal_crsp(signal_df, crsp_full)
        # Deciles innerhalb (month, group)
        df = _VWLongShortTables._assign_deciles_within_groups(df, ["month", group_col])

        # LS pro Gruppe
        grp_cols = ["month", group_col, "decile"]
        ret = (df.groupby(grp_cols).apply(_VWLongShortTables._vwret)
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
            for name, cols in _VWLongShortTables.MODEL_SPECS.items():
                a, t = _VWLongShortTables._fit_alpha(y, df_ts[cols])
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

    @staticmethod
    def long_short_alphas_neutral_aggregations(signal_df: pd.DataFrame,
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
        df = _VWLongShortTables._prepare_signal_crsp(signal_df, crsp_full)
        df = _VWLongShortTables._assign_deciles_within_groups(df, ["month", group_col])

        wide, mcap = _VWLongShortTables._build_group_decile_returns(df, group_col)

        fac = factors_full.copy()
        fac["month"] = pd.to_datetime(fac["DATE"]).dt.to_period("M").dt.to_timestamp("M")
        fac = fac.set_index("month").sort_index()
        fac[[c for c in fac.columns if c!="DATE"]] = fac[[c for c in fac.columns if c!="DATE"]] * 100

        rows = []
        for scheme_name, scheme in [("MarketCap-Weighted", "mcap"), ("Equal-Weighted", "equal")]:
            dec = _VWLongShortTables._aggregate_over_groups(wide, mcap, scheme)
            ls = dec[GROUP_NUM_SLICES] - dec[1]
            df_ts = pd.DataFrame({"LS": ls}).join(fac, how="inner")
            y = df_ts["LS"] - df_ts["rf_ff"]

            row = {"scheme": scheme_name}
            for name, cols in _VWLongShortTables.MODEL_SPECS.items():
                a, t = _VWLongShortTables._fit_alpha(y, df_ts[cols])
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
    print(22)

    alpha_by_industry = _VWLongShortTables.long_short_alphas_by_group(
        df, crsp_full, factors_full, group_col="ff12"
    )
    latex_alpha_by_industry = Formatter.alpha_table_to_latex(
        alpha_by_industry, "Industry", "Intercept estimates from times-series regressions of long-short portfolios for each industry using different factor models. " \
        "Each cell displays the monthly intercept with t-statistics in brackets. " \
        "The portfolios are constructed using value weighting for each industry."
    )

    alpha_industry_neutral = _VWLongShortTables.long_short_alphas_neutral_aggregations(
        df, crsp_full, factors_full, group_col="ff12"
    )
    latex_alpha_industry_neutral = Formatter.alpha_table_to_latex(
        alpha_industry_neutral, "", "Intercept values for the aggregated industry portfolios based on the market cap and equal weighting."
    )


    alpha_by_country = _VWLongShortTables.long_short_alphas_by_group(
        df, crsp_full, factors_full, group_col="country"
    )
    
    MAX_ROWS_SINGLE = 20
    country_caption = (
        "Intercept estimates from times-series regressions of long-short portfolios for each country using different factor models. " \
        "Each cell displays the monthly intercept with t-statistics in brackets. " \
        "The portfolios are constructed using value weighting for each country."
    )

    if len(alpha_by_country) > MAX_ROWS_SINGLE:
        latex_alpha_by_country = Formatter.alpha_table_to_latex_four_quarters_two_pages(
            alpha_by_country, "Country", country_caption
        )
    else:
        latex_alpha_by_country = Formatter.alpha_table_to_latex(
            alpha_by_country, "Country", country_caption
        )

    alpha_country_neutral = _VWLongShortTables.long_short_alphas_neutral_aggregations(
        df, crsp_full, factors_full, group_col="country"
    )
    latex_alpha_country_neutral = Formatter.alpha_table_to_latex(
        alpha_country_neutral, "", "Intercept values for the aggregated country portfolios based on the market cap and equal weighting."
    )


    print(3)
    escaped_signal = Formatter._latex_escape(signal_name)

    # Title-aware LaTeX for baseline
    baseline_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Baseline (All Industries)"
    try:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fmb_res, latex_alpha_by_industry, latex_alpha_industry_neutral,
            latex_alpha_by_country, latex_alpha_country_neutral,
            title=baseline_title,         
        )
    except TypeError:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fmb_res, latex_alpha_by_industry, latex_alpha_industry_neutral,
            latex_alpha_by_country, latex_alpha_country_neutral,
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

    results = {
        "ff3_equal.txt": ff3_equal,
        "ff5_equal.txt": ff5_equal,
        "q_equal.txt": q_equal,
        "long_short_equal.txt": long_short_equal,
        "ff3_value.txt": ff3_value,
        "ff5_value.txt": ff5_value,
        "q_value.txt": q_value,
        "long_short_value.txt": long_short_value,
        "fama_macbeth.txt": fmb_res_string,
        "output.tex": latex_output,
    }
    print(5)

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
