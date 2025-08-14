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

# Constants
#------------------

NUM_OF_SLICES = 10

#------------------


class BaseAnalyzer(ABC):
    """Implements an abstract analyzer for stock return predictors."""

    def __init__(self, input_file: pd.DataFrame, crsp: pd.DataFrame | None = None, factors: pd.DataFrame | None = None) -> None:
        """Creates a new abstract analyzer.
        
        Args:
            input_file (pd.DataFrame): Data frame of the input file (.csv or .xlsx) 
                                       which contains data corresponding to the portfolio strategy.
        """
        self.data = input_file
        self.crsp = crsp if crsp is not None else pd.read_csv("./data/dsws_crsp.csv")
        self.factors = factors if factors is not None else pd.read_csv("./data/Factors.csv")

    def prepare_signal_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize a 3-column input DataFrame to columns: permno, date, signal.
        """
        if df.shape[1] != 3:
            raise ValueError("Input data must have exactly 3 columns.")
        df_signal = df.copy()
        df_signal.columns = ['permno', 'date', 'signal']
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        return df_signal

    def compute_long_short_regression(self, port_returns: pd.DataFrame, df_factors: pd.DataFrame) -> dict:
        """
        Computes long-short portfolio (Slice 10 - Slice 1) returns and regresses on FF3, FF5, Q models.
        """
        pivot = port_returns.pivot(index='year_month', columns='slice', values='port_ret')

        if 1 not in pivot.columns or 10 not in pivot.columns:
            raise ValueError("Long-Short regression skipped: Slice 1 or 10 not found.")

        # Caluclate Long-short: High - Low
        pivot['long_short'] = pivot[10] - pivot[1]
        ls_returns = pivot[['long_short']].reset_index()

        df = pd.merge(ls_returns, df_factors, on='year_month', how='inner')
        df['excess_ret'] = df['long_short'] - df['RF']

        y = df['excess_ret']
        x_ff3 = sm.add_constant(df[['MKT', 'SMB', 'HML']])
        x_ff5 = sm.add_constant(df[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
        x_q   = sm.add_constant(df[['MKT', 'SIZE', 'IA', 'ROE']])

        res_ff3 = sm.OLS(y, x_ff3).fit()
        res_ff5 = sm.OLS(y, x_ff5).fit()
        res_q   = sm.OLS(y, x_q).fit()

        result = {
            "FF3": res_ff3,
            "FF5": res_ff5,
            "Q": res_q
        }

        return result



    @abstractmethod
    def analyze(self) -> Any:
        """
        Abstract method to perform analysis on the loaded data.
        """
        pass


class SimpleAnalyzer(BaseAnalyzer):

    def analyze(self) -> Tuple[str, str, str]:
        """
        Analyzes portfolio strategy by computing average next-month returns by slices.

        This method performs the following steps:
        1. Loads industry return data and given signal data.
        2. Prepares and formats both the signal and return datasets.
        3. Assigns each stock to a signal-based slice per month.
        4. Merges the signal data with the return data, aligning each stock's signal with its return in the following month.
        5. Calculates the average next-month return for each slice.
        6. Returns a formatted string showing the average return by slice.

        Returns:
            Tuple[str, str, str]: 
                - First parameter: A multiline string reporting the average next-month return for each signal slice.
                - Second parameter: A multiline string reporting the monthly returns per slice.
                - Third parameter: A multiline string reporting the mapping of each stock to the respective slice for every month.
        """
        df_signal = self.prepare_signal_df(self.data)
        df_return = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date', 'RET_USD': 'ret_usd'}, inplace=True)
        
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_return['date'] = pd.to_datetime(df_return['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        
        df_signal['slice'] = (
            df_signal.groupby('year_month')['signal']
            .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )
        
        df_signal['next_month'] = (df_signal['year_month'] + 1).dt.to_timestamp()
        df_return['year_month'] = df_return['date'].dt.to_period('M')

        df_signal['permno'] = pd.to_numeric(df_signal['permno'], errors='coerce')
        df_return['permno'] = pd.to_numeric(df_return['permno'], errors='coerce')

        df_signal = df_signal.dropna(subset=['permno'])
        df_return = df_return.dropna(subset=['permno'])
        df_signal['permno'] = df_signal['permno'].astype(int)
        df_return['permno'] = df_return['permno'].astype(int)
        
        merged = pd.merge(df_signal, df_return, left_on=['permno', 'next_month'], right_on=['permno', 'date'], how='inner')
        
        portfolio_returns = merged.groupby(['year_month_x', 'slice'])['ret_usd'].mean().reset_index()
        
        avg_returns = portfolio_returns.groupby('slice')['ret_usd'].mean()
        

        # Generate Output messages
        # --------------------------
        result_str = "Average next-month returns:\n" \
                     "---------------------------\n"
        for slice, avg_ret in avg_returns.items():
            result_str += f"Slice {slice}: {avg_ret:.4f}\n"
        
        monthly_avg = portfolio_returns.copy()
        monthly_avg['year_month_x'] = monthly_avg['year_month_x'].astype(str)

        monthly_avg_str = "Monthly Average Returns by Slices:\n" \
                          "------------------------------------\n"
        for _, row in monthly_avg.iterrows():
            monthly_avg_str += f"Month {row['year_month_x']}, Slice {row['slice']}: {row['ret_usd']:.4f}\n"


        slice_mapping = df_signal[['permno', 'year_month', 'slice']].dropna().copy()
        slice_mapping['year_month'] = slice_mapping['year_month'].astype(str)
        mapping_str = "Slice Mapping by Month:\n" \
                      "--------------------------\n"
        for _, row in slice_mapping.iterrows():
            mapping_str += f"permno {row['permno']}, month {row['year_month']}, slice {row['slice']}\n"

        return result_str, monthly_avg_str, mapping_str
    

class EqualWeightedFactorModelAnalyzer(BaseAnalyzer):
    """ Analyzer that performs Fama-French (3, 5, Q-Factor) regressions on signal-based portfolios. """
    def analyze(self, industry_code: int | None = None) -> Tuple[dict, dict]:
        # Prepare signal data
        df_signal = self.prepare_signal_df(self.data)
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')

        # Prepare CRSP data (filter by industry if requested)
        df_crsp = self.crsp
        if industry_code is not None:
            df_crsp = df_crsp.loc[df_crsp['ff12'] == industry_code]

        df_crsp = df_crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'}).copy()
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100  # percent to decimal

        # Merge signals with CRSP
        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')

        # Create slices based on signal
        merged['slice'] = (
            merged.groupby('year_month')['signal']
                  .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )

        # Equal-weighted portfolio returns
        port_returns = (
            merged.groupby(['year_month', 'slice'])['RET_USD']
                  .mean()
                  .reset_index()
                  .rename(columns={'RET_USD': 'port_ret'})
        )
        port_returns['year_month'] = port_returns['year_month'].astype('period[M]')

        # Prepare factors
        df_factors = self.factors.copy()
        df_factors['DATE'] = pd.to_datetime(df_factors['DATE'])
        df_factors['year_month'] = df_factors['DATE'].dt.to_period('M')
        df_factors = df_factors.rename(columns={
            'MKTRF_usd': 'MKT', 'SMB_usd': 'SMB', 'HML_usd': 'HML',
            'RMW_usd': 'RMW', 'CMA_usd': 'CMA', 'rf_ff': 'RF',
            'ME_usd': 'SIZE', 'ROE_usd': 'ROE', 'IA_usd': 'IA'
        })

        # Merge with factor returns
        model_data = pd.merge(port_returns, df_factors, on='year_month', how='inner')
        model_data['excess_ret'] = model_data['port_ret'] - model_data['RF']

        # Run regressions per slice
        results = {}
        for q in sorted(model_data['slice'].unique()):
            subset = model_data[model_data['slice'] == q]
            y = subset['excess_ret']
            x_ff3 = sm.add_constant(subset[['MKT', 'SMB', 'HML']])
            x_ff5 = sm.add_constant(subset[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
            x_q = sm.add_constant(subset[['MKT', 'SIZE', 'IA', 'ROE']])
            results[q] = {
                'FF3': sm.OLS(y, x_ff3).fit(),
                'FF5': sm.OLS(y, x_ff5).fit(),
                'Q':   sm.OLS(y, x_q).fit()
            }

        long_short_res = self.compute_long_short_regression(port_returns, df_factors)
        return results, long_short_res
        


class ValueWeightedFactorModelAnalyzer(BaseAnalyzer):
    """ Analyzer that performs Fama-French (3, 5, Q-Factor) regressions on value-weighted signal-based portfolios. """
    def analyze(self, industry_code: int | None = None) -> Tuple[str, str, str, str]:
        # Prepare signal data
        df_signal = self.prepare_signal_df(self.data)
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')

        # Prepare CRSP (filter by industry if requested)
        df_crsp = self.crsp
        if industry_code is not None:
            df_crsp = df_crsp.loc[df_crsp['ff12'] == industry_code]

        df_crsp = df_crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'}).copy()
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100  # percent to decimal

        # Merge signals with CRSP
        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')
        merged = merged.dropna(subset=['signal', 'RET_USD', 'size_lag'])

        # Create slices based on signal
        merged['slice'] = (
            merged.groupby('year_month')['signal']
                  .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )

        # Ensure weights numeric
        merged['size_lag'] = pd.to_numeric(merged['size_lag'], errors='coerce')
        merged = merged.dropna(subset=['size_lag'])

        # Value-weighted returns
        merged['weighted_ret'] = merged['RET_USD'] * merged['size_lag']
        port_returns = (
            merged.groupby(['year_month', 'slice'])
                  .agg(total_ret=('weighted_ret', 'sum'), total_size=('size_lag', 'sum'))
                  .reset_index()
        )
        port_returns['port_ret'] = port_returns['total_ret'] / port_returns['total_size']

        # Factors
        df_factors = self.factors.copy()
        df_factors['DATE'] = pd.to_datetime(df_factors['DATE'])
        df_factors['year_month'] = df_factors['DATE'].dt.to_period('M')
        df_factors = df_factors.rename(columns={
            'MKTRF_usd': 'MKT', 'SMB_usd': 'SMB', 'HML_usd': 'HML',
            'RMW_usd': 'RMW', 'CMA_usd': 'CMA', 'rf_ff': 'RF',
            'ME_usd': 'SIZE', 'ROE_usd': 'ROE', 'IA_usd': 'IA'
        })

        model_data = pd.merge(port_returns, df_factors, on='year_month', how='inner')
        model_data['excess_ret'] = model_data['port_ret'] - model_data['RF']

        # Regressions
        results = {}
        for q in sorted(model_data['slice'].unique()):
            subset = model_data[model_data['slice'] == q]
            y = subset['excess_ret']
            x_ff3 = sm.add_constant(subset[['MKT', 'SMB', 'HML']])
            x_ff5 = sm.add_constant(subset[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
            x_q = sm.add_constant(subset[['MKT', 'SIZE', 'IA', 'ROE']])
            results[q] = {
                'FF3': sm.OLS(y, x_ff3).fit(),
                'FF5': sm.OLS(y, x_ff5).fit(),
                'Q':   sm.OLS(y, x_q).fit()
            }

        long_short_res = self.compute_long_short_regression(port_returns, df_factors)
        return results, long_short_res


class FamaMacBethAnalyzer(BaseAnalyzer):
    """ Performs Fama-MacBeth regression of returns on signals. """
    def analyze(self, industry_code: int | None = None) -> tuple:
        df_signal = self.prepare_signal_df(self.data)
        df_crsp = self.crsp
        if industry_code is not None:
            df_crsp = df_crsp.loc[df_crsp['ff12'] == industry_code]

        df_crsp = df_crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'}).copy()
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100

        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')
        merged = merged.dropna(subset=['RET_USD', 'signal'])

        betas, dates = [], []
        for date, group in merged.groupby('year_month'):
            if group['signal'].nunique() > 1:
                X = sm.add_constant(group['signal'])
                y = group['RET_USD']
                res = sm.OLS(y, X).fit()
                betas.append(res.params['signal'])
                dates.append(date)

        betas_series = pd.Series(betas, index=pd.PeriodIndex(dates, freq='M'))
        beta_mean = betas_series.mean()
        beta_std = betas_series.std()
        n = len(betas_series)
        t_stat = beta_mean / (beta_std / np.sqrt(n)) if n > 1 and beta_std > 0 else np.nan
        return beta_mean, beta_std, t_stat, n


def _inject_title_fallback(doc: str, title: str) -> str:
    # If Formatter can't take a doc_title kwarg, inject it into the LaTeX string.
    # 1) Replace existing \title{...} if present; else
    # 2) Insert \title{...}\maketitle before \begin{document}.
    import re as _re
    if r'\title{' in doc:
        return _re.sub(r'\\title\{.*?\}', f'\\title{{{title}}}', doc, count=1, flags=_re.S)
    return doc.replace(r'\begin{document}', f'\\title{{{title}}}\n\\maketitle\n\\begin{document}', 1)

def _latex_escape(s: str) -> str:
    """
    Escape LaTeX special chars in strings for safe use inside LaTeX documents.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        s = s.replace(char, repl)
    return s

def run_analysis(df: pd.DataFrame, signal_name: str):
    # Load once & share
    crsp_full = pd.read_csv("./data/dsws_crsp.csv")
    factors_full = pd.read_csv("./data/Factors.csv")

    equal_factor_model_analyzer = EqualWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full)
    value_factor_model_analyzer = ValueWeightedFactorModelAnalyzer(df, crsp=crsp_full, factors=factors_full)
    fama_macbeth_analyzer = FamaMacBethAnalyzer(df, crsp=crsp_full, factors=factors_full)

    # ---------- Baseline (all industries) ----------
    results_equal, long_short_results_equal = equal_factor_model_analyzer.analyze()
    ff3_equal, ff5_equal, q_equal = Formatter.results_to_strings(results_equal)
    long_short_equal = Formatter.long_short_res_to_string(long_short_results_equal)
    latex_ff3_equal = Formatter.generate_latex_table(results_equal, "FF3")
    latex_ff5_equal = Formatter.generate_latex_table(results_equal, "FF5")
    latex_q_equal = Formatter.generate_latex_table(results_equal, "Q")
    latex_long_short_equal = Formatter.generate_long_short_latex_table(long_short_results_equal)

    results_value, long_short_results_value = value_factor_model_analyzer.analyze()
    ff3_value, ff5_value, q_value = Formatter.results_to_strings(results_value)
    long_short_value = Formatter.long_short_res_to_string(long_short_results_value)
    latex_ff3_value = Formatter.generate_latex_table(results_value, "FF3")
    latex_ff5_value = Formatter.generate_latex_table(results_value, "FF5")
    latex_q_value = Formatter.generate_latex_table(results_value, "Q")
    latex_long_short_value = Formatter.generate_long_short_latex_table(long_short_results_value)

    beta_mean, beta_std, t_stat, n = fama_macbeth_analyzer.analyze()
    fama_macbeth_res = Formatter.fama_macbeth_res_to_string(beta_mean, beta_std, t_stat, n)
    latex_fama_macbeth = Formatter.generate_fama_macbeth_latex_table(beta_mean, beta_std, t_stat, n)

    escaped_signal = _latex_escape(signal_name)

    # Title-aware LaTeX for baseline
    baseline_title = f"Global Stock Market Protocol Analysis Results for {escaped_signal} - Baseline (All Industries)"
    try:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fama_macbeth,
            title=baseline_title,         
        )
    except TypeError:
        latex_output = Formatter.create_complete_latex_document(
            latex_ff3_equal, latex_ff5_equal, latex_q_equal, latex_long_short_equal,
            latex_ff3_value, latex_ff5_value, latex_q_value, latex_long_short_value,
            latex_fama_macbeth
        )
        latex_output = _inject_title_fallback(latex_output, baseline_title)

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
        "baseline/fama_macbeth.txt": fama_macbeth_res,
        "baseline/output.tex": latex_output,
    }

    # ---------- Per-industry runs ----------
    industry_codes = sorted(pd.unique(crsp_full['ff12'].dropna().astype(int)))
    for code in industry_codes:
        code_int = int(code)

        res_eq_ind, ls_eq_ind = equal_factor_model_analyzer.analyze(industry_code=code_int)
        ff3_eq_i, ff5_eq_i, q_eq_i = Formatter.results_to_strings(res_eq_ind)
        ls_eq_i = Formatter.long_short_res_to_string(ls_eq_ind)

        res_vw_ind, ls_vw_ind = value_factor_model_analyzer.analyze(industry_code=code_int)
        ff3_vw_i, ff5_vw_i, q_vw_i = Formatter.results_to_strings(res_vw_ind)
        ls_vw_i = Formatter.long_short_res_to_string(ls_vw_ind)

        bmean_i, bstd_i, t_i, n_i = fama_macbeth_analyzer.analyze(industry_code=code_int)
        fm_i = Formatter.fama_macbeth_res_to_string(bmean_i, bstd_i, t_i, n_i)

        # Title-aware LaTeX per industry
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

                Formatter.generate_fama_macbeth_latex_table(bmean_i, bstd_i, t_i, n_i),
                title=industry_title,
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

                Formatter.generate_fama_macbeth_latex_table(bmean_i, bstd_i, t_i, n_i),
            )
            latex_output_i = _inject_title_fallback(latex_output_i, industry_title)

        prefix = f"industries/industry_ff12_{code_int}"
        results.update({
            f"{prefix}/ff3_equal.txt": ff3_eq_i,
            f"{prefix}/ff5_equal.txt": ff5_eq_i,
            f"{prefix}/q_equal.txt":   q_eq_i,
            f"{prefix}/long_short_equal.txt": ls_eq_i,
            f"{prefix}/ff3_value.txt": ff3_vw_i,
            f"{prefix}/ff5_value.txt": ff5_vw_i,
            f"{prefix}/q_value.txt":   q_vw_i,
            f"{prefix}/long_short_value.txt": ls_vw_i,
            f"{prefix}/fama_macbeth.txt": fm_i,
            f"{prefix}/output.tex": latex_output_i,
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