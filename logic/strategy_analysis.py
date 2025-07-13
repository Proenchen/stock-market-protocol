import numpy as np
import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Any


# Constants
#------------------

NUM_OF_SLICES = 10

#------------------


class BaseAnalyzer(ABC):
    """Implements an abstract analyzer for stock return predictors."""

    def __init__(self, input_file: pd.DataFrame) -> None:
        """Creates a new abstract analyzer.
        
        Args:
            input_file (pd.DataFrame): Data frame of the input file (.csv or .xlsx) 
                                       which contains data corresponding to the portfolio strategy.
        """
        self.data = input_file
        self.crsp = pd.read_csv("./data/dsws_crsp.csv")
        self.factors = pd.read_csv("./data/Factors.csv")

    def prepare_signal_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize a 3-column input DataFrame to columns: permno, date, signal.
        """
        if df.shape[1] != 3:
            raise ValueError("Input data must have exactly 3 columns: DSCD, DATE, SIGNAL.")
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


    # Methods for formatting results
    #-------------------------------------
    def results_to_strings(self, results: dict) -> Tuple[str, str, str]:
        ff3_str, ff5_str, q_str = "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Slice {q} ---\n{res['FF3'].summary().as_text()}\n"
            ff5_str += f"\n--- Slice {q} ---\n{res['FF5'].summary().as_text()}\n"
            q_str += f"\n--- Slice {q} ---\n{res['Q'].summary().as_text()}\n"

        return ff3_str, ff5_str, q_str

    def long_short_res_to_string(self, result: dict) -> str:
        output = ""
        output = "\n======= Long-Short Portfolio (Slice 10 - Slice 1) =======\n"
        output += "\n--- FF3 Regression ---\n" + result["FF3"].summary().as_text()
        output += "\n\n--- FF5 Regression ---\n" + result["FF5"].summary().as_text()
        output += "\n\n--- Q-Factor Regression ---\n" + result["Q"].summary().as_text()

        return output
    
    def fama_macbeth_res_to_string(self, beta_mean, beta_std, t_stat, n) -> str:
        result_str = "Fama-MacBeth Regression Result\n" \
                     "------------------------------\n" \
                     f"Mean Beta: {beta_mean:.4f}\n" \
                     f"Std Dev:   {beta_std:.4f}\n" \
                     f"T-Stat:    {t_stat:.4f}\n" \
                     f"N Months:  {n}\n"
        
        return result_str
                

    def generate_latex_table(self, results_dict: dict, model_name: str) -> str:
        """
        Generates a LaTeX table from regression results.

        Args:
            results_dict (dict): Dictionary mapping slice to statsmodels regression results.
            model_name (str): One of 'FF3', 'FF5', or 'Q'.

        Returns:
            str: LaTeX code for the table.
        """

        factor_order = {
            'FF3': ['const', 'MKT', 'SMB', 'HML'],
            'FF5': ['const', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'],
            'Q':   ['const', 'MKT', 'IA', 'ROE', 'SIZE']
        }

        factor_labels = {
            'const': r'$\alpha$',
            'MKT': r'$\beta_{MKT}$',
            'SMB': r'$\beta_{SMB}$',
            'HML': r'$\beta_{HML}$',
            'RMW': r'$\beta_{RMW}$',
            'CMA': r'$\beta_{CMA}$',
            'SIZE': r'$\beta_{SIZE}$',
            'IA': r'$\beta_{IA}$',
            'ROE': r'$\beta_{ROE}$'
        }

        # Table header
        headers = [factor_labels[p] for p in factor_order[model_name]]
        table_header = " & " + " & ".join(headers) + r" \\ \toprule" + "\n"

        # Table body
        body = ""
        for slice_id in sorted(results_dict.keys()):
            model = results_dict[slice_id][model_name]
            coef = model.params
            tvals = model.tvalues

            row = [str(slice_id)]
            for factor in factor_order[model_name]:
                if factor in coef:
                    val = f"{coef[factor]:.4f}"
                    tstat = f"{tvals[factor]:.2f}"
                    cell = r"\begin{tabular}{@{}c@{}}" + val +  r"\\\relax [" +  tstat + r"]\end{tabular}"
                else:
                    cell = ""
                row.append(cell)

            body += " & ".join(row) + r" \\[12pt]" + "\n"

        # Final LaTeX table
        table = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{l" + "c" * len(factor_order[model_name]) + "}" + "\n"
            r"\toprule" + "\n"
            "Slice" + table_header + body + r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            rf"\caption{{Regression Results: {model_name}. T-statistics are in brackets.}}" + "\n"
            r"\end{table}"
        )

        return table
    
    def generate_long_short_latex_table(self, results_dict: dict) -> str:
        """
        Generates a LaTeX table for the long-short regression results across FF3, FF5, and Q models.

        Args:
            res_ff3, res_ff5, res_q (RegressionResultsWrapper): Regression results for each model.

        Returns:
            str: LaTeX code for the table.
        """
        models = {
            'FF3': results_dict["FF3"],
            'FF5': results_dict["FF5"],
            'Q': results_dict["Q"]
        }

        # Desired fixed order of all potential factors
        all_factors = ['const', 'MKT', 'SMB', 'HML', 'RMW', 'CMA', 'IA', 'ROE', 'SIZE']

        # LaTeX-friendly labels
        factor_labels = {
            'const': r'$\alpha$',
            'MKT': r'$\beta_{MKT}$',
            'SMB': r'$\beta_{SMB}$',
            'HML': r'$\beta_{HML}$',
            'RMW': r'$\beta_{RMW}$',
            'CMA': r'$\beta_{CMA}$',
            'SIZE': r'$\beta_{SIZE}$',
            'IA': r'$\beta_{IA}$',
            'ROE': r'$\beta_{ROE}$'
        }

        # Table header
        headers = ["Model"] + [factor_labels.get(f, f) for f in all_factors]
        table_header = " & ".join(headers) + r" \\ \toprule" + "\n"

        # Table body with t-stats
        body = ""
        for model_name, res in models.items():
            row = [model_name]
            coef = res.params
            tvals = res.tvalues

            for factor in all_factors:
                if factor in coef:
                    val = f"{coef[factor]:.4f}"
                    tstat = f"{tvals[factor]:.2f}"
                    cell = r"\begin{tabular}{@{}c@{}}" + val +  r"\\\relax[" +  tstat + r"]\end{tabular}"
                else:
                    cell = ""
                row.append(cell)
            body += " & ".join(row) + r" \\[12pt]" + "\n"

        # Full table
        table = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{l" + "c" * len(all_factors) + "}" + "\n"
            r"\toprule" + "\n"
            + table_header + body +
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{Long-Short Regression Results (Slice 10 - Slice 1). T-statistics are in brackets.}" + "\n"
            r"\end{table}"
        )

        return table
    
    def generate_fama_macbeth_latex_table(self, beta_mean, beta_std, t_stat, n) -> str:
        table = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{lr}" + "\n"
            r"\toprule" + "\n"
            r"Ratio & Value \\" + "\n"
            r"\midrule" + "\n"
            rf"Mean Beta & {beta_mean:.4f} \\" + "\n"
            rf"Standard Deviation & {beta_std:.4f} \\" + "\n"
            rf"T-Statistic & {t_stat:.4f} \\" + "\n"
            rf"N (Months) & {n} \\" + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{Fama-MacBeth Regression Result}" + "\n"
            r"\end{table}"
        )

        return table


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
    """
    Analyzer that performs Fama-French (3, 5, Q-Factor) regressions on signal-based portfolios.
    """

    def analyze(self) -> Tuple[dict, dict]:
        """
        Performs:
        1. Merge signals with CRSP data.
        2. Build equal-weighted portfolios.
        3. Calculate monthly portfolio returns.
        4. Regress each slice return on FF3, FF5, and Q-Factor models. Also perform an analysis on a long-short portfolio.

        Returns:
            Tuple of results for 3-factor, 5-factor, Q-factor models and the long-short analysis.
        """
        # Prepare signal data
        df_signal = self.prepare_signal_df(self.data)
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')

        # Prepare CRSP data
        df_crsp = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'})
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')

        # Convert percentage in decimal
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100

        # Merge signals with CRSP
        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')

        # Create slices based on signal
        merged['slice'] = (
            merged.groupby('year_month')['signal']
            .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )

        # Portfolio returns (equal-weighted)
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
        df_factors = df_factors.rename(columns={'MKTRF_usd': 'MKT', 'SMB_usd': 'SMB', 'HML_usd': 'HML',
                                                'RMW_usd': 'RMW', 'CMA_usd': 'CMA', 'rf_ff': 'RF',
                                                'ME_usd': 'SIZE', 'ROE_usd': 'ROE', 'IA_usd': 'IA'})

        # Merge portfolio returns with factor returns
        model_data = pd.merge(port_returns, df_factors, on='year_month', how='inner')
        model_data['excess_ret'] = model_data['port_ret'] - model_data['RF']

        # Run regressions
        results = {}
        for q in sorted(model_data['slice'].unique()):
            subset = model_data[model_data['slice'] == q]
            
            y = subset['excess_ret']
            x_ff3 = sm.add_constant(subset[['MKT', 'SMB', 'HML']])
            x_ff5 = sm.add_constant(subset[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
            x_q = sm.add_constant(subset[['MKT', 'SIZE', 'IA', 'ROE']]) 

            res_ff3 = sm.OLS(y, x_ff3).fit()
            res_ff5 = sm.OLS(y, x_ff5).fit()
            res_q = sm.OLS(y, x_q).fit()

            results[q] = {
                'FF3': res_ff3,
                'FF5': res_ff5,
                'Q': res_q
            }
        
        long_short_res = self.compute_long_short_regression(port_returns, df_factors)

        return results, long_short_res
        


class ValueWeightedFactorModelAnalyzer(BaseAnalyzer):
    """
    Analyzer that performs Fama-French (3, 5, Q-Factor) regressions on value-weighted signal-based portfolios.
    """

    def analyze(self) -> Tuple[str, str, str, str]:
        # Prepare signal data
        df_signal = self.prepare_signal_df(self.data)
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')

        # Prepare CRSP data
        df_crsp = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'})
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')

        # Convert percentage in decimal
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100

        # Merge signals with CRSP
        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')

        # Drop missing values in critical columns
        merged = merged.dropna(subset=['signal', 'RET_USD', 'size_lag'])

        # Create slices based on signal
        merged['slice'] = (
            merged.groupby('year_month')['signal']
            .transform(lambda x: pd.qcut(x, NUM_OF_SLICES, labels=False, duplicates='drop') + 1)
        )

        # Ensure size_lag is numeric
        merged['size_lag'] = pd.to_numeric(merged['size_lag'], errors='coerce')
        merged = merged.dropna(subset=['size_lag'])

        # Value-weighted return = Return * lagged market cap
        merged['weighted_ret'] = merged['RET_USD'] * merged['size_lag']

        # Aggregate portfolio returns (value-weighted)
        port_returns = (
            merged.groupby(['year_month', 'slice']).agg(
                total_ret=('weighted_ret', 'sum'),
                total_size=('size_lag', 'sum')
            ).reset_index()
        )
        port_returns['port_ret'] = port_returns['total_ret'] / port_returns['total_size']

        # Prepare factor data
        df_factors = self.factors.copy()
        df_factors['DATE'] = pd.to_datetime(df_factors['DATE'])
        df_factors['year_month'] = df_factors['DATE'].dt.to_period('M')
        df_factors = df_factors.rename(columns={
            'MKTRF_usd': 'MKT', 'SMB_usd': 'SMB', 'HML_usd': 'HML',
            'RMW_usd': 'RMW', 'CMA_usd': 'CMA', 'rf_ff': 'RF',
            'ME_usd': 'SIZE', 'ROE_usd': 'ROE', 'IA_usd': 'IA'
        })

        # Merge portfolio returns with factor data
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

            res_ff3 = sm.OLS(y, x_ff3).fit()
            res_ff5 = sm.OLS(y, x_ff5).fit()
            res_q = sm.OLS(y, x_q).fit()

            results[q] = {
                'FF3': res_ff3,
                'FF5': res_ff5,
                'Q': res_q
            }

        long_short_res = self.compute_long_short_regression(port_returns, df_factors)

        return results, long_short_res


class FamaMacBethAnalyzer(BaseAnalyzer):
    """
    Performs Fama-MacBeth regression of returns on signals.
    """

    def analyze(self) -> tuple:
        # Merge signal and CRSP data
        df_signal = self.prepare_signal_df(self.data)
        df_crsp = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'})

        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])

        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')

        # Convert percentage in decimal
        df_crsp['RET_USD'] = df_crsp['RET_USD'] / 100

        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')

        # Drop rows with missing returns or signal
        merged = merged.dropna(subset=['RET_USD', 'signal'])

        # Store results
        betas = []
        dates = []

        # First-stage: Run time-series of cross-sectional regressions
        for date, group in merged.groupby('year_month'):
            if group['signal'].nunique() > 1:  # Avoid perfect collinearity
                X = sm.add_constant(group['signal'])
                y = group['RET_USD']
                res = sm.OLS(y, X).fit()
                betas.append(res.params['signal'])
                dates.append(date)

        # Second-stage: average and t-test
        betas_series = pd.Series(betas, index=pd.PeriodIndex(dates, freq='M'))
        beta_mean = betas_series.mean()
        beta_std = betas_series.std()
        n = len(betas_series)
        t_stat = beta_mean / (beta_std / np.sqrt(n))

        return beta_mean, beta_std, t_stat, n
