import numpy as np
import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod
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

    def compute_long_short_regression(self, port_returns: pd.DataFrame, df_factors: pd.DataFrame) -> str:
        """
        Computes long-short portfolio (Slice 10 - Slice 1) returns and regresses on FF3, FF5, Q models.
        """
        pivot = port_returns.pivot(index='year_month', columns='slice', values='port_ret')

        if 1 not in pivot.columns or 10 not in pivot.columns:
            return "Long-Short regression skipped: Slice 1 or 10 not found."

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

        result = "\n======= Long-Short Portfolio (Slice 10 - Slice 1) =======\n"
        result += "\n--- FF3 Regression ---\n" + res_ff3.summary().as_text()
        result += "\n\n--- FF5 Regression ---\n" + res_ff5.summary().as_text()
        result += "\n\n--- Q-Factor Regression ---\n" + res_q.summary().as_text()

        return result
    
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

    def analyze(self) -> Tuple[str, str, str, str]:
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
                'FF3': res_ff3.summary().as_text(),
                'FF5': res_ff5.summary().as_text(),
                'Q': res_q.summary().as_text()
            }

        # Format return strings
        ff3_str, ff5_str, q_str, long_short_str = "", "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Slice {q} ---\n{res['FF3']}\n"
            ff5_str += f"\n--- Slice {q} ---\n{res['FF5']}\n"
            q_str += f"\n--- Slice {q} ---\n{res['Q']}\n"

        long_short_str = self.compute_long_short_regression(port_returns, df_factors)

        return ff3_str, ff5_str, q_str, long_short_str


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
                'FF3': res_ff3.summary().as_text(),
                'FF5': res_ff5.summary().as_text(),
                'Q': res_q.summary().as_text()
            }

        # Format results into strings
        ff3_str, ff5_str, q_str, long_short_str = "", "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Slice {q} ---\n{res['FF3']}\n"
            ff5_str += f"\n--- Slice {q} ---\n{res['FF5']}\n"
            q_str += f"\n--- Slice {q} ---\n{res['Q']}\n"

        long_short_str = self.compute_long_short_regression(port_returns, df_factors)

        return ff3_str, ff5_str, q_str, long_short_str


class FamaMacBethAnalyzer(BaseAnalyzer):
    """
    Performs Fama-MacBeth regression of returns on signals.
    """

    def analyze(self) -> str:
        # Merge signal and CRSP data
        df_signal = self.prepare_signal_df(self.data)
        df_crsp = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'})

        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])

        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')

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

        result_str = "Fama-MacBeth Regression Result\n" \
                     "------------------------------\n" \
                     f"Mean Beta: {beta_mean:.4f}\n" \
                     f"Std Dev:   {beta_std:.4f}\n" \
                     f"T-Stat:    {t_stat:.4f}\n" \
                     f"N Months:  {n}\n"

        return result_str
