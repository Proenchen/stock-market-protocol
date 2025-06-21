import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod
from typing import Tuple, Any


class BaseAnalyzer(ABC):
    """Implements an abstract analyzer for stock return predictors."""

    def __init__(self, input_file: pd.DataFrame) -> None:
        """Creates a new abstract analyzer.
        
        Args:
            input_file (pd.DataFrame): Data frame of the input file (.csv or .xlsx) 
                                       which contains data corresponding to the portfolio strategy.
        """
        self.data = input_file

    @abstractmethod
    def analyze(self) -> Any:
        """
        Abstract method to perform analysis on the loaded data.
        """
        pass


class SimpleAnalyzer(BaseAnalyzer):

    def __init__(self, input_file: pd.DataFrame):
        """Creates a new analyzer which performs a simple analysis.
        
        Args:
            input_file (pd.DataFrame): Data frame of the input file (.csv or .xlsx) 
                                       which contains data corresponding to the portfolio strategy.
        """
        super().__init__(input_file)

    def analyze(self) -> Tuple[str, str, str]:
        """
        Analyzes portfolio strategy by computing average next-month returns by quintile.

        This method performs the following steps:
        1. Loads industry return data and given signal data.
        2. Prepares and formats both the signal and return datasets.
        3. Assigns each stock to a signal-based quintile per month.
        4. Merges the signal data with the return data, aligning each stock's signal with its return in the following month.
        5. Calculates the average next-month return for each quintile.
        6. Returns a formatted string showing the average return by quintile.

        Returns:
            Tuple[str, str, str]: 
                - First parameter: A multiline string reporting the average next-month return for each signal quintile.
                - Second parameter: A multiline string reporting the monthly returns per quintile.
                - Third parameter: A multiline string reporting the mapping of each stock to the respective quntile for every month.
        """
        df_signal = self.data
        df_return = pd.read_csv("./data/Industry_returns.csv")
        
        df_signal.rename(columns={'permno': 'permno', 'date': 'date', 'signal': 'signal'}, inplace=True)
        df_return.rename(columns={'DSCD': 'permno', 'DATE': 'date', 'RET_USD': 'ret_usd'}, inplace=True)
        
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_return['date'] = pd.to_datetime(df_return['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')
        
        df_signal['quintile'] = (
            df_signal.groupby('year_month')['signal']
            .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1)
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
        
        portfolio_returns = merged.groupby(['year_month_x', 'quintile'])['ret_usd'].mean().reset_index()
        
        avg_returns = portfolio_returns.groupby('quintile')['ret_usd'].mean()
        

        # Generate Output messages
        # --------------------------
        result_str = "Average next-month returns:\n" \
                     "---------------------------\n"
        for quintile, avg_ret in avg_returns.items():
            result_str += f"Quintile {quintile}: {avg_ret:.4f}\n"
        
        monthly_avg = portfolio_returns.copy()
        monthly_avg['year_month_x'] = monthly_avg['year_month_x'].astype(str)

        monthly_avg_str = "Monthly Average Returns by Quintile:\n" \
                          "------------------------------------\n"
        for _, row in monthly_avg.iterrows():
            monthly_avg_str += f"Month {row['year_month_x']}, Quintile {row['quintile']}: {row['ret_usd']:.4f}\n"


        quintile_mapping = df_signal[['permno', 'year_month', 'quintile']].dropna().copy()
        quintile_mapping['year_month'] = quintile_mapping['year_month'].astype(str)
        mapping_str = "Quintile Mapping by Month:\n" \
                      "--------------------------\n"
        for _, row in quintile_mapping.iterrows():
            mapping_str += f"permno {row['permno']}, month {row['year_month']}, quintile {row['quintile']}\n"

        return result_str, monthly_avg_str, mapping_str
    

class EqualWeightedFactorModelAnalyzer(BaseAnalyzer):
    """
    Analyzer that performs Fama-French (3, 5, Q-Factor) regressions on signal-based portfolios.
    """

    def __init__(self):

        self.data = pd.read_csv("./data/ML_Predictions_Full.csv")
        self.crsp = pd.read_csv("./data/dsws_crsp.csv")
        self.factors = pd.read_csv("./data/Factors.csv")

    def analyze(self) -> Tuple[str, str, str]:
        """
        Performs:
        1. Merge signals with CRSP data.
        2. Build equal-weighted portfolios (quintiles).
        3. Calculate monthly portfolio returns.
        4. Regress each quintile return on FF3, FF5, and Q-Factor models.

        Returns:
            Tuple of results for 3-factor, 5-factor, Q-factor models.
        """
        # Prepare signal data
        df_signal = self.data.rename(columns={'DSCD': 'permno', 'DATE': 'date', 'ENSEMBLE_raw': 'signal'})
        df_signal['date'] = pd.to_datetime(df_signal['date'])
        df_signal['year_month'] = df_signal['date'].dt.to_period('M')

        # Prepare CRSP data
        df_crsp = self.crsp.rename(columns={'DSCD': 'permno', 'DATE': 'date'})
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        df_crsp['year_month'] = df_crsp['date'].dt.to_period('M')

        # Merge signals with CRSP
        merged = pd.merge(df_signal, df_crsp, on=['permno', 'year_month'], how='inner')

        # Build Quintiles
        merged['quintile'] = (
            merged.groupby('year_month')['signal']
            .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1)
        )

        # Portfolio returns (equal-weighted)
        port_returns = (
            merged.groupby(['year_month', 'quintile'])['RET_USD']
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
                                                'ME_usd': 'SIZE', 'ROE_usd': 'ROE'})

        # Merge portfolio returns with factor returns
        model_data = pd.merge(port_returns, df_factors, on='year_month', how='inner')
        model_data['excess_ret'] = model_data['port_ret'] - model_data['RF']

        # Run regressions
        results = {}
        for q in sorted(model_data['quintile'].unique()):
            subset = model_data[model_data['quintile'] == q]
            
            y = subset['excess_ret']
            x_ff3 = sm.add_constant(subset[['MKT', 'SMB', 'HML']])
            x_ff5 = sm.add_constant(subset[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
            x_q = sm.add_constant(subset[['MKT', 'SIZE', 'CMA', 'ROE']])  # Use Q-Factor here if different

            res_ff3 = sm.OLS(y, x_ff3).fit()
            res_ff5 = sm.OLS(y, x_ff5).fit()
            res_q = sm.OLS(y, x_q).fit()

            results[q] = {
                'FF3': res_ff3.summary().as_text(),
                'FF5': res_ff5.summary().as_text(),
                'Q': res_q.summary().as_text()
            }

        # Format return strings
        ff3_str, ff5_str, q_str = "", "", ""
        for q, res in results.items():
            ff3_str += f"\n--- Quintile {q} ---\n{res['FF3']}\n"
            ff5_str += f"\n--- Quintile {q} ---\n{res['FF5']}\n"
            q_str += f"\n--- Quintile {q} ---\n{res['Q']}\n"

        return ff3_str, ff5_str, q_str


if __name__ == '__main__':
    analyzer = EqualWeightedFactorModelAnalyzer()
    ff3, ff5, q = analyzer.analyze()
    print(ff3)
    print("----------------------------------")
    print(ff5)
    print("----------------------------------")
    print(q)