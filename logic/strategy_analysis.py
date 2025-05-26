import pandas as pd
from abc import ABC, abstractmethod


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
    def analyze(self) -> str:
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

    def analyze(self) -> str:
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
            str: A multiline string reporting the average next-month return for each signal quintile.
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
        
        result_str = "Durchschnittliche Folgemonatsrendite nach Signal-Quintilen:\n"
        for quintile, avg_ret in avg_returns.items():
            result_str += f"Quintil {quintile}: {avg_ret:.4f}\n"
        
        return result_str