from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd


@dataclass
class SharedContext:
    """Container for shared datasets used by analyzers.

    Attributes
    ----------
    crsp : pd.DataFrame
        The main CRSP-like security-level panel (e.g., prices/returns).
    factors : pd.DataFrame
        Factor returns (e.g., Fama-French, momentum, etc.).
    fm : pd.DataFrame
        Cross-sectional or time-series regression outputs (Fama-MacBeth or similar).
    """

    crsp: pd.DataFrame
    factors: pd.DataFrame
    fm: pd.DataFrame


@dataclass
class AnalyzerOutput:
    """Standardized output understood by the downstream "Composer".

    Use this to return both human-readable artifacts (LaTeX blocks) and any
    raw text files that should be bundled into an export archive.

    Attributes
    ----------
    name : str
        Display name used in logs or protocol outputs.
    raw_texts : Dict[str, str]
        Mapping of filename to content for any auxiliary plain-text files.
    latex_blocks : List[str]
        Ordered list of LaTeX fragments to be injected into the report.
    meta : Dict[str, Any] | None
        Optional metadata for programmatic consumers (e.g., summary stats).
    """

    name: str  # Display name in the protocol/log
    raw_texts: Dict[str, str]  # filename -> content
    latex_blocks: List[str]  # LaTeX blocks appearing in order in the report
    meta: Dict[str, Any] | None = None  # Optional metadata


class BaseAnalyzer(ABC):
    """Abstract analyzer for stock return predictors.

    Subclasses should implement :meth:`analyze` and :meth:`generate_output`.
    The constructor provides common access to the shared datasets and the
    input DataFrame that contains the signal being analyzed.

    Class Attributes
    ----------------
    ENABLED : bool
        Whether this analyzer should be discovered and run.
    ORDER : int
        Sort order for sections in the report.
    TITLE : str
        Section title for the PDF/report.
    """

    ENABLED: bool
    ORDER: int  
    TITLE: str  

    def __init__(self, ctx: SharedContext, df_input: pd.DataFrame, signal_name: str) -> None:
        """Initialize the analyzer with shared context and inputs.

        Parameters
        ----------
        ctx : SharedContext
            Shared datasets passed to all analyzers.
        df_input : pd.DataFrame
            Input data which contains the user signal.
        signal_name : str
            Human-readable name of the signal being analyzed.
        """
        self.data = df_input
        self.crsp = ctx.crsp
        self.factors = ctx.factors
        self.fm = ctx.fm
        self.signal_name = signal_name

    @abstractmethod
    def analyze(self) -> Any:
        """Run the core analysis and return intermediate results.

        Implementations may return any structure that is helpful for
        :meth:`generate_output`, such as summary tables or model fits.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_output(self) -> AnalyzerOutput:
        """Execute the full workflow and build the :class:`AnalyzerOutput`.

        Returns
        -------
        AnalyzerOutput
            A standardized bundle of LaTeX blocks, raw texts, and metadata
            that downstream components can consume to assemble the report.
        """
        raise NotImplementedError


class AutoRegistered:
    """Mixin that auto-registers concrete subclasses.

    Any subclass with ``AUTO_REGISTER = True`` (the default) and without
    abstract methods will be added to the class-level registry.

    Attributes
    ----------
    _registry : List[type]
        Discovered concrete subclasses in import order.
    """

    _registry: List[type] = []

    def __init_subclass__(cls, **kwargs): 
        """Hook invoked on subclass creation; registers concrete analyzers."""
        super().__init_subclass__(**kwargs)
        # Skip abstract classes/adapters:
        if getattr(cls, "AUTO_REGISTER", True) and not getattr(cls, "__abstractmethods__", None):
            AutoRegistered._registry.append(cls)
