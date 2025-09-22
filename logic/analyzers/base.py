from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

@dataclass
class SharedContext:
    crsp: pd.DataFrame
    factors: pd.DataFrame
    fm: pd.DataFrame


@dataclass
class AnalyzerOutput:
    """Standardisierte Ausgabe, die der Composer versteht."""
    name: str                           # Anzeigename im Protokoll
    # Rohtexte, falls du weiterhin .txt in die ZIP legen willst:
    raw_texts: Dict[str, str]           # filename -> content
    # LaTeX-Blöcke, die im Protokoll erscheinen sollen (in Reihenfolge)
    latex_blocks: List[str]
    # Optionale Metadaten
    meta: Dict[str, Any] | None = None


class BaseAnalyzer(ABC):
    """Abstract analyzer for stock return predictors."""

    ENABLED: bool
    ORDER: int                          # Sortierreihenfolge im Protokoll
    TITLE: str                          # Abschnittstitel im PDF

    def __init__(self, ctx: SharedContext, df_input: pd.DataFrame, signal_name: str) -> None:
        self.data = df_input
        self.crsp = ctx.crsp 
        self.factors = ctx.factors
        self.fm = ctx.fm 
        self.signal_name = signal_name

    @abstractmethod
    def analyze(self) -> Any:
        pass
    
    @abstractmethod
    def generate_output(self) -> AnalyzerOutput:
        """Führt den Algorithmus aus und baut LaTeX-Blöcke + raw_texts."""
        pass


# Auto-Registry per __init_subclass__
class AutoRegistered:
    _registry: List[type] = []
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Abstrakte/Adapter auslassen:
        if getattr(cls, "AUTO_REGISTER", True) and not getattr(cls, "__abstractmethods__", None):
            AutoRegistered._registry.append(cls)
