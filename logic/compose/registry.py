import pkgutil
import importlib
from typing import List, Type, Iterable

from logic.analyzers.base import AutoRegistered, BaseAnalyzer


class Registry:
    @staticmethod
    def list_all_analyzers(package: str = "logic.analyzers") -> List[Type[BaseAnalyzer]]:
        """Alle registrierten Analyzer zurückgeben – unabhängig von ENABLED."""
        pkg = importlib.import_module(package)
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(modname)

        return sorted(AutoRegistered._registry, key=lambda c: getattr(c, "ORDER", 100))


    @staticmethod
    def discover_selected_analyzers(selected_fullnames: Iterable[str], package: str = "logic.analyzers") -> List[Type[BaseAnalyzer]]:
        """Nur die vom User ausgewählten Analyzer (per qualifiziertem Klassenname) liefern.

        Args:
        selected_fullnames: Iterable voll qualifizierter Klassennamen
        (z. B. "logic.analyzers.performance.PerformanceAnalyzer")
        """
        selected_set = set(selected_fullnames or [])
        all_cls = Registry.list_all_analyzers(package)
        picked = [c for c in all_cls if f"{c.__module__}.{c.__name__}" in selected_set]
        return picked
