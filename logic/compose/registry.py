import pkgutil
import importlib
from typing import List, Type

from logic.analyzers.base import AutoRegistered, BaseAnalyzer


class Registry:
    @staticmethod
    def discover_analyzers(package: str = "logic.analyzers") -> List[Type[BaseAnalyzer]]:
        """Discover and return all enabled analyzers in a package.

        Parameters
        ----------
        package : str, optional
            The package path (default is ``"logic.analyzers"``).

        Returns
        -------
        list of Type[BaseAnalyzer]
            A list of analyzer classes discovered and enabled, sorted by their
            ``ORDER`` attribute (defaulting to 100 if missing).
        """
        # Import the base package
        pkg = importlib.import_module(package)
        # Walk through submodules and import each one to trigger class registration
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(modname)

        # Collect registered analyzers that are enabled
        candidates: list[Type[BaseAnalyzer]] = []
        for cls in AutoRegistered._registry:
            if hasattr(cls, "ENABLED") and getattr(cls, "ENABLED"):
                candidates.append(cls)

        # Ensure consistent ordering by ORDER attribute
        candidates.sort(key=lambda c: getattr(c, "ORDER", 100))
        return candidates
