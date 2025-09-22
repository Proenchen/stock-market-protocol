import pkgutil, importlib
from typing import List, Type
from logic.analyzers.base import AutoRegistered, BaseAnalyzer

def discover_analyzers(package="logic.analyzers") -> List[Type[BaseAnalyzer]]:
    # Importiere alle Submodule (nur 1x beim Start)
    pkg = importlib.import_module(package)
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(modname)

    candidates = []
    for cls in AutoRegistered._registry:
        if hasattr(cls, "ENABLED") and getattr(cls, "ENABLED"):
            candidates.append(cls)
    # Reihenfolge konsistent
    candidates.sort(key=lambda c: getattr(c, "ORDER", 100))
    return candidates
