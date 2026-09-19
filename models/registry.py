"""
Model Registry Submodule (`models/registry.py`).
Scans models/ directory using inspect and importlib to auto-discover all BaseBlastModel subclasses.
Exposes singleton get_registry() for Streamlit UI dropdown auto-discovery.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Tuple, Optional
from models.base import BaseBlastModel, ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Auto-discovering ModelRegistry scanning models/ directory for BaseBlastModel subclasses.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._registry: Dict[str, Type[BaseBlastModel]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._load_errors: List[Dict[str, str]] = []

    def discover(self) -> None:
        """Scan models_dir and register all BaseBlastModel subclasses."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory '{self.models_dir}' does not exist.")
            return

        for py_file in self.models_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            if "test" in py_file.name:
                continue
            module_path = self._file_to_module(py_file)
            self._try_register_module(module_path)

    def _try_register_module(self, module_path: str) -> None:
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            logger.warning(f"Failed to import {module_path}: {e}")
            self._load_errors.append({"module": module_path, "error": str(e)})
            return

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseBlastModel)
                and obj is not BaseBlastModel
                and obj.__module__ == module_path):
                try:
                    metadata = obj.get_metadata()
                    self._registry[metadata.name] = obj
                    self._metadata[metadata.name] = metadata
                    logger.info(f"Registered model: {metadata.name}")
                except Exception as e:
                    logger.warning(f"Failed to register {name}: {e}")
                    self._load_errors.append({"module": module_path, "class": name, "error": str(e)})

    def _file_to_module(self, py_file: Path) -> str:
        rel = py_file.relative_to(self.models_dir.parent)
        return str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")

    def list_models(self) -> List[ModelMetadata]:
        return list(self._metadata.values())

    def get_available_models(self) -> List[Tuple[str, str]]:
        """Returns list of (display_name, model_name) tuples for Streamlit dropdown selection."""
        return sorted([(meta.display_name, meta.name) for meta in self._metadata.values()], key=lambda x: x[0])

    def get_model(self, name: str) -> Type[BaseBlastModel]:
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not registered. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def get_metadata(self, name: str) -> ModelMetadata:
        if name not in self._metadata:
            raise KeyError(f"Metadata for model '{name}' not found.")
        return self._metadata[name]

    def get_load_errors(self) -> List[Dict[str, str]]:
        return self._load_errors


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.discover()
    return _registry


def register_model(name: str):
    """Decorator compatibility helper."""
    def decorator(cls: Type[BaseBlastModel]):
        reg = get_registry()
        meta = cls.get_metadata()
        reg._registry[meta.name] = cls
        reg._metadata[meta.name] = meta
        return cls
    return decorator
