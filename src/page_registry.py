"""
Re-export shim for PAGE_REGISTRY and get_compatible_models.
"""

from src.components.page_registry import PAGE_REGISTRY, get_compatible_models

__all__ = ["PAGE_REGISTRY", "get_compatible_models"]
