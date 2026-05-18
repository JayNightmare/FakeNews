"""Dataset loader package.

Each loader implements the DatasetLoader interface and converts
a source dataset into UnifiedRecord instances.
"""

from src.datasets.base import DatasetLoader

__all__ = ["DatasetLoader"]
