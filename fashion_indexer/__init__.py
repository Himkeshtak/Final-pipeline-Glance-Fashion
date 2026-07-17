"""Multi-expert fashion image indexing package."""

from fashion_indexer.config import COLLECTION_NAME, ModelSettings, PipelineSettings
from fashion_indexer.experts import NamedEmbeddings

__all__ = [
    "COLLECTION_NAME",
    "ModelSettings",
    "NamedEmbeddings",
    "PipelineSettings",
]

__version__ = "1.0.0"
