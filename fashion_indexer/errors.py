"""Domain-specific exceptions raised by the indexing pipeline."""


class FashionIndexerError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(FashionIndexerError):
    """Raised when runtime configuration is invalid."""


class ImageDirectoryError(FashionIndexerError):
    """Raised when the configured image directory cannot be read."""


class InvalidImageError(FashionIndexerError):
    """Raised when an image is corrupt, unsafe, or cannot be decoded."""


class DuplicateImageError(FashionIndexerError):
    """Raised when an image has the same raw-byte MD5 as an earlier image."""


class DeviceSelectionError(FashionIndexerError):
    """Raised when a requested accelerator or precision is unavailable."""


class ModelInitializationError(FashionIndexerError):
    """Raised when an expert model or its processor cannot be loaded."""


class ModelInferenceError(FashionIndexerError):
    """Raised when a model cannot produce a valid embedding."""


class DatabaseError(FashionIndexerError):
    """Raised for Qdrant initialization or operation failures."""


class DatabaseConflictError(DatabaseError):
    """Raised when another process or incompatible storage conflicts with Qdrant."""


class CollectionSchemaError(DatabaseError):
    """Raised when an existing collection has an incompatible vector schema."""
