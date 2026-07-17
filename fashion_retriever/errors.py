"""Expected failures raised by the retrieval pipeline."""


class RetrieverError(Exception):
    """Base class for expected retrieval failures."""


class RetrieverConfigurationError(RetrieverError):
    """Raised when runtime retrieval configuration is invalid."""


class QueryPlanError(RetrieverError):
    """Raised when an LLM response cannot satisfy the query-routing contract."""


class LLMServiceError(RetrieverError):
    """Raised when the configured LLM service cannot produce a query plan."""


class LLMTimeoutError(LLMServiceError):
    """Raised when an LLM request exceeds its configured timeout."""


class TextEncodingError(RetrieverError):
    """Raised when a text expert cannot produce a valid query vector."""


class RetrievalDatabaseError(RetrieverError):
    """Raised when Qdrant cannot execute or validate a retrieval operation."""


class EmptyCollectionError(RetrievalDatabaseError):
    """Raised when retrieval is attempted before any images have been indexed."""
