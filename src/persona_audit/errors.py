class ManifestError(ValueError):
    """Raised when a versioned data contract is invalid."""


class InferenceError(RuntimeError):
    """Raised when an anchor score cannot be recovered safely."""
