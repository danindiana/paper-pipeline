"""Shared domain exceptions with no transport or pipeline dependencies."""


class ShutdownRequested(Exception):
    """Raised when graceful shutdown interrupts active work."""


class OllamaUnavailable(Exception):
    """Raised when the configured Ollama service cannot be reached or used."""


class EmptyGenerationError(RuntimeError):
    """Raised when Ollama completes a generation with no `response` text.

    Distinct from a timeout or transport failure: the HTTP call succeeded
    and streamed to `done`, but no answer text arrived — e.g. the model
    spent its entire context budget on `thinking` content, or hit an
    exhausted/corrupted context checkpoint. Subclasses RuntimeError so
    existing broad `except Exception` handlers still catch it.
    """


class LeaseLostError(RuntimeError):
    """Raised when a worker no longer owns its processing lease."""
