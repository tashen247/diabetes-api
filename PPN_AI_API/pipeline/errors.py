"""Failures the screening pipeline knows how to route."""

from __future__ import annotations


class ScreeningError(Exception):
    """Base error for intake, workers, and reporting."""


class BadImageError(ScreeningError):
    """The image cannot be read. This is permanent and goes to the dead-letter queue."""

    def __init__(self, message: str, reason: str = "bad_image"):
        super().__init__(message)
        self.reason = reason


class TransientModelError(ScreeningError):
    """The model or GPU is unavailable. The message is delayed and retried."""

    def __init__(self, message: str, reason: str = "model_down"):
        super().__init__(message)
        self.reason = reason


class BrokerUnavailable(ScreeningError):
    """RabbitMQ cannot accept the publication. The study stays stored."""


class MissingImage(ScreeningError):
    """An image id in the message is not in the image store."""
