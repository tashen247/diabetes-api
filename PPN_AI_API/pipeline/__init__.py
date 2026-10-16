"""Resilient fundus screening: one study, eight models.

A model being down delays its result. The result is not dropped, and the
report stays PENDING until all eight are in.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
