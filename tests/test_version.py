"""Smoke test: the package imports and exposes a version string."""

import fiber_mosaic


def test_version_is_string():
    """`__version__` is a non-empty string."""
    assert isinstance(fiber_mosaic.__version__, str)
    assert fiber_mosaic.__version__
