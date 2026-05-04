# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
