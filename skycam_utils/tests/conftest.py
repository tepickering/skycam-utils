"""Shared fixtures for the skycam_utils test suite."""

import sys
from types import ModuleType

import pytest


@pytest.fixture
def patch_alcor(monkeypatch):
    """Replace a name in every ``skycam_utils.alcor`` submodule that binds it.

    ``alcor`` used to be one 5700-line module, so ``monkeypatch.setattr(alcor,
    name, value)`` reached every caller at once -- there was only one namespace.
    Now that it is a package, each submodule holds its own binding (``from .io
    import load_alcor_fits``), and patching only the package's re-export would
    leave the real consumers running the real function. The test would still
    pass, for the wrong reason, which is the failure mode worth engineering
    against.

    So this patches the name wherever it is bound, restoring exactly what the
    single-namespace patch used to mean. Patching nothing is an error rather
    than a silent no-op, since that is how a typo'd name would otherwise turn
    into a vacuously passing test.
    """

    def patch(name, value):
        targets = [
            mod
            for mod_name, mod in sorted(sys.modules.items())
            if isinstance(mod, ModuleType)
            and (mod_name == "skycam_utils.alcor"
                 or mod_name.startswith("skycam_utils.alcor."))
            and name in vars(mod)
        ]
        if not targets:
            raise AttributeError(
                f"no skycam_utils.alcor module binds {name!r}"
            )
        for mod in targets:
            monkeypatch.setattr(mod, name, value)
        return value

    return patch
