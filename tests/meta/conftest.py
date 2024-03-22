import pytest


def _clear_all():
    import supy.meta as meta

    meta.models.clear()
    meta.events.clear()
    meta.supermodels.clear()
    meta.assimilation_algorithms.clear()


@pytest.fixture(autouse=True)
def _clear_register():
    """Remove all the registered models, supermodels and assimilation algorithms."""

    _clear_all()
    yield
    _clear_all()
