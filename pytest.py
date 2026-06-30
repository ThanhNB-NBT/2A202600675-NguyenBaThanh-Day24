from __future__ import annotations


def fixture(func=None, **kwargs):
    def decorate(f):
        f.__pytest_fixture__ = True
        return f

    return decorate(func) if func is not None else decorate
