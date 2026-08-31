"""Optional C++ extension. Install still succeeds if a compiler is missing."""

from __future__ import annotations

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext as _pybind_build
except ImportError:
    Pybind11Extension = None  # type: ignore[misc, assignment]
    _pybind_build = _build_ext


class OptionalBuildExt(_pybind_build):  # type: ignore[misc]
    def build_extensions(self) -> None:
        try:
            super().build_extensions()
        except Exception as exc:
            print("GIG C++ extension not built (Python fallback active):", exc)
            self.extensions = []


ext_modules = []
if Pybind11Extension is not None:
    ext_modules = [
        Pybind11Extension(
            "gig._speed",
            ["src/cpp/speed.cpp"],
            cxx_std=17,
        )
    ]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
