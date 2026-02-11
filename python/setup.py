from __future__ import annotations

from pathlib import Path
from setuptools import Extension, setup
from pybind11.setup_helpers import build_ext
import pybind11

ROOT = Path(__file__).resolve().parent.parent

ext_modules = [
    Extension(
        "last_vector_core",
        [
            str(ROOT / "cpp/src/python_bindings.cpp"),
            str(ROOT / "cpp/src/sim.cpp"),
            str(ROOT / "cpp/src/observation.cpp"),
            str(ROOT / "cpp/src/upgrades.cpp"),
        ],
        include_dirs=[str(ROOT / "cpp/include"), pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++20"],
    )
]

setup(
    name="last-vector-core",
    version="0.1.0",
    description="Python bindings for Last-Vector deterministic simulator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
