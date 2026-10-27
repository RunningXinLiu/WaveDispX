from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys


extensions = [
    Pybind11Extension(
        "wavedispx._dwm_core",
        ["cpp/dwm_core.cpp"],
        cxx_std=17,
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
        define_macros=[("_USE_MATH_DEFINES", "1")],
    )
]


setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})

