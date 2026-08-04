from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


extensions = [
    Pybind11Extension(
        "wavedispx._dwm_core",
        ["cpp/dwm_core.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]


setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})

