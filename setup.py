from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules = cythonize("pyrossgeo/*.pyx", language_level=3),
    zip_safe=False
)
