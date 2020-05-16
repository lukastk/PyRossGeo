import numpy as np
import os, sys 
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name='PyRossGeo',
    version='1.0.0',
    url='https://github.com/lukastk/PyRossGeo.git',
    author='The PyRossGeo team',
    license='MIT',
    description='python library for spatial numerical simulation of infectious diseases',
    long_description='pyrossgeo is a library for spatially resolved mathematical modelling of infectious diseases',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize([ Extension('pyrossgeo/*', ["pyrossgeo/*.pyx"],
        include_dirs=[np.get_include()],
        ), Extension('pyrossgeo.mft', ["pyrossgeo/mft/*.pyx"],
        include_dirs=[np.get_include()],
        )],
        compiler_directives={"language_level": 3},#sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyrossgeo'],
    package_data={'pyrossgeo': ['*.pxd']},
)
