import numpy                                                                                                    
import os, re, sys
#from distutils.core import setup                                                                                      
#from distutils.extension import Extension                                                                             
from setuptools import setup, Extension
from Cython.Build import cythonize                                                                                    
import Cython.Compiler.Options                                                                                        
Cython.Compiler.Options.annotate = True                                                                               
                                                                                                                      
                                                                                                                      
if 'darwin'==(sys.platform).lower():                                                                                  
    extension1 = Extension('pyrossgeo/*', ['pyrossgeo/*.pyx'],                                                           
        include_dirs=[numpy.get_include()],                                                                           
        extra_compile_args=['-mmacosx-version-min=10.9'],                                                             
        extra_link_args=['-mmacosx-version-min=10.9'],                                                                
    )                                                                                                                 
else:                                                                                                                 
    extension1 = Extension('pyrossgeo/*', ['pyrossgeo/*.pyx'],                                                           
        include_dirs=[numpy.get_include()],                                                                           
    )                                                                                                                 
                                                                                                                      
if 'darwin'==(sys.platform).lower():                                                                                  
    extension2 = Extension('pyrossgeo.mft.*', ['pyrossgeo/mft/*.pyx'],                                                       
        include_dirs=[numpy.get_include()],                                                                           
        extra_compile_args=['-mmacosx-version-min=10.9'],                                                             
        extra_link_args=['-mmacosx-version-min=10.9'],                                                                
    )                                                                                                                 
else:                                                                                                                 
    extension2 = Extension('pyrossgeo.mft.*', ['pyrossgeo/mft/*.pyx'],
        include_dirs=[numpy.get_include()],                                                                           
    )                                                                                                                 
                                                                                                                      
                                                                                                                      
with open("README.md", "r") as fh:
    long_description = fh.read()


cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'pyrossgeo', '__init__.py')) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError('Unable to find own __version__ string')


setup(                                                                                                                
    name='PyRossGeo',                                                                                                 
    version=version,                                                                                                  
    url='https://github.com/lukastk/PyRossGeo.git',                                                                   
    project_urls={
            "Documentation": "https://pyrossgeo.readthedocs.io",
            "Source": "https://github.com/lukastk/PyRossGeo.git",
            },
    author='The PyRossGeo team',                                                                                      
    license='MIT',                                                                                                    
    description='python library for spatial numerical simulation of infectious diseases',                             
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',                                 
    ext_modules=cythonize([extension1, extension2],                                                                   
        compiler_directives={"language_level": 3},                                              
        ),                                                                                                            
    libraries=[],                                                                                                     
    packages=['pyrossgeo', 'pyrossgeo/mft'],                                                                                           
    package_data={
                'pyrossgeo': ['*pyx', '*.pxd'],                                                                            
                'pyrossgeo/mft': ['*pyx', '*.pxd']},                                                                            
    classifiers=[
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.7',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Intended Audience :: Science/Research',
                'Intended Audience :: Education',
                ],
)                                                                                                                     
