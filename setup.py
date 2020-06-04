import numpy as np                                                                                                    
import os, sys                                                                                                        
from distutils.core import setup                                                                                      
from Cython.Build import cythonize                                                                                    
from distutils.extension import Extension                                                                             
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
    extension2 = Extension('pyrossgeo/mft/*', ['pyrossgeo/mft/*.pyx'],                                                       
        include_dirs=[numpy.get_include()],                                                                           
        extra_compile_args=['-mmacosx-version-min=10.9'],                                                             
        extra_link_args=['-mmacosx-version-min=10.9'],                                                                
    )                                                                                                                 
else:                                                                                                                 
    extension2 = Extension('pyrossgeo/mft/*', ['pyrossgeo/mft/*.pyx'],                                                          
        include_dirs=[numpy.get_include()],                                                                           
    )                                                                                                                 
                                                                                                                      
                                                                                                                      
setup(                                                                                                                
    name='PyRossGeo',                                                                                                 
    version='1.0.0',                                                                                                  
    url='https://github.com/lukastk/PyRossGeo.git',                                                                   
    author='The PyRossGeo team',                                                                                      
    license='MIT',                                                                                                    
    description='python library for spatial numerical simulation of infectious diseases',                             
    long_description='pyrossgeo is a library for spatially resolved mathematical modelling of infectious diseases',   
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',                                 
    ext_modules=cythonize([extension1, extension2],                                                                   
        compiler_directives={"language_level": 3},#sys.version_info[0]},                                              
        ),                                                                                                            
    libraries=[],                                                                                                     
    packages=['pyrossgeo'],                                                                                           
    package_data={'pyrossgeo': ['*.pxd']},                                                                            
)                                                                                                                     
