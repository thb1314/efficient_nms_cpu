from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

curdir = os.path.dirname(os.path.realpath(__file__))

setup(
    name="cython_nms",
    version="1.0",
    ext_modules=cythonize(os.path.join(curdir, 'cython_nms.pyx')),
    include_dirs=[np.get_include()]    
)
