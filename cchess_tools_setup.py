from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='cchess tools',
    ext_modules=cythonize('cchess_tools.pyx'),
    include_dirs=[numpy.get_include()]
)