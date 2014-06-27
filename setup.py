
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[Extension('features_base',['features_base.pyx'], include_dirs=[np.get_include()])]

setup(
    name='features_base',
    cmdclass={'build_ext':build_ext},
    ext_modules=ext_modules
)
