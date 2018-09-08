from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("GaussNewtonRoutines_Cython", ["VisualOdometry/GaussNewtonRoutines_Cython.pyx"]),
    ]

#setup(
#  name = 'GaussNewtonRoutines_Cython',
#  ext_modules = cythonize(extensions),
#)

setup(
  ext_modules = cythonize(extensions)
)