import os
import shutil
from subprocess import check_call, call

try:
    os.remove('GaussNewtonRoutines_Cython.c')
except OSError:
    pass

try:
    os.remove('GaussNewtonRoutines_Cython.so')
except OSError:
    pass

shutil.rmtree('build', ignore_errors=True)

check_call(r'python setup.py build_ext --inplace'.split())

import GaussNewtonRoutines_Cython # ignore - will be dynamically linked

print("*" * 80)
print(GaussNewtonRoutines_Cython.hello_world())
print("*" * 80)