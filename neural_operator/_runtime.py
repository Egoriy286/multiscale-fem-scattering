"""Keep the local ARM Linux PyTorch wheel compatible with system PETSc."""

import ctypes
import ctypes.util
import platform
import sys


def prepare_native_runtime():
    # The CPU torch wheel bundles an older libgfortran with the same SONAME.
    # Load the system runtime first; PETSc requires its GFORTRAN_10 symbols.
    if sys.platform == "linux" and platform.machine() == "aarch64":
        library = ctypes.util.find_library("gfortran")
        if library:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
