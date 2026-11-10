FROM multiscale-fem-scattering/dolfinx-complex:local

# The base image supplies complex DOLFINx/PETSc, Gmsh and Jupyter.
# Install the neural dependencies into the SAME /dolfinx-env interpreter.
COPY requirements.txt /tmp/reference-sno-requirements.txt
RUN python3 -m pip install --no-cache-dir 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir -r /tmp/reference-sno-requirements.txt \
    && python3 -c "import dolfinx, gmsh, torch; from petsc4py import PETSc; import numpy as np; assert np.dtype(PETSc.ScalarType).kind == 'c'"
