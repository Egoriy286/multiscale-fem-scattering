FROM dolfinx/lab:stable@sha256:ce646f5b359e60bdd3af040fc1240236df6281786a96ff86274a2e0b744bea25

ENV PETSC_ARCH=linux-gnu-complex128-32 \
    PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.12/dist-packages:/usr/local/lib \
    LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib \
    PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig \
    CMAKE_PREFIX_PATH=/usr/local/dolfinx-complex/lib/cmake

RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb \
    && rm -rf /var/lib/apt/lists/*

# Existing notebooks use pyvista.start_xvfb(), available in PyVista 0.46.
RUN python3 -m pip install --no-cache-dir meshio==5.3.5 h5py==3.16.0 'pyvista[jupyter]==0.46.5' \
    && python3 -m ipykernel install --sys-prefix --name=python3 \
       --display-name="Python (DOLFINx Complex)"
