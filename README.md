# multiscale-fem-scattering
Master's degree research on generalized multiscale finite element methods (GMsFEM) for wave scattering problems.

Нейрооператоры SVD-SNO, Chebyshev-SNO и FNO: [структура, данные и запуск](neural_operator/README.md).

## DOLFINx Complex

Окружение Docker содержит DOLFINx 0.11.0.post0, PETSc с типом
`numpy.complex128`, JupyterLab, Gmsh, Meshio, HDF5 и PyVista 0.46.5.
Комплексный режим включён по умолчанию для Python, терминала и Jupyter.

Запуск из папки проекта:

```bash
docker compose up -d --build
docker compose exec dolfinx-complex jupyter server list
```

Вторая команда показывает ссылку с токеном для входа в
[JupyterLab](http://localhost:8888/lab). Папка проекта подключена в
`/workspace/multiscale-fem-scattering`; изменения файлов сохраняются на компьютере.
Для блокнотов выберите ядро **Python (DOLFINx Complex)**.

Для работы в VS Code выполните **Dev Containers: Reopen in Container**.
Терминал контейнера также доступен командой:

```bash
docker compose exec dolfinx-complex bash
```

Проверка комплексного режима:

```bash
docker compose exec dolfinx-complex python3 -c 'from petsc4py import PETSc; print(PETSc.ScalarType)'
```

Остановка: `docker compose stop`. Закрытие VS Code оставляет контейнер работающим.

`laba4/lr4.py` использует библиотеку `dolfin` (FEniCS Legacy) и запускается
в отдельном окружении Legacy.
