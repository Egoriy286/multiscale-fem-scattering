# Локальные neural operators для Helmholtz scattering

Полный локальный workflow: генерация FEM-данных, диагностика exact/randomized SVD, обучение SVD-SNO, Chebyshev-SNO и FNO, самостоятельный inference, сравнение с FEM/GMsFEM и отдельный высокочастотный эксперимент. Colab, Google Drive и JAX не требуются. Математические определения и ограничения: [methods.md](docs/methods.md), первичные статьи и протокол: [references.md](docs/references.md), аудит прежнего notebook: [change_log.md](docs/change_log.md).

Это небольшое воспроизводимое исследование, а не обещание высокой точности. В первом SVD pilot выполнено 58 эпох, лучшая эпоха — 8; средняя test FEM L2 ошибка ≈0.6858, POD reconstruction error ≈0.02668. Значительный разрыв показывает ошибку предсказания коэффициентов сверх ошибки сжатия. Все восемь notebooks выполнены; итоговые результаты и ограничения приведены в [отчёте локальных запусков](docs/local_results.md). Данные и шесть обученных моделей сохранены в `data/` и `runs/`.

## Рекомендуемое окружение

Запускайте команды из корня репозитория в уже существующем Docker service `dolfinx-complex`. Он содержит complex DOLFINx/PETSc и Gmsh. На текущем локальном host Python 3.15 нет NumPy; host-интерпретатор не используется для этого workflow.

```bash
docker compose up -d dolfinx-complex
docker compose exec -T dolfinx-complex python3 -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

CPU PyTorch 2.15.0 уже установлен в используемом service; SciPy, Matplotlib и nbclient также доступны. При необходимости остальные Python-зависимости можно установить так:

```bash
docker compose exec -T dolfinx-complex python3 -m pip install -r neural_operator/requirements.txt
```

Установка пакетов внутри контейнера сохраняется при stop/start того же контейнера, но не при его пересоздании или rebuild. После пересоздания повторите команду установки PyTorch и при необходимости requirements. Корневой Dockerfile этим workflow не изменяется. Для notebooks используйте Jupyter из этого контейнера и его kernel; адрес/token показывает `docker compose logs dolfinx-complex`.

На ARM Linux пакет при импорте `neural_operator` заранее загружает системную `libgfortran` через `_runtime.py`. Это устраняет обнаруженный конфликт: bundled runtime CPU PyTorch не предоставлял требуемые PETSc символы `GFORTRAN_10`. Если `torch` уже импортирован до `neural_operator`, перезапустите kernel и выполняйте notebook с первой ячейки. Исправление находится в Python-пакете, менять системные библиотеки не нужно.

Отдельный venv на Python 3.11/3.12 допустим для обучения/инференса над скопированным готовым датасетом и bundle. Для генерации FEM и classical benchmark используйте Docker: complex DOLFINx не устанавливается обычным `pip install dolfinx` из requirements.

## Последовательность notebooks

| Notebook | Назначение |
| --- | --- |
| [Dataset.ipynb](Dataset.ipynb) | Общая конформная Gmsh-сетка, физические DG0 материалы, FEM-снимки и групповой split |
| [SVD-data-checks.ipynb](SVD-data-checks.ipynb) | Exact/randomized SVD, спектры, ошибки реконструкции, orthogonality/subspace checks и grid transfer |
| [SVD-SNO.ipynb](SVD-SNO.ipynb) | Train-only weighted POD, rank selection на val, coefficient network и test FEM L2 |
| [Chebyshev-SNO.ipynb](Chebyshev-SNO.ipynb) | DCT-I, ряд Чебышёва и нелинейность в пространстве коэффициентов |
| [FNO.ipynb](FNO.ipynb) | FFT spectral layers и физические нелинейности на регулярной сетке |
| [Inference.ipynb](Inference.ipynb) | Загрузка готового `model.pt`, single/batch prediction и save/load parity |
| [Comparison.ipynb](Comparison.ipynb) | Paired FEM/GMsFEM/neural comparison и отдельно полный neural test |
| [High-frequency.ipynb](High-frequency.ipynb) | Frozen low-band модели, самостоятельное обучение на k∈[4,8] и FEM mesh refinement |

Первый запуск training notebook обучает модель. Повторный вызов `run_experiment(..., retrain=False)` загружает завершённый run только при совпадении всей конфигурации и dataset fingerprint. При несовпадении возникает явная ошибка: выберите другую папку либо `retrain=True`. `retrain=True` запускает обучение заново; автоматического resume optimizer/незавершённой эпохи нет. Сначала выполняйте notebooks по порядку; inference после создания bundle не требует обучения.

## CLI без Jupyter

Эквивалентный pilot использует 48 сред × 4 направления, h=0.1, coarse grid 4×4, k∈[1,2]. Обучение по умолчанию: CPU, два PyTorch threads, максимум 200 эпох, patience=50, min_epochs=50, лучший checkpoint по validation. Epoch cap не означает подтверждённую сходимость.

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.run data --n-media 48 --n-directions 4 --h 0.1
docker compose exec -T dolfinx-complex python3 -m neural_operator.run svd
docker compose exec -T dolfinx-complex python3 -m neural_operator.run train --method all --epochs 200
docker compose exec -T dolfinx-complex python3 -m neural_operator.run compare --method all
docker compose exec -T dolfinx-complex python3 -m neural_operator.run high-frequency --method all --epochs 200
```

Для отдельного метода используйте `--method svd`, `chebyshev` или `fno`; для свежего переобучения — `--retrain`. Параметр `--dataset PATH` выбирает каталог с `fem_dataset.npz` и `geometry.npz`; `--output PATH` меняет папку результатов выбранного действия. CLI `compare`/`high-frequency` берут low-band модели из стандартного `neural_operator/runs/low_frequency/{method}`: для нестандартных путей используйте Python API/notebook с явным словарём run directories.

```python
from neural_operator.data import load_dataset
from neural_operator.training import TrainConfig, run_experiment

dataset = load_dataset("neural_operator/data/low_frequency")
predictor, metrics, history = run_experiment(
    dataset, "neural_operator/runs/low_frequency/svd",
    TrainConfig(method="svd", epochs=200, threads=2),
    retrain=False,
)
```

## Данные и сохранённые модели

Новый dataset сохраняет узловые A,n, комплексное `u_scat`, θ,k, `medium_id`, split, точные `A_cells`/`n_cells`, геометрию и исходную `.msh`. Среды целиком разделены на train/val/test; в pilot это 32/8/8 сред и 128/32/32 снимка. Базисы и normalization stats строятся только по train, rank и лучшая эпоха выбираются по val.

Старый `fem_dataset.npz` пригоден для neural workflow при наличии совместимого `geometry.npz` с `dof_coords` и `triangles`. Для честного classical benchmark нужны новые точные DG0 arrays, physical cell tags и `.msh`; восстановление материалов из сглаженных узловых значений не подменяет исходный FEM.

Каждая model run directory содержит `model.pt`, `metrics.json`, `history.json` и ошибки тестовых снимков. Bundle хранит architecture config, state_dict, preprocessing/bases, normalization и геометрию; загрузка использует `torch.load(..., weights_only=True)`. Для inference не нужны training data, FEM-решатель или повторный fit базисов, но нужны входные A,n в исходном порядке FEM-узлов и k,θ:

```python
from neural_operator.training import load_predictor

predictor = load_predictor("neural_operator/runs/low_frequency/svd/model.pt")
# A_input, n_input: массивы [n_nodes] в сохранённом порядке DOF.
u_scat = predictor.predict(A_input, n_input, theta=0.3, k0=1.5)
```

У `predict_dataset` есть проверка совпадения coordinates/connectivity. При прямом `predict` соответствие порядка узлов обеспечивает вызывающий код. Предсказание на новой сетке требует явного переноса полей; reshape FEM-DOF в квадратную картинку недопустим.

## Как читать результаты

Основная ошибка — consistent P1 FEM L2 комплексного поля. SVD-базисы строятся в lumped-weighted норме, а loss вычисляется после реконструкции в consistent норме. Поэтому POD reconstruction error — диагностическая ошибка сжатия, не строгая lower bound для consistent loss. Входные A,n и выходные Re/Im сохраняют собственные numerical ranks; выходные каналы дополняются нулями до общего размера сети. Нормированные коэффициенты возвращаются к физическому масштабу при decoding.

Chebyshev и FNO используют явные FEM↔tensor transfers; отдельно измеряются transfer и compression errors. SVD-вариант — локальная POD адаптация coefficient-space SNO; это не заявленное точное воспроизведение Fanaskov–Oseledets или PODNO.

`Comparison` сохраняет `paired_samples.csv`/`paired_summary.csv` на трёх held-out средах, по одному направлению каждой, и отдельно `neural_full_test.csv`/`neural_full_test_summary.csv` на всех 32 neural test снимках. Эти разные множества нельзя смешивать в одной сравнительной средней. GMsFEM строит базисы для соответствующей среды/k через существующий `gmsfem.offline`; времена assembly, offline basis construction, training и online prediction записываются раздельно.

Высокочастотный workflow сравнивает все три frozen low-band модели с отдельно переобученными на k∈[4,8]. Он сохраняет ошибки по частотам, POD projection diagnostics, mesh resolution и refinement study для трёх held-out сред с наибольшими k. Совпадающие media IDs между полосами сохраняют split. Число узлов на длину волны и refinement difference — диагностика качества FEM reference, а не доказательство точности нейросети; OOD и retrained режимы читаются отдельно.

Проверки модулей:

```bash
docker compose exec -T dolfinx-complex python3 -m pytest neural_operator/tests -q
```
