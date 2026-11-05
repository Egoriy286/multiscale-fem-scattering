# Нейрооператоры для рассеяния волн

Три метода решают одну FEM-задачу: **SVD-SNO**, **Chebyshev-SNO** и **FNO**. Каждый метод имеет свою папку; данные, обучение и оценка общие. Full и randomized — два алгоритма POD для SVD-SNO. [Карта проекта](docs/structure.md) · [Датасеты](docs/datasets.md) · [Формулы](docs/methods.md).

Готовые randomized SVD-SNO модели дают среднюю test FEM L2 ошибку **0.96%** на k∈[1,2] и **5.54%** на k∈[4,8]. Это удержанные частоты **одной фиксированной среды**, а не перенос на новые материалы. [Результаты и ограничения](randomized/RESULTS.md), [проверка других сеток](docs/cross_mesh_inference.md). Остальные профили обучения оцениваются отдельно.

## Быстрый запуск

Команды выполняются из корня репозитория. Генерация FEM требует существующего Docker service с complex DOLFINx/PETSc и Gmsh:

```bash
docker compose up -d dolfinx-complex
docker compose exec -T dolfinx-complex python3 -m pip install -r neural_operator/requirements.txt
```

Для notebooks выберите Jupyter kernel из этого контейнера; адрес показывает `docker compose logs dolfinx-complex`. В контейнере уже установлен CPU PyTorch. При пересоздании контейнера зависимости устанавливаются повторно. Если `torch` импортирован до `neural_operator` и возник конфликт `libgfortran`, перезапустите kernel и начните с первой ячейки: пакет сам загружает совместимый runtime.

Далее команды показаны для `python3` внутри контейнера; с host добавьте перед ними `docker compose exec -T dolfinx-complex`.

```bash
# 1. Посмотреть план без FEM, затем создать или загрузить low-band данные.
python3 -m neural_operator.run data --band low --dry-run
python3 -m neural_operator.run data --band low
python3 -m neural_operator.run inspect --dataset neural_operator/data/static_low_frequency_1000x8

# Небольшой отдельный набор: 6 частот × 2 направления, без обучения.
python3 -m neural_operator.run data --band low --n-frequencies 6 --n-directions 2 --dataset neural_operator/data/static_low_frequency_6x2

# 2. Проверить POD, затем отдельно запустить обучение SVD-SNO.
python3 -m neural_operator.run svd --pod-method randomized
python3 -m neural_operator.run train --method svd --pod-method randomized --epochs 200

# 3. Сравнить готовую модель с FEM/GMsFEM.
python3 -m neural_operator.run compare --method svd --pod-method randomized
```

`--method chebyshev|fno|all` выбирает другие методы; `all` в сравнении требует готовности всех выбранных моделей. `--band high` выбирает k∈[4,8]; генерация high-band не запускает обучение. `high-frequency` отдельно обучает high-band модели и проверяет перенос frozen low-band. `--help` показывает параметры каждого действия. Старый `python3 -m neural_operator.randomized.run` сохранён как совместимый вход.

Завершённый run повторно используется только при совпадении конфигурации и fingerprint данных. При несовпадении выберите другой `--output` для обучения; `--retrain` явно начинает обучение заново. Автоматического resume нет. Для сравнения `--runs` задаёт каталог моделей, `--output` — каталог отчёта.

## Notebooks

| Задача | Notebook |
| --- | --- |
| Данные и FEM-проверки | [Dataset](notebooks/Dataset.ipynb) |
| POD и перенос на сетку | [SVD-data-checks](notebooks/SVD-data-checks.ipynb) |
| SVD-SNO | [Full-SVD](sno/Full-SVD.ipynb), [Randomized-SVD](sno/Randomized-SVD.ipynb) |
| Chebyshev-SNO | [Train](chebyshev/Train.ipynb) |
| FNO | [Train](fno/Train.ipynb) |
| Самостоятельный inference | [Inference](notebooks/Inference.ipynb) |
| FEM/GMsFEM/neural comparison | [Comparison](notebooks/Comparison.ipynb) |
| Частотный перенос и уточнение FEM | [High-frequency](notebooks/High-frequency.ipynb) |
| Frozen inference на других сетках | [Cross-mesh-inference](notebooks/Cross-mesh-inference.ipynb) |

В общих notebooks выберите `POD_METHOD="full"` или `"randomized"` в первой ячейке. Сохранённые результаты относятся к randomized; данные остаются общими. Comparison и High-frequency начинают с `METHODS=['svd']`; расширяйте список после обучения других методов. Training notebooks и High-frequency обучают модели при запуске соответствующих ячеек.

## Inference и проверка

После обучения `model.pt` содержит веса, базисы, нормировку и геометрию. Для prediction не нужны training data или FEM solver:

```python
from neural_operator.training import load_predictor

predictor = load_predictor(
    "neural_operator/randomized/runs/low_frequency_static_1000x8/svd/model.pt"
)
u_scat = predictor.predict(A_input, n_input, theta=0.3, k0=1.5)
```

Поля A,n передаются в сохранённом порядке FEM-узлов; `predict_dataset` дополнительно проверяет сетку. У текущих static моделей входные POD-ранги равны нулю: они используют k,θ и не реагируют на новую среду. На другой сетке переносятся прежние P1 mean/modes. [Протокол изменения включений](docs/changed_inclusions.md) описывает это ограничение.

Основная метрика — consistent комплексная FEM L2. POD reconstruction и transfer error измеряются отдельно; они не доказывают точность сети или сходимость FEM. [Исторический pilot](docs/local_results.md), [спецификация среды](docs/static_medium.md), [история переработки](docs/change_log.md), [источники](docs/references.md).

```bash
docker compose exec -T dolfinx-complex python3 -m pytest neural_operator/tests -q
```
