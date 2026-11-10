# Нейрооператоры для рассеяния волн

Три метода решают одну FEM-задачу: **SVD-SNO**, **Chebyshev-SNO** и **FNO**. Каждый метод имеет свою папку; данные, обучение и оценка общие. Full и randomized — два алгоритма POD для SVD-SNO. [Карта проекта](docs/structure.md) · [Датасеты](docs/datasets.md) · [Формулы](docs/methods.md) · [Методичка по SVD, SNO и FNO](docs/svd_sno_guide.md).

Текущие результаты относятся к **static-low**, k∈[1,2], одной фиксированной среде и 1280 test-полям. Средняя complex consistent FEM L2 ошибка: **SVD-SNO v2 — 0.9475%**, **FNO — 1.6599%**, **Chebyshev-SNO modes8 — 7.9364%**. [Формулы, полные конфигурации и SHA-256](docs/operator_revision.md). Для Chebyshev validation decoder floor при modes8 равен 7.6143%; выбор выходного пространства оценивается отдельно от обучения.

Это один seed и удержанные частоты той же среды; результаты не подтверждают перенос на другие материалы или точность относительно непрерывной PDE.

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
python3 -m neural_operator.run train --method svd --pod-method randomized --epochs 200 --sno-architecture-version 2 --output neural_operator/runs/revision_2026-11-08/static_low

# 3. Сравнить готовую модель с FEM/GMsFEM.
python3 -m neural_operator.run compare --method svd --pod-method randomized --runs neural_operator/runs/revision_2026-11-08/static_low
```

`--method chebyshev|fno|all` выбирает другие методы; `all` в сравнении требует готовности всех выбранных моделей. `--band high` выбирает k∈[4,8]; генерация high-band не запускает обучение. `high-frequency` отдельно обучает high-band модели и проверяет перенос frozen low-band. `--help` показывает параметры каждого действия. Старый `python3 -m neural_operator.randomized.run` сохранён как совместимый вход.

`--sno-architecture-version 2` — новый coefficient-token lift и параллельные локальная/модальная ветви; `1` явно выбирает legacy. Старые checkpoint без версии загружаются как v1. Новый v2 run требует собственного `--output`; старые веса не переиспользуются для другой архитектуры. Два SVD training notebooks уже выбирают отдельные v2-каталоги. High-frequency notebook — шаблон нового исследования с явным выбором сетки и отдельными каталогами; сохранённых результатов в нём нет.

Обучение принимает `--device` (по умолчанию `cpu`). На совместимом Mac в локальном Python-окружении с зависимостями проекта FNO можно обучать через `python3 -m neural_operator.run train --method fno --pod-method randomized --band low --epochs 200 --device mps`; для второй полосы замените `low` на `high`. Устройство обучения сохраняется в конфигурации. Готовые веса загружаются на CPU через `load_predictor(path, device="cpu")`; FNO notebook проверяет настройки и fingerprint, затем использует доступное устройство для inference.

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
| Текущие static-low SNO v2/FNO/Chebyshev: метрики и Re/Im/Abs | [Static-predictions](notebooks/Static-predictions.ipynb) |
| FEM/GMsFEM/neural comparison | [Comparison](notebooks/Comparison.ipynb) |
| Частотный перенос и уточнение FEM | [High-frequency](notebooks/High-frequency.ipynb) |
| Frozen inference на других сетках | [Cross-mesh-inference](notebooks/Cross-mesh-inference.ipynb) |

В общих notebooks выберите `POD_METHOD="full"` или `"randomized"` в первой ячейке. Данные остаются общими. Execution outputs очищены. Inference и Static-predictions читают текущие static-low checkpoints; training notebooks запускают обучение при отсутствии совместимого run. Шаблон High-frequency требует явного включения нового исследования.

## Inference и проверка

После обучения `model.pt` содержит веса, базисы, нормировку и геометрию. Для prediction не нужны training data или FEM solver:

```python
from neural_operator.training import load_predictor

predictor = load_predictor(
    "neural_operator/runs/revision_2026-11-08/static_low/svd/model.pt"
)
result = predictor.predict_components(A_input, n_input, theta=0.3, k0=1.5)
re_u = result["real"]  # Re(u_scattering)
im_u = result["imag"]  # Im(u_scattering)
abs_u = result["abs"]  # Abs(u_scattering) = sqrt(re_u**2 + im_u**2)
```

Этот интерфейс общий для SVD-SNO, Chebyshev-SNO и FNO. Для одного входа каждая компонента имеет форму `[nodes]`, для batch — `[samples, nodes]`. Метод `predict(...)` по-прежнему возвращает комплексный массив `u_scattering`; его можно разобрать функцией `neural_operator.fields.scattering_components`. Метод `predict_dataset_components(dataset, indices)` возвращает те же три компоненты для выборки. Модуль вычисляется после восстановления и денормировки поля; переобучение сохранённых моделей для такого вывода не требуется.

Общий `plot_prediction` показывает три строки **Re / Im / Abs** и три столбца **FEM / prediction / абсолютная ошибка соответствующей компоненты**. Для FEM и prediction используются одинаковые цветовые шкалы внутри каждой строки. Ошибка модуля — `abs(abs(u_pred) - abs(u_FEM))`; комплексная FEM L2 метрика дополнительно учитывает фазу. Новые `test_predictions.npz` сохраняют `prediction_real`, `prediction_imag`, `prediction_abs` вместе с комплексным `prediction`.

Поля A,n передаются в сохранённом порядке FEM-узлов; `predict_dataset` дополнительно проверяет сетку. У текущих static SVD-SNO моделей входные POD-ранги равны нулю: они используют k,θ и не реагируют на новую среду. FNO получает A,n на регулярной сетке, но обучение на одной среде само по себе не проверяет перенос на другие материалы. У SVD-SNO на другой сетке переносятся прежние P1 mean/modes. Новая геометрия и материалы требуют отдельной физической проверки.

Основная метрика — consistent комплексная FEM L2. POD reconstruction и transfer error измеряются отдельно; они не доказывают точность сети или сходимость FEM. [спецификация среды](docs/static_medium.md), [история переработки](docs/change_log.md), [источники](docs/references.md).

```bash
docker compose exec -T dolfinx-complex python3 -m pytest neural_operator/tests -q
```
