# Локальные neural operators: фиксированная среда

Готовы две локально обученные randomized SVD-SNO модели: средняя test FEM L2 ошибка **0.96%** для k∈[1,2] и **5.54%** для k∈[4,8]. Данные относятся к одной фиксированной среде. [Результаты и ограничения](randomized/RESULTS.md) включают частотный перенос, уточнение FEM и независимый inference.


Текущий профиль использует **одну неизменную физическую среду**: геометрия и все значения A,n фиксированы, меняются только частота k и направление падающей волны. На каждую из полос k∈[1,2] и k∈[4,8] приходится **1000 уникальных частот × 8 направлений = 8000 FEM-полей**, всего 16000. Это 1000 частот, а не 1000 разных сред. Полная спецификация и протокол проверки: [static_medium.md](docs/static_medium.md).

Корневая последовательность использует **full SVD**. Отдельный вариант [randomized/](randomized/README.md) содержит восемь notebooks и собственные пути результатов, использует **randomized SVD**, но читает тот же физический датасет. Слово randomized относится к численному алгоритму разложения, а не к геометрии или материалам.

Старые variable-medium datasets `data/low_frequency_1000x8`, `data/high_frequency_1000x8` и pilot 48×4 вместе с его шестью моделями сохранены отдельно. [Старый отчёт](docs/local_results.md) описывает pilot и не подтверждает точность новой статичной постановки. Создание нового датасета и каталога randomized само по себе не означает обучение новых моделей.

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

## Данные и постановка split

Фиксируется **точно среда 0 прежнего dynamic генератора** при seed=2026: прежние активные круги и прежние умеренные контрасты A∈[0.6,0.9], n∈[1.5,3] во включениях. Материалы выбираются один раз; фон A=n=1. Статичный генератор воспроизводит последовательность RNG прежнего medium draw, включая начальную выборку частоты. Это сохранение существующей среды, а не новый набор случайных материалов.

`medium_id=0` для всех 8000 строк каждой полосы. `k` задаётся равномерным `linspace` из 1000 уникальных значений; для каждой частоты восемь равномерных направлений получают воспроизводимый угловой offset. `frequency_id` группирует направления для split: **670/170/160 частот**, то есть **5360/1360/1280 полей** train/val/test. Одна физическая среда присутствует во всех split намеренно. Тест оценивает удержанные волновые параметры в этой среде; он не проверяет generalization на новые материалы или геометрию.

| Артефакты | Низкая полоса | Высокая полоса |
| --- | --- | --- |
| Общие данные | `data/static_low_frequency_1000x8` | `data/static_high_frequency_1000x8` |
| Full-SVD model runs | `runs/static_low_frequency_1000x8` | `runs/static_high_frequency_1000x8` |
| Randomized-SVD model runs | `randomized/runs/low_frequency_static_1000x8` | `randomized/runs/high_frequency_static_1000x8` |

Ссылки на параметры: [low config](data/static_low_frequency_1000x8/config.json), [high config](data/static_high_frequency_1000x8/config.json). Фактические counts/hashes: [low dataset_info](data/static_low_frequency_1000x8/dataset_info.json), [high dataset_info](data/static_high_frequency_1000x8/dataset_info.json).

В `.npz` узловые A,n и DG0 `A_cells`/`n_cells` хранятся **одной строкой**. Loader предоставляет broadcast views формы `[samples,nodes/cells]` общему API без физического дублирования материалов. Комплексные решения сохраняются по одному на пару (k,θ), `complex128`; геометрия, connectivity, physical tags и `.msh` сохраняются рядом. Не изменяйте broadcast views на месте.

## Последовательность notebooks

| Notebook | Назначение |
| --- | --- |
| [Dataset.ipynb](Dataset.ipynb) | Статичная среда, FEM-поля и split по частотам |
| [SVD-data-checks.ipynb](SVD-data-checks.ipynb) | Спектры, exact/randomized диагностика, reconstruction и grid transfer |
| [SVD-SNO.ipynb](SVD-SNO.ipynb) | Train-only POD и coefficient-space модель, full SVD |
| [Chebyshev-SNO.ipynb](Chebyshev-SNO.ipynb) | DCT-I и нелинейность над коэффициентами |
| [FNO.ipynb](FNO.ipynb) | Fourier layers и физические нелинейности |
| [Inference.ipynb](Inference.ipynb) | Независимая загрузка bundle и single/batch prediction |
| [Comparison.ipynb](Comparison.ipynb) | Paired FEM/GMsFEM/neural comparison |
| [High-frequency.ipynb](High-frequency.ipynb) | Frozen low-band, отдельное high-band обучение и mesh refinement |

Обучение выполняется только при отдельном запуске соответствующих ячеек/CLI actions. При первом `Run All` training notebook создаёт модель в новых static run paths. `run_experiment(..., retrain=False)` повторно загружает завершённый run только при совпадении всей конфигурации и dataset fingerprint. Иначе возникает явная ошибка: нужна другая папка либо `retrain=True`. Последний вариант начинает обучение заново, автоматического resume нет. Full/randomized runs не перезаписывают друг друга или старые pilot веса.

Для неизменных A,n центрированный входной POD имеет нулевой ранг. Это корректно: сеть получает три изменяемых параметра `(cos θ,sin θ,k)`, а не искусственные моды численного шума. Базисы выходов и нормировки по-прежнему строятся только по train, rank и checkpoint выбираются по val. Проверки нулевого входного ранга и inference относятся к корректности реализации, а не к измерению точности полного обучения.

## CLI и генерация без обучения

Из корня репозитория создать статичный low-band dataset с текущими defaults:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.run data
```

`Dataset.ipynb` создаёт и просматривает низкую полосу. Высокую полосу без обучения можно создать отдельно:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.run data --dataset neural_operator/data/static_high_frequency_1000x8 --k-range 4 8
```

Генератор принимает `--n-frequencies`, `--n-directions`, `--fixed-medium-id` и `--h`. Для другой конфигурации задайте отдельный `--dataset PATH`.

Последующие этапы запускаются отдельно; `train` и `high-frequency` обучают модели:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.run svd
docker compose exec -T dolfinx-complex python3 -m neural_operator.run train --method all --epochs 200
docker compose exec -T dolfinx-complex python3 -m neural_operator.run compare --method all
docker compose exec -T dolfinx-complex python3 -m neural_operator.run high-frequency --method all --epochs 200
```

Стандартные настройки обучения: CPU, два PyTorch threads, максимум 200 эпох, patience=50, min_epochs=50. Достижение лимита не доказывает сходимости; точность и время full training новой постановки пока не заявлены. `--dataset PATH` выбирает данные, `--output PATH` — результаты; для changed generation config используйте новый каталог. Список действующих параметров показывает `python3 -m neural_operator.run --help` внутри контейнера. Randomized-вариант запускается через `python3 -m neural_operator.randomized.run` с теми же actions и собственными settings.

## Самостоятельный inference и чтение результатов

[Changed-inclusions.ipynb](randomized/Changed-inclusions.ipynb) отдельно проверяет смещение центров кругов, безопасный jitter и изменения A,n с независимым target FEM solve. [Протокол](docs/changed_inclusions.md) поясняет ограничение frozen static моделей: при нулевых входных рангах они не реагируют на новую среду; новая сетка только задаёт точки вычисления прежнего source prediction. Численные результаты добавляются после запуска.

Для проверки других FEM-сеток добавлен [randomized/Cross-mesh-inference.ipynb](randomized/Cross-mesh-inference.ipynb). Frozen модели вычисляют перенесённые source P1 mean/modes на новых узлах той же области и среды; это эквивалентно переносу прежнего prediction. Сравнение с независимо решённым target FEM и ограничения описаны в [cross_mesh_inference.md](docs/cross_mesh_inference.md). Notebook выполнен: на шести выбранных запросах каждой полосы mean model/target FEM error составила 1.63–1.76% low и 10.86–20.74% high на новых triangulations. Это отдельная малая выборка; прежние 0.96%/5.54% относятся к полному test на сетке обучения. Таблица также показывает FEM discretization gap; checkpoint не изменён.


Bundle `model.pt` сохраняет веса, architecture config, базисы, normalization и геометрию; загружается с `weights_only=True`. После отдельного обучения ему не нужны training data или FEM solver, нужны входные A,n в сохранённом порядке узлов и k,θ:

```python
from neural_operator.training import load_predictor

# Файл появится после отдельного обучения static full-SVD модели.

predictor = load_predictor(
    "neural_operator/runs/static_low_frequency_1000x8/svd/model.pt"
)
u_scat = predictor.predict(A_input, n_input, theta=0.3, k0=1.5)
```

При прямом `predict` порядок FEM-узлов обеспечивает вызывающий код; `predict_dataset` проверяет coordinates/connectivity. Основная метрика — consistent P1 FEM L2 комплексного поля. Lumped-weighted POD reconstruction error — диагностика сжатия, не строгая lower bound для consistent loss. Сравнение всех 1280 neural test полей хранится отдельно от малого paired classical subset. В статичной постановке classical subset выбирается по частотным группам, а не по единственному medium ID.

High-band сетка h=0.1 не уточняется при увеличении числа снимков. В старом variable-medium pilot refinement difference была примерно 11–15%; это историческое ограничение, а не измерение ошибки новой фиксированной среды. Для физической точности нового reference нужна собственная mesh-refinement проверка. Frozen low-band и переобученный high-band режимы читаются отдельно.

Формулы и источники: [methods.md](docs/methods.md), [references.md](docs/references.md). SVD-вариант — локальная POD адаптация coefficient-space SNO, не заявленное точное воспроизведение SNO/PODNO. Chebyshev/FNO transfers и compression errors должны измеряться отдельно от ошибки сети.

```bash
docker compose exec -T dolfinx-complex python3 -m pytest neural_operator/tests -q
```

## Память при локальном обучении

CLI обучения читает u_scat через read-only mmap. Первый запуск создаёт проверяемый по исходному архиву кэш NPY в .array_cache внутри каталога набора; повторный использует его. Проверки и хеширование идут по блокам без полного копирования broadcast-материалов. Randomized POD центрирует одну рабочую матрицу; для SVD loss не создаётся двухканальная копия всех FEM-целей — они читаются порциями. Метрика сохраняется consistent FEM L2, её Gram-форма проверена по значениям и градиентам относительно прямой интеграции.
