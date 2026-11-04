# Randomized SVD на общей фиксированной среде

Готовы две локально обученные randomized SVD-SNO модели: средняя test FEM L2 ошибка **0.96%** для k∈[1,2] и **5.54%** для k∈[4,8]. Данные относятся к одной фиксированной среде. [Результаты и ограничения](RESULTS.md) включают частотный перенос, уточнение FEM и независимый inference.


Здесь находится отдельная последовательность восьми notebooks: [Dataset.ipynb](Dataset.ipynb), [SVD-data-checks.ipynb](SVD-data-checks.ipynb), [SVD-SNO.ipynb](SVD-SNO.ipynb), [Chebyshev-SNO.ipynb](Chebyshev-SNO.ipynb), [FNO.ipynb](FNO.ipynb), [Inference.ipynb](Inference.ipynb), [Comparison.ipynb](Comparison.ipynb), [High-frequency.ipynb](High-frequency.ipynb). Они используют общие Python-модули `neural_operator` и тот же физический датасет, что корневые notebooks. **Randomized означает алгоритм SVD, а не случайную геометрию или новые материалы.**

Физическая среда 0 при seed=2026 фиксирована. По 1000 уникальных частот × 8 направлений дают 8000 решений в каждой полосе k∈[1,2] и k∈[4,8]. Общие данные читаются из `neural_operator/data/static_low_frequency_1000x8` и `neural_operator/data/static_high_frequency_1000x8`; повторная копия не создаётся. A,n и DG0 coefficients неизменны, `medium_id=0` во всех строках. Split по `frequency_id`: 670/170/160 частот, 5360/1360/1280 полей. Это тест удержанных волновых параметров одной среды; он не оценивает перенос на новые среды.

Собственные settings задают `POD_METHOD='randomized'` и model paths:

- `neural_operator/randomized/runs/low_frequency_static_1000x8/{method}`;
- `neural_operator/randomized/runs/high_frequency_static_1000x8/{method}`.

Корневая версия использует `POD_METHOD='full'` и другие run paths. Chebyshev и FNO сами по себе не становятся randomized: отдельные notebooks позволяют сравнить весь workflow, а алгоритм POD/SVD изменяется у SVD-варианта. Exact/randomized сравниваются при одинаковых данных, split, метрике, ранге и сохранённых random seed/oversampling/power iterations.

Запускайте из корня репозитория в существующем Docker service:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.randomized.run data
docker compose exec -T dolfinx-complex python3 -m neural_operator.randomized.run svd
```

`data` создаёт/загружает общий low-band dataset. Обучение и последующий benchmark запускаются отдельно:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.randomized.run train --method all --epochs 200
docker compose exec -T dolfinx-complex python3 -m neural_operator.randomized.run compare --method all
docker compose exec -T dolfinx-complex python3 -m neural_operator.randomized.run high-frequency --method all --epochs 200
```

Первый training run обучает новые модели; повторный `run_experiment(retrain=False)` использует cache только при совпадении config и dataset fingerprint. `--retrain` начинает обучение заново; automatic resume отсутствует. В каталоге уже есть две SVD-SNO модели; остальные training notebooks создают свои модели при запуске. Старые variable-medium и pilot результаты сохранены в прежних каталогах и не относятся к этому static run.

Инструкции Docker/PyTorch, обнаруженный ARM Linux `libgfortran` fix, порядок notebooks и независимый bundle inference описаны в [корневом README](../README.md). Если `torch` импортирован до `neural_operator`, перезапустите kernel. Спецификация неизменного материала, компактного хранения и проверок: [static_medium.md](../docs/static_medium.md).

## Память при локальном обучении

CLI обучения читает u_scat через read-only mmap. Первый запуск создаёт проверяемый по исходному архиву кэш NPY в .array_cache внутри каталога набора; повторный использует его. Проверки и хеширование идут по блокам без полного копирования broadcast-материалов. Randomized POD центрирует одну рабочую матрицу; для SVD loss не создаётся двухканальная копия всех FEM-целей — они читаются порциями. Метрика сохраняется consistent FEM L2, её Gram-форма проверена по значениям и градиентам относительно прямой интеграции.

Comparison и High-frequency по умолчанию запускают готовый SVD-SNO workflow; список METHODS можно дополнить после обучения Chebyshev/FNO. Inference использует выбранный METHOD и не требует готовности всех остальных моделей.

## Проверка других FEM-сеток

[Cross-mesh-inference.ipynb](Cross-mesh-inference.ipynb) дополняет основную последовательность: те же frozen low/high SVD-SNO weights проверяются на более грубой, другой и более мелкой triangulation той же физической среды, а также при перестановке source-узлов. На каждую полосу выбираются три held-out частоты × два направления; reference заново решается на target-сетке. Нового обучения нет.

Переносятся source P1 mean/modes с сохранёнными коэффициентами; это эквивалентно интерполяции source prediction и не добавляет новых fine-scale физических компонент. Результаты разделяют model error, source/target FEM gap, roundtrip и commutation checks. [Формулы и интерпретация](../docs/cross_mesh_inference.md). Notebook полностью выполнен: mean model/target FEM error на новых triangulations — 1.63–1.76% low и 10.86–20.74% high. На fine-сетке high-band FEM gap составил 7.79%. Это шесть выбранных запросов на полосу; указанные выше 0.96%/5.54% относятся ко всему test на исходной сетке. Checkpoints остались неизменными.

## Другие координаты включений

[Changed-inclusions.ipynb](Changed-inclusions.ipynb) проверяет исходную среду, перенос всех центров (0.18,−0.12), безопасные отдельные смещения и варианты A-only, n-only, translation+both. Радиусы фиксированы; для новых центров строится физическая сетка, DG0 материалы назначаются по circle ID, reference решается независимо. Используются те же frozen low/high модели и шесть held-out запросов на полосу, без обучения.

У этих static моделей входные ранги A,n равны нулю: изменение среды при тех же k,θ не меняет предсказанные коэффициенты. На новой сетке вычисляется прежнее source P1-поле. Поэтому это диагностика ограничения, а не подтверждение переноса на новые среды. [Протокол и интерпретация](../docs/changed_inclusions.md); численные результаты будут добавлены после запуска.
