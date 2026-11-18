# Одна SVD-SNO для k ∈ [0.1,4.0]

Отдельный эксперимент по просьбе пользователя: одна сеть, общие POD-базисы,
общая нормировка волнового числа и один checkpoint для всего диапазона.
Предыдущий `reference_sno_gmsfem` остаётся источником FEM-полей и проверенных
GMsFEM-решений. Его данные, модели и результаты не изменяются.

**Обучение и сравнение выполнены:** 500 эпох, выбран checkpoint эпохи 493,
61 POD-мода на каждую компоненту, 50 648 обучаемых параметров.
На одних и тех же 424 test-полях средние ошибки к reference FEM:

| Метод | Полное поле, % | Рассеянное поле, % |
|---|---:|---:|
| Одна SVD-SNO [0.1,4] | 2.3656 | 3.3271 |
| Лучший GMsFEM: 20×20, Type2, M16 | 3.2439 | 4.2925 |

Для SNO p95 ошибки полного поля — 7.3718%, максимум — 27.5856%:
среднее качество не означает равномерную точность по всем k и направлениям.
79 tests и 8 subtests прошли; все 424 предсказания повторены из checkpoint,
четыре новые контрольные задачи FEM/GMsFEM совпали с сохранёнными метриками.

Подробные результаты и рисунки: [results/REPORT.md](results/REPORT.md),
полная таблица: [results/comparison.csv](results/comparison.csv),
проверки: [results/validation.json](results/validation.json),
[results/tests.json](results/tests.json).

**Визуальная сводка:** [галерея графиков](results/visuals/README.md) —
ошибки по k с разбросом по направлениям, сравнение среднего и p95,
распределение и попарное сравнение ошибок, обучение, карты Re/Im/Abs.
Четыре новых графика доступны в PNG и SVG. Для их пересоздания:

```bash
docker compose exec -T dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run visualize
```

## Физическая задача

Ω=[−2,2]², все 24 включения сохранённой GMsFEM-геометрии. Фон α=n=1,
во включениях A=0.2I, n=10. Падающая волна `exp(ik d·x)`,
`d=(cosθ,sinθ)`; предсказывается комплексное рассеянное поле.
Reference P1 FEM: `Kα − k²Mn − ikB`, 53 246 узлов, 105 690 треугольников,
сохранённая сетка h=0.02, поглощающее условие первого порядка.

## Общий датасет

Используются точные запросы прежних low/high наборов без новой генерации.
Получается 2568 полей на 320 уникальных частотах:

| Split | Поля | Частоты |
|---|---:|---:|
| Train | 1712 | 214 |
| Validation | 432 | 54 |
| Test | 424 | 52 |

Прежнее разбиение сохранено для каждой строки. k=2 содержит 16 разных
направлений; все они относятся к одному глобальному frequency_id и test.
В samples.npz хранятся `source_band` и `source_index`, связывающие каждую
строку с исходным полем. Частотная сетка — объединение прежних сеток,
а не равномерная сетка из 320 точек. Конфликт splits на одном физическом k
и дубли физических запросов отклоняются.

Исходные и объединённые labels проверяются streaming SHA256. Среда,
порядок DOF, связность и DG0-материалы должны совпадать точно. Поля
хранятся в read-only mmap; незавершённое копирование можно продолжить.
Для подготовки обучающих данных новых PDE solves нет.

## Архитектура

Переиспользуется проверенная SVD-SNO v2 из `neural_operator.sno.model`.
Вход `[c_A,c_n,cosθ,sinθ,scaled k]` превращается в coefficient tokens
`[batch,R,5]`, где `R=max(r_out,r_A,r_n)`. Shared lift 5→48 и шесть блоков:

`Z_next = Z + GELU(Z W_local + b + B Z W_channel)`.

Общая голова 48→2 выдаёт коэффициенты Re/Im; затем выполняются обратная
нормировка и два фиксированных POD-декодера. Все части общие для диапазона.
Модуль получается из восстановленных Re/Im. При фиксированном материале
`r_A=r_n=0`: фактически обучается `(k,θ)→u_s` одной среды.

Train-only randomized POD, rank cap 96, validation target 0.1% среднего
projection error. Фактические ранг и floor записываются в metrics.json.
Validation выбирает checkpoint; test не участвует в оптимизации и выборе
гиперпараметров. Лимит 500 эпох, width 48, depth 6, batch 32, AdamW lr .002,
weight decay 1e-5, seed 42, CPU threads 2. По validation применяется
ReduceLROnPlateau; loss — complex consistent FEM L2 с точным POD Gram loss.
Используется прежний проверенный training loop, без копирования алгоритма.

## Сравнение и проверка

GMsFEM: сетки 10×10 и 20×20, Type 1/2, 16 локальных мод; исходная fine-сетка
h=0.04. Дополнительно показывается FEM h=0.04, чтобы видеть вклад различия
fine/reference сеток. Метрики всех методов считаются на тех же 424 test
queries в reference-квадратурных точках с аналитической падающей волной.

2120 исходных baseline-строк переиндексируются по точным физическим запросам,
без интерполяции метрик. Настройки, source hashes и исходная табличная
репликация проверяются. Затем в k=.1,1,2,4 заново решаются FEM h=0.04 и
GMsFEM 20×20 Type2 M16 и сверяются все метрики. Все 424 SNO-предсказания
повторяются из checkpoint; также заново проверяются все публикуемые
квадратурные метрики. Отчёт отклоняет stale verification и несовпадающие
результаты. Рисунки полей при k=1 и k=4 используют реальные решения.

Отдельные intervals в отчёте — диагностика одной общей модели. Исторические
GMsFEM timings не выдаются за новый benchmark общей вычислительной цепочки.

## Запуск из корня репозитория

Используется существующий healthy Docker service `dolfinx-complex` с complex
DOLFINx/PETSc и PyTorch. Обучающие артефакты предыдущего reference-эксперимента
должны уже присутствовать локально. Все новые артефакты находятся здесь.

```bash
docker compose exec -T dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run prepare
docker compose exec -T -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run train --epochs 500
docker compose exec -T dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run baselines
docker compose exec -T dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run verify
docker compose exec -T dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run report
```

Эквивалентный запуск всех стадий:

```bash
docker compose exec -T -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 dolfinx-complex python3 -m experiments.shared_sno_gmsfem.run all
```

Повторный train загружает уже завершённый совместимый bundle. Изменённые
данные или frozen protocol приводят к ошибке; несовместимые результаты
не перезаписываются автоматически. Прогноз из сохранённого bundle:

```python
from experiments.reference_sno_gmsfem.dataset import load_dataset
from neural_operator.training import load_predictor

data = load_dataset("experiments/shared_sno_gmsfem/data")
model = load_predictor("experiments/shared_sno_gmsfem/runs/shared/model.pt")
index = data.split_indices()[2][0]
u_s = model.predict_dataset(data, [index])[0]

# Непрерывный запрос: k не обязан совпадать с частотами датасета.
u_new = model.predict(data.A[0], data.n[0], theta=0.3, k0=2.731)
```

Проверки:

```bash
docker compose exec -T dolfinx-complex python3 -m pytest experiments/shared_sno_gmsfem/tests experiments/reference_sno_gmsfem/tests neural_operator/tests/test_operators.py neural_operator/tests/test_pod_fem_loss.py gmsfem/tests -q
```

Основные файлы: `dataset.py` — verified union, `training.py` — training adapter,
`comparison.py` — paired import/reload/fresh solves, `report.py` — отчёт и
рисунки, `run.py` — CLI. `data/` и `runs/` исключены из Git из-за размера;
результаты и контрольные суммы сохраняются в `results/`.
