# Multiscale FEM Scattering

**GMsFEM и спектральные нейрооператоры для двумерного рассеяния волн.**

Проект объединяет комплексный P1 FEM, локальную многомасштабную редукцию GMsFEM и три обучаемые модели: **SVD-SNO v2, Chebyshev-SNO и FNO**. Основной neural-эксперимент использует одну фиксированную среду и проверяет предсказание для удержанных частот и направлений падающей волны.

[Саммари проекта](SUMMARY.md) · [Учебный маршрут GMsFEM](gmsfem/docs/learning_path.md) · [Методы и результаты](neural_operator/docs/operator_revision.md) · [Статья](paper/svd_sno_revision_2026-11-06/main.pdf)

**English overview.** Research code for 2D heterogeneous Helmholtz scattering: complex-valued finite elements, generalized multiscale finite elements, POD/SVD coefficient-space SNO, Chebyshev spectral operators, and Fourier Neural Operators. The retained neural study evaluates held-out wave parameters in one fixed physical medium.

## Задача и метрика

Область Ω=[−2,2]² содержит 24 допустимых круговых включения. Материалы описываются положительными скалярами α и n; A=αI. Для падающей волны uⁱ=exp(ik d·x) полное поле uᵗᵒᵗ=uⁱ+uˢ удовлетворяет

$$\nabla\cdot(\alpha\nabla u^{tot})+k^2nu^{tot}=0.$$

Решается рассеянное поле uˢ с поглощающим условием первого порядка на внешней границе. Контрастная слабая правая часть учитывает материальные интерфейсы. [Вывод постановки](gmsfem/docs/method.md).

Основная neural-метрика — относительная комплексная P1 FEM L2-норма с согласованной матрицей масс M:

$$e_M=\sqrt{(\hat u^s-u^s)^*M(\hat u^s-u^s)}\,/\,\sqrt{(u^s)^*Mu^s}.$$

Re и Im восстанавливаются моделью; Abs вычисляется из комплексного результата. Ошибка модуля и комплексная ошибка различаются. [Формулы методов](neural_operator/docs/methods.md).

## Сохранённый neural-эксперимент

Одна среда, k∈[1,2], 1000 частот × 8 направлений = **8000 FEM-полей** на 3221 узле. Split по частотным группам: **5360/1360/1280** полей train/validation/test. Базисы и нормировки строятся по train; ранги и checkpoints выбираются по validation.

| Модель | Mean test FEM L2 | p95 | Параметры | Выполнено / лучшая эпоха |
| --- | ---: | ---: | ---: | ---: |
| SVD-SNO v2 | **0,9475%** | 1,6534% | 4022 | 165 / 115 |
| FNO | **1,6599%** | 1,9505% | 445010 | 142 / 92 |
| Chebyshev-SNO | **7,9364%** | 9,8811% | 2426 | 200 / 199 |

Все три метода сохранены. SNO v2 и Chebyshev обучены заново; существующий FNO повторно проверен текущим CPU-кодом. Ошибки проверены по всем 1280 test-полям. Конфигурации, SHA256, повторная загрузка и ограничения: [отчёт](neural_operator/docs/operator_revision.md), [JSON](neural_operator/docs/operator_revision.json), [FNO provenance](neural_operator/docs/fno_static_low_verification_2026-11-08.json).

У SNO постоянные материалы дают нулевые входные POD-ранги: сеть учит зависимость поля от k и направления. Эта оценка не проверяет новые материалы. У Chebyshev с 8×8 модами измеренный mean validation decoder floor равен 7,6143%; увеличение числа эпох само по себе не устраняет ограничение пространства. Разные размеры моделей и устройства не задают равный вычислительный бюджет. Ошибки относятся к FEM-меткам, а не к аналитическому решению PDE.

Старые экспериментальные серии удалены целиком из текущего дерева репозитория: их отчёты, снимки результатов, рисунки и сохранённые notebook outputs не входят в публикационный комплект. [Состав оставшихся данных](neural_operator/docs/dataset_audit.md).

## GMsFEM: проверяемая редукция

Для сохранённой серии `notebook`, k=1, fine FEM содержит **17549 неизвестных**. Type 2 с 16 модами на крупную вершину даёт:

| Крупная сетка | Неизвестных | Комплексная L2 полного поля к fine FEM |
| --- | ---: | ---: |
| 10×10 | 1936 | 0,5092% |
| 20×20 | 7056 | 0,1696% |

Это отдельная материальная постановка и ошибка редукции к той же тонкой сетке. [Сохранённый отчёт](gmsfem/results/notebook/report.md), [учебный маршрут с самопроверкой](gmsfem/docs/learning_path.md), [численная ревизия](gmsfem/docs/audit.md).

## Запуск и быстрая проверка

Все команды выполняются из корня репозитория, в одном MPI-процессе.

```bash
docker compose up -d --build dolfinx-complex
docker compose exec -T dolfinx-complex python3 -m pip install -r neural_operator/requirements.txt

docker compose exec -T dolfinx-complex python3 -m gmsfem.learning verify
docker compose exec -T dolfinx-complex python3 -m neural_operator.run data --band low --dry-run

docker compose exec -T dolfinx-complex python3 -m unittest discover -s gmsfem/tests -v
docker compose exec -T dolfinx-complex python3 -m pytest neural_operator/tests -q
```

Для воспроизведения neural-эксперимента после клонирования:

```bash
docker compose exec -T dolfinx-complex python3 -m neural_operator.run data --band low --dataset neural_operator/data/static_low_frequency_1000x8

docker compose exec -T dolfinx-complex python3 -m neural_operator.run train --band low --method svd --pod-method randomized --epochs 200 --sno-architecture-version 2 --output neural_operator/runs/revision_2026-11-08/static_low

docker compose exec -T dolfinx-complex python3 -m neural_operator.run train --band low --method chebyshev --pod-method randomized --epochs 200 --resolution 32 --modes 8 --output neural_operator/runs/revision_2026-11-08/static_low
```

FNO и все подробные настройки: [neural README](neural_operator/README.md). Данные и веса — локальные генерируемые артефакты в игнорируемых `data/` и `runs/`; Git хранит исходники, компактные результаты и доказательства проверки. Для другой конфигурации используйте новый каталог результатов.

## Карта проекта

| Путь | Назначение |
| --- | --- |
| [SUMMARY.md](SUMMARY.md) | Научное и практическое саммари |
| [gmsfem/](gmsfem/README.md) | FEM, snapshots, локальный спектр, глобальная проекция, учебные примеры |
| [neural_operator/](neural_operator/README.md) | Общие данные, три метода, обучение и inference |
| [templates/](templates/README.md) | Лабораторные 4 и 6: блоки, подсетки и окрестности |
| [paper/](paper/svd_sno_revision_2026-11-06/README.md) | Статья, рисунки, проверка и воспроизведение |
| [llms.txt](llms.txt) | Краткий индекс для исследовательских агентов |

Шаблон лабораторной 4 и Legacy-вариант лабораторной 6 требуют DOLFIN 2019; основной workflow работает в complex DOLFINx/PETSc. Локальные реализации адаптируют методы к геометрии проекта. Методологические источники: [GMsFEM](https://doi.org/10.1016/j.cam.2022.114977), [SNO](https://arxiv.org/abs/2205.10573), [FNO](https://arxiv.org/abs/2010.08895). [Список первичных источников](neural_operator/docs/references.md).

**Автор:** Egoriy Monastyrev. **Лицензия:** [MIT](LICENSE). При цитировании указывайте commit, конфигурацию и отчёт запуска: [CITATION.cff](CITATION.cff).
