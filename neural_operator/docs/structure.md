# Структура neural_operator

[Методичка по SVD, SNO и выходам Re/Im/Abs](svd_sno_guide.md) объясняет вычислительный путь и текущую реализацию SNO v2.

| Путь | Назначение |
| --- | --- |
| `sno/` | SVD-SNO: модель, POD-базис, Full-SVD и Randomized-SVD notebooks |
| `chebyshev/` | Chebyshev-SNO: модель и Train notebook |
| `fno/` | FNO: модель и Train notebook |
| `datasets/` | Конфигурация, план, генерация и чтение данных |
| `notebooks/` | Общие данные, диагностика, inference и исследования |
| `training.py`, `benchmark.py`, `diagnostics.py` | Общее обучение и оценка методов |
| `run.py`, `settings.py` | Единый CLI и стандартные пути full-профиля |
| `randomized/` | Совместимый CLI/settings, готовые runs, SVD-SNO [RESULTS.md](../randomized/RESULTS.md) и [проверка static-low FNO](operator_revision.md) |
| `data/`, `runs/` | Локальный static-low набор и актуальные результаты |
| `tests/`, `docs/` | Проверки и подробные протоколы |

SVD-SNO обучается над POD-коэффициентами, Chebyshev-SNO — над коэффициентами DCT-I, FNO применяет Fourier layers с нелинейностью в физическом пространстве. Общие split, FEM loss, обучение и bundle inference не копируются между папками методов. Импорты через прежние `models.py` и `basis.py` остаются совместимыми.

`--pod-method full|randomized` выбирает алгоритм POD и совместимый профиль путей. Randomized не означает случайную геометрию и не создаёт новые определения Chebyshev/FNO. Данные обоих профилей общие; full runs находятся в `runs/static_*`, randomized — в `randomized/runs/*_static_*`. Текущие paths checkpoints перечислены в [отчёте](operator_revision.md).

Общие notebooks используют единые данные и модели. Для SVD-SNO оставлены full/randomized training notebooks; Chebyshev/FNO имеют по одному. Сохранённые execution outputs очищены. [Static-predictions](../notebooks/Static-predictions.ipynb) читает текущие static-low SNO v2, FNO и Chebyshev, без обучения. [Cross-mesh-inference](../notebooks/Cross-mesh-inference.ipynb) и [High-frequency](../notebooks/High-frequency.ipynb) служат шаблонами новых исследований, без сохранённых экспериментальных результатов.

Каждый notebook ищет корень репозитория среди текущего каталога и его родителей, поэтому запуск из `sno/`, `chebyshev/`, `fno/` или `notebooks/` не зависит от фиксированной глубины пути. Kernel должен работать внутри репозитория и использовать окружение из [README](../README.md).

Изменения кода перечислены в [change_log.md](change_log.md). Изменение структуры само по себе не является новым обучением или измерением точности.
