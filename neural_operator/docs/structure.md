# Структура neural_operator

| Путь | Назначение |
| --- | --- |
| `sno/` | SVD-SNO: модель, POD-базис, Full-SVD и Randomized-SVD notebooks |
| `chebyshev/` | Chebyshev-SNO: модель и Train notebook |
| `fno/` | FNO: модель и Train notebook |
| `datasets/` | Конфигурация, план, генерация и чтение данных |
| `notebooks/` | Общие данные, диагностика, inference и исследования |
| `training.py`, `benchmark.py`, `diagnostics.py` | Общее обучение и оценка методов |
| `run.py`, `settings.py` | Единый CLI и стандартные пути full-профиля |
| `randomized/` | Совместимый CLI/settings, готовые runs и RESULTS.md |
| `data/`, `runs/` | Локальные данные и результаты; пути существующих артефактов сохранены |
| `tests/`, `docs/` | Проверки и подробные протоколы |

SVD-SNO обучается над POD-коэффициентами, Chebyshev-SNO — над коэффициентами DCT-I, FNO применяет Fourier layers с нелинейностью в физическом пространстве. Общие split, FEM loss, обучение и bundle inference не копируются между папками методов. Импорты через прежние `models.py` и `basis.py` остаются совместимыми.

`--pod-method full|randomized` выбирает алгоритм POD и совместимый профиль путей. Randomized не означает случайную геометрию и не создаёт новые определения Chebyshev/FNO. Данные обоих профилей общие; full runs находятся в `runs/static_*`, randomized — в `randomized/runs/*_static_*`. Существующие checkpoints и отчёты не перемещаются.

Общие notebooks объединяют два прежних набора wrappers; параметр `POD_METHOD` выбирается в первой ячейке. Для SVD-SNO оставлены два коротких notebooks с явно разными алгоритмами. Выполненные randomized outputs сохранены в общем наборе и помечены как randomized. Chebyshev/FNO имеют по одному training notebook. [Cross-mesh-inference](../notebooks/Cross-mesh-inference.ipynb) сохраняет записанные исследования без нового обучения.

Каждый notebook ищет корень репозитория среди текущего каталога и его родителей, поэтому запуск из `sno/`, `chebyshev/`, `fno/` или `notebooks/` не зависит от фиксированной глубины пути. Kernel должен работать внутри репозитория и использовать окружение из [README](../README.md).

Исторический аудит и научные ограничения сохранены в [change_log.md](change_log.md). Изменение структуры само по себе не является новым обучением или измерением точности.
