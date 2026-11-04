# Совместимость randomized-профиля

Randomized — алгоритм POD/SVD для **SVD-SNO**. Данные, Chebyshev-SNO и FNO используют общую реализацию. Отдельный набор повторяющихся notebooks заменён [общими notebooks](../notebooks/) и [Randomized-SVD](../sno/Randomized-SVD.ipynb).

Основной вход теперь общий:

```bash
python3 -m neural_operator.run train --method svd --pod-method randomized
python3 -m neural_operator.run compare --method svd --pod-method randomized
```

Старые `python3 -m neural_operator.randomized.run`, `settings.py` и пути артефактов сохранены:

- `runs/low_frequency_static_1000x8/{method}`;
- `runs/high_frequency_static_1000x8/{method}`;
- диагностические и исследовательские результаты в `runs/`.

Готовые SVD-SNO модели: **0.96%** low-band и **5.54%** high-band mean test FEM L2 на одной фиксированной среде. [Полный отчёт](RESULTS.md). Это не оценка переноса на новые материалы.

[Запуск и окружение](../README.md) · [Структура](../docs/structure.md) · [Данные](../docs/datasets.md) · [Cross-mesh notebook](../notebooks/Cross-mesh-inference.ipynb).
