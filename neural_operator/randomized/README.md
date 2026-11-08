# Randomized POD/SVD

Randomized — алгоритм построения POD-базисов для **SVD-SNO**. Данные, Chebyshev-SNO и FNO используют общую реализацию; выбор этого профиля не добавляет POD в Fourier/Chebyshev-слои.

Актуальный static-low SNO v2 находится в `../runs/revision_2026-11-08/static_low/svd`; FNO — в `runs/low_frequency_static_1000x8/fno`. Оба пути используются в [Static-predictions](../notebooks/Static-predictions.ipynb).

```bash
python3 -m neural_operator.run train --method svd --pod-method randomized --sno-architecture-version 2 --output neural_operator/runs/revision_2026-11-08/static_low
```

Генерация, обучение и сравнение используют общий CLI `neural_operator.run`; прежний `neural_operator.randomized.run` остаётся совместимым входом. [Результаты](RESULTS.md) · [Запуск и окружение](../README.md) · [Структура](../docs/structure.md) · [Данные](../docs/datasets.md).
