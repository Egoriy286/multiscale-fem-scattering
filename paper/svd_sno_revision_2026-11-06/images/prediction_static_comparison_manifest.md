# Статические SVD-SNO и FNO: проверенное сравнение

Текущие PyTorch-модели используют одну фиксированную среду в каждой полосе: 8000 полей = 1000 частот x 8 направлений на частоту. Восемь различных направлений выбираются отдельно для каждой частоты, общей угловой сетки нет. Train/validation/test: 5360/1360/1280 полей, 670/170/160 непересекающихся частот. Проверены одинаковые dataset fingerprints, mesh и порядок узлов обеих моделей.

Пример выбирается исключительно по SVD-SNO: ближайшая к медиане ошибка среди всех 1280 сохранённых test-полей, первая позиция при равенстве. FNO не влияет на выбор; выбор не минимизирует ошибку одной из архитектур. Этот выбор типичен по ошибке SNO, но не гарантирует медианную ошибку FNO. Средние ошибки используют все test-поля.

## low

- Датасет: `neural_operator/data/static_low_frequency_1000x8`.
- Fingerprint: `786ee3849e909083d98213653affa47907312d0f626e8a60da76e88c70de0ca5`.
- Общий test index: 2211; k=1.276276276276; theta=2.782013344835 рад.
- SVD-SNO: пример **0.882475%**; средняя test **0.961395%**; эпох 200, лучшая 170; training device `cpu`, versions `{'torch': '2.15.0+cpu', 'numpy': '2.4.6'}`; checkpoint `neural_operator/randomized/runs/low_frequency_static_1000x8/svd/model.pt`.
- FNO: пример **1.748813%**; средняя test **1.659892%**; эпох 142, лучшая 92; training device `mps`, versions `{'torch': '2.15.0', 'numpy': '2.5.4'}`; checkpoint `neural_operator/randomized/runs/low_frequency_static_1000x8/fno/model.pt`.
- `prediction_static_low_comparison_components.pdf/.png`: обзор 3x5 (FEM, SVD-SNO, ошибка SVD-SNO, FNO, ошибка FNO).
- `prediction_static_low_comparison_sno_components.pdf/.png` и `prediction_static_low_comparison_fno_components.pdf/.png`: отдельные 3x3 рисунки для статьи с теми же шкалами.

Строки: Re, Im, Abs рассеянного поля. Модуль вычисляется из реконструированных Re/Im. Ошибки компонент: abs(Re(pred)-Re(FEM)), abs(Im(pred)-Im(FEM)), abs(abs(pred)-abs(FEM)); последняя не является модулем комплексной ошибки. Цветовые шкалы полей общие для FEM и обеих моделей в каждой строке; шкалы абсолютных ошибок общие для обеих моделей. Диапазоны полные, без отсечения. Рисунки используют исходные FEM-треугольники.

Обе модели проверены по fingerprint, configuration и metadata checkpoint, split, mesh, всем 1280 сохранённым test indices, пересчитанным consistent complex FEM L2 ошибкам и mean/median/p95/max; новый инференс показанного примера сверен с архивом. Configuration совпадает между моделями за исключением method и, возможно, device. Точные config и версии обучения сохранены отдельно для каждой модели; устройства обучения могут отличаться (CPU/MPS), поэтому сравнение времени и производительности не проводится. Проверочный инференс обоих checkpoint выполняется на CPU. Сравнение относится к этим обученным моделям и настройкам, а не устанавливает универсальный рейтинг архитектур.

Подробные SHA256, конфигурации, проверки и численные шкалы: `prediction_static_comparison_provenance.json`. Метрики всех четырёх моделей: `prediction_static_comparison_summary.json` и `.csv` (ошибки в процентах).

```sh
docker compose exec -T dolfinx-complex python paper/svd_sno_revision_2026-11-06/prepare_static_comparison.py --band low
```

Скрипт не обучает модели и не выполняет FEM solves.
