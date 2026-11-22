# Рассеяние волн при переменных коэффициентах среды

[Статья](main.pdf) · [Исходный текст](main.tex) · [SVD-SNO](sno_summary.csv) · [GMsFEM](gms/summary.csv) · [Приращения](sensitivity.csv)

Одна строка данных соответствует одной среде и одному направлению при k=2. В каждой среде восемь направлений. Коэффициенты alpha1, n1 относятся к малым и средним кругам (теги 2 и 3), alpha2, n2 к крупным (тег 4). Все четыре величины в каждой строке заданы в data/queries.npz. Для фоновых ячеек оба скаляра равны 1. Архив reference_problem.npz хранит геометрию и материалы исторической задачи, использованной для сверки решателя. Материалы новых расчётов восстанавливаются по запросам и тегам.

Строки с кодами split 0, 1, 2 и 3 соответствуют обучению, валидации, тесту и отдельному парному контролю. Эталонные решения находятся в data/fields.npy, их контрольные суммы в data/complete.json. Тестовые результаты сопоставляются по row в sno_cases.csv и index в gms/per_case.csv.

## Воспроизведение

Из корня репозитория с работающим сервисом dolfinx-complex:

```sh
docker compose exec -T dolfinx-complex python3 paper/variable_media_2026/generate.py
VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 .venv/bin/python paper/variable_media_2026/train.py
docker compose exec -T -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 dolfinx-complex python3 -m paper.variable_media_2026.gms.run all
.venv/bin/python paper/variable_media_2026/plots.py
.venv/bin/python paper/variable_media_2026/results.py
cd paper/variable_media_2026
latexmk -xelatex -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

Для сборки PDF достаточно готовых каталогов figures и tables. Исходник схемы сети находится в figures/architecture.tex.
