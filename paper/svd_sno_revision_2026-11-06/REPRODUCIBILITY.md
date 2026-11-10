# Воспроизведение текущей статьи

Все команды выполняются из корня репозитория. Используется Docker-сервис `dolfinx-complex`. Данные: `neural_operator/data/static_low_frequency_1000x8`; материал и сетка фиксированы, k∈[1,2]. Dataset fingerprint:

`786ee3849e909083d98213653affa47907312d0f626e8a60da76e88c70de0ca5`.

## Проверить три завершённых запуска

Точные каталоги:

| Метод | Run |
|---|---|
| SVD-SNO v2 | `neural_operator/runs/revision_2026-11-08/static_low/svd` |
| FNO | `neural_operator/randomized/runs/low_frequency_static_1000x8/fno` |
| Chebyshev-SNO | `neural_operator/runs/revision_2026-11-08/static_low/chebyshev` |

Для проверки существующих весов и построения всех трёх рисунков:

```sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_predictions.py
```

Скрипт не обучает сеть и не решает FEM. Он проверяет fingerprint и mesh ordering, все 1280 test-индексов, consistent FEM-ошибки сохранённых предсказаний и mean/median/p95/max. Затем повторно вычисляет первый test-запрос каждой моделью на CPU. Все методы используют один индекс, выбранный до просмотра ошибок. Точные k,θ, индивидуальные ошибки и хеши записываются в [provenance](images/static_low_predictions_provenance.json).

## Конкретный материал и разбиение

Активны три круга: центры(-0,6;0,1),(0,4;0),(-0,3;-0,3), радиусы 0,05/0,10/0,05. Точные коэффициенты сохранены в исходных DG0-массивах и приведены в статье. Сетка содержит 3221 узел и 6280 треугольников.1000 равномерных частот сочетаются с восемью направлениями на частоту. Train/validation/test=670/170/160 частотных групп, или 5360/1360/1280 полей. Все восемь направлений одной частоты остаются вместе.

```sh
docker compose exec -T dolfinx-complex python3 -m neural_operator.run inspect --band low
```

Материал воспроизводится с seed 2026; сеть и перемешивание пакетов используют seed 42. Все data-dependent преобразования строятся по train. Test не участвует в выборе ранга, эпохи или decoder modes.

## Повторить обучение

Сохранённые SVD-SNO v2 и Chebyshev-SNO получены следующими настройками:

```sh
docker compose exec -T dolfinx-complex python3 -m neural_operator.run train \
  --method svd --pod-method randomized --band low --epochs 200 \
  --sno-architecture-version 2 \
  --output neural_operator/runs/revision_2026-11-08/static_low

docker compose exec -T dolfinx-complex python3 -m neural_operator.run train \
  --method chebyshev --pod-method randomized --band low --epochs 200 \
  --resolution 32 --modes 8 \
  --output neural_operator/runs/revision_2026-11-08/static_low
```

CLI повторно использует соответствующий завершённый run; для независимого обучения укажите другой `--output`. SVD-SNO требует version2. Для Chebyshev параметр `pod-method` выбирает профиль каталогов, POD внутри модели не выполняется. FNO существующего run использует resolution 32,modes 8,padding 8,width 24,depth 3,batch 32,lr0,002,seed 42 и MPS; её повторная CPU-проверка не является новым обучением.

Все модели используют AdamW, weight_decay=1e-5, gradient clipping=1, patience 50, min_epochs 50. SVD-SNO завершила 165 эпох,best 115; FNO142,best 92; Chebyshev200,best 199. Устройства и ёмкости отличаются, поэтому времена отдельных runs не задают равный budget или переносимый speedup.

## Диагностика представления

SVD-SNO использует train-only randomized POD,10 дополнительных направлений и две power iterations. Общий выходной ранг 10 выбран по validation reconstruction target1%; A,n имеют нулевые входные ранги. Loss вычисляет consistent FEM-норму через Gram-матрицу, а не MSE нормированных коэффициентов.

Для фактического Chebyshev-декодера 8×8 мод/grid 32 оптимальная mean validation-ошибка равна 7,614263%,p95=9,526629%. Текущий trained test mean7,936406% — другая характеристика и другая часть выборки. Диагностические значения находятся в [snapshot](evidence/current_static_low/chebyshev_decoder_validation.json). Строгая нижняя граница относится к согласованной проекции в фактический decoder span с билинейным переносом в FEM-узлы.

## Статья, схемы и проверка

```sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_figures.py
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_architecture.py
sh paper/svd_sno_revision_2026-11-06/build.sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/validate_paper.py
```

Скрипт проверки требует только NumPy. Он читает [snapshots](evidence/current_static_low/sources.json), проверяет исходные хеши, три строки результатов, сохранённые completed/best epochs, фигуры/ссылки/библиографию, DCT-I и число параметров. Намеренное обновление источников требует `--refresh-evidence`.

После фактического просмотра всех финальных страниц:

```sh
python3 paper/svd_sno_revision_2026-11-06/validate_paper.py --visual-pages-reviewed N
sh paper/svd_sno_revision_2026-11-06/build.sh --clean
```

Замените N реальным числом просмотренных страниц. Скрипт лишь записывает факт проведённой проверки и хеш PDF; он сам не выполняет визуальный просмотр. Текущий [отчёт](evidence/validation_report.json) различает проверку исходника и завершённую проверку PDF. Авторская исходная папка `original/` не служит источником текущих измеренных результатов.
