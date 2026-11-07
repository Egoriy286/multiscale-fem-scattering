# Текущие статические SVD-SNO: происхождение рисунков

Рисунки используют **одну фиксированную среду**, по отдельному randomized SVD-SNO checkpoint на каждую полосу. Это текущие PyTorch-результаты с разбиением по частотам; они отличаются от исторического JAX-эксперимента статьи и от архивного пилота с 48 средами. На момент генерации подходящего обученного FNO checkpoint для этих статических данных в проекте нет.

В каждой полосе: 8000 полей = 1000 частот x 8 направлений на частоту; направления выбираются отдельно для каждой частоты, общей сетки из 8 углов нет. Train/validation/test = 5360/1360/1280 полей (670/170/160 непересекающихся частот). Physical media = 1; входные POD-ранги A,n = 0. Выбран пример, ближайший к медиане сохранённой complex FEM L2 ошибки по всем 1280 test-полям: `np.argmin(abs(saved_errors - median(saved_errors)))`, при равенстве берётся первая позиция. Это типичный по величине ошибки пример, не лучший. Средняя test-ошибка и ошибка показанного примера подписаны раздельно.

## low

- Датасет: `neural_operator/data/static_low_frequency_1000x8`.
- Checkpoint: `neural_operator/randomized/runs/low_frequency_static_1000x8/svd/model.pt`.
- Fingerprint: `786ee3849e909083d98213653affa47907312d0f626e8a60da76e88c70de0ca5`.
- Пример: test index 2211, frequency_id 276, medium_id 0, k=1.276276276276, theta=2.782013344835 рад.
- Ошибка примера: **0.882475%**; средняя по всем 1280 test-полям: **0.961395%**, complex consistent FEM L2.
- Выполнено эпох: 200; лучшая: 170; ранги: `{'A': 0, 'n': 0, 'real': 10, 'imag': 10}`.
- `prediction_static_low_components.pdf/.png`: строки Re/Im/Abs, столбцы FEM / SVD-SNO / абсолютная ошибка соответствующей компоненты.

## high

- Датасет: `neural_operator/data/static_high_frequency_1000x8`.
- Checkpoint: `neural_operator/randomized/runs/high_frequency_static_1000x8/svd/model.pt`.
- Fingerprint: `198c3b0ad024b64aa349ddb7750183352e6a5065626895709b35fbb4d64a096a`.
- Пример: test index 6077, frequency_id 759, medium_id 0, k=7.039039039039, theta=4.245364524507 рад.
- Ошибка примера: **5.066963%**; средняя по всем 1280 test-полям: **5.537526%**, complex consistent FEM L2.
- Выполнено эпох: 200; лучшая: 198; ранги: `{'A': 0, 'n': 0, 'real': 48, 'imag': 48}`.
- `prediction_static_high_components.pdf/.png`: строки Re/Im/Abs, столбцы FEM / SVD-SNO / абсолютная ошибка соответствующей компоненты.

Модуль вычисляется из реконструированных Re и Im. Ошибка Abs - `abs(abs(prediction)-abs(FEM))`, а не модуль комплексной ошибки. Для FEM и модели используется общая полная цветовая шкала в каждой строке, без отсечения экстремумов. Поле отображается на исходных FEM-треугольниках.

Проверены fingerprint модели и данных, config и metadata, геометрия и порядок узлов, 1000 частот с 8 направлениями, непересекающиеся frequency splits и их размеры, все 1280 test indices, ошибки и их summaries, совпадение нового инференса с сохранённым полем. Все локальные model.pt проверены на наличие подходящего FNO по dataset fingerprint. SHA256 файлов, sample/mean errors, конфигурации, проверки и аудит FNO: `prediction_static_provenance.json`.

Воспроизведение из корня репозитория:

```sh
docker compose exec -T dolfinx-complex python paper/svd_sno_revision_2026-11-06/prepare_static_predictions.py
```

Скрипт не выполняет обучение или FEM solves.
