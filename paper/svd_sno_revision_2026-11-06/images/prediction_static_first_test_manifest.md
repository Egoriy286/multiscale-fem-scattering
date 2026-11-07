# Архив первых test-примеров: статические SVD-SNO

Рисунки используют **одну фиксированную среду**, по отдельному randomized SVD-SNO checkpoint на каждую полосу. Это текущие PyTorch-результаты с разбиением по частотам; они отличаются от исторического JAX-эксперимента статьи и от архивного пилота с 48 средами. Подходящего обученного FNO checkpoint для этих статических данных в проекте нет.

В каждой полосе: 8000 полей = 1000 частот x 8 направлений; train/validation/test = 5360/1360/1280 полей (670/170/160 частот). Physical media = 1; входные POD-ранги A,n = 0. Первый test index выбирается без просмотра ошибки. Средняя test-ошибка и ошибка показанного примера подписаны раздельно.

## low

- Датасет: `neural_operator/data/static_low_frequency_1000x8`.
- Checkpoint: `neural_operator/randomized/runs/low_frequency_static_1000x8/svd/model.pt`.
- Fingerprint: `786ee3849e909083d98213653affa47907312d0f626e8a60da76e88c70de0ca5`.
- Пример: test index 8, frequency_id 1, medium_id 0, k=1.001001001001, theta=0.033123693433 рад.
- Ошибка примера: **1.741020%**; средняя по всем 1280 test-полям: **0.961395%**, complex consistent FEM L2.
- Выполнено эпох: 200; лучшая: 170; ранги: `{'A': 0, 'n': 0, 'real': 10, 'imag': 10}`.
- `prediction_static_low_first_test_components.pdf/.png`: строки Re/Im/Abs, столбцы FEM / SVD-SNO / абсолютная ошибка соответствующей компоненты.

## high

- Датасет: `neural_operator/data/static_high_frequency_1000x8`.
- Checkpoint: `neural_operator/randomized/runs/high_frequency_static_1000x8/svd/model.pt`.
- Fingerprint: `198c3b0ad024b64aa349ddb7750183352e6a5065626895709b35fbb4d64a096a`.
- Пример: test index 8, frequency_id 1, medium_id 0, k=4.004004004004, theta=0.033123693433 рад.
- Ошибка примера: **14.912563%**; средняя по всем 1280 test-полям: **5.537526%**, complex consistent FEM L2.
- Выполнено эпох: 200; лучшая: 198; ранги: `{'A': 0, 'n': 0, 'real': 48, 'imag': 48}`.
- `prediction_static_high_first_test_components.pdf/.png`: строки Re/Im/Abs, столбцы FEM / SVD-SNO / абсолютная ошибка соответствующей компоненты.

Модуль вычисляется из реконструированных Re и Im. Ошибка Abs - `abs(abs(prediction)-abs(FEM))`, а не модуль комплексной ошибки. Для FEM и модели используется общая полная цветовая шкала в каждой строке, без отсечения экстремумов. Поле отображается на исходных FEM-треугольниках.

Проверены fingerprint модели и данных, config, геометрия и порядок узлов, все 1280 test indices, ошибки и их summaries, совпадение нового инференса с сохранённым полем. SHA256 файлов, sample/mean errors, конфигурации и проверки: `prediction_static_first_test_provenance.json`.

Это сохранённые иллюстрации первого test index 8. Текущие основные рисунки и скрипт используют детерминированный выбор около медианы ошибки; этот архив не является основным результатом статьи. Поля первых примеров находятся в исходном `test_predictions.npz`; точная привязка записана в JSON выше.
