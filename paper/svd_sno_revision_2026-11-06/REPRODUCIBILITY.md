# Воспроизводимые примеры статьи

Все команды ниже выполняются из корня репозитория. Научное окружение — сервис `dolfinx-complex` из корневого compose. Источники не следует перезаписывать ради получения той же таблицы: новые обучения сохраняются в новые каталоги.

## Пример 1: фиксированный материал и удержанные частоты

Три активных круга: центры (-0,6;0,1), (0,4;0), (-0,3;-0,3), радиусы 0,05/0,10/0,05. Точные DG0-параметры и узловой порядок сохранены в `neural_operator/data/static_low_frequency_1000x8/`. Частотные группы: train/val/test=670/170/160; восемь направлений на частоту; 3221 FEM-узел. Полоса high использует тот же материал и split, но новые k и поля.

Точные метаданные без обучения или FEM:

```sh
docker compose exec -T dolfinx-complex python3 -m neural_operator.run inspect --band low
docker compose exec -T dolfinx-complex python3 -m neural_operator.run inspect --band high
```

Рисунки legacy SVD-SNO, выбранные по близости к медиане всех 1280 тестовых ошибок:

```sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_predictions.py
```

Скрипт проверяет fingerprint, mesh/DOF ordering, split, все сохранённые ошибки и повторное предсказание выбранного поля. Не обучает модель и не запускает FEM. Источники: `neural_operator/randomized/runs/{low,high}_frequency_static_1000x8/svd/`. Выбраны low sample 2211 (k=1,276276…, θ=2,782013…) и high sample 6077 (k=7,039039…, θ=4,245364…). Точные числа находятся в [provenance](images/prediction_static_provenance.json).

Низкочастотный FNO-рисунок использует тот же sample 2211 и общие цветовые шкалы. Его источник — `neural_operator/randomized/runs/low_frequency_static_1000x8/fno/`, 142 эпохи, best=92. Полный CPU-повторный инференс существующего checkpoint описан в `neural_operator/docs/fno_static_low_verification_2026-11-08.json`; обучение при этой проверке не выполнялось. Средняя ошибка округляется до тех же 1,6599%. High-band FNO не включена в таблицы статьи, поскольку завершённый подтверждённый результат отсутствует.

## Пример 2: новая версия SVD-SNO

Legacy-веса используют последовательное смешивание каналов и мод. Версия 2 использует помодовый вход и две ветви. Существующие bundles без номера версии интерпретируются как v1, а не меняют архитектуру при загрузке.

Реальный новый low-band запуск v2:

```sh
docker compose exec -T dolfinx-complex python3 -m neural_operator.run train \
  --method svd --pod-method randomized --band low --epochs 200 \
  --sno-architecture-version 2 \
  --output neural_operator/runs/revision_2026-11-08/static_low
```

Если соответствующий завершённый запуск существует, CLI повторно использует его. Для независимого обучения используйте другой `--output` и сохраните seed/config. Результат v2: 165 выполненных эпох, best=115, 4022 параметра, mean test=0,9474554%. Это новое обучение; оно не меняет legacy mean=0,9613946%. Оба используют один dataset fingerprint и те же train-only POD-ранги 0/0/10/10. Сопоставление одного seed не задаёт статистического ранжирования архитектур.

На том же наборе завершено отдельное обучение Chebyshev-SNO:

```sh
docker compose exec -T dolfinx-complex python3 -m neural_operator.run train \
  --method chebyshev --pod-method randomized --band low --epochs 200 \
  --output neural_operator/runs/revision_2026-11-08/static_low
```

200 эпох, best=199, 2426 параметров, mean test=7,936406%, p95=9,881075%. Моды 8×8, сетка 32×32. Оптимальный согласованный mean validation decoder floor=7,614263% ограничивает достижимую точность в этом фиксированном пространстве. Полный повторный CPU-инференс всех 1280 полей Chebyshev и v2 совпал с архивом; источник — `neural_operator/docs/operator_revision.json`, snapshot в evidence. Высокая ошибка не исключена из таблицы. Три метода обучены не в одном парном benchmark времени; общий рейтинг по скорости не заявляется.

## Пример 3: неудачный перенос и уточнение FEM

Точный машинный источник — `neural_operator/randomized/static_results.json`, копия в [evidence](evidence/revision_2026-11-08/static_results.json). Frozen low→high mean=144,1781%, POD reconstruction=96,7713%. Отдельная high-band сеть даёт 5,5375%. Их базисы и веса различаются.

High-band h=0,1→0,05, 3221→8958 узлов: сохранённые изменения 13,6564%, 13,5935%, 13,3219%. Поле h интерполируется на уточнённую сетку, а ошибка считается в её consistent P1 mass относительно refined FEM. Это не точная PDE-ошибка. Нельзя выдавать 5,54% surrogate-to-coarse-reference за установленную физическую точность.

## Пример 4: пилот новых материалов и Chebyshev/FNO

Сохранённые `neural_operator/data/{low,high}_frequency/` содержат по 192 поля, 48 материалов и четыре направления. Разбиение 32/8/8 сред, или 128/32/32 поля. Архивный контрольный эксперимент использовал width=24/depth=3/cap=48, Chebyshev/FNO resolution=32/modes=8. Сводки статьи независимо пересчитываются из [архивных per-sample CSV](evidence/revision_2026-11-08/pilot_low_samples.csv), [high CSV](evidence/revision_2026-11-08/pilot_high_samples.csv) и [paired CSV](evidence/revision_2026-11-08/pilot_paired_samples.csv).

Архивный low SVD mean=68,5807% относится к 68 666 параметрам. Фактически доступные low SVD-веса — другой run: width64/depth4/rank85, 328 550 параметров, mean=74,0115%. Нельзя запустить текущий `runs/low_frequency/svd/model.pt` и объявить это воспроизведением старой строки 68,58%. В таблице честно сохранена архивная серия, её статистика повторно проверена по отдельным строкам; отсутствие прежних весов обозначено явно.

## Проверка статьи, сборка и рисунки

```sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/validate_paper.py
sh paper/svd_sno_revision_2026-11-06/build.sh
```

`validate_paper.py` требует только NumPy. Проверяет SHA256 snapshots, 20 архивных агрегатов, LaTeX-метки и библиографические ключи, восемь включённых рисунков, численные строки статической таблицы, DCT-I для размеров 3/8/32, формулы числа параметров и неизменность оригинального исходника. Если доступен `build/main.log`, проверяет ошибки LaTeX, overfull, неопределённые ссылки и отсутствующие символы.

Намеренное обновление snapshots при выборе новых источников требует `--refresh-evidence`. Простая проверка не читает удалённые сервисы, не обновляет эксперименты и не переобучает сети.

Схемы и графики:

```sh
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_figures.py
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_architecture.py
docker compose exec -T dolfinx-complex python3 paper/svd_sno_revision_2026-11-06/prepare_fno_architecture.py
```

После сборки страницы необходимо отрендерить Poppler и просмотреть; успешный LaTeX сам по себе не является визуальной проверкой. Последний завершённый аудит фиксируется в [отчёте](evidence/revision_2026-11-08/validation_report.json). Исторический JAX не переобучался: доступны notebook extracts и журнал в `evidence/`, а его dataset/checkpoint не входят в комплект.
