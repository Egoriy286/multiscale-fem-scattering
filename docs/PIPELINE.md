# Docker и пайплайн экспериментов

Все команды — из корня проекта. На macOS сначала запустите OrbStack.
Для Linux достаточно работающего Docker Engine с Compose v2 или новее.

```bash
make docker-build
make docker-up
make doctor
make pipeline
```

Один образ содержит complex DOLFINx/PETSc, Gmsh, CPU PyTorch 2.9.1 и
зависимости анализа. Базовый FEM-образ закреплён SHA256 и поддерживает ARM64
и AMD64. Healthcheck проверяет совместный импорт FEM/Torch, комплексный
тип PETSc и двумерное FFT. Jupyter доступен на `http://localhost:8888`;
токен можно посмотреть через `bash scripts/docker.sh compose logs`.
Сервис автоматически запускается после перезапуска Docker (`unless-stopped`).
Для явной остановки: `bash scripts/docker.sh compose stop dolfinx-complex`;
для следующего запуска: `make docker-up`.

По умолчанию запускается незавершённый опыт `mscale_fno_100`:

1. Проверка контейнера и численных тестов.
2. `prepare`: проверка или возобновление генерации 100 FEM-полей.
3. `train`: FNO GELU, FNO sine и MscaleFNO, seeds 11/29/47, по 300 эпох.
4. `baselines`: FEM и GMsFEM Type 1/2, сетка 20×20, M=8/16 на тех же 15 test-частотах.
5. `verify`: повторная загрузка всех девяти checkpoints, пересчёт метрик и контроль полноты сравнения.
6. `report`: таблицы и рисунки после успешной проверки.

Данные и веса остаются в `experiments/mscale_fno_100/{data,runs}`.
Результаты — в `results/REPORT.md`, `summary.csv`, `verification.json` и PNG.
Среда, исходники и настройки фиксируются в JSON; выбор checkpoint использует
только validation. Неудачный критерий точности остаётся численным результатом
и явно указывается в отчёте.

```bash
make status
python3 scripts/pipeline.py run --from-stage verify
make pipeline EXPERIMENT=shared_sno_gmsfem
make pipeline EXPERIMENT=mscale_sno_100
```

`mscale_sno_100` — отдельный POD/SVD-SNO опыт с обычными GELU/sine SNO и
масштабированными sine-ветвями, тремя seeds и validation early stopping.
Он использует те же физические запросы, FEM-метки и GMsFEM Type 1/2 M=8/16,
что `mscale_fno_100`; существующие файлы копируются после проверки всех SHA256.
Происхождение данных и baseline сохраняется отдельно от нового обучения.
При отсутствии готовых данных FEM-генерация выполняется в собственном каталоге.
Готовые поля позволяют обучить и проверить опыт без DOLFINx на хосте:

```bash
.venv/bin/python -m experiments.mscale_sno_100.run all
```

Итоговый отчёт и парное сравнение создаются в
`experiments/mscale_sno_100/results/REPORT.md` после успешного `verify`.

## Серия по размеру train-набора

`make sample-data` готовит 1200 новых FEM-полей в Docker. `make sample-study`
обучает четыре метода на вложенных train-наборах 400/600/800/1000 с общими
100 validation и 100 test, выполняет replay и независимый физический аудит.
`make sample-status` показывает число завершённых запусков из 48.

Этот отдельный workflow использует native macOS `.venv/bin/python`: CPU для
SNO и MPS для FNO. Docker остаётся средой complex FEM и GMsFEM. Настройки,
ресурсное ограничение 100 эпох и отличия от статьи описаны в
`experiments/multiscale_sample_study/PROTOCOL.md`. Источником точных физических
частот служит сохранённый `samples.npz`, а не заново округлённый планировщик
из другой версии NumPy. Модели не возобновляются с весов меньшего train-набора.

Статус и логи каждой попытки сохраняются в `experiments/<имя>/pipeline/`:
`latest.json` содержит текущую стадию, exit code и путь к логу; отдельный
каталог содержит полный manifest и логи завершённых стадий. При ошибке
следующие стадии не запускаются. Одновременная запись двух pipeline-процессов
в один опыт запрещена файловой блокировкой.

Повторный запуск с тем же протоколом использует готовые данные, baseline и
завершённые модели. MscaleFNO-training сохраняет после каждой эпохи модель,
optimizer, scheduler, RNG, историю и лучший checkpoint в `resume.pt`.
Прерванная эпоха вычисляется заново. Неполный запуск без `resume.pt`, изменение
исходников, числа эпох, seeds или данных требует отдельного каталога опыта;
пайплайн не смешивает несовместимые результаты. Сохранение `resume.pt` может
добавлять время и дисковое место; это учитывается в реальном training time.

`--epochs` и `--seeds` задаются до первого обучения и затем должны оставаться
теми же. `--skip-tests` предназначен для продолжения после уже пройденных
тестов. Для пересборки вместе с запуском есть `--build`.

На этом Mac системный `docker-credential-osxkeychain` зависал при сборке.
`scripts/docker.sh` использует отдельный `tmp/docker-cli` для публичных
образов, без обращения к Keychain и изменения настроек пользователя.
Явно заданный `DOCKER_CONFIG` имеет приоритет, например для авторизованного
registry. Если заданы `DOCKER_CONTEXT` или `DOCKER_HOST`, wrapper их сохраняет;
иначе на macOS использует доступный сокет OrbStack.

## GitHub Actions

`.github/workflows/experiments.yml` собирает тот же Docker-образ, проверяет
healthcheck и запускает численные тесты на push и pull request. Через
`workflow_dispatch` с `full_experiment=true` можно запустить все стадии опыта.
Логи и компактные отчёты сохраняются как artifact на 14 дней. Обучение не
включается на каждый push; существующие локальные датасеты и веса в CI не
передаются. Полный запуск в чистом checkout генерирует поля заново.

Workflow начинает работать после отправки файлов в GitHub. Локальная
проверка тех же команд не подтверждает отдельный запуск GitHub Actions.
Используются официальные [checkout](https://github.com/actions/checkout)
и [upload-artifact](https://github.com/actions/upload-artifact).
