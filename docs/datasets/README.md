# Датасеты для агентов

Пути указаны относительно корня репозитория. В Git хранятся протоколы, метрики и контрольные суммы. Массивы полей, веса и часть файлов предобработки находятся в локальных игнорируемых каталогах.

| Датасет | Локальный каталог | FEM-поля | Источник для сверки |
| --- | --- | ---: | --- |
| Static-low, 1000 частот × 8 направлений | `neural_operator/data/static_low_frequency_1000x8/` | 8000 | [Аудит NPZ](../../neural_operator/docs/dataset_audit.json) |
| Reference SNO, low | `experiments/reference_sno_gmsfem/data/low/` | 1288 | [Аудит двух полос](../../experiments/reference_sno_gmsfem/results/dataset_verification.json) |
| Reference SNO, high | `experiments/reference_sno_gmsfem/data/high/` | 1280 | [Аудит двух полос](../../experiments/reference_sno_gmsfem/results/dataset_verification.json) |
| Shared SNO, объединение low и high | `experiments/shared_sno_gmsfem/data/` | 2568 | [Происхождение данных и подготовка](../../experiments/shared_sno_gmsfem/README.md#общий-датасет) |
| Multiscale sample study | `experiments/multiscale_sample_study/data/` | 1200 | [Протокол](../../experiments/multiscale_sample_study/protocol.json) |
| High-k SVD-SNO | `experiments/highk_svd_gmsfem/data/` | 2400 | [Протокол](../../experiments/highk_svd_gmsfem/protocol.json) |

Для объединённого Shared SNO нужны обе исходные полосы Reference SNO. Стадия `prepare` копирует готовые FEM-поля и сохраняет принадлежность к split. High-k дополнительно использует данные, POD-базис и три checkpoint из sample study. Их точные пути и хеши записаны в `inherited_source_sha256` протокола high-k.

Серия по сеточной устойчивости от 16 ноября 2026 года закрыта в статусе `retired`. Локальные данные, сетки и журналы серии удалены. [Протокол использовался в учебном пособии](../../experiments/svd_sno_mesh_stability_2026_11_16/README.md).

## Хранение и восстановление через Google Drive

1. Для архива возьмите каталог из таблицы целиком. Вместе с полями сохраните сетку, конфигурацию, `dataset_info.json` и `samples.npz`, если набор использует этот файл. Восстанавливаемый `.array_cache/` можно исключить. Для сохранённых моделей включите исходные веса и файлы предобработки, перечисленные в протоколе.
2. Из папки с архивом выполните `shasum -a 256 snapshot.tar.gz > SHA256SUMS`, подставив имя своего архива. Загрузите оба файла в Google Drive. Запишите рядом ссылку на архив, исходный путь из таблицы и commit проекта.
3. Скачайте архив и контрольную сумму во временный каталог. Выполните `shasum -a 256 -c SHA256SUMS` из этого каталога. Распакуйте архив во временный каталог и проверьте файлы по манифесту датасета. Затем разместите проверенный набор по исходному пути, указанному для снимка.

## Проверка перед запуском

Для наборов с `u_scat.npy` используйте `sha256` из `dataset_info.json`. При сверке sample study и high-k сопоставьте этот словарь с `dataset_sha256` из протокола. Для static-low сверяйте SHA256 `fem_dataset.npz` с аудитом NPZ в таблице.

Пример проверки набора с `u_scat.npy` из корня репозитория на Python 3.11 или новее. Для проверки после распаковки укажите путь к временной папке.

```sh
python3 - experiments/multiscale_sample_study/data <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
info = json.loads((dataset / "dataset_info.json").read_text())
if info.get("complete") is not True or not info.get("sha256"):
    raise SystemExit("Датасет не завершён или отсутствуют контрольные суммы")
for name, expected in info["sha256"].items():
    with (dataset / name).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Расхождение SHA256 для {name}")
print(f"Проверено файлов — {len(info['sha256'])}")
PY
```

Для повторения опубликованного запуска используйте сохранённые частоты, split и порядок FEM-узлов. Точные параметры генерации, обучения и проверки приведены в README соответствующего эксперимента и в [описании пайплайна](../PIPELINE.md).
