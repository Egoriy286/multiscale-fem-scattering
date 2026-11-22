# Датасеты для агентов

Пути указаны относительно корня репозитория. В Git хранятся протоколы, метрики и контрольные суммы. Массивы полей, веса и часть файлов предобработки находятся в локальных игнорируемых каталогах.

| Датасет | Архив и папка с частями в Drive | Частей | Локальный каталог | FEM-поля | Источник для сверки |
| --- | --- | ---: | --- | ---: | --- |
| Static-low, 1000 частот × 8 направлений | [static-low-1000x8.tar.zst](https://drive.google.com/drive/folders/1Klxqr4JxmLmt_2Bga_fbMYHQIWx6Cd80) | 4 | `neural_operator/data/static_low_frequency_1000x8/` | 8000 | [Аудит NPZ](../../neural_operator/docs/dataset_audit.json) |
| Reference SNO, low | [reference-sno-low.tar.zst](https://drive.google.com/drive/folders/1I-3ZMWRK9w3alGDZrWM8AEOEL3nwFkHd) | 11 | `experiments/reference_sno_gmsfem/data/low/` | 1288 | [Аудит двух полос](../../experiments/reference_sno_gmsfem/results/dataset_verification.json) |
| Reference SNO, high | [reference-sno-high.tar.zst](https://drive.google.com/drive/folders/1veWoxq_WczCDuBOZL9qHsy-DeD5427XR) | 11 | `experiments/reference_sno_gmsfem/data/high/` | 1280 | [Аудит двух полос](../../experiments/reference_sno_gmsfem/results/dataset_verification.json) |
| Shared SNO, объединение low и high | [shared-sno.tar.zst](https://drive.google.com/drive/folders/1JoSIfaBeU1APFY8bEvkMFyGyd9yeiv8R) | 21 | `experiments/shared_sno_gmsfem/data/` | 2568 | [Происхождение данных и подготовка](../../experiments/shared_sno_gmsfem/README.md#общий-датасет) |
| Multiscale sample study | [multiscale-sample-study.tar.zst](https://drive.google.com/drive/folders/18W2rKrTds4ZI8NaLRuhws876oV0mcmt4) | 10 | `experiments/multiscale_sample_study/data/` | 1200 | [Протокол](../../experiments/multiscale_sample_study/protocol.json) |
| High-k SVD-SNO | [highk-svd-sno.tar.zst](https://drive.google.com/drive/folders/1Il35Ds6peu45nrsgsHZxcUm5yfg4Wg05) | 45 | `experiments/highk_svd_gmsfem/data/` | 2400 | [Протокол](../../experiments/highk_svd_gmsfem/protocol.json) |

Для объединённого Shared SNO нужны обе исходные полосы Reference SNO. Стадия `prepare` копирует готовые FEM-поля и сохраняет принадлежность к split. High-k дополнительно использует данные, POD-базис и три checkpoint из sample study. Их точные пути и хеши записаны в `inherited_source_sha256` протокола high-k.

Серия по сеточной устойчивости от 16 ноября 2026 года закрыта в статусе `retired`. Локальные данные, сетки и журналы серии удалены. [Протокол использовался в учебном пособии](../../experiments/svd_sno_mesh_stability_2026_11_16/README.md).

## Снимок для Google Drive

Снимок соответствует коммиту `624516e95cc9d393e4695a3b9318ac81627791db`. В архивах сохранены пути относительно корня репозитория, конфигурации, FEM-сетки, разбиения выборки и массивы полей. Веса и предобработка для повторения обученных запусков перечислены в протоколах. [Код снимка](https://github.com/Egoriy286/multiscale-fem-scattering/tree/624516e95cc9d393e4695a3b9318ac81627791db) · [Папка снимка в Google Drive](https://drive.google.com/drive/folders/1wWsmUZ_fj4xJYgPCjibQDOsgc2I1aQm6)

Для каждого набора предусмотрена подпапка с именем архива без `.tar.zst`. Скачайте из неё все файлы `.part-NNN` и `PARTS.sha256`, а из общей папки скачайте `SHA256SUMS`. Сохраните их в одном временном каталоге. [Контрольные суммы архивов](https://drive.google.com/file/d/1s78En0oN0dBqEZDP6Yf6ruutYzNa9YiV/view?usp=drivesdk)

Из каталога загрузки выполните команды, подставив имя архива из таблицы. Для распаковки нужны `zstd` и `tar`.

```bash
set -euo pipefail
export LC_ALL=C
archive=multiscale-sample-study.tar.zst
shasum -a 256 -c PARTS.sha256
cat "$archive".part-[0-9][0-9][0-9] > "$archive"
awk -v name="$archive" '$2 == name' SHA256SUMS | shasum -a 256 -c
restore_dir=$(mktemp -d)
zstd -dc "$archive" | tar -xf - -C "$restore_dir"
printf '%s\n' "$restore_dir"
```

Проверьте распакованные файлы по манифесту датасета и перенесите их в локальный каталог из таблицы. Если каталог уже существует, сверяйте хеши перед заменой.

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
