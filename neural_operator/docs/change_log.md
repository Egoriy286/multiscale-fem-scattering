# Изменения текущей реализации

## SNO v2 и общий static-low workflow

- Сохранены SVD-SNO, Chebyshev-SNO и FNO с общими данными, split, consistent complex FEM L2 loss и переносимым inference bundle.
- SNO v2 использует coefficient tokens и параллельные локальную/модальную ветви. Входные POD-моды не обрезаются, если их больше выходных. Совместимость с форматом v1 сохранена в коде загрузчика.
- Исправлены zero/zero loss, привязка отрицательных Fourier-весов при clipping, проверка размеров DCT-I и явный отказ от комплексного ввода в вещественный POD.
- CLI предоставляет `--sno-architecture-version`, `--resolution`, `--modes`. Генерация FEM, диагностика, обучение и inference запускаются отдельно.
- Актуальные результаты относятся к одной фиксированной среде и k∈[1,2]: mean test FEM L2 **0.9475% SNO v2**, **1.6599% FNO**, **7.9364% Chebyshev modes8**. Полные конфигурации и hashes находятся в [отчёте](operator_revision.md).
- Для Chebyshev вычислены validation-only consistent decoder floors; они отделены от ошибок обученной сети. Все три метода сохраняют два независимых выхода Re/Im и вычисляют Abs после физического декодирования.
- В notebooks нет сохранённых execution outputs. [Static-predictions](../notebooks/Static-predictions.ipynb) читает только текущие static-low checkpoints. Шаблоны новых исследований сеток и частот не содержат опубликованных результатов.

Текущий протокол сохраняет train-only preprocessing, validation selection, итоговый test и проверку checkpoint из чистого процесса. Результаты и ограничения описаны в [методичке](svd_sno_guide.md), [математическом протоколе](methods.md) и [карте проекта](structure.md).
