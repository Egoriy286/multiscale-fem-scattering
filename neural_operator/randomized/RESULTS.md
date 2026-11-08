# Актуальные результаты static-low

Одна фиксированная среда, k∈[1,2], 8000 полей; split по частотам: 5360 train / 1360 validation / 1280 test. Метрика — complex consistent P1 FEM L2 рассеянного поля.

| Метод | Mean test | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| SVD-SNO v2 | 0.9475% | 0.8671% | 1.6534% | 2.8962% |
| FNO | 1.6599% | 1.6329% | 1.9505% | 2.2305% |
| Chebyshev-SNO modes8 | 7.9364% | 7.6838% | 9.8811% | 11.1065% |

Randomized train-only POD используется только SNO. Входные ранги A,n равны нулю, выходные Re/Im — 10; модель получает три изменяемых признака волны. FNO обучена на MPS и проверена на CPU; SNO/Chebyshev обучены на CPU. Это разные архитектуры с разным числом параметров и завершённых эпох.

[Полный отчёт, ограничения и команды](../docs/operator_revision.md) · [Конфигурации и SHA-256](../docs/operator_revision.json) · [FNO provenance](../docs/fno_static_low_verification_2026-11-08.json) · [Validation floors Chebyshev](../docs/chebyshev_decoder_validation_2026-11-08.json) · [Inference notebook](../notebooks/Static-predictions.ipynb).

Эти значения сохранены без изменения и относятся только к текущему static-low test. Они не подтверждают перенос на другие материалы или точность относительно непрерывной PDE.
