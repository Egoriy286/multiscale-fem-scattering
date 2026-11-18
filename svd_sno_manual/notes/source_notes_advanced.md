# Источники и границы главы 03

Текст оригинальный, написан по локальному коду и описаниям; внешние источники и интернет не использовались. Production-файлы не изменялись; FEM, обучение и большие вычисления не запускались.

## Прочитанные исходники

- `experiments/mscale_sno_100/model.py`: re-export SpectralSurrogate и ComplexPOD, а не собственная новая архитектура.
- `experiments/mscale_sno_100/dataset.py`: checksum-проверенная независимая копия common 100-field benchmark, физический manifest и сравнение запросов.
- `experiments/mscale_sno_100/training.py`: train-only preprocessing, scalar s_a, физическая loss через ортогональный остаток, matching widths по параметрам, validation checkpoint, replay/provenance.
- `experiments/mscale_sno_100/METHOD.md` и `README.md`: постановка и ограничения именно серии 100 полей; к численным итогам глава не привязана.
- `experiments/gms_sno_100/model.py`: реальный class SpectralSurrogate, CoefficientBranch, CoefficientBlock; dense lift 3->RW, sine/GELU, D^(-1/2), независимые головы, обучаемые scales и gamma.
- `experiments/gms_sno_100/pod.py`: комплексный snapshot consistent mass POD, центрирование через offsets, порог 32 eps n scale, отсечение n-1, energy/rank/max_rank, Cholesky correction, encode/decode.
- `experiments/highk_svd_gmsfem/README.md`: native high-order reference vs сохранённая P1 representation, ограничения выводов и протокол high-k.
- `experiments/highk_svd_gmsfem/training.py`: fit_streaming_pod и TransferredSurrogate, train relative RMS rank rule, projection нового/старого decoder, zero-initialized residual heads, physical-k gate. Собственного model.py в этой папке нет: импортируется тот же SpectralSurrogate.
- `experiments/svd_sno_mesh_stability_2026_11_16/README.md`: независимые native CG2 сетки, fixed geometry interfaces, fresh training, common quadrature, заранее объявленные thresholds, статус незавершённой серии по README.
- `experiments/svd_sno_mesh_stability_2026_11_16/training.py`: импорт SpectralSurrogate, независимый train-only POD на каждой сетке, fixed rank192/W8/D4/J8 и протокол. Отдельного model.py нет.
- `neural_operator/mesh_inference.py`: source P1 transfer, отдельный перенос среднего и мод для SVD, static вход только при нулевых material ranks.
- `neural_operator/models.py`, `neural_operator/sno/model.py`, `neural_operator/sno/basis.py`: сравнение basic SVDSNO v2 с отдельными material coefficients/tokens, real POD и диагональной weighted метрикой.

## Проверяемые тонкости

1. c_j положительны только при инициализации. Во время обучения параметр ничем не ограничен.
2. gamma не softmax, сумма не обязана равняться 1. Среднее decoder добавляется один раз.
3. s_a = sqrt(mean |a|^2) по train и модам, не std отдельного канала/моды.
4. Cholesky correction сохраняет span, но слегка меняет точные spectral coordinates. Формула коэффициентов всегда через фактические сохранённые моды.
5. Energy tolerance в mscale_sno_100 и train relative RMS tolerance в highk — разные правила.
6. POD floor — граница для фиксированного affine decoder, не для всех возможных моделей.
7. highk README сообщает 1 255 744 trainable parameters итоговой модели. По формуле одного SpectralSurrogate с R192,W8,D4,J8 получается 1 233 312; в главе эти числа не отождествлены. Highk использует составной TransferredSurrogate. Точные параметры итогового checkpoint родитель может подтвердить отдельно по его config/state.
8. Никакого успеха незавершённой native mesh stability серии не заявлено. Из пилота FEM не выводится устойчивость сети.
9. Точность переноса T(mean+Phi a) = Tmean+(TPhi)a не означает сходимость относительно PDE.
10. Наблюдения P1 и native CG2 DOF — разные представления, также различаются их mass/quadrature метрики.

## Редакционная проверка

- Использованы только отдельные файлы `03_multiscale_practice.tex` и этот source notes.
- Метки имеют префикс `adv:`.
- 11 упражнений с выполненными решениями; простой rank-one SVD пример не повторяется.
- Числа в упражнениях 2, 5–7 и формула parameter count проверены арифметически.
- Листинг содержит только ASCII.
- Открытые вопросы к сборке: фактический объём страниц и ширины таблиц проверяются при сборке всей методички; зависимости preamble (tabularx/longtable/listings/path/AMS и пользовательские среды) предоставляет родитель.
