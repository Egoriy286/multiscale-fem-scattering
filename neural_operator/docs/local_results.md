# Результаты локального запуска — 2026-11-04

Этот отчёт фиксирует выполненный ранее пилот **48×4**. Текущие notebooks уже настроены на расширенный набор **1000×8**; новые данные и их проверка описаны в [dataset_expansion.md](dataset_expansion.md). Приведённые здесь ошибки и модели относятся к пилоту.

Все восемь notebooks выполнены последовательно через nbclient, каждый в отдельном чистом kernel локального Docker `dolfinx-complex`. Все шесть моделей действительно обучены на локальном CPU; cloud-сервисы не использовались. Повторный запуск использует сохранённые артефакты, совпадающие по полной конфигурации и data fingerprint.

В репозитории был исходный notebook с Drive-путями, но готового исходного Drive-датасета не было. Здесь сгенерирован новый физический pilot: 48 сред × 4 направления, 192 комплексных поля на 3221 CG1-узле и 6280 треугольниках. Материалы DG0 лежат в сохранённых `A_cells`, `n_cells`; узловые A,n предназначены для нейрооператоров. Split: 32/8/8 сред, то есть 128/32/32 train/val/test снимка. Полосы k=[1,2] и [4,8] используют идентичные среды, направления, сетку и split; высокочастотные решения вычислены заново.

Основная метрика во всех таблицах — относительная consistent complex FEM L2 ошибка рассеянного поля в процентах. Это ошибка относительно дискретного FEM reference. FEM не является аналитическим точным решением. Направления одной среды зависимы; интервалы неопределённости по этой одиночной seed-серии не оцениваются.

## Обученная полоса k∈[1,2]

| Метод | Средняя test ошибка | Медиана | p95 | Параметры | Выполнено / лучшая эпоха | Обучение CPU, с |
|---|---:|---:|---:|---:|---:|---:|
| SVD-SNO | 68.58% | 63.06% | 106.59% | 68666 | 58 / 8 | 4.99 |
| Chebyshev-SNO | 70.62% | 72.68% | 82.70% | 2426 | 200 / 196 | 17.74 |
| FNO | 38.84% | 37.20% | 59.65% | 445010 | 200 / 194 | 78.46 |

SVD-POD reconstruction error при r=48: mean **2.67%**, p95 4.74%. Входные ранги A,n равны 24, выходные Re/Im — 48. Validation-цель 1% при cap=48 не достигнута и не объявляется достигнутой. Диагностика до rank=96 находится в [SVD notebook](../notebooks/SVD-data-checks.ipynb). Lumped-POD reconstruction не является строгой нижней границей consistent FEM loss.

На этом небольшом числе сред SVD-SNO переобучается: train loss уменьшается, лучшая validation-эпоха наступает рано. Разрыв ошибки сети и POD-реконструкции показывает, что одно увеличение POD rank не решает задачу обобщения. У Chebyshev и FNO достигнут лимит 200 эпох; это не утверждение о сходимости. Число параметров указано в вещественных степенях свободы: один комплексный Fourier-вес считается за два. Числа параметров различаются; одинаковый максимум эпох не является одинаковым FLOP budget и не позволяет заявить универсальное превосходство архитектуры.

![Сохранённые спектры и validation-кривые](../runs/svd_diagnostics/svd_diagnostics.png)

## Paired сравнение с классическими методами

Ниже одна и та же тройка held-out сред, одно направление на среду. Это **другое множество**, чем полный neural test из 32 снимков выше. Сохранены все per-sample результаты. Coarse baseline: Q1 traces в fine P1 с Galerkin solve, а не независимая coarse P1 mesh. GMsFEM строится существующим `gmsfem.offline` для конкретных DG0-материалов/k; глобальные reference-решения при построении basis не используются.

| Метод | Средняя paired ошибка | Среднее warm online, мс | Offline basis/preprocessing, с |
|---|---:|---:|---:|
| Fine FEM reference | 0.00% | 6.660 | 0.000 |
| Coarse Q1 traces in fine P1 | 35.28% | 0.604 | 0.000 |
| GMsFEM Type 1, M=1 | 34.82% | 0.598 | 0.114 |
| GMsFEM Type 1, M=2 | 29.70% | 1.047 | 0.114 |
| GMsFEM Type 1, M=4 | 9.46% | 2.569 | 0.114 |
| GMsFEM Type 2, M=1 | 58.45% | 0.595 | 0.114 |
| GMsFEM Type 2, M=2 | 53.86% | 1.059 | 0.114 |
| GMsFEM Type 2, M=4 | 8.53% | 2.547 | 0.114 |
| SVD-SNO | 61.84% | 0.547 | 0.269 |
| Chebyshev-SNO | 67.49% | 0.906 | 0.086 |
| FNO | 39.53% | 1.803 | 0.094 |

Fine FEM имеет нулевую ошибку к собственным меткам по определению. Его assembly/read cost и LU solve сохранены раздельно. Для GMsFEM full online включает проекцию, solve, реконструкцию и residual validation; один offline basis до max M=4 переиспользуется внутри серии M. Для сети online включает входное кодирование и реконструкцию, а training cost приведён отдельно выше. Малые времена чувствительны к cache и общей загрузке VM; их не следует трактовать как переносимый коэффициент ускорения.

![Paired ошибки и online времена](../runs/comparison/paired_comparison.png)

## Ошибка входного и выходного переноса

Validation FEM→grid→FEM, resolution=32. Эти ошибки не равны neural error и не являются строгими lower bounds для модели с train-mean correction.

| Сетка | Поле | Mean FEM L2 | p95 |
|---|---|---:|---:|
| uniform | A | 1.07% | 1.33% |
| uniform | n | 4.71% | 5.93% |
| uniform | u_scat | 2.67% | 3.75% |
| chebyshev | A | 1.68% | 2.05% |
| chebyshev | n | 7.25% | 9.72% |
| chebyshev | u_scat | 4.04% | 5.20% |

## Высокочастотный эксперимент k∈[4,8]

Frozen = модели и preprocessing из low band без изменений. Retrained = новые train-only bases/stats и отдельное обучение на high band с тем же budget, seed и непересекающимися train/val/test средами.

| Метод | Режим | Mean test FEM L2 | p95 | POD reconstruction mean |
|---|---|---:|---:|---:|
| SVD-SNO | frozen_low_band | 112.56% | 133.85% | 84.24% |
| Chebyshev-SNO | frozen_low_band | 116.69% | 149.62% | — |
| FNO | frozen_low_band | 163.57% | 271.57% | — |
| SVD-SNO | retrained_high_band | 115.69% | 134.58% | 24.26% |
| Chebyshev-SNO | retrained_high_band | 100.89% | 106.22% | — |
| FNO | retrained_high_band | 93.53% | 106.15% | — |

При frozen SVD меняется не только отображение коэффициентов: low-band basis плохо представляет high-band поля. Переобучение уменьшает POD-реконструкцию, но для малого датасета network error остаётся большой. Эти результаты демонстрируют ограничение конкретного pilot; они не устанавливают невозможность высокочастотного neural operator.

![Ошибка в зависимости от k](../runs/high_frequency/comparison/frequency_errors.png)

## Проверка FEM reference на высокой частоте

Исходный h=0.1 сравнивается с h=0.05: 3221 → 8958 DOF. Исходное P1-поле переносится в узлы более мелкой сетки, разность интегрируется с refined consistent mass. Это оценка изменения дискретного решения, включающая перенос; не строгая апостериорная оценка ошибки относительно PDE.

| Test sample | k | Изменение h→h/2 |
|---:|---:|---:|
| 16 | 7.7795 | 14.94% |
| 28 | 7.4677 | 11.61% |
| 124 | 7.4228 | 11.50% |

**Изменение 11–15% означает, что h=0.1 недостаточно для уверенного вывода о высокой физической точности reference в этой полосе.** High-band таблица характеризует обучение к этим дискретным меткам. Для исследования точности непрерывной PDE нужны более тонкие общие данные и дальнейшая h-серия. Номинальное число точек на кратчайшую длину волны здесь примерно 5.37–7.76 по cellwise kh и само по себе не устраняет Helmholtz pollution error.

## Воспроизводимость и проверка

- CPU: Linux aarch64 в локальном Docker, 2 PyTorch threads; seed данных 2026, seed моделей 42. Версии сохранены в [environment.json](environment.json).
- Максимум 200 эпох, batch 32, AdamW lr=0.002, width=24, depth=3; POD cap=48, input cap=32; tensor grid=32², modes=8 по оси. Best epoch выбирается только по validation.
- **27 numerical/integration tests passed**: независимая UFL-сборка K/M/B/RHS, homogeneous scattering, group split, weighted POD, DCT-I/полиномы, signed Fourier bands/gradients, consistent loss и model bundle round trip.
- **8/8 notebooks выполнены без error outputs**, каждый с нового kernel. Inference не выполняет fit или FEM solve; single/batch parity проверена для всех трёх методов.
- Проверены сохранённые PNG спектров, полей, paired comparison и частотного эксперимента.
- `data/` и `runs/` — локальные воспроизводимые артефакты, исключённые из Git; исходники, notebooks с новыми outputs и этот отчёт доступны рядом.

Команды и порядок notebooks: [README](../README.md). [Первичные источники](references.md), [математические соглашения](methods.md), [аудит прежнего notebook](change_log.md).

Основные сохранённые таблицы: [paired_summary.csv](../runs/comparison/paired_summary.csv), [neural_full_test_summary.csv](../runs/comparison/neural_full_test_summary.csv), [frequency_summary.csv](../runs/high_frequency/comparison/frequency_summary.csv), [refinement.csv](../runs/high_frequency/comparison/refinement.csv).

Data fingerprint low band: `643f882b18d6e82cb75552e8dacb245c72eab30da13a1eff40ee0f97d0ce9746`. High band: `b3ab143dfd3b8a7de40ac06def5aba72b50a5cd13d055c2287fd84e4ae6ff7fe`.
