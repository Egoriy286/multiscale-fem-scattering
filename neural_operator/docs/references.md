# Первичные источники и протокол экспериментов

Дата проверки: 2026-11-04. Формулы FEM-метрики, weighted POD и разбиения в [methods.md](methods.md) служат локальным экспериментальным протоколом; численные результаты из статей не переносятся на эту задачу.

| Источник | Что использовано и граница сопоставления |
| --- | --- |
| V. Fanaskov, I. Oseledets, **Spectral Neural Operators**, arXiv:2205.10573 (2022), [статья](https://arxiv.org/abs/2205.10573), [авторский код](https://github.com/VLSF/SNO) | §3 и приложение C.1.3: коэффициенты рядов Чебышёва/Фурье, смешивание мод и каналов. Замена фиксированного спектрального базиса на POD и применение к FEM scattering — локальная адаптация, а не заявленная репликация авторских экспериментов. |
| Z. Li et al., **Fourier Neural Operator for Parametric Partial Differential Equations**, ICLR 2021, [статья](https://arxiv.org/abs/2010.08895) | §3: Fourier kernel, truncation спектральных мод, локальное channel mixing и нелинейность на физической сетке. Оригинальные тесты Burgers/Darcy/Navier–Stokes не дают гарантии для Helmholtz scattering и данного переноса FEM↔grid. |
| L. Lu et al., **A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data**, CMAME 393 (2022), 114778, [препринт](https://arxiv.org/abs/2111.05512), [DOI](https://doi.org/10.1016/j.cma.2022.114778) | POD-DeepONet: фиксированные POD-моды выходного поля вместо trunk network. Это первичный пример data-adaptive выходного базиса. Его архитектура отличается от coefficient-space SNO. |
| Z. Cheng, Z. Wang, L.-L. Wang, M. Azaiez, **PODNO: Proper Orthogonal Decomposition Neural Operators**, [препринт](https://arxiv.org/abs/2504.18513), [журнальная версия](https://doi.org/10.1137/25M1751372), [авторский код](https://github.com/Zilan-Cheng/PODNO) | PODNO меняет Fourier-преобразования FNO-подобного оператора на ортонормированные POD-преобразования; исследуются осциллирующие дисперсионные задачи. Это релевантное сравнение, но не источник для утверждения, что произвольная сеть над SVD coefficients воспроизводит PODNO. |
| U. Kalachikova, M. Vasilyeva, I. Harris, E. T. Chung, **Generalized Multiscale Finite Element Method for scattering problem in heterogeneous media**, JCAM 424 (2023), 114977, [DOI](https://doi.org/10.1016/j.cam.2022.114977), [страница издателя](https://www.sciencedirect.com/science/article/abs/pii/S0377042722005751) | Локальная PDF-копия `references/10 GMsFEM for scattering problem.pdf`, стр. 3–7: scattering с absorbing boundary, fine FEM, два snapshot spaces (elliptic/full Helmholtz), локальная spectral reduction и global coarse solve. Для сравнения требуется одинаковая дискретная задача. |

## Что означает GMsFEM baseline

В статье сначала строятся локальные harmonic extensions граничных delta-функций. Type 1 использует эллиптическую часть, Type 2 — полный Helmholtz operator. Затем решается локальная обобщённая eigenproblem в snapshot space; отобранные моды умножаются на partition-of-unity functions. Если столбцы P содержат глобальные multiscale modes, coarse solve имеет вид

\[
 (P^*H_hP)c=P^*f_h,\qquad u_{ms}=Pc,
\]

где H_h — полная комплексная FEM-матрица. Это решение PDE в сокращённом пространстве; POD surrogate предсказывает коэффициенты обученной сетью. Для реальных базисов P*=Pᵀ. Простая выборка узлов, POD проекция или произвольная coarse mesh сами по себе не являются GMsFEM из статьи. Если локальный baseline упрощён, указывается конкретное отличие и время построения базиса.

## Протокол воспроизводимого сравнения

1. Зафиксировать configuration, seed, mesh/DOF ordering, DG0 coefficients, частотную полосу, число сред/направлений и групповые train/val/test IDs. Хешировать геометрию, connectivity, dataset и split. При отсутствии реального FEM-датасета использовать явно помеченный smoke fixture; его метрики не являются физической валидацией.
2. Проверить нулевое рассеяние, конечность полей и совпадение mesh hash. На ограниченном числе сред проверить fine-FEM mesh convergence отдельно от surrogate error. Не называть fine-FEM reference «точным решением».
3. Запустить SVD-диагностику без обучения сети: exact и randomized при одинаковых данных, метрике, ранге; spectra, cumulative energy, reconstruction curves, orthogonality, subspace angles, время и память. Val определяет r; test не используется для перебора.
4. Для SVD adaptation, Chebyshev coefficient-space модели и FNO использовать одинаковые среды и параметры PDE. Согласовать tuning budget; записать число параметров, epochs, batch size, seed, patience и причину остановки. Лучшая эпоха выбирается по val; достижение лимита epochs не равно сходимости.
5. Отдельно сохранить pure-POD/полиномиальную projection error и FEM→grid→FEM transfer error. Основную ошибку всех методов вычислять в common FEM P1 space с consistent mass matrix. Lumped/nodal scores хранить отдельными колонками.
6. Inference запускать из чистого процесса по сохранённому bundle. Bundle содержит веса, архитектуру, базисы/их определение, mean/std, mesh hash, норму, порядок признаков и версии. Проверить single/batch и save/load parity.
7. Для GMsFEM использовать те же A,n,k,d, границы и fine mesh; сообщать coarse size, число локальных мод, тип snapshots и стоимость local basis construction. Если параметры среды/частоты меняются, учитывать необходимую перестройку базисов.
8. В таблицу результатов включить mean/median/p95/max FEM L2, абсолютные ошибки почти нулевых полей и paired per-medium results. Несколько направлений одной среды — зависимые наблюдения; интервалы неопределённости рассчитывать с ресемплингом сред.
9. Замерять отдельно dataset generation, preprocessing, basis construction, training, JIT/warm-up, transfer и warm inference. Синхронизировать accelerator. Для batch=1 и batch inference указать hardware и число повторов. Не сравнивать warm inference с FEM-временем, включающим только у одного метода startup, и не скрывать offline costs.
10. Повторить на высокой частоте: обученная полоса, OOD частота с frozen weights/bases и отдельный retraining experiment. В каждом случае нужны новые test media, FEM resolution diagnostics и projection floor. Никакой итог о высокочастотной точности не формулируется до этих запусков.

Числа «1%», rank cap или выбранная полоса k — параметры эксперимента, а не выводы перечисленных статей. Измеренные физические результаты необходимо публиковать с конфигурацией и provenance артефактов.
