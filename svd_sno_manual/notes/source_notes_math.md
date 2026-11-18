# Математическая часть: источники и оговорки

Проверены локальные файлы:
- neural_operator/fem.py: знак operator K-k*k*Mn-1j*k*B, rhs с (1-alpha) и (n-1), DG0, P1 mass, boundary edge mass, квадратура 8x8 Duffy, LU reuse.
- neural_operator/sno/basis.py: row snapshots, weighted SVD, отдельный real-only fit, stable-offset mean, total centered energy, rank0, randomized oversampling10/power2/fallback, numerical threshold с field_scale, encode/decode.
- neural_operator/training.py::_svd_preprocessing: train-only fit; unique train medium representatives для A,n; validation mean consistent error выбирает общий ранг, компоненты сохраняют независимые реальные ранги; нулевое дополнение.
- neural_operator/data.py: consistent M и lumped row sums, complex FEM L2, split invariants.
- neural_operator/docs/methods.md: PDE и трактовка границы, предупреждение lumped projection versus consistent optimum, статическая среда.

Файл docs/methods.md отсутствует в корне; фактический путь neural_operator/docs/methods.md.

Глава 1: волна, PDE, weak form с промежуточным Green identity, FEM и точный P1 пример, нормы и материалы. Глава 2: POD от интуиции до row-convention weighted derivation, числовой rank1 пример, ограниченная формулировка Eckart-Young, full/randomized/numerical rank, четыре базиса, rank selection/train-only.

Все числа в POD примере учебные, не результаты эксперимента. Его W=diag(1,2,1) специально выбран, не приписан FEM-сетке. Никаких утверждений о достигнутом качестве сети нет. Проекция через lumped encode не названа строгим consistent floor. Текст не использует интернет и не заявляет воспроизведение статьи. Проверка компиляции и пагинации остаётся интеграционной задачей основного агента.
