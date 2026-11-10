# Независимый архитектурный обзор — Xhigh

Изучены sno/model.py, _layers.py, training.py, datasets/config.py,
datasets/generation.py, data.py, fem.py и POD basis. Core не менялся.

Ключевые выводы:

1. Variable recipe генерирует одно случайное k на среду; hierarchical M×F×D
   пока не реализован. Dataset metadata недостаточно для cohort/lineage учёта.
2. При 16 train-media центрированный material POD имеет ранг≤15. Это
   ограничение входного представления не снимается глубиной.
3. Для фиксированных 24 кругов известные mask contrasts дают прозрачный
   контроль без потери 24-мерного материального подпространства. Нужен новый
   input adapter; текущий Predictor автоматически его не поддерживает.
4. Output POD каждого scout arm строится только по его собственным train
   полям. Для depth study один bundle и POD seed фиксируются для всех seeds.
5. Current train_model автоматически оценивает test после fit. До HPO
   требуется val-only режим без доступа к test labels и отдельный evaluator.
6. Current rank selector измеряет реконструкцию lumped-POD через FEM norm;
   нужен oracle floor фактического decoder через consistent Gram matrix.
7. Глубины 3/6/12,H64 сравниваются отдельно от shallow/wide matched-parameter
   controls. Формула P=8 H+2+L(2 H²+H+R²) проверена по реальным Linear-слоям.
8. Геометрия с подвижными центрами/радиусами требует отдельного multi-mesh
   протокола и группирования по lineage; смена active mask не равна этому.

Векторы DG0/nodal материалов надо хранить один раз наmedium. Для 9600 полей
на 3221 узле complex128 payload≈494.75 MB; повторение A/n добавляет≈1.46 GB.
Refined QA, sparse LU, optimizer и SVD workspace в эти числа не входят.
LU переиспользуется для направлений при одинаковых medium,k,mesh.
