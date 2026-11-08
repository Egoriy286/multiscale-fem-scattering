# Вердикт независимого судьи — Xhigh

**ACCEPT staged planning protocol. REVISE прямоугольные рекомендации.
REJECT25×100×4 как основной доказательный опыт переноса на новые среды.**
Это решение о дизайне исследования; FEM и новое обучение не выполнены.

Рассмотрены независимые предложения dataset/architecture и первичный
[NSNO источник](https://arxiv.org/html/2401.13494 v1#S6.SS2). Он поддерживает
many-media/multi-direction постановку, но не выбирает наш optimum или depth.

Причины решения:

- 25 сред при нынешнемsplit дают 16 train/4 val/5 test; independent information
  и input POD rank ограничены числом сред, а не 10000 повторными запросами.
- 100×25×4 и 120×20×4 разумны как кандидаты, но не установлены данными как
  оптимальные. Числоtest-media и QA расходы надо фиксировать отдельно.
- Полные competing10000-field datasets не помещаются в один globalcap.
  Нужен nested scout с подсчётом union.

Принятый предел: **7200 train +800 val +1600 test +до 400 QA =до 10000**.
Scout arms20×20×4,40×10×4,80×5×4 имеют 1600 полей каждый и union3200.
Оставшиеся 4000 train идут в frequency fallback90×20×4 или diversity extension
по заранее объявленному val-only правилу. Все scout labels сохраняются.

20 validation и 40 test-media независимы от train. Evaluation40 queries/media:
20 shared-query,12 reserved-frequency,8 shifted-angle. Угловые/frequency cohorts
условны на новых средах, не изолируют перенос на уже обученных материалах.
Test закрыт до dataset/preprocessing/model/optimizer freeze.

Hard gates: exact DG0/geometry/hash/zero-scattering/linear-residual checks;
стратифицированная mesh-stability проверка; input reconstruction и exact
consistent decoder floor; arm-own train-only POD в scout; общий frozen POD
в depth study; независимые seeds; matched-parameter controls без сменыrank;
macro metrics и paired bootstrap по средам. CI, пересекающий ноль, означает
inconclusive. Заданные effect thresholds — policy, не статистический закон.

Текущий core не готов к прямому запуску протокола: нужны hierarchical
recipe/storage, native-mask adapter, val-only training и oracle rank selector.
Metadata planner допустим сейчас. Числа 9600 coarseRHS/2280 LU исключают
refined QA; стоимостьfine solves измеряется отдельно.400 QA — верхний резерв,
не утверждение, что 400 физических контрольных запросов уже определены/решены.

Что неизвестно до scout: достаточный output rank/mesh, реальные cost/peak
memory, победитель completion и выигрыш глубины. Универсальный optimum
не заявляется. Координатор зафиксировал более строгую новую label gate
(предпочтительно 0.5%,hard1%) как явную проверяемую policy, не как PDE bound.
