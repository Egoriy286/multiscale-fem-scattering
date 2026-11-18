# Проверка главы реализации SVD-SNO v2

Файл: `chapters/02_implementation.tex`. Проверено по текущему дереву проекта, без интернет-источников. Production-файлы не менялись; обучение и FEM-генерация не запускались.

## Прочитанные источники

- `neural_operator/sno/model.py`: версии 1/2, независимые input_ranks, token_modes=max(output_modes,*input_ranks), lift Linear(5,H)+GELU, padding, forward, readout Linear(H,2).
- `neural_operator/_layers.py`: фактическое положение CoefficientBlock (файла sno/_layers.py нет), обе ветви и residual; legacy _ModeBlock.
- `neural_operator/sno/basis.py`: train centering, lumped SVD, numeric tolerance, full/randomized fallback, encode/decode, zero rank.
- `neural_operator/training.py`: preprocessing, validation selection, statistics, predictor, AdamW, checkpoint/reuse.
- `neural_operator/losses.py`: float32 decoder buffers -> float64 quadratic integration; direct residual energy; target rounding; epsilon=1e-24.
- `neural_operator/data.py`: consistent mass, lumped weights, validation, grouped split, fingerprint helpers, mmap and inspect.
- `neural_operator/run.py`, `neural_operator/settings.py`, root README, neural_operator README, compose.yaml.
- `neural_operator/tests/test_pod_fem_loss.py`: actual test path (root tests/test_pod_fem_loss.py does not exist).

## Существенные уточнения

1. Optimizer is AdamW, not plain Adam. No learning-rate scheduler exists in train_model; lr is only logged. Weight decay=1e-5, gradient norm clip=1.
2. Four independent bases: A, n, real, imag; actual ranks can differ; zero input ranks are legitimate.
3. Rank selection judges mean validation consistent FEM L2 of lumped-projected reconstruction. It does not guarantee p95, per-sample error, test error, or network error. Fallback retains available rank and records target_attained=False.
4. Stable loss is e/sqrt(max(e,eps))/sqrt(max(q,eps)), after e>=0 clamping. It gives exact zero at e=0; flooring numerator under sqrt would not.
5. Loss precomputes consistent optimal projection, not PODBasis.encode projection. Floor is directly integrated from residual; no cancellation-prone total-minus-projected-energy computation.
6. Bundle format_version=1 for both architecture versions. Unversioned SVD model_kwargs default to architecture_version=1. No silent v1->v2 conversion.
7. Best state is deep-copied in RAM and saved after training/testing, not an every-epoch optimizer resume checkpoint. Optimizer/scheduler/RNG state is not saved.
8. CLI exposes epochs/device/version/pod-method, not width/depth/rank/batch-size. Explicit Python API example is provided for those.
9. CLI --output is a method root; run.py appends /svd. Python run_experiment output_dir is the actual method directory.
10. predict_components returns real/imag/abs of scattered field; abs is derived after physical reconstruction, no learned third channel.

## Выполненные проверки и ограничения окружения

- `python3 -m neural_operator.run train --help` executed successfully on host; documented flags match.
- Attempted `docker compose exec -T dolfinx-complex python3 -m neural_operator.run train --help`: Docker daemon unavailable; unix /Users/egoriy/.orbstack/run/docker.sock missing. No Docker execution success claimed.
- Attempted safe `data ... --dry-run` on host: stops at missing NumPy (host Python 3.15). No data creation or FEM solve occurred. The source confirms --dry-run bypasses generation after plan.
- Docker command service/workdir/volume mapping verified in compose.yaml. Training commands are user instructions, not execution claims.
- Code listings contain only ASCII. CoefficientBlock snippet omits only docstring; coefficient_tokens snippet is exact computational core, with explicit explanation that checks were omitted; full SVDSNO.forward copied verbatim.
- No extra tests were written or heavy tests run.

## Вопросы для редактора / финальной сборки

- The parent manual must provide lstlisting, longtable, idea, pitfall and worked environments, AMS macros and path. Chapter is fragment without preamble.
- All labels begin code:. Chapter does not cite internet or introduce bibliography entries.
- Page count depends on final typesetting; chapter has substantial explanatory prose and tables. Render/compile in parent workflow should determine exact chapter extent.
- The training CLI cannot set rank/width/depth: retain explicit distinction if editing commands.
