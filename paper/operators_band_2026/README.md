# Четыре нейрооператора для рассеяния при переменных материале и волновом числе

[Рукопись](main.pdf) · [LaTeX](main.tex) · [Сводные проверки](audit.json)

В основном опыте 124 среды по 10 полей, разбиение по среде 100/12/12. Все модели используют обучающие префиксы из 400/600/800/1000 полей. Статическая нормировка задаёт нулевое рассеянное поле. Дополнительная серия из шести положительных низкочастотных задач используется только после фиксации моделей. Обе серии используют тот же физический материал в строках, указанных в запросах. В нескольких файлах reference_problem.npz сохранены также коэффициенты исторической задачи проверки решателя; для новых меток используются значения alpha1,alpha2,n1,n2 из queries.npz.

## Воспроизведение

Из корня проекта:

```sh
docker compose up -d dolfinx-complex
docker compose exec -T dolfinx-complex python3 paper/operators_band_2026/generate.py
docker compose exec -T dolfinx-complex python3 paper/operators_band_2026/check_mesh.py
docker compose exec -T dolfinx-complex python3 paper/operators_band_2026/nearzero.py
```

Обучение, выбор эпохи и оценка приведены в протоколах [SVD-SNO](svd/README.md), [FNO](fno/README.md), [Chebyshev-SNO](chebyshev/README.md) и [Fourier-SNO](fourier/README.md). Каждая директория содержит summary.csv и cases.csv. Ключ row относится к одному и тому же индексу в data/queries.npz. В nearzero.csv индексы относятся к data/nearzero.npz. Большие данные и веса хранятся локально.

Сборка численных таблиц и карт выполняется командой `.venv/bin/python paper/operators_band_2026/report.py`. Для компиляции статьи достаточно готовых каталогов tables и figures.

```sh
cd paper/operators_band_2026
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```
