# Методичка по multiscale SVD-SNO

Основной документ: [печатная PDF-версия](output/pdf/multiscale_svd_sno_manual.pdf). Редактируемый текст: [METHOD.md](METHOD.md). Последняя страница PDF специально отведена примечанию об отличиях от обычного SVD-SNO.

Разобраны постановка рассеяния и FEM-эталон, комплексный массовый POD, размерности тензоров, восемь ветвей multiscale SVD-SNO, один coefficient block, физическая функция потерь, ранги, нормировки и защита от утечки. Отдельно объяснено фактическое высокочастотное расширение с дообучаемой старой моделью, фиксированным переносом декодеров и новой residual-сетью.

Сопоставлены обычный SVD-SNO, multiscale SVD-SNO, FNO, Chebyshev-SNO и GMsFEM Type 1/2. Для каждой серии приведены её метрики и условия. В static-low [1,2] использовано 1280 тестовых полей, в sample-study [0.1,4] — 100 тестовых частот, в high-k [0.05,20] — 200 тестовых частот. Высокочастотное сравнение включает пять размеров выборки дообучения и GMsFEM Type 2 с M=16 и M=32. [Итоговый отчёт high-k](../experiments/highk_svd_gmsfem/results/REPORT.md).

## Файлы

- `chapters.json` - основной структурированный текст для сборки PDF.
- `METHOD.md` - экспорт текста, таблиц и формул в Markdown.
- `assets/` - схемы PNG/SVG, графики и зафиксированный численный снимок `metrics.json`.
- `build_assets.py` - воспроизводимая генерация схем и графиков из сохранённых результатов.
- `build_manual.py` - вёрстка PDF средствами ReportLab.
- `export_markdown.py` - синхронизация Markdown из `chapters.json`.
- `tmp/` - промежуточные формулы и рендеры страниц для визуальной проверки.

## Сборка

Из корня репозитория:

```sh
VECLIB_MAXIMUM_THREADS=2 .venv/bin/python method_define/build_assets.py
.venv/bin/python method_define/export_markdown.py
/Users/egoriy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 method_define/build_manual.py
```

Для правки печатной версии изменяйте `chapters.json`, затем повторяйте экспорт и сборку. После изменения текста или схем PDF нужно снова отрендерить и проверить все страницы. Пути bundled Python/Poppler зависят от локальной установки; здесь записана фактически использованная среда.

Графики построены по сохранённым измерениям. `build_assets.py` фиксирует дату сборки и хеши источников, а данные high-k берёт из проверенной итоговой серии.
