"""Render scientific solution/error maps and an offline comparison viewer.

First run visual_fields; then run this module in the same container.
All displayed fields originate from the saved, frozen experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "visuals"
COMPONENTS = {"real": (np.real, "Re"), "imag": (np.imag, "Im"), "abs": (np.abs, "|u|")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 11, "figure.titlesize": 15,
                     "savefig.facecolor": "white"})


def decorate(axis, circles, *, outline=True):
    axis.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="x", ylabel="y", aspect="equal")
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_yticks([-2, -1, 0, 1, 2])
    if outline:
        for cx, cy, radius, *_ in circles:
            axis.add_patch(Circle((cx, cy), radius, fill=False, edgecolor="#6b7280",
                                  linewidth=.5, alpha=.6))


def draw_map(axis, data, circles, *, cmap, low, high, title):
    dx, dy = 4/(data.shape[1]-1), 4/(data.shape[0]-1)
    artist = axis.imshow(data, origin="lower", extent=(-2-dx/2, 2+dx/2, -2-dy/2, 2+dy/2),
                         cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    decorate(axis, circles)
    axis.set_title(title)
    return artist


def save(fig, path):
    fig.savefig(path, dpi=145)
    plt.close(fig)


def inputs_figures(inputs):
    xy, triangles, circles = (inputs[k] for k in ("coordinates", "triangles", "circles"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained")
    for axis, key, label, values, colors in (
        (axes[0], "alpha_cells", "α: коэффициент градиентного члена", [.2, 1.], ["#d94801", "#edf0f3"]),
        (axes[1], "n_cells", "n: коэффициент при k²u", [1., 10.], ["#edf0f3", "#0868ac"]),
    ):
        cmap = ListedColormap(colors)
        bounds = [values[0] - (values[1]-values[0])/2,
                  sum(values)/2, values[1] + (values[1]-values[0])/2]
        artist = axis.tripcolor(xy[:, 0], xy[:, 1], triangles, facecolors=inputs[key],
                                cmap=cmap, norm=BoundaryNorm(bounds, 2), shading="flat",
                                rasterized=True)
        decorate(axis, circles)
        axis.set_title(label)
        fig.colorbar(artist, ax=axis, ticks=values, fraction=.047, pad=.03)
    fig.suptitle("Входная среда: фон α=n=1; все 24 включения α=0.2, n=10")
    save(fig, OUTPUT / "inputs.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    for axis in axes:
        axis.triplot(xy[:, 0], xy[:, 1], triangles, color="#64748b", lw=.20, alpha=.55)
        for v in np.linspace(-2, 2, 21):
            axis.axhline(v, color="#c2410c", lw=.65, alpha=.7)
            axis.axvline(v, color="#c2410c", lw=.65, alpha=.7)
        decorate(axis, circles)
    axes[0].set_title("Эталон МКЭ: 53 246 узлов, 105 690 треугольников")
    cx, cy, radius = circles[np.argmax(circles[:, 2]), :3]
    margin = max(.24, radius * 2.4)
    axes[1].set(xlim=(cx-margin, cx+margin), ylim=(cy-margin, cy+margin))
    axes[1].set_xticks(np.round(np.linspace(cx-margin, cx+margin, 5), 2))
    axes[1].set_yticks(np.round(np.linspace(cy-margin, cy+margin, 5), 2))
    axes[1].set_title("Увеличение у включения; оранжевым — сетка 20×20")
    fig.suptitle("Сетка, на которой получены метки и восстанавливается SNO")
    save(fig, OUTPUT / "mesh.png")


def fields_for(fields, case, field):
    identifier = case["id"]
    incident = fields[identifier + "_incident"]
    values = {name: fields[identifier + "_" + name + "_scattered"]
              for name in ("reference", "sno", "gms")}
    if field == "total":
        values = {name: value + incident for name, value in values.items()}
    return values, incident


def case_title(case, field):
    label = "Полное поле u = uᶦ + uˢ" if field == "total" else "Рассеянное поле uˢ"
    return f"{label}  ·  k={case['k']:.6g}, θ={case['theta_deg']:.3f}°  ·  test #{case['index']}"


def overview(fields, case, field, component, circles):
    values, incident = fields_for(fields, case, field)
    transform, symbol = COMPONENTS[component]
    mapped = {key: transform(value) for key, value in values.items()}
    maximum = max(float(np.max(np.abs(v))) for v in mapped.values())
    maximum = max(maximum, 1e-12)
    low = 0 if component == "abs" else -maximum
    cmap = "viridis" if component == "abs" else "RdBu_r"
    errors = {key: np.abs(values[key] - values["reference"]) for key in ("sno", "gms")}
    error_max = max(float(np.max(v)) for v in errors.values())
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.0), layout="constrained")
    for axis, name, label in zip(axes[0], ("reference", "sno", "gms"),
                                 ("МКЭ — эталон", f"SNO {case['band']}", "GMsFEM 20×20 · Type 2 · M=16")):
        artist = draw_map(axis, mapped[name], circles, cmap=cmap, low=low, high=maximum,
                          title=f"{label}\n{symbol} решения")
    cb = fig.colorbar(artist, ax=list(axes[0]), shrink=.9, pad=.015, fraction=.025)
    cb.set_label("Общая шкала трёх решений")
    wave_label = "|uᶦ|" if component == "abs" else f"{symbol}(uᶦ)"
    wave = draw_map(axes[1, 0], transform(incident), circles, cmap=cmap,
                    low=0 if component == "abs" else -1, high=1,
                    title=f"Входная падающая волна: {wave_label}")
    d = case["direction"]
    axes[1, 0].annotate("", xy=(-1.5+.8*d[0], -1.5+.8*d[1]), xytext=(-1.5, -1.5),
                        arrowprops=dict(arrowstyle="->", color="#111827", linewidth=2.5))
    axes[1, 0].text(-1.7, -1.84, f"d = ({d[0]:.3f}, {d[1]:.3f})", fontsize=9,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    fig.colorbar(wave, ax=axes[1, 0], shrink=.9, pad=.02, fraction=.045)
    for axis, name, label in zip(axes[1, 1:], ("sno", "gms"), ("SNO", "GMsFEM")):
        error_artist = draw_map(axis, errors[name], circles, cmap="magma", low=0,
                                high=max(error_max, 1e-12),
                                title=f"|u_{label} − u_МКЭ|\nКомплексная ошибка: амплитуда и фаза")
    cb = fig.colorbar(error_artist, ax=list(axes[1, 1:]), shrink=.9, pad=.015, fraction=.035)
    cb.set_label("Общая шкала ошибок; 0 — точное совпадение")
    sno_error = case["sno_metrics"][field + "_e_complex_ref"]
    gms_error = case["gms_metrics"][field + "_e_complex_ref"]
    fig.suptitle(case_title(case, field) +
                 f"\nКомплексная относительная L² этого примера: SNO {sno_error:.3f}%; GMsFEM {gms_error:.3f}%")
    path = OUTPUT / f"{case['id']}_{field}_{component}.png"
    save(fig, path)


def signed_differences(fields, case, field, circles):
    values, _ = fields_for(fields, case, field)
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 13), layout="constrained")
    names = ("sno", "gms")
    for row, (transform, symbol) in enumerate(COMPONENTS.values()):
        differences = {name: transform(values[name])-transform(values["reference"]) for name in names}
        limit = max(max(float(np.max(np.abs(v))) for v in differences.values()), 1e-12)
        for axis, name, label in zip(axes[row], names, ("SNO", "GMsFEM")):
            difference_label = ("|u_модель| − |u_МКЭ|" if symbol == "|u|"
                                else f"{symbol}(u_модель) − {symbol}(u_МКЭ)")
            artist = draw_map(axis, differences[name], circles, cmap="RdBu_r", low=-limit, high=limit,
                              title=f"{label}: {difference_label}")
        cb = fig.colorbar(artist, ax=list(axes[row]), pad=.02, fraction=.033, shrink=.9)
        cb.set_label("Общая шкала разностей этой строки")
    fig.suptitle(case_title(case, field) + "\nЗнаковые разности: синий < 0; белый ≈ 0; красный > 0")
    save(fig, OUTPUT / f"{case['id']}_{field}_differences.png")


def extrapolation_figure(fields, case, circles):
    incident = fields[case["id"] + "_incident"]
    reference = fields[case["id"] + "_reference_scattered"] + incident
    trained = fields[case["id"] + "_sno_scattered"] + incident
    frozen = fields[case["id"] + "_sno_low_scattered"] + incident
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.4), layout="constrained")
    limit = max(float(np.max(np.abs(reference.real))), float(np.max(np.abs(trained.real))), 1e-12)
    for axis, value, label in zip(axes[0], (reference, trained), ("МКЭ — эталон", "SNO high: обучена в [2,4]")):
        artist = draw_map(axis, value.real, circles, cmap="RdBu_r", low=-limit, high=limit, title=label)
    fig.colorbar(artist, ax=list(axes[0]), pad=.02, fraction=.03, label="Общая шкала Re(u)")
    frozen_max = max(float(np.max(np.abs(frozen.real))), 1e-12)
    artist = draw_map(axes[1, 0], frozen.real, circles, cmap="RdBu_r", low=-frozen_max,
                      high=frozen_max, title="SNO low: перенос за диапазон\nОтдельная шкала из-за большой ошибки")
    fig.colorbar(artist, ax=axes[1, 0], pad=.02, fraction=.045)
    error = np.abs(frozen-reference)
    artist = draw_map(axes[1, 1], error, circles, cmap="magma", low=0, high=max(float(error.max()), 1e-12),
                      title="|u_low − u_МКЭ|\nАбсолютная комплексная ошибка")
    fig.colorbar(artist, ax=axes[1, 1], pad=.02, fraction=.045)
    fig.suptitle(f"Замороженная low-модель на k={case['k']:.6g}, θ={case['theta_deg']:.3f}°"
                 f"\nПолное поле: high {case['sno_metrics']['total_e_complex_ref']:.2f}%; "
                 f"low {case['ood_metrics']['total_e_complex_ref']:.2f}%")
    save(fig, OUTPUT / f"{case['id']}_extrapolation.png")


def write_guide(cases):
    lines = ["# Решения, разности и входные параметры", "",
             "Откройте [наглядное сравнение с переключателями](index.html). "
             "Каждый рисунок — реальные решения сохранённого эксперимента, без нового обучения.", "",
             "![Коэффициенты среды](inputs.png)", "",
             "В фоне α=n=1, во всех 24 включениях α=0.2 и n=10. "
             "α — коэффициент градиентного члена, n — коэффициент при k²u. "
             "Точные DG0-коэффициенты показаны по ячейкам; узловое усреднение не используется.", "",
             "Падающая волна uᶦ=exp(ik d·x), направление d=(cosθ,sinθ). "
             "Полное поле u=uᶦ+uˢ; рассеянное uˢ показывает отклик включений. "
             "Re и Im — две фазовые компоненты, |u| — амплитуда.", "",
             "| Пример | k | θ, градусы | Индекс test |",
             "|---|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['id']} | {c['k']:.9g} | {c['theta_deg']:.6f} | {c['index']} |")
    lines += ["", "У примеров разные направления: это не серия изменения только k. "
              "k≈3 на рисунках означает фактическую test-частоту 3.0062893081761004.", "",
              "На основных рисунках сверху стоят МКЭ, SNO и GMsFEM с общей цветовой шкалой. "
              "Снизу слева — входная падающая волна и стрелка её направления; "
              "снизу справа — две карты |u_модель−u_МКЭ| с общей шкалой. "
              "Чем ближе цвет ошибки к нулю, тем ближе поле к МКЭ. "
              "Ошибка включает амплитуду и фазу, поэтому она не равна разности модулей.", "",
              "Отдельные карты знаковых разностей показывают Re(u_модель)−Re(u_МКЭ), "
              "Im(u_модель)−Im(u_МКЭ) и |u_модель|−|u_МКЭ|. "
              "Последняя строка характеризует амплитуду и не заменяет комплексную ошибку. "
              "Общая падающая волна сокращается в комплексной разности полного и рассеянного полей; "
              "их абсолютные комплексные ошибки одинаковы, но относительные L²-ошибки различаются.", "",
              "Каждое комплексное P1-поле оценивается из своей исходной сетки на общей "
              "отображаемой сетке 401×401. Только после этого вычисляются модуль и разности. "
              "Проценты L² взяты из исходной FEM-квадратуры и проверены повторным расчётом; "
              "они не посчитаны по пикселям рисунка. Цветовые шкалы между разными k могут "
              "различаться — смотрите численные пределы colorbar.", "",
              "GMsFEM: 20×20, Type 2, 16 мод на крупную вершину, исходная fine-сетка 17 549 DOF. "
              "SNO: замороженные low/high checkpoint на reference-сетке 53 246 DOF. "
              "Фактический вход нейросети в этой фиксированной среде — cosθ, sinθ и "
              "нормированное k; входные POD-ранги материалов равны нулю. В viewer показана "
              "нормировка отдельно для выбранной модели.", "",
              "![Эталонная сетка](mesh.png)", ""]
    for c in cases:
        lines += [f"## {c['id']}: k={c['k']:.6g}, θ={c['theta_deg']:.3f}°", "",
                  f"![Решения и ошибки]({c['id']}_total_real.png)", "",
                  f"[Знаковые разности]({c['id']}_total_differences.png) · "
                  f"[Модуль решения]({c['id']}_total_abs.png) · "
                  f"[Рассеянное поле]({c['id']}_scattered_real.png)", ""]
    lines += ["## Воспроизведение", "", "Из корня проекта, в исправленном complex-контейнере:", "",
              "```bash", "docker compose exec -T dolfinx-complex python3 -m experiments.reference_sno_gmsfem.visual_fields",
              "docker compose exec -T dolfinx-complex python3 -m experiments.reference_sno_gmsfem.visualize_fields", "```", "",
              "`cases.json` содержит фактические параметры, проверку метрик и SHA256 входов. "
              "`fields.npz` хранит комплексные поля отображения, `inputs.npz` — исходные коэффициенты и сетку.", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main():
    metadata = json.loads((OUTPUT / "cases.json").read_text())
    if metadata.get("immutable_inputs_unchanged") is not True:
        raise ValueError("Visual source verification is missing")
    with np.load(OUTPUT / "inputs.npz", allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    circles = inputs["circles"]
    inputs_figures(inputs)
    with np.load(OUTPUT / "fields.npz", allow_pickle=False) as fields:
        for case in metadata["cases"]:
            for field in ("total", "scattered"):
                for component in COMPONENTS:
                    overview(fields, case, field, component, circles)
                signed_differences(fields, case, field, circles)
            if case["band"] == "high":
                extrapolation_figure(fields, case, circles)
            print(f"Rendered {case['id']}: solutions, signed differences and inputs", flush=True)
    write_guide(metadata["cases"])
    template = (HERE / "visual_viewer.html").read_text()
    # Metadata is data, not executable HTML. Escape script end delimiters.
    payload = json.dumps(metadata, ensure_ascii=False).replace("</", "<\\/")
    (OUTPUT / "index.html").write_text(template.replace("__CASE_DATA__", payload))
    print(f"Viewer and scientific PNGs ready in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
