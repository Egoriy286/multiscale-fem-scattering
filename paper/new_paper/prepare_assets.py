"""Produce all paper plots and tables from archived, traceable numerical data.

No model fitting, FEM solve, smoothing, or selected-error example search occurs.
The two spatial examples are fixed k=1 and k=4 from the existing baseline check.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.tri as mtri
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
EVIDENCE = HERE / "evidence"
GENERATED = HERE / "generated"
SHARED = ROOT / "experiments/shared_sno_gmsfem/results"
STATIC = ROOT / "paper/svd_sno_revision_2026-11-06/evidence/current_static_low"
REF = ROOT / "experiments/reference_sno_gmsfem/results"
SNO = "SVD-SNO shared"
GMS = "GMsFEM 20x20 Type 2 M=16"
FEM = "FEM h=0.04"
COLORS = {SNO: "#0072B2", GMS: "#D55E00", FEM: "#777777"}
LABELS = {SNO: "Shared SVD-SNO", GMS: "GMsFEM 20 x 20, type 2", FEM: "FEM h = 0.04"}
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.035,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .6,
                         "grid.alpha": .25, "grid.linewidth": .5,
                         "legend.frameon": False, "lines.linewidth": 1.25,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})


def summarize(rows):
    result = []
    for method in dict.fromkeys(r["method"] for r in rows):
        subset = [r for r in rows if r["method"] == method]
        row = {"method": method, "samples": len(subset)}
        for field in ("total", "scattered"):
            values = np.array([r[f"{field}_e_complex_ref"] for r in subset])
            row.update({f"{field}_{key}_percent": float(value) for key, value in (
                ("mean", values.mean()), ("median", np.median(values)),
                ("p95", np.percentile(values, 95)), ("max", values.max()))})
        result.append(row)
    return result


def method_figure(examples):
    fig = plt.figure(figsize=(6.75, 2.35))
    ax = fig.add_axes([.01, .1, .24, .78])
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    for x, y, radius, _ in examples["circles"]:
        ax.add_patch(Circle((x, y), radius, fc="#BDD9E8", ec="#24688D", lw=.5))
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.spines[["top", "right"]].set_visible(True)
    ax.set_title("(a) Fixed heterogeneous medium", fontsize=8)
    ax = fig.add_axes([.29, .02, .70, .96]); ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis("off")
    ax.text(5, 4.8, "(b) Learned coefficients and physical coarse solve", ha="center", fontsize=9)

    def box(x, y, w, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, .8, boxstyle="round,pad=0.08,rounding_size=0.08",
                                   fc=color, ec="#45515A", lw=.55))
        ax.text(x+w/2, y+.4, text, ha="center", va="center", fontsize=7)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=7, lw=.65, color="#45515A"))
    box(.15, 2.9, 1.6, r"$k,\cos\theta,\sin\theta$", "#F2F4F5")
    box(2.3, 2.9, 2.05, "SVD-SNO v2\n6 coefficient blocks", "#E4F0F7")
    box(4.9, 2.9, 2.2, "61 coefficients\nper real component", "#E4F0F7")
    box(7.65, 2.9, 2.1, "Fixed POD decoder\ncomplex P1 field", "#E4F0F7")
    for x1, x2 in ((1.75, 2.3), (4.35, 4.9), (7.1, 7.65)): arrow(x1, 3.3, x2, 3.3)
    ax.text(8.7, 4.1, "Train-only snapshots + lumped weights", ha="right", fontsize=7)
    arrow(8.7, 4.0, 8.7, 3.75)
    box(.15, 1.2, 2.35, "Local snapshots\nelliptic / Helmholtz", "#FBECE5")
    box(3.0, 1.2, 2.75, "Local eigenvectors\n16 modes per coarse vertex", "#FBECE5")
    box(6.25, 1.2, 3.5, r"$P^* H_h(k)P c=P^* f_h$" + "\nreconstruction $Pc$", "#FBECE5")
    arrow(2.5, 1.6, 3., 1.6); arrow(5.75, 1.6, 6.25, 1.6)
    ax.text(5, .44, "Common reference quadrature: phase-sensitive complex L2 error", ha="center", fontsize=8)
    save(fig, "method")


def frequency_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), layout="constrained")
    for ax, field, title in zip(axes, ("total", "scattered"), ("(a) Total field", "(b) Scattered field")):
        for method in (SNO, GMS, FEM):
            selected = [r for r in rows if r["method"] == method]
            ks = np.unique([r["k"] for r in selected])
            errors = [np.array([r[f"{field}_e_complex_ref"] for r in selected if r["k"] == k]) for k in ks]
            ax.plot(ks, [a.mean() for a in errors], color=COLORS[method],
                    ls="--" if method == FEM else "-", label=LABELS[method])
            if method != FEM:
                ax.fill_between(ks, [np.percentile(a, 10) for a in errors],
                                [np.percentile(a, 90) for a in errors], color=COLORS[method], alpha=.12, lw=0)
        ax.set(xlim=(.1, 4), yscale="log", xlabel="Wavenumber $k$", ylabel="Relative complex L2 error (%)", title=title)
        ax.grid(); ax.legend(loc="upper left")
    save(fig, "frequency_errors")


def distributions(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), layout="constrained")
    tables = {method: {r["index"]: r for r in rows if r["method"] == method} for method in (SNO, GMS, FEM)}
    ids = sorted(tables[SNO])
    assert all(set(t) == set(ids) for t in tables.values())
    for method, table in tables.items():
        errors = np.sort([r["total_e_complex_ref"] for r in table.values()])
        axes[0].step(errors, np.arange(1, len(errors)+1)/len(errors), where="post",
                     color=COLORS[method], label=LABELS[method], ls="--" if method == FEM else "-")
    axes[0].set(xscale="log", ylim=(0, 1.02), xlabel="Total-field error (%)", ylabel="Empirical cumulative fraction",
                title="(a) All 424 test queries")
    axes[0].legend(loc="lower right"); axes[0].grid()
    for i in ids:
        assert all(tables[m][i][key] == tables[SNO][i][key] for m in tables for key in ("k", "theta"))
    x = np.array([tables[SNO][i]["total_e_complex_ref"] for i in ids])
    y = np.array([tables[GMS][i]["total_e_complex_ref"] for i in ids])
    k = np.array([tables[SNO][i]["k"] for i in ids])
    scatter = axes[1].scatter(x, y, c=k, cmap="viridis", s=9, alpha=.8, edgecolors="none", rasterized=True)
    lim = (min(x.min(), y.min())*.7, max(x.max(), y.max())*1.4)
    axes[1].plot(lim, lim, color="#555555", ls="--", lw=.7)
    axes[1].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Shared SVD-SNO error (%)",
                ylabel="GMsFEM error (%)", title="(b) Paired total-field errors")
    axes[1].text(.05, .95, f"SNO lower on {np.count_nonzero(x < y)}/{len(x)} queries",
                 transform=axes[1].transAxes, va="top", fontsize=7)
    axes[1].grid(); fig.colorbar(scatter, ax=axes[1], label="$k$", fraction=.045, pad=.02)
    save(fig, "error_distributions")
    return dict(sno_lower_total_queries=int(np.count_nonzero(x < y)), test_queries=len(x))


def field_figure(examples):
    fig, axes = plt.subplots(2, 5, figsize=(6.75, 3.45), layout="constrained")
    mesh = mtri.Triangulation(*examples["coordinates"].T, triangles=examples["triangles"])
    for row, k in enumerate(examples["k"]):
        true, sno, gms = [examples[name][row] for name in ("reference", "sno", "gms")]
        fields = [true.real, sno.real, gms.real, np.abs(sno-true), np.abs(gms-true)]
        scale = max(np.max(np.abs(v)) for v in fields[:3])
        escale = max(np.max(v) for v in fields[3:])
        for col, value in enumerate(fields):
            ax = axes[row, col]
            im = ax.tripcolor(mesh, value, shading="gouraud", rasterized=True,
                              cmap="RdBu_r" if col < 3 else "magma",
                              vmin=-scale if col < 3 else 0, vmax=scale if col < 3 else escale)
            ax.set(aspect="equal", xlim=(-2,2), ylim=(-2,2), xticks=[-2,0,2], yticks=[-2,0,2])
            if row == 0: ax.set_title(["Reference Re$(u^s)$", "SNO Re$(u^s)$", "GMsFEM Re$(u^s)$", "SNO $|e^s|$", "GMsFEM $|e^s|$"][col], fontsize=7)
            if col == 0: ax.set_ylabel(f"$k={k:g}$\n$x_2$")
            else: ax.set_yticklabels([])
            if row == 1: ax.set_xlabel("$x_1$")
            ax.spines[["top", "right"]].set_visible(True)
            if col == 2: fig.colorbar(im, ax=axes[row, :3], orientation="horizontal", fraction=.06, pad=.07, shrink=.9, label="Real scattered field")
            if col == 4: fig.colorbar(im, ax=axes[row, 3:], orientation="horizontal", fraction=.06, pad=.07, shrink=.9, label="Absolute complex difference")
    save(fig, "field_examples")


def representation_figure(static, floor):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), layout="constrained")
    names = ["SVD-SNO", "FNO", "Chebyshev-SNO"]
    values = [static[k]["test_fem_l2"]["mean"]*100 for k in ("sno", "fno", "chebyshev")]
    bars = axes[0].bar(range(3), values, color=["#0072B2", "#777777", "#D55E00"], width=.55)
    axes[0].set(xticks=range(3), xticklabels=names, ylim=(0,9.4), ylabel="Mean test scattered-field error (%)",
                title="(a) Separate static-medium study")
    for rect, value in zip(bars, values): axes[0].text(rect.get_x()+rect.get_width()/2, value+.15, f"{value:.3f}", ha="center", fontsize=8)
    axes[0].axhline(static["sno"]["consistent_projection_floor_fem_l2"]["mean"]*100,
                    lw=.8, color="#0072B2", ls=":")
    axes[0].grid(axis="y"); axes[0].set_axisbelow(True)
    modes = [r["modes_per_axis"] for r in floor["records"]]
    errors = [r["validation_fem_l2"]["mean"]*100 for r in floor["records"]]
    axes[1].plot(modes, errors, "o-", color="#D55E00", markersize=4)
    for m, e in zip(modes, errors): axes[1].annotate(f"{e:.3f}%", (m,e), xytext=(0,7), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set(xticks=modes, xlim=(7,17), ylim=(4,8.5), xlabel="Chebyshev modes per axis", ylabel="Mean validation decoder floor (%)",
                title="(b) Best representation on grid 32 x 32")
    axes[1].grid(); save(fig, "representation")


def extrapolation_figure(rows, shared):
    fig, ax = plt.subplots(figsize=(3.25, 2.3), layout="constrained")
    selected = [r for r in rows if r["method"] == "SNO low extrapolation"]
    shared_by_query = {(r["k"], r["theta"]): r for r in shared if r["method"] == SNO}
    assert len(selected) == 200
    for subset, color, label in ((selected, "#D55E00", "Frozen low-band SNO"),
                                 ([shared_by_query[r["k"], r["theta"]] for r in selected], "#0072B2", "Shared full-band SNO")):
        ks = np.unique([r["k"] for r in subset])
        ax.plot(ks, [np.mean([r["total_e_complex_ref"] for r in subset if r["k"] == k]) for k in ks], color=color, label=label)
    ax.set(xlim=(2,4), yscale="log", xlabel="Wavenumber $k$", ylabel="Mean total-field error (%)")
    ax.legend(); ax.grid(); save(fig, "extrapolation")
    return dict(samples=len(selected), frozen_total_mean_percent=float(np.mean([r["total_e_complex_ref"] for r in selected])),
                shared_total_mean_percent=float(np.mean([shared_by_query[r["k"],r["theta"]]["total_e_complex_ref"] for r in selected])))


def training_figure(history, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout="constrained")
    epochs = np.arange(1, len(history["train"])+1)
    for key, label, color in (("train", "Train", "#0072B2"), ("val", "Validation", "#D55E00")):
        axes[0].plot(epochs, np.array(history[key])*100, label=label, color=color, lw=.8)
    axes[0].axvline(metrics["best_epoch"], color="#666666", ls="--", lw=.7, label="Selected epoch 493")
    axes[0].set(yscale="log", xlabel="Epoch", ylabel="Mean relative scattered-field loss (%)")
    axes[0].legend(); axes[0].grid()
    axes[1].step(epochs, history["lr"], where="post", color="#0072B2")
    axes[1].set(yscale="log", xlabel="Epoch", ylabel="AdamW learning rate"); axes[1].grid()
    save(fig, "training")


def diagnostics_figure(static, floor, ood, shared):
    """Three panels at their final paper size, without shrinking label fonts."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.25), layout="constrained")
    values = [static[k]["test_fem_l2"]["mean"]*100 for k in ("sno","fno","chebyshev")]
    bars = axes[0].bar(range(3), values, color=["#0072B2","#777777","#D55E00"], width=.55)
    axes[0].set(xticks=range(3), xticklabels=["SVD-\nSNO","FNO","Chebyshev-\nSNO"], ylim=(0,9.5),
                ylabel="Mean test error (%)", title="(a) Static-medium predictors")
    for rect,value in zip(bars,values): axes[0].text(rect.get_x()+rect.get_width()/2,value+.15,f"{value:.2f}",ha="center",fontsize=7)
    axes[0].axhline(static["sno"]["consistent_projection_floor_fem_l2"]["mean"]*100, color="#0072B2",ls=":",lw=.8)
    axes[0].grid(axis="y"); axes[0].set_axisbelow(True)
    modes=[r["modes_per_axis"] for r in floor["records"]]
    errors=[r["validation_fem_l2"]["mean"]*100 for r in floor["records"]]
    axes[1].plot(modes,errors,"o-",color="#D55E00",markersize=3)
    for m,e in zip(modes,errors): axes[1].annotate(f"{e:.2f}",(m,e),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    axes[1].set(xticks=modes,xlim=(7,17),ylim=(4,8.5),xlabel="Modes per axis",ylabel="Validation decoder floor (%)",title="(b) Chebyshev representation")
    selected=[r for r in ood if r["method"]=="SNO low extrapolation"]
    table={(r["k"],r["theta"]):r for r in shared if r["method"]==SNO}
    for subset,color,label in ((selected,"#D55E00","Frozen low-band"),([table[r["k"],r["theta"]] for r in selected],"#0072B2","Shared full-band")):
        ks=np.unique([r["k"] for r in subset])
        axes[2].plot(ks,[np.mean([r["total_e_complex_ref"] for r in subset if r["k"]==k]) for k in ks],color=color,label=label)
    axes[2].set(xlim=(2,4),yscale="log",xlabel="Wavenumber $k$",ylabel="Mean total-field error (%)",title="(c) Low-band extrapolation")
    axes[2].legend(loc="upper left",fontsize=6.5)
    for ax in axes[1:]: ax.grid()
    save(fig,"diagnostics")


def tables(summary, static):
    names = {SNO: "Shared SVD-SNO", FEM: "FEM ($h=0.04$)"}
    for n in (10,20):
        for kind in (1,2): names[f"GMsFEM {n}x{n} Type {kind} M=16"] = f"GMsFEM ${n}\\times{n}$, type {kind}"
    lines=[]
    for r in summary:
        dofs = "50,648 weights" if r["method"] == SNO else "17,549" if r["method"] == FEM else "1,936" if "10x10" in r["method"] else "7,056"
        vals = [r[f"{field}_{stat}_percent"] for field,stat in (("total","mean"),("total","median"),("total","p95"),("total","max"),("scattered","mean"),("scattered","p95"))]
        lines.append(names[r["method"]]+" & "+dofs+" & "+" & ".join(f"{v:.3f}" for v in vals)+r" \\")
    (GENERATED/"shared_table.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    lines=[]
    for key,label in (("sno","SVD-SNO v2"),("fno","FNO"),("chebyshev","Chebyshev-SNO")):
        r=static[key]
        lines.append(f"{label} & {r['parameters']:,} & {r['test_fem_l2']['mean']*100:.4f} & {r['test_fem_l2']['p95']*100:.4f} & {r['completed_epochs']}/{r['best_epoch']}"+r" \\")
    (GENERATED/"static_table.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")


def main():
    for path in (FIG,EVIDENCE,GENERATED): path.mkdir(exist_ok=True)
    style()
    comparison = read(SHARED/"comparison.json")
    canonical = read(SHARED/"sno.json")["rows"]+read(SHARED/"baselines.json")["rows"]
    assert comparison["rows"] == canonical
    summary = summarize(canonical)
    assert summary == comparison["summary"]
    metrics = read(SHARED/"sno_metrics.json"); history = read(SHARED/"sno_history.json")
    read(SHARED/"verification.json"); read(SHARED/"protocol.json")
    read(SHARED/"dataset_config.json"); read(SHARED/"dataset_dataset_info.json")
    read(STATIC/"dataset_config.json"); read(STATIC/"fno_verification.json")
    static = {key:read(STATIC/f"{key}_metrics.json") for key in ("sno","fno","chebyshev")}
    floor = read(STATIC/"chebyshev_decoder_validation.json")
    ood = read(REF/"high/extrapolation.json")["rows"]
    # Keep a compact paper-local data snapshot for independent plotting.
    for name, rows in (("shared_scores",canonical),("ood_scores",ood)):
        with (EVIDENCE/f"{name}.csv").open("w",newline="") as stream:
            fields=["method","index","k","theta","total_e_complex_ref","scattered_e_complex_ref"]
            writer=csv.DictWriter(stream,fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    for name, value in (("shared_metrics",metrics),("shared_history",history),("static_metrics",static),("chebyshev_floor",floor)):
        (EVIDENCE/f"{name}.json").write_text(json.dumps(value,indent=2)+"\n")
    with np.load(EVIDENCE/"field_examples.npz",allow_pickle=False) as examples:
        method_figure(examples); field_figure(examples)
    frequency_figure(canonical)
    derived = distributions(canonical)
    derived["extrapolation_paired"] = extrapolation_figure(ood,canonical)
    representation_figure(static,floor); training_figure(history,metrics)
    diagnostics_figure(static,floor,ood,canonical); tables(summary,static)
    (EVIDENCE/"derived_statistics.json").write_text(json.dumps(derived,indent=2)+"\n")
    manifest = dict(date="2026-11-13", source_sha256=SOURCES,
                    field_examples_sha256=sha(EVIDENCE/"field_examples.npz"),
                    generated_sha256={str(p.relative_to(HERE)):sha(p) for directory in (FIG,GENERATED) for p in sorted(directory.iterdir()) if p.is_file()},
                    plots="Deterministic data-derived figures; no synthetic physical observations",
                    summary=summary)
    (EVIDENCE/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Generated {len(list(FIG.glob('*.pdf')))} PDF/PNG figure pairs and two tables")
    print(json.dumps(derived,indent=2))


if __name__ == "__main__": main()
