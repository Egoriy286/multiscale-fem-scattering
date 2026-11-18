"""Check paper provenance, paired summaries, generated assets, and LaTeX output."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool(name):
    direct = shutil.which(name)
    if direct:
        return direct
    bundled = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin"/name
    if bundled.exists():
        return str(bundled)
    raise RuntimeError(f"Install Poppler ({name}) to check PDF text and page dimensions")


def main():
    import numpy as np
    manifest = json.loads((HERE/"evidence/manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert sha(ROOT/name) == digest, f"Changed numerical source: {name}"
    for name, digest in manifest["generated_sha256"].items():
        assert sha(HERE/name) == digest, f"Changed generated asset: {name}"
    assert sha(HERE/"evidence/field_examples.npz") == manifest["field_examples_sha256"]
    fresh = json.loads((HERE/"evidence/operator_checks.json").read_text())
    assert fresh["passed"] and fresh["reloaded_test_fields"] == 424
    assert fresh["field_examples_sha256"] == manifest["field_examples_sha256"]
    test_results = json.loads((HERE/"evidence/test_results.json").read_text())
    assert test_results["passed"] == 91 and test_results["subtests_passed"] == 2
    rows = list(csv.DictReader((HERE/"evidence/shared_scores.csv").open()))
    for row in rows:
        row["index"] = int(row["index"])
        for key in ("k","theta","total_e_complex_ref","scattered_e_complex_ref"):
            row[key] = float(row[key])
    from prepare_assets import summarize, SNO, GMS
    calculated = summarize(rows)
    assert calculated == manifest["summary"]
    assert len(rows) == 424*6
    table = (HERE/"generated/shared_table.tex").read_text()
    assert len(table.splitlines()) == 7
    for item,line in zip(calculated,table.splitlines()):
        assert item["samples"] == 424
        for field,stat in (("total","mean"),("total","median"),("total","p95"),("total","max"),("scattered","mean"),("scattered","p95")):
            assert f"{item[f'{field}_{stat}_percent']:.3f}" in line
    paired={m:{r["index"]:r for r in rows if r["method"]==m} for m in (SNO,GMS)}
    wins = sum(paired[SNO][i]["total_e_complex_ref"] < paired[GMS][i]["total_e_complex_ref"] for i in paired[SNO])
    assert wins == 139
    for i in paired[SNO]:
        assert all(paired[SNO][i][key] == paired[GMS][i][key] for key in ("k","theta"))
    derived=json.loads((HERE/"evidence/derived_statistics.json").read_text())
    assert derived["sno_lower_total_queries"] == wins
    assert derived["extrapolation_paired"]["samples"] == 200
    assert abs(derived["extrapolation_paired"]["shared_total_mean_percent"]-4.560418600295925) < 1e-12
    tex=(HERE/"main.tex").read_text()
    assert "\u2014" not in tex
    assert not re.search(r"\betc\.|TODO|TBD|lorem ipsum",tex,re.I)
    bib=(HERE/"references.bib").read_text()
    keys=set(re.findall(r"@\w+\{([^,]+),",bib))
    citations={key for group in re.findall(r"\\cite\w*\{([^}]+)\}",tex) for key in group.split(',')}
    assert citations <= keys
    log=(HERE/"main.log").read_text()
    assert not re.search(r"Overfull \\[hv]box|undefined|Fatal error|LaTeX Error",log,re.I)
    document=subprocess.check_output([tool("pdftotext"),"-layout",str(HERE/"main.pdf"),"-"]).decode()
    assert "AUTHORERR" not in document and "??" not in document
    pages=[page for page in document.split('\f') if page.strip()]
    reference_page=next(i+1 for i,page in enumerate(pages) if re.search(r"^\s*References\b",page,re.M))
    assert (HERE/"main.tex").stat().st_mtime <= (HERE/"main.pdf").stat().st_mtime, "Recompile after editing main.tex"
    assert reference_page-1 <= 8
    assert "Preprint. November 13, 2026." in pages[0]
    info=subprocess.check_output([tool("pdfinfo"),str(HERE/"main.pdf")]).decode()
    assert re.search(r"Page size:\s+612 x 792 pts",info)
    assert "Research preprint; not a conference publication" in info
    figure_paths=re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",tex)
    assert all((HERE/"figures"/name).is_file() for name in figure_paths)
    report=dict(passed=True,date="2026-11-13",numerical_sources=len(manifest["source_sha256"]),
                paired_queries=424,baseline_methods=5,sno_lower_total_queries=wins,
                tests=test_results["passed"],subtests=test_results["subtests_passed"],
                figure_pairs=len(list((HERE/"figures").glob("*.pdf"))),
                total_pdf_pages=len(pages),main_body_pages=reference_page-1,
                undefined_references=False,overfull_boxes=False,pdf_author_errors=False,
                letter_page_size=True,latex_sha256=sha(HERE/"main.tex"),pdf_sha256=sha(HERE/"main.pdf"),
                visual_review="Recorded separately in visual_review.json after inspecting final page renders")
    (HERE/"evidence/validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
