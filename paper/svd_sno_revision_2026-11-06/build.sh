#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p build
if command -v latexmk >/dev/null 2>&1; then
    latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
else
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
fi
cp build/main.pdf main.pdf
printf '\nPDF: %s/main.pdf\n' "$PWD"
