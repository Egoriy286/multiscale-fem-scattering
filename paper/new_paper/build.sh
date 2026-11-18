#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export TEXINPUTS="./vendor:${TEXINPUTS:-}"
export BSTINPUTS="./vendor:${BSTINPUTS:-}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
