#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p tmp/latex output/pdf
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory=tmp/latex main.tex > tmp/latex/build_stdout.log
done
cp tmp/latex/main.pdf output/pdf/svd_sno_student_manual.pdf
echo "PDF: $(pwd)/output/pdf/svd_sno_student_manual.pdf"
