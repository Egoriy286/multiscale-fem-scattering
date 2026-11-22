#!/bin/sh
set -eu

case "${1:-all}" in
  all|calculate|score) stage="${1:-all}" ;;
  *) printf '%s\n' 'Usage: reproduce.sh [all|calculate|score]' >&2; exit 2 ;;
esac

docker compose exec -T -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 dolfinx-complex \
  python3 -u -m paper.reference_modes_2026.gms.run "$stage"
if [ "$stage" != calculate ]; then
  docker compose exec -T dolfinx-complex python3 -m paper.reference_modes_2026.gms.example
fi
