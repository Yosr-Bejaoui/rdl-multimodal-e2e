#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
for cfg in configs/*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== $name ==="
  python -m rdl_e2e.cli --config "$cfg" --mode both --out "results/${name}.json"
done

echo "Done. Build the comparison table with: python scripts/make_report.py"
