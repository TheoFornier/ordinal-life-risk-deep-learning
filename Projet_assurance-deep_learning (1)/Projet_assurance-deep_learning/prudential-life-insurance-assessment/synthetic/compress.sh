#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
csv_files=("$script_dir"/*.csv)

if [ "${#csv_files[@]}" -eq 0 ]; then
  echo "No CSV files found in $script_dir"
  exit 0
fi

for csv_file in "${csv_files[@]}"; do
  base="$(basename "$csv_file" .csv)"
  echo "Compressing ${base}.csv..."
  zip -j "$script_dir/${base}.zip" "$csv_file"
done

echo "Finished compressing ${#csv_files[@]} file(s)."
