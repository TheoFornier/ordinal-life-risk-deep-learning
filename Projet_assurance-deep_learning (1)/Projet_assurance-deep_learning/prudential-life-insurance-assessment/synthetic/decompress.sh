#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
zip_files=("$script_dir"/*.zip)

if [ "${#zip_files[@]}" -eq 0 ]; then
  echo "No zip files found in $script_dir"
  exit 0
fi

for zip_file in "${zip_files[@]}"; do
  echo "Extracting $(basename "$zip_file")..."
  unzip -o "$zip_file" -d "$script_dir"
done

echo "Finished extracting ${#zip_files[@]} archive(s)."
