#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/01.Base/PINN_Single/PINN + 固定权重/100k"

echo "root=$ROOT"
if [ ! -d "$ROOT" ]; then
  echo "missing_reference_dir"
  exit 0
fi

echo "===== directories ====="
find "$ROOT" -maxdepth 2 -type d | sed "s#^$ROOT#.#" | sort | head -200

echo "===== file type counts ====="
find "$ROOT" -type f | awk '
  {
    n=$0
    sub(/^.*\//, "", n)
    if (n ~ /\.[^.]+$/) {
      ext=n
      sub(/^.*\./, ".", ext)
    } else {
      ext="[no_ext]"
    }
    count[ext]++
  }
  END {
    for (ext in count) print count[ext], ext
  }
' | sort -nr

echo "===== sample files ====="
find "$ROOT" -maxdepth 3 -type f | sed "s#^$ROOT/##" | sort | head -250
