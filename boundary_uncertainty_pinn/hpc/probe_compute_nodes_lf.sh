#!/usr/bin/env bash
set -euo pipefail

NODES=(10.6.234.22 10.6.234.23 10.6.234.24 10.6.234.25 10.6.234.27 10.6.234.28 10.6.234.29 10.6.234.30)

echo "manage_host=$(hostname)"
for NODE in "${NODES[@]}"; do
  echo "===== $NODE ====="
  ssh -o BatchMode=yes -o ConnectTimeout=6 "wxtian@$NODE" '
    echo "host=$(hostname)"
    echo "ip=$(hostname -I 2>/dev/null | awk "{print \$1}")"
    echo "date=$(date "+%F %T")"
    screen -ls 2>/dev/null | sed -n "1,8p" || true
    pgrep -af "train_decoupled_pfnn.py|train_deepxde_pfnn_v2.py|train_material_field.py" | sed -n "1,8p" || true
  ' || true
done
