#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
NODE=${NODE:-10.6.234.22}
RUN_DIR=${RUN_DIR:?set RUN_DIR from launch output}

ssh "$NODE" "set -e; echo '--- processes ---'; ps -u wxtian -f | grep train_deepxde_pfnn.py | grep -v grep || true; echo '--- log timestamp ---'; stat -c '%y %n' '$RUN_DIR/txt/train.log' 2>/dev/null || true; echo '--- latest steps ---'; grep -E 'Step|STEP|RUN_COMPLETE|Traceback|Error' '$RUN_DIR/txt/train.log' 2>/dev/null | tail -40 || true; echo '--- artifacts ---'; find '$RUN_DIR' -maxdepth 2 -type f | sed 's#^#$NODE:#' | head -80"
