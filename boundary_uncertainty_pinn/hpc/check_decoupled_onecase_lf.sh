#!/usr/bin/env bash
set -euo pipefail

NODE=${1:?Usage: check_decoupled_onecase_lf.sh NODE RUN_DIR [SLEEP_SECONDS]}
RUN_DIR=${2:?Usage: check_decoupled_onecase_lf.sh NODE RUN_DIR [SLEEP_SECONDS]}
SLEEP_SECONDS=${3:-20}

ssh "$NODE" "set -euo pipefail
  echo '===== node ====='
  hostname
  echo '===== live process ====='
  ps -u \"\${USER:-wxtian}\" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= | grep train_decoupled_pfnn.py | grep '$RUN_DIR' | grep -v grep || true
  echo '===== log timestamp before ====='
  stat -c '%Y %y %s %n' '$RUN_DIR/txt/train.log' 2>/dev/null || true
  echo '===== latest steps before ====='
  grep -E '^[[:space:]]*[0-9]+[[:space:]]|STEP|RUN_COMPLETE|Traceback|Error|ERROR' '$RUN_DIR/txt/train.log' 2>/dev/null | tail -20 || true
  sleep '$SLEEP_SECONDS'
  echo '===== log timestamp after ====='
  stat -c '%Y %y %s %n' '$RUN_DIR/txt/train.log' 2>/dev/null || true
  echo '===== latest steps after ====='
  grep -E '^[[:space:]]*[0-9]+[[:space:]]|STEP|RUN_COMPLETE|Traceback|Error|ERROR' '$RUN_DIR/txt/train.log' 2>/dev/null | tail -25 || true
  echo '===== artifacts ====='
  find '$RUN_DIR' -maxdepth 2 -type f 2>/dev/null | sed 's#^#file #g' | sort | head -120 || true
"
