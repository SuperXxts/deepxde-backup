#!/usr/bin/env bash
set -euo pipefail

STAMP=${1:-20260702_dec_formal01}
NODE=${2:-10.6.234.27}

ssh "wxtian@$NODE" "
  hostname
  echo '--- screens ---'
  screen -ls | grep A8_boundary || true
  screen -ls | grep '$STAMP' || true
  echo '--- processes ---'
  ps -u \"\${USER:-wxtian}\" -o pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd= \
    | grep train_decoupled_pfnn.py \
    | grep A8_boundary \
    | grep '$STAMP' \
    | grep -v grep || true
"
