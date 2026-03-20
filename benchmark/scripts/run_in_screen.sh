#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash benchmark/scripts/run_in_screen.sh <screen_name> <command ...>"
  exit 1
fi

SCREEN_NAME=$1
shift
screen -S "$SCREEN_NAME" -dm bash -lc "$*"
screen -ls | grep "$SCREEN_NAME" || true
