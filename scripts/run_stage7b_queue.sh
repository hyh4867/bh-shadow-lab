#!/bin/bash
# Stage 7 ladder B queue (PREREGISTRATION 12.7). Run inside screen with caffeinate -i.
cd "$(dirname "$0")/.."
for s in stage7b stage7_select; do
  echo "=== $s $(date)" >> logs/stage7_queue.log
  .venv/bin/python scripts/run_$s.py > logs/$s.log 2>&1
  rc=$?
  echo "=== $s exit $rc $(date)" >> logs/stage7_queue.log
  [ $rc -eq 0 ] || exit $rc
done
