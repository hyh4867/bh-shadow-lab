#!/bin/bash
# Stage 7 queue (PREREGISTRATION 12.6): one heavy job at a time. Run inside screen with caffeinate -i.
cd "$(dirname "$0")/.."
mkdir -p logs
for s in modemap models select; do
  echo "=== $s $(date)" >> logs/stage7_queue.log
  .venv/bin/python scripts/run_stage7_$s.py > logs/stage7_$s.log 2>&1
  rc=$?
  echo "=== $s exit $rc $(date)" >> logs/stage7_queue.log
  [ $rc -eq 0 ] || exit $rc
done
