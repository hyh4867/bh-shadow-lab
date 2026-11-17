#!/bin/bash
# Stage 6 queue (PREREGISTRATION 11.9): one heavy job at a time, after run_stage6_models.py has finished.
cd "$(dirname "$0")/.."
until [ -f results/stage6_models.json ]; do sleep 60; done
until ! screen -ls | grep -q bh6m; do sleep 30; done
for s in tau systematics xsring 2018 rml; do
  echo "=== $s $(date)" >> logs/stage6_queue.log
  .venv/bin/python scripts/run_stage6_$s.py > logs/stage6_$s.log 2>&1
  echo "=== $s exit $? $(date)" >> logs/stage6_queue.log
done
