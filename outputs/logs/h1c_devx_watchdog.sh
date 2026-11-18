#!/usr/bin/env bash
# BTC-H1c-DEVX watchdog (owner condition 9): stop the Modal app at the first container startup failure.
LOG=outputs/logs/h1c_devx_run.log
OUT=outputs/logs/h1c_devx_watchdog.log
echo "$(date -u +%FT%TZ) watchdog started" >> $OUT
while true; do
  if [ -f "$LOG" ] && grep -q "failed during startup\|Runner failed" "$LOG"; then
    APP=$(grep -o "ap-[A-Za-z0-9]*" "$LOG" | head -1)
    echo "$(date -u +%FT%TZ) startup failure detected; stopping $APP" >> $OUT
    .venv/Scripts/modal app stop "$APP" --yes >> $OUT 2>&1
    echo "$(date -u +%FT%TZ) stop issued" >> $OUT
    exit 0
  fi
  if [ -f "$LOG" ] && grep -q "^EXIT=" "$LOG"; then
    echo "$(date -u +%FT%TZ) run finished ($(grep '^EXIT=' $LOG)); watchdog exits" >> $OUT
    exit 0
  fi
  sleep 3
done
