#!/bin/bash
# 명령을 nice 10 으로 돌리고 온도 감시를 붙인다. 사용: bash run_guarded.sh <기록 접두> <명령 ...>
P=$1; shift
nice -n 10 "$@" > "$P.out" 2>&1 &
PID=$!
bash "$(dirname "$0")/thermal_guard.sh" $PID "$P.thermal" &
wait $PID; RC=$?
sleep 6
echo "rc=$RC $(date +%T)" >> "$P.out"
tail -1 "$P.thermal" >> "$P.out"
