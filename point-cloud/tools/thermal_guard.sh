#!/bin/bash
# CPU 온도 기록 · 과열 멈춤: 5 초마다 x86_pkg_temp 를 적고, 90 °C 이상이면 대상 PID 를 SIGSTOP, 75 °C 아래로 식으면 SIGCONT.
# 사용: bash thermal_guard.sh <pid> <기록 파일>   (대상이 끝나면 최고 온도를 적고 끝난다)
PID=$1; LOG=$2
Z=""
for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = x86_pkg_temp ] && Z=$z && break; done
[ -z "$Z" ] && { echo "$(date +%T) x86_pkg_temp 없음" >> "$LOG"; exit 0; }
paused=0
trap '[ "$paused" = 1 ] && kill -CONT $PID 2>/dev/null' EXIT
echo "$(date +%T) 시작 pid $PID (멈춤 90 °C 이상, 재개 75 °C 아래)" >> "$LOG"
while kill -0 $PID 2>/dev/null; do
  t=$(( $(cat $Z/temp) / 1000 ))
  echo "$(date +%T) $t" >> "$LOG"
  if [ "$t" -ge 90 ] && [ "$paused" = 0 ]; then kill -STOP $PID; paused=1; echo "$(date +%T) PAUSE $t" >> "$LOG"
  elif [ "$t" -le 75 ] && [ "$paused" = 1 ]; then kill -CONT $PID; paused=0; echo "$(date +%T) RESUME $t" >> "$LOG"; fi
  sleep 5
done
awk '$2 ~ /^[0-9]+$/ {n++; s+=$2; if ($2>m) m=$2} END {printf "끝: 최고 %d °C · 평균 %.1f °C (%d 회)\n", m, (n? s/n:0), n}' "$LOG" >> "$LOG"
