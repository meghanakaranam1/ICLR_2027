#!/bin/bash
# Self-healing wrapper for run_isocompute_final.py --calibrate-only, same
# stall-detection pattern already proven elsewhere in this repo (see
# mistral_aegis_autoheal.sh): 3 consecutive 60s ticks with no growth in the
# calibration checkpoint files = confirmed Ollama-server-degradation stall
# (an observed, not hypothetical, failure mode on this machine), kill -9 the
# worker and relaunch -- safe because run_isocompute_final.py now persists
# every (candidate, example) score to a per-pair checkpoint file as it goes.

cd /Users/meghanakarnam/Desktop/MLHC/ICLR_2027 || exit 1
LOG=/Users/meghanakarnam/Desktop/MLHC/ICLR_2027/isocompute_calibration_clean.log
WORKER_PID=""

log() { echo "$(date): $1" >> "$LOG"; }

checkpoint_rows() {
    total=0
    for f in analysis/risk_coverage/isocompute_calib_checkpoint_*.jsonl; do
        [ -f "$f" ] && total=$((total + $(wc -l < "$f" 2>/dev/null || echo 0)))
    done
    echo $total
}

launch_worker() {
    python3 -u run_isocompute_final.py --calibrate-only >> "$LOG" 2>&1 &
    WORKER_PID=$!
    log "worker launched, pid=$WORKER_PID"
}

restart_ollama() {
    log "restarting ollama service (stall recovery)"
    launchctl kickstart -k gui/$(id -u)/homebrew.mxcl.ollama 2>&1 | while read l; do log "  $l"; done
    sleep 5
}

log "=== isocompute calibration autoheal started ==="
launch_worker
sleep 20
last=$(checkpoint_rows)
stall_ticks=0

while true; do
    sleep 90
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        wait "$WORKER_PID" 2>/dev/null
        exit_code=$?
        if [ "$exit_code" -eq 0 ] && [ -f analysis/risk_coverage/isocompute_frozen_budgets.json ]; then
            log "ALL DONE -- frozen budgets written"
            exit 0
        fi
        log "worker exited (code=$exit_code), relaunching"
        launch_worker
        sleep 20
        last=$(checkpoint_rows)
        stall_ticks=0
        continue
    fi

    cur=$(checkpoint_rows)
    if [ "$cur" == "$last" ]; then
        stall_ticks=$((stall_ticks + 1))
        log "no growth ($cur checkpoint rows), stall_ticks=$stall_ticks/3"
    else
        stall_ticks=0
        log "progress ($last -> $cur checkpoint rows)"
    fi
    last=$cur

    if [ "$stall_ticks" -ge 3 ]; then
        log "STALL CONFIRMED -- killing pid=$WORKER_PID, restarting ollama, relaunching"
        kill -9 "$WORKER_PID" 2>/dev/null
        sleep 2
        restart_ollama
        launch_worker
        sleep 20
        last=$(checkpoint_rows)
        stall_ticks=0
    fi
done
