#!/bin/bash
# Show live per-job progress for a paper_sweep run.
#
#   analysis/prog.sh            # the real sweep (results/paper)
#   analysis/prog.sh smoke      # any other --tag
#   watch -n 30 analysis/prog.sh
#
# Reads the tqdm bar out of each job's log. tqdm overwrites its line with
# carriage returns, so the log has to be split on \r before the last bar can
# be picked off; a plain tail shows one unreadable mega-line.

cd "$(dirname "$0")/.." || exit 1

TAG=${1:-paper}
LOG_DIR="results/$TAG/_logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "No logs yet at $LOG_DIR (sweep not started, or wrong tag)."
    exit 0
fi

running=0
finished=0

for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)

    variant=${name%_seed_*}
    seed=${name##*_seed_}

    if [ -f "results/$TAG/$variant/seed_$seed/status.json" ]; then
        state="done"
        finished=$((finished + 1))
    else
        state=""
        running=$((running + 1))
    fi

    bar=$(tr '\r' '\n' < "$f" | grep -a "%|" | tail -1 | sed 's/.*: *\([0-9]*%\)/\1/')
    printf "%-24s %-46s %s\n" "$name" "${bar:-starting...}" "$state"
done

echo "---"
echo "$finished finished, $running in flight / started"
