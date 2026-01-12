#!/bin/bash
# Run scene generation in parallel by splitting prompts across workers
#
# Usage: ./scripts/run_parallel_generation.sh <csv_file> <num_workers> [extra_args...]
#
# Examples:
#   # Generate scenes from prompts.csv with 4 workers
#   ./scripts/run_parallel_generation.sh prompts.csv 4
#
#   # With custom results directory and mode
#   ./scripts/run_parallel_generation.sh ~/SceneEval/input/annotations.csv 8 \
#       --results_dir ./results/my_run \
#       --mode one_shot \
#       --skip_existing
#
#   # Full example with all options
#   ./scripts/run_parallel_generation.sh prompts.csv 4 \
#       --asset_source curated \
#       --mode finetuned \
#       --render_final \
#       --skip_existing
#
# Output:
#   - Each scene creates its own output directory
#   - Worker stdout/stderr is captured in logs/worker_*.log

set -e

# Parse arguments
CSV_FILE=${1:-""}
NUM_WORKERS=${2:-4}

if [ $# -lt 2 ] || [ -z "$CSV_FILE" ] || [ ! -f "$CSV_FILE" ]; then
    echo "Usage: $0 <csv_file> <num_workers> [extra_args...]"
    echo ""
    echo "Arguments:"
    echo "  csv_file     Path to CSV file with prompts (must have ID, Description columns)"
    echo "  num_workers  Number of parallel workers"
    echo "  extra_args   Additional arguments passed to run_from_csv.py"
    echo ""
    echo "Extra args options:"
    echo "  --results_dir DIR      Directory to save results"
    echo "  --asset_source TYPE    'curated' (674 assets) or 'full' (50K assets)"
    echo "  --mode MODE            'finetuned' (default), 'one_shot', etc."
    echo "  --render_final         Render final scene with 3D assets"
    echo "  --save_blend           Save .blend files (requires --render_final)"
    echo "  --skip_existing        Skip scenes that already have results"
    echo "  --max_retries N        Max retries for failed scenes (default: 10)"
    echo ""
    echo "Example: $0 prompts.csv 4 --skip_existing --mode finetuned"
    exit 1
fi

shift 2  # Remove first two args, rest are passed to run_from_csv.py

# Get script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create unique run ID and log directory
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${PROJECT_DIR}/logs/run_${RUN_ID}"
mkdir -p "$LOG_DIR"

# Count total prompts in CSV (excluding header)
TOTAL_PROMPTS=$(($(wc -l < "$CSV_FILE") - 1))

if [ "$TOTAL_PROMPTS" -le 0 ]; then
    echo "Error: CSV file has no prompts (only header or empty)"
    exit 1
fi

# Calculate prompts per worker
PROMPTS_PER_WORKER=$(( (TOTAL_PROMPTS + NUM_WORKERS - 1) / NUM_WORKERS ))  # Ceiling division
PIDS=()

# Cleanup function to kill all workers
cleanup() {
    echo ""
    echo "Caught signal, terminating workers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    # Wait briefly for graceful shutdown, then force kill
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing worker $pid..."
            kill -9 "$pid" 2>/dev/null
        fi
    done
    echo "Workers terminated."
    exit 130
}

# Trap signals to ensure cleanup
trap cleanup INT TERM EXIT

echo "========================================"
echo "Parallel Scene Generation"
echo "========================================"
echo "Run ID: $RUN_ID"
echo "CSV file: $CSV_FILE"
echo "Total prompts: $TOTAL_PROMPTS"
echo "Workers: $NUM_WORKERS"
echo "Prompts per worker: ~$PROMPTS_PER_WORKER"
echo "Extra args: $@"
echo "Log directory: $LOG_DIR"
echo "========================================"
echo ""

# Get min and max IDs from CSV
MIN_ID=$(tail -n +2 "$CSV_FILE" | cut -d',' -f1 | sort -n | head -1)
MAX_ID=$(tail -n +2 "$CSV_FILE" | cut -d',' -f1 | sort -n | tail -1)

echo "ID range in CSV: $MIN_ID to $MAX_ID"
echo ""

# Launch workers with ID ranges
cd "$PROJECT_DIR"

for ((i=0; i<NUM_WORKERS; i++)); do
    START_ID=$((MIN_ID + i * PROMPTS_PER_WORKER))
    END_ID=$((MIN_ID + (i + 1) * PROMPTS_PER_WORKER - 1))

    # Last worker gets all remaining
    if [ $i -eq $((NUM_WORKERS - 1)) ]; then
        END_ID=$MAX_ID
    fi

    # Skip if start is beyond max
    if [ "$START_ID" -gt "$MAX_ID" ]; then
        echo "Worker $i: No prompts to process (start_id $START_ID > max_id $MAX_ID)"
        continue
    fi

    # Clamp end to max
    if [ "$END_ID" -gt "$MAX_ID" ]; then
        END_ID=$MAX_ID
    fi

    echo "Starting worker $i: IDs [$START_ID, $END_ID]"

    uv run python run_from_csv.py \
        --csv_file "$CSV_FILE" \
        --start_id "$START_ID" \
        --end_id "$END_ID" \
        "$@" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/worker_*.log"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_DIR}/worker_*.log"
echo ""
echo "Waiting for completion (Ctrl+C to cancel)..."

# Wait for all workers and track failures
FAILED=0
FAILED_WORKERS=()

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if ! wait $PID; then
        echo "Worker $i (PID $PID) FAILED"
        FAILED=1
        FAILED_WORKERS+=($i)
    else
        echo "Worker $i (PID $PID) completed"
    fi
done

echo ""
echo "========================================"

# Disable EXIT trap for normal completion
trap - EXIT

if [ $FAILED -eq 1 ]; then
    echo "SOME WORKERS FAILED: ${FAILED_WORKERS[*]}"
    echo "Check ${LOG_DIR}/worker_*.log for details"
    exit 1
else
    echo "ALL WORKERS COMPLETED SUCCESSFULLY"
fi
echo "Logs: ${LOG_DIR}/"
echo "========================================"
