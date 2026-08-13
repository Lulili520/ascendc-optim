#!/usr/bin/env bash
set -uo pipefail

cd /data/lu/ascendc-optim || exit 1
source /data/lu/activate_evokernel.sh || exit 1

queue_file=kernel_workspace/KernelBench910B/level1_remaining_queue.json
progress_log=kernel_workspace/KernelBench910B/level1_remaining_collection.tsv

while IFS= read -r op_name; do
    project_dir="kernel_workspace/KernelBench910B/level1/${op_name}_0"
    workspace_file="$project_dir/workspace.json"
    status=$(jq -r '.status' "$workspace_file")
    printf 'CHECK\t%s\t%s\t%s\n' "$op_name" "$status" "$(date -u +%FT%TZ)" >> "$progress_log"

    if [[ "$status" == "PREPARED" || "$status" == "BUILD_FAILED" ]]; then
        python .codex/skills/kernel-precision/scripts/validate_precision.py \
            "$op_name" --project-dir "$project_dir"
        precision_rc=$?
        printf 'PRECISION\t%s\t%s\t%s\n' "$op_name" "$precision_rc" "$(date -u +%FT%TZ)" >> "$progress_log"
        status=$(jq -r '.status' "$workspace_file")
    fi

    if [[ "$status" == "PRECISION_PASS" ]]; then
        python .codex/skills/kernel-performance/scripts/collect_performance.py \
            "$op_name" --device 0 --project-dir "$project_dir"
        performance_rc=$?
        printf 'PERFORMANCE\t%s\t%s\t%s\n' "$op_name" "$performance_rc" "$(date -u +%FT%TZ)" >> "$progress_log"
    fi
done < <(jq -r '.items[].operator' "$queue_file")

printf 'QUEUE_DONE\t-\t0\t%s\n' "$(date -u +%FT%TZ)" >> "$progress_log"
