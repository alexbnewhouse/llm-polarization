#!/usr/bin/env bash
# Ollama LLM Benchmark Suite for AMD Ryzen AI MAX+ 395 / Radeon 8060S
# Tests: tokens/sec, TTFT, GPU utilization, model loading time

set -uo pipefail

RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_DIR/benchmark_$TIMESTAMP.json"
SUMMARY="$RESULTS_DIR/summary_$TIMESTAMP.txt"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*"; }

# ── Prompt library ───────────────────────────────────────────────────────────
SHORT_PROMPT="What is the capital of France? Answer in one word."
MEDIUM_PROMPT="Explain the difference between TCP and UDP networking protocols in about 3 paragraphs."
LONG_PROMPT="Write a comprehensive tutorial on Python list comprehensions, including basic syntax, nested comprehensions, conditional filtering, and 10 practical examples with explanations."
REASONING_PROMPT="Think step by step: A train leaves Station A at 9am traveling at 60mph. Another train leaves Station B (300 miles away) at 10am traveling at 80mph toward Station A. At what time do they meet?"

# ── Models to benchmark (edit this list as needed) ──────────────────────────
declare -A MODEL_CATEGORIES
MODEL_CATEGORIES["small"]="llama3.1:8b deepseek-r1:8b"
MODEL_CATEGORIES["medium"]="mistral-small:latest gemma3:27b qwen3:32b"
MODEL_CATEGORIES["large"]="qwen3-coder:latest qwen3.5:35b"
MODEL_CATEGORIES["xl"]="qwen3-coder-next:latest deepseek-r1:70b"

# Default: run small and medium unless --all or --category specified
RUN_CATEGORIES="${RUN_CATEGORIES:-small medium}"

# ── Argument parsing ─────────────────────────────────────────────────────────
QUICK=false
ALL_MODELS=false
SPECIFIED_MODELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)     QUICK=true; shift ;;
        --all)       ALL_MODELS=true; shift ;;
        --model)     SPECIFIED_MODELS="$SPECIFIED_MODELS $2"; shift 2 ;;
        --category)  RUN_CATEGORIES="$2"; shift 2 ;;
        *) echo "Usage: $0 [--quick] [--all] [--model <name>] [--category small|medium|large|xl]"; exit 1 ;;
    esac
done

if $ALL_MODELS; then
    RUN_CATEGORIES="small medium large xl"
fi

# ── GPU Monitoring ───────────────────────────────────────────────────────────
GPU_SAMPLES=()
start_gpu_monitor() {
    GPU_SAMPLES=()
    (while true; do
        pct=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk '{print $NF}' | tr -d '%')
        echo "${pct:-0}" >> /tmp/gpu_samples_$$
        sleep 0.5
    done) &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    kill $GPU_MONITOR_PID 2>/dev/null || true
    wait $GPU_MONITOR_PID 2>/dev/null || true
    if [[ -f /tmp/gpu_samples_$$ ]]; then
        mapfile -t GPU_SAMPLES < /tmp/gpu_samples_$$
        rm -f /tmp/gpu_samples_$$
    fi
}

avg_gpu_use() {
    if [[ ${#GPU_SAMPLES[@]} -eq 0 ]]; then echo "0"; return; fi
    local sum=0
    for v in "${GPU_SAMPLES[@]}"; do sum=$((sum + ${v:-0})); done
    echo $((sum / ${#GPU_SAMPLES[@]}))
}

peak_gpu_use() {
    if [[ ${#GPU_SAMPLES[@]} -eq 0 ]]; then echo "0"; return; fi
    local peak=0
    for v in "${GPU_SAMPLES[@]}"; do [[ ${v:-0} -gt $peak ]] && peak=${v:-0}; done
    echo $peak
}

# ── Single benchmark run ─────────────────────────────────────────────────────
run_single() {
    local model="$1" prompt="$2" label="$3"
    local tmp_out="/tmp/ollama_bench_$$"

    # Ensure model is loaded (warm-up call, discarded)
    if ! ollama list | grep -q "^${model}"; then
        warn "Model $model not found locally, skipping"
        echo "SKIP"
        return
    fi

    start_gpu_monitor

    local t_start t_end
    t_start=$(date +%s%3N)

    # Run inference, capture full response + stats
    ollama run "$model" "$prompt" > "$tmp_out" 2>&1
    local exit_code=$?

    t_end=$(date +%s%3N)
    stop_gpu_monitor

    if [[ $exit_code -ne 0 ]]; then
        err "Model $model failed"
        cat "$tmp_out"
        rm -f "$tmp_out"
        echo "ERROR"
        return
    fi

    local elapsed_ms=$(( t_end - t_start ))
    local response
    response=$(cat "$tmp_out")
    local word_count
    word_count=$(echo "$response" | wc -w)
    # Rough token estimate: words * 1.3
    local token_estimate=$(( word_count * 13 / 10 ))
    local tps=0
    if [[ $elapsed_ms -gt 0 ]]; then
        tps=$(( token_estimate * 1000 / elapsed_ms ))
    fi

    local avg_gpu peak_gpu
    avg_gpu=$(avg_gpu_use)
    peak_gpu=$(peak_gpu_use)

    rm -f "$tmp_out"

    echo "${elapsed_ms}|${token_estimate}|${tps}|${avg_gpu}|${peak_gpu}|${label}"
}

# ── Accurate benchmark using /api/generate with eval stats ──────────────────
run_api_bench() {
    local model="$1" prompt="$2" label="$3"

    if ! ollama list | grep -q "^${model}"; then
        warn "Model $model not found, skipping" >&2
        return 1
    fi

    log "  Benchmarking [$label] prompt..." >&2
    start_gpu_monitor

    local t_start t_end
    t_start=$(date +%s%3N)

    local response
    response=$(curl -s http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"prompt\": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"stream\": false}" \
        2>/dev/null)

    t_end=$(date +%s%3N)
    stop_gpu_monitor

    if [[ -z "$response" ]]; then
        err "No response from API for $model"
        return 1
    fi

    local total_duration eval_count eval_duration prompt_eval_count prompt_eval_duration
    total_duration=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_duration',0))" 2>/dev/null || echo "0")
    eval_count=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
    eval_duration=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo "0")
    prompt_eval_count=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo "0")
    prompt_eval_duration=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('prompt_eval_duration',0))" 2>/dev/null || echo "0")

    local gen_tps=0 prompt_tps=0 ttft_ms=0 total_ms=0
    if [[ $eval_duration -gt 0 && $eval_count -gt 0 ]]; then
        gen_tps=$(python3 -c "print(round($eval_count / ($eval_duration / 1e9), 1))")
    fi
    if [[ $prompt_eval_duration -gt 0 && $prompt_eval_count -gt 0 ]]; then
        prompt_tps=$(python3 -c "print(round($prompt_eval_count / ($prompt_eval_duration / 1e9), 1))")
    fi
    if [[ $prompt_eval_duration -gt 0 ]]; then
        ttft_ms=$(python3 -c "print(round($prompt_eval_duration / 1e6, 1))")
    fi
    if [[ $total_duration -gt 0 ]]; then
        total_ms=$(python3 -c "print(round($total_duration / 1e6, 1))")
    fi

    local avg_gpu peak_gpu
    avg_gpu=$(avg_gpu_use)
    peak_gpu=$(peak_gpu_use)

    echo "${model}|${label}|${gen_tps}|${prompt_tps}|${ttft_ms}|${total_ms}|${eval_count}|${avg_gpu}|${peak_gpu}"
}

# ── Model load time ──────────────────────────────────────────────────────────
measure_load_time() {
    local model="$1"
    log "  Measuring cold-load time for $model..." >&2

    # Unload model first
    ollama stop "$model" 2>/dev/null || true
    sleep 2

    local t_start t_end
    t_start=$(date +%s%3N)
    curl -s http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"prompt\": \"Hi\", \"stream\": false}" > /dev/null 2>&1
    t_end=$(date +%s%3N)
    echo $(( t_end - t_start ))
}

# ── Main benchmark loop ──────────────────────────────────────────────────────
declare -a JSON_ENTRIES=()
declare -a SUMMARY_LINES=()

SUMMARY_LINES+=("=== Ollama Benchmark Results — $(date) ===")
SUMMARY_LINES+=("Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151) / 128GB unified memory")
SUMMARY_LINES+=("")
SUMMARY_LINES+=("$(printf '%-32s %-6s %-6s %-8s %-8s %-8s' 'Model' 'GenTPS' 'PrmTPS' 'TTFT(ms)' 'AvgGPU%' 'PkGPU%')")
SUMMARY_LINES+=("$(printf '%.0s-' {1..80})")

models_to_run=()
if [[ -n "$SPECIFIED_MODELS" ]]; then
    IFS=' ' read -ra models_to_run <<< "$SPECIFIED_MODELS"
else
    for cat in $RUN_CATEGORIES; do
        if [[ -n "${MODEL_CATEGORIES[$cat]:-}" ]]; then
            IFS=' ' read -ra cat_models <<< "${MODEL_CATEGORIES[$cat]}"
            models_to_run+=("${cat_models[@]}")
        fi
    done
fi

for model in "${models_to_run[@]}"; do
    echo ""
    log "=== Benchmarking: $model ==="

    if ! ollama list | grep -q "^${model}"; then
        warn "Skipping $model (not installed)"
        continue
    fi

    # Measure cold load time
    load_ms=$(measure_load_time "$model")
    ok "Load time: ${load_ms}ms"

    declare -A bench_results
    bench_results=()

    # Short prompt
    result=$(run_api_bench "$model" "$SHORT_PROMPT" "short")
    if [[ -n "$result" ]]; then
        bench_results["short"]="$result"
    fi

    if ! $QUICK; then
        # Medium prompt
        result=$(run_api_bench "$model" "$MEDIUM_PROMPT" "medium")
        if [[ -n "$result" ]]; then
            bench_results["medium"]="$result"
        fi

        # Long prompt
        result=$(run_api_bench "$model" "$LONG_PROMPT" "long")
        if [[ -n "$result" ]]; then
            bench_results["long"]="$result"
        fi
    fi

    # Build JSON entry
    json_entry="{\"model\": \"$model\", \"load_ms\": $load_ms, \"results\": ["
    first=1
    for key in short medium long; do
        [[ -z "${bench_results[$key]:-}" ]] && continue
        IFS='|' read -ra f <<< "${bench_results[$key]}"
        # Fields: model|label|gen_tps|prompt_tps|ttft_ms|total_ms|eval_count|avg_gpu|peak_gpu
        lbl="${f[1]:-?}" gen_tps="${f[2]:-0}" prm_tps="${f[3]:-0}" ttft="${f[4]:-0}" total="${f[5]:-0}" tokens="${f[6]:-0}" avg_gpu="${f[7]:-0}" pk_gpu="${f[8]:-0}"
        [[ $first -eq 0 ]] && json_entry+=","
        json_entry+="{\"label\": \"$lbl\", \"gen_tps\": $gen_tps, \"prompt_tps\": $prm_tps, \"ttft_ms\": $ttft, \"total_ms\": $total, \"tokens\": $tokens, \"avg_gpu_pct\": $avg_gpu, \"peak_gpu_pct\": $pk_gpu}"
        first=0

        # Summary line
        SUMMARY_LINES+=("$(printf '%-32s %-6s %-6s %-8s %-8s %-8s' "${model}(${lbl})" "$gen_tps" "$prm_tps" "$ttft" "$avg_gpu" "$pk_gpu")")
    done
    json_entry+="]}"
    JSON_ENTRIES+=("$json_entry")

    ollama stop "$model" 2>/dev/null || true
done

# ── Write JSON log ───────────────────────────────────────────────────────────
{
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"hardware\": {"
    echo "    \"cpu\": \"AMD Ryzen AI MAX+ 395\","
    echo "    \"gpu\": \"Radeon 8060S (gfx1151)\","
    echo "    \"ram_gb\": 128,"
    echo "    \"rocm\": true"
    echo "  },"
    echo "  \"results\": ["
    IFS=','; echo "    $(IFS=$'\n'; echo "${JSON_ENTRIES[*]}" | paste -sd',')"
    echo "  ]"
    echo "}"
} > "$LOG"

# ── Write summary ─────────────────────────────────────────────────────────────
printf '%s\n' "${SUMMARY_LINES[@]}" > "$SUMMARY"
printf '%s\n' "${SUMMARY_LINES[@]}"

echo ""
ok "Results saved:"
echo "  JSON: $LOG"
echo "  Summary: $SUMMARY"
