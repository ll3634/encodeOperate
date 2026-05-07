#!/usr/bin/env bash
#
# run_norm_sweep_all.sh — 归一化公平对比 + 多 Random Baseline
#
# Phase 1: search-post 和 V12 做 max_rho sweep (各 ~2h)
# Phase 2: 5 个 random directions 只跑 rho=0.5 和 2.0 两个关键点 (各 ~48min, 共 ~4h)
# Phase 3: 汇总对比，输出 random 分布的 mean/std/95%CI
#
# 总计 ~8h。串行运行，共用 GPU。
#
# 用法:
#   cd tmc/scripts/e2e_agent
#   nohup bash scripts/run_norm_sweep_all.sh > results/norm_sweep_all.log 2>&1 &
#   tail -f results/norm_sweep_all.log   # 监控
#
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="results/norm_sweep_${TIMESTAMP}"
mkdir -p "$RESULTS_ROOT"

# --- 共享参数 ---
DATA="data/hotpotqa/hotpot_dev_distractor_v1.json"
CORPUS="data/hotpotqa/corpus.jsonl"
DATASET="hotpotqa"
TYPE_FILTER="bridge"
# --- 样本数：默认 200；可通过环境变量 N_SAMPLES 显式覆盖 ---
N_SAMPLES="${N_SAMPLES:-200}"
SEED=42
TAU="0.0"
RHO_SWEEP="0.5 1.0 1.5 2.0 3.0"
ALPHA_MAX=8.0
NORM_RMS=1.0
LAYER=12

# --- 模型：默认直接使用 HF model ID；如有需要可通过环境变量 MODEL 显式覆盖 ---
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
echo "Using model: ${MODEL}"

# --- 方向文件 ---
DIR_SEARCH="steering/directions/direction_search_post.npz"
DIR_V12="steering/directions/direction_v12_post_scaled.npz"

# --- Random baseline 参数 ---
N_RANDOM=5                     # 5个 random seeds，N_VC=23 下统计上已足够
RHO_RANDOM="0.5 2.0"           # 只跑两个关键点：noise floor(0.5) 和去饱和区(2.0)
RANDOM_DIR_ROOT="${RESULTS_ROOT}/random_directions"
mkdir -p "$RANDOM_DIR_ROOT"

echo "============================================================"
echo "  NORMALIZED SWEEP + MULTI-RANDOM BASELINE"
echo "  Started: $(date)"
echo "  Results: ${RESULTS_ROOT}"
echo "============================================================"
echo ""
echo "Config:"
echo "  N=${N_SAMPLES}, seed=${SEED}, tau=${TAU}"
echo "  max_rho sweep (search/V12): ${RHO_SWEEP}"
echo "  max_rho (random): ${RHO_RANDOM}"
echo "  alpha_max=${ALPHA_MAX}, normalize_rms=${NORM_RMS}"
echo "  Directions: search-post, V12 (full sweep); ${N_RANDOM}x random (rho=${RHO_RANDOM})"
echo ""

# --- Helper: run one direction ---
run_one() {
    local label="$1"
    local dir_path="$2"
    local rho_values="$3"
    local out_dir="${RESULTS_ROOT}/${label}"
    local start_ts=$(date +%s)

    echo "============================================================"
    echo "  [${label}] START — $(date)"
    echo "============================================================"

    python scripts/run_verify_critical_pipeline.py \
        --data-path "$DATA" \
        --corpus-path "$CORPUS" \
        --direction-path "$dir_path" \
        --dataset "$DATASET" \
        --type-filter "$TYPE_FILTER" \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED" \
        --tau-sweep $TAU \
        --max-rho-sweep $rho_values \
        --alpha-max "$ALPHA_MAX" \
        --normalize-rms "$NORM_RMS" \
        --layer "$LAYER" \
        --model "$MODEL" \
        --out "$out_dir"

    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    echo ""
    echo "  [${label}] DONE in ${elapsed}s ($((elapsed/60))min)"
    echo ""
}

TOTAL_START=$(date +%s)

# ================================================================
# Phase 1: Targeted directions with full rho sweep
# ================================================================
echo ""
echo "########  PHASE 1: Search-Post & V12 (full rho sweep)  ########"
echo ""

run_one "search_post" "$DIR_SEARCH" "$RHO_SWEEP"
run_one "v12_post" "$DIR_V12" "$RHO_SWEEP"

# ================================================================
# Phase 2: Multi-random baseline
# ================================================================
echo ""
echo "########  PHASE 2: ${N_RANDOM} Random Directions  ########"
echo ""

# Step 2a: Generate N_RANDOM random direction files
echo "[Phase 2a] Generating ${N_RANDOM} random directions..."
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '.')
from steering.directions import load_direction

# Load search_post as reference (for shape only; random is NOT orthogonalized)
ref, _ = load_direction('${DIR_SEARCH}', normalize_rms=None)
dim = ref.shape[0]

for seed_i in range(${N_RANDOM}):
    rng = np.random.RandomState(seed_i + 2000)  # offset to avoid collision with data seed
    rand_dir = rng.randn(dim).astype(np.float32)
    # Save with key='decision_direction' so load_direction() works unchanged
    out_path = '${RANDOM_DIR_ROOT}/random_seed{}.npz'.format(seed_i)
    np.savez(out_path, decision_direction=rand_dir)
    rms = float(np.sqrt(np.mean(rand_dir**2)))
    print(f'  Saved {out_path}  (raw RMS={rms:.6f}, will be normalized to ${NORM_RMS})')
print('Done generating ${N_RANDOM} random directions.')
"
echo ""

# Step 2b: Run each random direction
for i in $(seq 0 $((N_RANDOM - 1))); do
    run_one "random_seed${i}" "${RANDOM_DIR_ROOT}/random_seed${i}.npz" "$RHO_RANDOM"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

# ================================================================
# Phase 3: Cross-direction comparison + random distribution
# ================================================================
echo ""
echo "============================================================"
echo "  CROSS-DIRECTION COMPARISON"
echo "  Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED/60))min)"
echo "============================================================"
echo ""

python3 scripts/compare_norm_sweep.py "$RESULTS_ROOT" --n-random "$N_RANDOM"

echo ""
echo "Results directory: ${RESULTS_ROOT}"
echo "Done at: $(date)"

