#!/bin/bash
# 多 seed 批量生成脚本
# 用法: bash run_multi_seed.sh [run次数] [输出根目录] [script yaml]
#   例: bash run_multi_seed.sh 3
#       bash run_multi_seed.sh 5 /tmp/my_outputs
#       bash run_multi_seed.sh 3 ./out_aba test_aba_scene.yaml

set -e

RUNS=${1:-3}
BASE_OUTPUT=${2:-"/root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/phase1_poc/output_multi_seed"}
SCRIPT_NAME=${3:-"test_night_rider.yaml"}

SCRIPT="/root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/configs/${SCRIPT_NAME}"
CONFIG="/root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/configs/config.yaml"
NGPU=8

mkdir -p "$BASE_OUTPUT"

echo "=========================================="
echo "  多 seed 批量生成"
echo "  runs      : $RUNS"
echo "  script    : $SCRIPT"
echo "  base_output: $BASE_OUTPUT"
echo "  nproc_per_node: $NGPU"
echo "=========================================="

# 预定义 seed 列表（可重复使用、结果可复现）
SEEDS=(1337 2025 314159 777 99999 12345 54321 66666 11111 42)

for i in $(seq 1 "$RUNS"); do
    # 优先使用预定义 seed，超出列表范围则随机生成
    if [ $((i - 1)) -lt ${#SEEDS[@]} ]; then
        SEED=${SEEDS[$((i - 1))]}
    else
        SEED=$((RANDOM * RANDOM + i * 12345))
    fi

    RUN_DIR="$BASE_OUTPUT/run_$(printf '%03d' $i)_seed${SEED}"

    echo ""
    echo "------------------------------------------"
    echo "  Run $i / $RUNS  |  seed=$SEED"
    echo "  output: $RUN_DIR"
    echo "------------------------------------------"

    torchrun --nproc_per_node=$NGPU \
        /root/paddlejob/workspace/env_run/output/lyx/T2V_Grounding/phase1_poc/run_demo.py \
        --script  "$SCRIPT" \
        --config  "$CONFIG" \
        --output  "$RUN_DIR" \
        --backend phantom \
        --seed    "$SEED"

    echo "  [OK] Run $i 完成 -> $RUN_DIR"
done

echo ""
echo "=========================================="
echo "  全部完成！输出目录:"
ls -1d "$BASE_OUTPUT"/run_*/
echo "=========================================="
