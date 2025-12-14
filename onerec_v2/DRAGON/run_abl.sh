#!/bin/bash
# ===================================
#  Ablation Study Runner for DRAGON
#  Saves logs & results under ./ablation/
# ===================================

DATASET="baby"
MODEL="DRAGON"

# 设置保存路径
BASE_DIR="./ablation"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/results"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# 遍历四个开关组合（此处你目前仅测试 False，可以改成 True False）
for use_homogeneity in True False; do
for use_diversity in True False; do
for use_align_loss in True False; do
for use_residual in True False; do

    # === ✅ 改进后的文件名，无空格无等号 ===
    CONFIG_NAME="${MODEL}_${DATASET}_homo-${use_homogeneity}_div-${use_diversity}_align-${use_align_loss}_res-${use_residual}"

    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
    RESULT_FILE="${RESULT_DIR}/${CONFIG_NAME}.txt"

    echo "=================================================="
    echo " Running config: ${CONFIG_NAME}"
    echo " Log file: ${LOG_FILE}"
    echo "=================================================="

    # === ✅ 执行实验并保存日志输出 ===
    python main.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --use_homogeneity "$use_homogeneity" \
        --use_diversity "$use_diversity" \
        --use_align_loss "$use_align_loss" \
        --use_residual "$use_residual" \
        > "$LOG_FILE" 2>&1

    # === ✅ 提取 best valid/test 结果部分 ===
    if grep -q "BEST" "$LOG_FILE"; then
        sed -n '/█████████████ BEST ████████████████/,$p' "$LOG_FILE" > "$RESULT_FILE"
        echo "✅ Result saved (from BEST marker) to $RESULT_FILE"
    else
        echo "⚠️ WARNING: No 'BEST' marker found in $LOG_FILE"
        {
            echo "use_homogeneity=${use_homogeneity}, use_diversity=${use_diversity}, use_align_loss=${use_align_loss}, use_residual=${use_residual}"
            echo "No BEST result found (check ${LOG_FILE})"
        } > "$RESULT_FILE"
    fi

done
done
done
done

echo "=================================================="
echo "✅ All ablation experiments completed!"
echo "Results saved in:  ${RESULT_DIR}/"
echo "Logs saved in:     ${LOG_DIR}/"
echo "=================================================="