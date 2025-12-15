#!/bin/bash
# ===================================
#  Ablation Study Runner for DRAGON
#  Supports list-type parameters (safe string passing to Python)
#  Saves logs & results under ./ablation/
# ===================================

# 模型和数据集
MODEL="DRAGON"

# 设置输出目录
BASE_DIR="./ablation"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/results"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# 遍历数据集
for DATASET in "baby" "sports" "clothing"; do

    # === 每个 dataset 对应的参数定义 ===
    if [ "$DATASET" == "baby" ]; then
        lr="[0.0005]"                         
        reg_weight="0.001"
        mix_bpr_weight_loss="[0.1]"
        dragon_bpr_weight="[0.1]"
        align_weight_loss="[0.1]"
        diver_weight_loss="[0.1]"
    elif [ "$DATASET" == "sports" ]; then
        lr="[0.0005]"
        reg_weight="0.001"
        mix_bpr_weight_loss="[0.1]"
        dragon_bpr_weight="[0.1]"
        align_weight_loss="[0.1]"
        diver_weight_loss="[0.1]"
    elif [ "$DATASET" == "clothing" ]; then
        lr="[0.0005]"
        reg_weight="0.001"
        mix_bpr_weight_loss="[0.1]"
        dragon_bpr_weight="[0.1]"
        align_weight_loss="[0.1]"
        diver_weight_loss="[0.1]"
    fi

    # === 四个逻辑开关 ===
    for use_homogeneity in True False; do
    for use_diversity in True False; do
    for use_align_loss in True False; do
    for use_residual in True False; do

        # === 构造输出文件命名 ===
        CONFIG_NAME="${MODEL}_${DATASET}_homo-${use_homogeneity}_div-${use_diversity}_align-${use_align_loss}_res-${use_residual}"
        LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
        RESULT_FILE="${RESULT_DIR}/${CONFIG_NAME}.txt"

        echo "=================================================="
        echo " Running config: ${CONFIG_NAME}"
        echo " Log file: ${LOG_FILE}"
        echo "=================================================="

        # === 执行实验 ===
        python main.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --use_homogeneity "$use_homogeneity" \
            --use_diversity "$use_diversity" \
            --use_align_loss "$use_align_loss" \
            --use_residual "$use_residual" \
            --lr "$lr" \
            --reg_weight "$reg_weight" \
            --mix_bpr_weight_loss "$mix_bpr_weight_loss" \
            --dragon_bpr_weight "$dragon_bpr_weight" \
            --align_weight_loss "$align_weight_loss" \
            --diver_weight_loss "$diver_weight_loss" \
            > "$LOG_FILE" 2>&1

        # === 提取结果 ===
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
done

echo "=================================================="
echo "✅ All ablation experiments completed!"
echo "Results saved in:  ${RESULT_DIR}/"
echo "Logs saved in:     ${LOG_DIR}/"
echo "=================================================="