# --- 1. 配置参数 (请在此处修改) ---

# **核心配置**
# ------------------------------------------------------------------------------
# 存放所有待评测推理文件 (.jsonl) 的文件夹路径
# 脚本将处理这个文件夹下的所有 .jsonl 文件
PREDICTIONS_DIR="/home/u2021110842/arxiv-appendix-extractor-v4/fill_mask_v0/output_result"

# 所有评测任务共享的、唯一的真值文件路径
GROUND_TRUTH_FILE="/home/u2021110842/arxiv-appendix-extractor-v4/fill_mask_v0/output_interline_mask_v3.jsonl"

# -- API 与性能配置 --
# ------------------------------------------------------------------------------
# 用于评估的 API 密钥
API_KEY="sk-YOUR_GPT4O_API_KEY_HERE"

# API 的 Base URL
BASE_URL="https://aihubmix.com/v1"

# 最大并发API请求数
PARALLEL_SIZE=32


# --- 2. 自动化执行逻辑 (通常无需修改) ---

# 你的Python脚本文件名
PYTHON_SCRIPT="api_eval_passk_ali.py"

# 检查预测文件夹是否存在
if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "❌ 错误：找不到预测文件夹: $PREDICTIONS_DIR"
    exit 1
fi

# 检查真值文件是否存在
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "❌ 错误：找不到真值文件: $GROUND_TRUTH_FILE"
    exit 1
fi

echo "===================================================="
echo "🤖 启动智能批量评估..."
echo "将处理 '$PREDICTIONS_DIR' 目录下的所有 .jsonl 文件"
echo "===================================================="
echo ""

# 查找并遍历目录下的所有 .jsonl 文件
# 使用 find 命令以更好地处理文件名中可能包含空格等特殊字符的情况
find "$PREDICTIONS_DIR" -maxdepth 1 -type f -name "*.jsonl" | while read -r PRED_FILE; do

    # 智能生成输出报告文件名
    # 例如: /path/model_A.jsonl -> /path/model_A-report.json
    OUTPUT_FILE="${PRED_FILE%.jsonl}-report.json"

    echo "----------------------------------------------------"
    echo "▶️  正在处理文件: $(basename "$PRED_FILE")"
    echo "   - 真值文件: $(basename "$GROUND_TRUTH_FILE")"
    echo "   - 输出报告将保存为: $(basename "$OUTPUT_FILE")"
    echo "----------------------------------------------------"

    python "$PYTHON_SCRIPT" \
        --prediction_file "$PRED_FILE" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --output_file "$OUTPUT_FILE" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --parallel_size "$PARALLEL_SIZE" \
        # --resume  # <-- 如果需要为单个任务恢复，请删除本行开头的 '#'

    # 检查上一个任务的执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 成功完成: $(basename "$PRED_FILE")"
    else
        echo "⚠️  处理失败: $(basename "$PRED_FILE"). 请检查上方日志。将继续处理下一个文件..."
    fi
    echo "" # 增加空行以提高可读性

done

echo "===================================================="
echo "🎉 所有评估任务已执行完毕。"
echo "===================================================="
