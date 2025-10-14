#!/bin/bash

# ==============================================================================
# 全功能 Pass@k 评估脚本 (api_eval_passk_robust.py) 的启动器
#
# 功能:
# - 具备断点续传功能
# - 可配置 API 端点
# - 计算两种 Pass@k 指标
#
# 使用方法:
# 1. 修改下方的 "配置参数" 部分，填入你的文件路径和API密钥。
# 2. (如果需要) 将 'api_eval_passk_robust.py' 替换为你的Python脚本实际文件名。
# 3. 在终端中给予脚本执行权限: chmod +x run_robust_evaluation.sh
# 4. 首次运行: ./run_robust_evaluation.sh
# 5. 若需恢复中断的任务，取消下面命令中 '--resume' 参数的注释再运行。
# ==============================================================================

# --- 1. 配置参数 (请在此处修改) ---

# -- 文件路径 --
# 推理脚本生成的、包含多次尝试的预测结果文件
PREDICTION_FILE="/home/u2021110842/arxiv-appendix-extractor-v4/fill_mask_v0/output_result/qwen3-max_20251011_211800_run_final_result.jsonl"

# 包含标准答案的原始数据文件
GROUND_TRUTH_FILE="/home/u2021110842/arxiv-appendix-extractor-v4/fill_mask_v0/output_interline_mask.jsonl"

# 保存最终评估报告的输出文件路径
OUTPUT_FILE="/home/u2021110842/arxiv-appendix-extractor-v4/fill_mask_v0/output_result/qwen3-max_20251011_211800_report.json"

# -- API 与模型配置 --
# 用于评估的 API 密钥
API_KEY="sk-YOUR_GPT4O_API_KEY_HERE"

# (可选) 用于评估的模型名称
MODEL_NAME="gpt-4o-2024-05-13"

# (可选) API 的 Base URL，方便接入不同服务
BASE_URL="https://aihubmix.com/v1"

# -- 性能配置 --
# (可选) 最大并发API请求数，根据你的API速率限制调整
PARALLEL_SIZE=32


# --- 2. 构造并执行命令 (通常无需修改) ---

# 你的Python脚本文件名
PYTHON_SCRIPT="api_eval_passk_ali.py"

echo "==================================="
echo "🚀 启动 Pass@k 评估任务..."
echo "  - 预测文件: $PREDICTION_FILE"
echo "  - 标准答案: $GROUND_TRUTH_FILE"
echo "  - 输出报告: $OUTPUT_FILE"
echo "  - API 端点: $BASE_URL"
echo "  - 并发数: $PARALLEL_SIZE"
echo "==================================="

python "$PYTHON_SCRIPT" \
    --prediction_file "$PREDICTION_FILE" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --output_file "$OUTPUT_FILE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --base_url "$BASE_URL" \
    --parallel_size "$PARALLEL_SIZE" \
    # --resume  # <-- 如果需要从断点恢复，请删除本行开头的 '#'

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "✅ 评估任务成功完成。"
    echo "详细报告已保存到: $OUTPUT_FILE"
    echo "==================================="
else
    echo "==================================="
    echo "❌ 评估任务失败，请检查上面的错误信息。"
    echo "==================================="
fi
