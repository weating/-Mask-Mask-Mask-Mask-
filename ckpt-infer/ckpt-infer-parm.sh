#!/bin/bash

# =================================================================
# Python 推理脚本的启动器
#
# 使用方法:
# 1. 修改下面的 "配置参数" 部分。
# 2. 将 'your_script_name.py' 替换为你的 Python 脚本的实际文件名。
# 3. 在终端中给予脚本执行权限: chmod +x run_inference.sh
# 4. 运行脚本: ./run_inference.sh
# =================================================================

# --- 1. 配置参数 (请在此处修改) ---

# API 和模型配置
#-------------------------------------------------
# 要使用的模型名称 (例如: deepseek-coder, deepseek-v2, gpt-4o 等)
MODEL_NAME="qwen3-max"

# 你的 API Key
API_KEY="sk-4JIlYr6BX1NX5fQ0A2C6B066CdF645A78f4f892859A93389" # <-- 替换为你的真实 API Key

# API 的 Base URL
BASE_URL="https://aihubmix.com/v1"


# 数据和目录配置
#-------------------------------------------------
# 输入的数据文件路径 (.jsonl 格式)
DATA_FILE="./output_interline_mask.jsonl" # <-- 替换为你的输入文件路径

# 保存输出结果的文件夹路径
OUTPUT_DIR="./test-qwen3.jsonl" # <-- 替换为你的输出文件夹路径

# 用作输出文件和检查点文件的前缀名称
OUTPUT_PREFIX="qwen3_test_run"
INPUT_KEY="question"

# 运行与性能配置
#-------------------------------------------------
# 并行请求数 (线程数)
PARALLEL_SIZE="[200]"

# 每个输入样本生成的回应数量
N_RESPONSES="[32]"


# --- 2. 构造并执行命令 (通常无需修改) ---

echo "==================================="
echo "启动推理任务..."
echo "  - 模型: $MODEL_NAME"
echo "  - 输入文件: $DATA_FILE"
echo "  - 输出目录: $OUTPUT_DIR"
echo "==================================="

# 将 'your_script_name.py' 替换为你的 Python 脚本的实际文件名
python ckpt-infer-parm.py \
    --data_dir "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --output_prefix "$OUTPUT_PREFIX" \
    --inference_model "$MODEL_NAME" \
    --api_key "$API_KEY" \
    --input_key "$INPUT_KEY" \
    --base_url "$BASE_URL" \
    --parallel_size "$PARALLEL_SIZE" \
    --n_responses "$N_RESPONSES" \
    --max_tokens 8192 \
    --temperature 0.0 \
    --resume # 如果任务中断后需要从断点恢复，请保留此行。如果想从头开始，请在行首添加 '#' 注释掉此行。

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "推理任务成功完成。"
    echo "结果已保存到: $OUTPUT_DIR"
    echo "==================================="
else
    echo "==================================="
    echo "推理任务失败，请检查上面的错误信息。"
    echo "==================================="
fi
