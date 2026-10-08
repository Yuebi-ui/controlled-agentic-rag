#!/bin/bash
# Phase 0 Step 3: 部署 ORPO v4 INT4 模型到线上
# 保留原 SFT INT4 做备份

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_MODEL_PATH="LLaMA-Factory-main/output/qwen3_orpo_v4_int4"
OLD_MODEL_PATH="LLaMA-Factory-main/output/qwen3_lora_sft_int4"
BACKUP_PATH="LLaMA-Factory-main/output/qwen3_lora_sft_int4.bak"

echo "========================================"
echo "  ORPO v4 模型部署"
echo "========================================"

cd "$PROJECT_DIR"

# 1. 检查新模型是否存在
if [ ! -d "$NEW_MODEL_PATH" ]; then
    echo "❌ 新模型不存在: $NEW_MODEL_PATH"
    echo "   请先运行:"
    echo "   python scripts/merge_orpo_v4.py"
    echo "   python scripts/quantize_orpo_v4.py"
    exit 1
fi

echo "新模型: $NEW_MODEL_PATH"
echo "旧模型: $OLD_MODEL_PATH"
echo ""

# 2. 备份原 SFT 模型
if [ -d "$OLD_MODEL_PATH" ] && [ ! -d "$BACKUP_PATH" ]; then
    echo "[1/3] 备份原 SFT 模型..."
    cp -r "$OLD_MODEL_PATH" "$BACKUP_PATH"
    echo "  ✓ 备份完成: $BACKUP_PATH"
else
    echo "[1/3] 备份已存在或旧模型不存在，跳过"
fi

# 3. 更新 start.sh 中的模型路径
echo "[2/3] 更新 start.sh..."
sed -i.bak "s|qwen3_lora_sft_int4|qwen3_orpo_v4_int4|g" start.sh
echo "  ✓ start.sh 模型路径已更新"
grep "vllm serve" start.sh

# 4. 清理合并信物
echo "[3/3] 清理临时文件..."
BF16_PATH="LLaMA-Factory-main/output/qwen3_orpo_v4_bf16"
if [ -d "$BF16_PATH" ]; then
    echo "  保留 BF16 合并模型: $BF16_PATH (可用于后续训练)"
fi

echo ""
echo "========================================"
echo "  部署完成！"
echo ""
echo "  后续步骤:"
echo "  1. 重启 vLLM（或运行 bash start.sh）"
echo "  2. 测试: curl -X POST http://localhost:9000/chat -H 'Content-Type: application/json' -d '{\"query\":\"充电口在哪\"}'"
echo "  3. 如需回滚: bash scripts/rollback_orpo_v4.sh"
echo "========================================"
