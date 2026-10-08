#!/bin/bash
# 回滚 ORPO v4 → 恢复 SFT INT4

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "回滚到 SFT 模型..."

# 恢复 start.sh
if [ -f start.sh.bak ]; then
    mv start.sh.bak start.sh
    echo "✓ start.sh 已恢复"
else
    # 手动恢复
    sed -i "s|qwen3_orpo_v4_int4|qwen3_lora_sft_int4|g" start.sh
    echo "✓ start.sh 路径已手动恢复"
fi

echo ""
echo "回滚完成。重启服务生效："
echo "  bash start.sh"
