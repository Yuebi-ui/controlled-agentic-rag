#!/bin/bash
# ========================================
# Agentic Hybrid RAG — 一键启动脚本
# 使用方法: bash start.sh
# ========================================

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ------ 1. 环境配置 ------
# 自动检测 conda 路径（优先使用 CONDA_PREFIX 环境变量，否则尝试常见路径）
if [ -n "$CONDA_PREFIX" ]; then
    CONDA_BASE="$(dirname "$CONDA_PREFIX")"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "错误：未找到 conda 安装路径，请设置 CONDA_PREFIX 环境变量"
    exit 1
fi
CONDA_ENV=rag
PYTHON=$CONDA_BASE/envs/$CONDA_ENV/bin/python
VLLM=$CONDA_BASE/envs/$CONDA_ENV/bin/vllm

export PATH=$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
export HF_ENDPOINT=https://hf-mirror.com
unset OMP_NUM_THREADS

# 加载 .env 环境变量（使用 source 方式，兼容引号/特殊字符）
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# 千问 API（兼容 LLM_API_KEY 和 DASHSCOPE_API_KEY 两种命名，不覆盖已有值）
export LLM_API_KEY="${LLM_API_KEY:-${DASHSCOPE_API_KEY:-}}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen-plus}"

# ------ 2. 清理旧进程 ------
echo "[清理] 停止旧服务进程..."
pkill -f "web_demo.py" 2>/dev/null && echo "  已停止旧 Web Demo"
pkill -f "semantic_chunk.py" 2>/dev/null && echo "  已停止旧语义分块"
pkill -f "vllm serve" 2>/dev/null && echo "  已停止旧 vLLM"
sleep 2

# ------ 3. 创建必要目录 ------
mkdir -p data/mongodb/data data/mongodb/log log

# ------ 4. 启动 MongoDB ------
echo "[1/5] 启动 MongoDB..."
$PROJECT_DIR/mongodb-7.0.20/bin/mongod --port=27017 \
    --dbpath=$PROJECT_DIR/data/mongodb/data \
    --logpath=$PROJECT_DIR/data/mongodb/log/mongodb.log \
    --bind_ip=0.0.0.0 --fork
echo "  MongoDB 已启动"

# ------ 5. 启动语义分块服务 ------
echo "[2/5] 启动语义分块服务..."
nohup $PYTHON $PROJECT_DIR/src/server/semantic_chunk.py > $PROJECT_DIR/log/semantic_chunk.log 2>&1 &
sleep 5
echo "  semantic_chunk 已启动 (PID: $!)"

# ------ 6. 启动 vLLM ------
echo "[3/5] 启动 vLLM (Qwen3-8B ORPO v4 BF16)..."
nohup $VLLM serve $PROJECT_DIR/LLaMA-Factory-main/output/qwen3_orpo_v4_bf16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    > $PROJECT_DIR/log/qwen3-7b.log 2>&1 &
VLLM_PID=$!
echo "  vLLM 已启动 (PID: $VLLM_PID)，等待加载..."

# 轮询 vLLM 健康检查，最多等待 5 分钟
MAX_WAIT=300
ELAPSED=0
POLL_INTERVAL=5
while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  vLLM 加载完成（耗时 ${ELAPSED}s）"
        break
    fi
    echo "  等待中...（${ELAPSED}s/${MAX_WAIT}s）"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "  警告: vLLM 未在 ${MAX_WAIT}s 内就绪，继续启动后续服务"
fi

# ------ 7. 启动 Web Demo（Gradio）------
echo "[4/5] 启动 Web Demo (http://localhost:7860)..."
nohup $PYTHON -u $PROJECT_DIR/web_demo.py > $PROJECT_DIR/log/web_demo.log 2>&1 &
echo "  Web Demo 已启动 (PID: $!)"

# ------ 8. 启动 FastAPI ------
echo "[5/5] 启动 FastAPI (http://localhost:9000)..."
nohup $PYTHON -m uvicorn src.api.main:app --host 0.0.0.0 --port 9000 > $PROJECT_DIR/log/fastapi.log 2>&1 &
echo "  FastAPI 已启动 (PID: $!)"

echo ""
echo "========================================"
echo "  全部启动完成！"
echo "  Web Demo: http://localhost:7860"
echo "  FastAPI:  http://localhost:9000"
echo "  API 文档: http://localhost:9000/docs"
echo "  日志目录: $PROJECT_DIR/log/"
echo "========================================"
