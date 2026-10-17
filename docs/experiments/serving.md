# Safety, serving and capacity


服务测试应固定并发集、流式输出、相同 prompt/output token 分布、warmup 与观测窗口、重试策略、成功/失败统计和预先声明的 SLA。在每个并发条件下保存客户端单调时间戳、真实 TTFT/TPOT、GPU/DCGM 采样、峰值显存和 request-level traces。错误、超时和拒答应分类计入分母，而不是只在成功请求上计算吞吐。

实际 Prompt Injection/KB 越权测试必须独立判断“攻击目标达成”与“边界保护是否生效”，将来源文档指令、合法性和是否输出未授权内容分别保存。确定性 Guard 测试不能替代模型级红队评审。所需脚本：`scripts/http_bench.py`、`eval/benchmark/locustfile.py`、`eval/safety/run_control_probes.py`。

