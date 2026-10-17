# Qwen3-8B DPO-first AnswerModel Adapter


## Why DPO is the target

`SFT -> DPO` is the intended training lineage. Production should call `Qwen3DPOAdapter`; `Qwen3SFTAdapter` exists only for baseline comparisons or an **explicit** manual selection. There is **no silent SFT fallback** when DPO is absent.

```
Evidence Gate = Sufficient
    -> sealed final_evidence + atomic requirements + original question
    -> Qwen3-8B DPO (non-streaming StructuredAnswer JSON)
    -> Pydantic contract check + allowed evidence ID check (adapter)
    -> Citation Guard (document/index/request/permission checks)
    -> Renderer (source metadata comes from backend, never the LLM)
```

## Configuration

Set the following in `.env` when you deploy:

```dotenv
ANSWER_MODEL_VARIANT=dpo
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
QWEN3_DPO_MODEL_NAME=qwen3-8b-dpo
QWEN3_SFT_MODEL_NAME=qwen3-8b-sft
QWEN3_OUTPUT_MODE=structured_outputs
```

The `*_MODEL_NAME` must match **vLLM's API served name**, not necessarily the filesystem path. Optional `QWEN3_DPO_BASE_URL` and `QWEN3_SFT_BASE_URL` allow separate services. For legacy vLLM 0.9, set `QWEN3_OUTPUT_MODE=guided_json`; newer versions use `structured_outputs={"json": schema}`. Both still require strict Pydantic validation. See [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) and [vLLM LoRA adapters](https://docs.vllm.ai/en/latest/features/lora/).

Example after *retraining and exporting weights* (replace placeholders):

```bash
# Merged model weights:
vllm serve /your/private/models/qwen3-8b-dpo-merged --served-model-name qwen3-8b-dpo --port 8000

# OR a LoRA adapter (if compatible with your vLLM version and base model):
vllm serve /your/private/models/qwen3-8b-base --enable-lora --lora-modules qwen3-8b-dpo=/your/private/adapters/qwen3-8b-dpo --port 8000
```

No weights or local datasets should be committed to GitHub.

## Adapter usage (already bridged into the new Controlled RAG orchestrator)

```python
from controlled_answer import (
    EvidenceItem, GenerationRequest, Qwen3DPOAdapter,
)

# IMPORTANT: Caller must first pass the Evidence Gate and seal the final snapshot.
request = GenerationRequest(
    original_query="如何启用这个功能？",
    atomic_requirements=("解释开启步骤",),
    final_evidence=(EvidenceItem("E1", "在控制菜单中开启此功能。"),),
)
result = Qwen3DPOAdapter().generate(request)
# -> StructuredAnswer; NOT user-facing text until Citation Guard and Renderer pass.
```


Only `answer_blocks[].text` is the model-authored factual answer. The model is forbidden to invent filenames, page numbers, provenance, permissions, or index/document versions. The adapter rejects invalid JSON, contradictory abstention states, duplicate/fabricated IDs, interrupted output and serving errors; it never retries or changes models automatically. The production Citation Guard **still** needs to validate request-scoped provenance, allowed KB/index/document versions, permissions and metadata, which this model adapter does **not** implement.

## Development

```bash
python -m pytest -q tests/test_qwen3_dpo_adapter.py
```

