# Evidence Router


| 指标 | Threshold rule | LR-like scenario |
|---|---:|---:|
|  Macro-F1 | 0.745 | 0.901 |
|  False-Sufficient | 0.113 | 0.087 |
|  Top-label 10-bin ECE | 0.115 | 0.039 |
|  Brier (unscaled) | 0.417 | 0.174 |

三态定义：`sufficient / partial / failed`。False-Sufficient 分母**仅为 Gold 不充分**的 Query，而不是所有 Query；校准阈值只能从 Dev 拟合，Test 必须冻结。
