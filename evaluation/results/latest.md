# Evaluation results (2026-11-01 11:46)

## Retrieval (top 5 passages)

| Strategy | Hit rate | MRR |
|---|---|---|
| vector | 68% | 0.62 |
| hybrid | 86% | 0.69 |

## Answers (strategy: hybrid, strict mode)

| Metric | Value |
|---|---|
| Answer accuracy (answerable questions) | 77% |
| Faithfulness (LLM judge) | 100% |
| Correctly refused (unanswerable) | 100% |
| Wrongly refused (answerable) | 5% |
| Latency average / p95 | 6057 ms / 8780 ms |
