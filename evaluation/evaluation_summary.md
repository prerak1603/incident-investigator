# Evaluation Results

## CV evaluation (Stage 3 segments vs ground truth)

- mAP@0.3: 1.0
- mAP@0.5: 1.0
- mAP@0.7: 1.0

## Memory ablation

| memory config | avg confidence shift | % verdict changed | LLM calls |
|---|---|---|---|
| none | -0.356 | 100% | 6 |
| short_term | -0.290 | 100% | 6 |
| short_and_long_term | -0.580 | 100% | 6 |
