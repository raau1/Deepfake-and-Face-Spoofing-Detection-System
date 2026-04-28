# DFDC Out-of-Family Evaluation

- Partitions: `dfdc_train_part_1`, `dfdc_train_part_2`
- Clips sampled: 676 (balanced 338 REAL / 338 FAKE)
- Seed: 42
- Failed clips: 0

| Model | Clips | AUC-ROC | EER | Video Acc |
|---|---|---|---|---|
| mixed | 676 | 0.7161 | 0.3595 | 0.6701 |
| robust | 676 | 0.6462 | 0.4053 | 0.5769 |
| ensemble | 676 | 0.6946 | 0.3521 | 0.6169 |
| temporal | 676 | 0.6107 | 0.4009 | 0.5754 |
