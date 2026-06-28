# Composed comparison — runs: colab, local

Public baselines are FIXED across runs; **ours** changes with the trained checkpoint, so its row is
repeated per run to show training's effect against the same baselines.

| method | run | F closed | F open | SDF-err | normal-cons | parts (open) | time (s) |
|---|---|---|---|---|---|---|---|
| SPSR | all | 98.8 | 78.8 | -- | 0.738 | 33 | 4.20 |
| BPA | all | 99.9 | 94.2 | -- | 0.858 | 1498 | 0.14 |
| APSS | all | 96.9 | 77.4 | -- | 0.739 | 1 | 0.57 |
| RIMLS | all | 96.4 | 65.2 | -- | 0.805 | 1 | 0.85 |
| tori | all | 77.7 | 81.3 | -- | 0.619 | 301 | 8.47 |
| ours | colab | 71.0 | 77.6 | -- | 0.546 | 1009 | 1.85 |
| ours | local | 82.0 | 85.6 | -- | 0.648 | 540 | 1.84 |
