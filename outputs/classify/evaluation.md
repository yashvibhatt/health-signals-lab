# study_type evaluation vs human gold labels

Gold rows: 50 · scored: 50 · missing prediction: 0

- **Accuracy:** 0.740
- **Macro F1** (classes present in gold): 0.705
- **Cohen's kappa:** 0.648

## Per-class precision / recall

|  | precision | recall | f1 | support |
|---|---|---|---|---|
| forecasting | 0.929 | 0.929 | 0.929 | 14 |
| surveillance | 0.577 | 1.0 | 0.732 | 15 |
| clinical | 1.0 | 0.333 | 0.5 | 9 |
| review | 1.0 | 1.0 | 1.0 | 4 |
| other | 0.667 | 0.25 | 0.364 | 8 |

## Confusion matrix (rows = gold, columns = predicted)

| gold | forecasting | surveillance | clinical | review | other |
|---|---|---|---|---|---|
| forecasting | 13 | 0 | 0 | 0 | 1 |
| surveillance | 0 | 15 | 0 | 0 | 0 |
| clinical | 0 | 6 | 3 | 0 | 0 |
| review | 0 | 0 | 0 | 4 | 0 |
| other | 1 | 5 | 0 | 0 | 2 |

Disagreements: 13 (see `evaluation_errors.csv`).
