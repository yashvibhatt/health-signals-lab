# Labeling guide for data/gold_labels.csv

Fill `study_type` for all 50 rows with exactly one of: `forecasting`, `surveillance`, `clinical`, `review`, `other`.
Label from the title and abstract only, choosing the study's PRIMARY aim. Use `labeler_notes` for borderline cases. Do not look at model outputs while labeling.

- **forecasting**: Primary aim is predicting future (or nowcasting current, not-yet-reported) disease activity, e.g. building or evaluating a forecasting/nowcasting model.
- **surveillance**: Primary aim is describing, monitoring or detecting disease activity from surveillance data (trends, burden, seasonality, early warning, system evaluation) without forecasting as the main contribution.
- **clinical**: Patient-level clinical research: treatment, vaccine efficacy/effectiveness trials, diagnosis, outcomes or risk factors in patients.
- **review**: Narrative, systematic or scoping review, or meta-analysis of existing literature.
- **other**: Anything else (e.g. lab/virology, genomics, transmission modelling without forecasting, policy, methods unrelated to the categories above).
