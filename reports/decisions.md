# Data decisions

| Decision | Reason |
|---|---|
| Drop `fnlwgt` | Survey sampling weight, not an attribute of the person |
| Drop `education` | Redundant with `education-num` (same information, numeric) |
| Impute missing values in the pipeline, do not drop rows | workclass (2,799), occupation (2,809) and native-country (857) are missing; dropping about 3,000 rows loses data and may bias the sample |
| Drop 52 exact duplicate rows before splitting | Prevents the same row appearing in both train and test |
| Evaluate with F1 and ROC-AUC, not accuracy alone | 76.1% / 23.9% class imbalance makes accuracy misleading |
| Sensitive attributes (`race`, `sex`) | TODO: decide whether to include them and write your reasoning here |
