# Data decisions

| Decision | Reason |
|---|---|
| Drop `fnlwgt` | Survey sampling weight, not an attribute of the person |
| Drop `education` | Redundant with `education-num` (same information, numeric) |
| Impute missing values in the pipeline, do not drop rows | workclass (2,799), occupation (2,809) and native-country (857) are missing; dropping about 3,000 rows loses data and may bias the sample |
| Drop 52 exact duplicate rows before splitting | Prevents the same row appearing in both train and test |
| Evaluate with F1 and ROC-AUC, not accuracy alone | 76.1% / 23.9% class imbalance makes accuracy misleading |
| Sensitive attributes (`race`, `sex`) | Excluded from the model. Adding them changed validation AUC by about 0.0001 (0.9336 vs 0.9337), so they add no measurable accuracy. Proxies such as relationship, marital-status and occupation remain, so excluding them does not remove all bias; per-sex positive prediction rates (8.0% female vs 26.5% male) and recall (0.62 vs 0.67) are reported in the README. |

