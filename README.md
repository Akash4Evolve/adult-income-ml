

## Error analysis (validation split)
Miss rate is the share of true >50K people predicted <=50K; false-alarm rate is the share of true <=50K people predicted >50K. Groups under 100 rows are omitted.

**By sex**
| Group | n | Share >50K | Miss rate | False-alarm rate |
|---|---|---|---|---|
| Female | 3195 | 0.1 | 0.382 | 0.021 |
| Male | 6563 | 0.307 | 0.328 | 0.084 |

**Occupations with the highest miss rate**
| Group | n | Share >50K | Miss rate | False-alarm rate |
|---|---|---|---|---|
| Handlers-cleaners | 422 | 0.088 | 0.784 | 0.003 |
| Machine-op-inspct | 585 | 0.109 | 0.766 | 0.008 |
| Other-service | 991 | 0.035 | 0.743 | 0.001 |
