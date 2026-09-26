# HDB Rental Data Analysis

## 1. Scope and safeguards

This report analyses only `data/train.csv` (150,000 rows and 11 columns).

- The analysis does not overwrite, clean, impute, or export a modified dataset.
- Derived variables and analytical subsets exist only in memory for descriptive analysis.
- `test.csv` is not used for distributions, feature selection, or analytical conclusions.
- Auxiliary datasets are not joined in this report.
- Repeated property-month records are treated as review candidates, not automatically as errors, because multiple genuine rental approvals can share the same property characteristics.

## 2. Dataset overview

The observed approval period is **2021-01 to 2025-03**, covering **51 months**.



|Column              |Semantic_type                            |R_type    | Missing| Missing_pct| Unique|
|:-------------------|:----------------------------------------|:---------|-------:|-----------:|------:|
|RENT_APPROVAL_DATE  |Temporal (monthly)                       |character |       0|           0|     51|
|TOWN                |Categorical location                     |character |       0|           0|     26|
|BLOCK               |High-cardinality identifier              |character |       0|           0|   2720|
|STREET              |High-cardinality location                |character |       0|           0|   1146|
|FLAT_TYPE           |Categorical property                     |character |       0|           0|     11|
|FLAT_MODEL          |Categorical property                     |character |       0|           0|     20|
|FLOOR_AREA_SQM      |Numerical property                       |numeric   |       0|           0|    156|
|FURNISHED           |Categorical property                     |character |       0|           0|      1|
|LEASE_COMMENCE_DATE |Year / property age                      |integer   |       0|           0|     57|
|FEE                 |Numerical; meaning requires confirmation |numeric   |       0|           0|      1|
|MONTHLY_RENT        |Numerical target                         |integer   |       0|           0|    608|

### Interpretation

`BLOCK` and `STREET` are high-cardinality location fields. Their raw group medians should not be interpreted without minimum sample-size rules. `FEE` requires confirmation from the competition documentation because its business meaning cannot be inferred safely from the column name alone.

## 3. Data quality audit



|Check                                              | Count| Percent|
|:--------------------------------------------------|-----:|-------:|
|Exact duplicate rows                               |   425|    0.28|
|Repeated property-month key after the first record | 15547|   10.37|
|MONTHLY_RENT <= 0                                  |     0|    0.00|
|FLOOR_AREA_SQM <= 0                                |     0|    0.00|
|Missing or invalid RENT_APPROVAL_DATE              |     0|    0.00|
|LEASE_COMMENCE_DATE later than approval year       |     1|    0.00|
|Derived flat age < 0                               |     1|    0.00|
|Derived flat age > 99                              |     0|    0.00|

### Rare-category exposure



|Feature    | Categories| Categories_under_20_records| Records_in_rare_categories|
|:----------|----------:|---------------------------:|--------------------------:|
|TOWN       |         26|                           0|                          0|
|FLAT_TYPE  |         11|                           1|                          8|
|FLAT_MODEL |         20|                           3|                         32|
|FURNISHED  |          1|                           0|                          0|
|STREET     |       1146|                         217|                       2231|
|BLOCK      |       2720|                        1097|                      11076|

### Formatting consistency



|Feature   | Raw_unique| Normalised_unique| Difference|
|:---------|----------:|-----------------:|----------:|
|FLAT_TYPE |         11|                 6|          5|
|STREET    |       1146|               576|        570|



|flat_type_std |Raw_forms      | Raw_form_count|
|:-------------|:--------------|--------------:|
|1 room        |1 room, 1-room |              2|
|2 room        |2 room, 2-room |              2|
|3 room        |3 room, 3-room |              2|
|4 room        |4 room, 4-room |              2|
|5 room        |5 room, 5-room |              2|

### Interpretation

There are **425 exact duplicate rows** and **15,547 repeated property-month keys after the first record**. The latter may represent parallel genuine transactions and should not be deleted without record-level evidence. `FLAT_TYPE` contains hyphen/space variants and `STREET` contains case variants. Subsequent grouped analysis uses disclosed in-memory standardised labels so the same business category is not split; the source values remain unchanged.

## 4. Target analysis



|Statistic          |    Value|
|:------------------|--------:|
|Count              | 150000.0|
|Mean               |   2719.0|
|Median             |   2700.0|
|Standard deviation |    749.7|
|0%                 |    300.0|
|1%                 |   1300.0|
|5%                 |   1650.0|
|25%                |   2100.0|
|50%                |   2700.0|
|75%                |   3200.0|
|95%                |   4000.0|
|99%                |   4600.0|
|100%               |   7600.0|

![Monthly rent distribution](figures/01_target_distribution.png)

### Tail comparison



|Tail      | Records| Median_rent| Median_area|Most_common_town |Most_common_flat_type |
|:---------|-------:|-----------:|-----------:|:----------------|:---------------------|
|Top 1%    |    1576|        4800|         110|bukit merah      |5 room                |
|Bottom 1% |    1694|        1200|          67|woodlands        |3 room                |

### Observation and modelling implication

The target median is **SGD 2,700**, while the mean is **SGD 2,719**. The difference and upper tail mean RMSE will be sensitive to high-rent records. These records should be retained unless a concrete data error is found, then examined separately during model error analysis.

## 5. Temporal analysis

![Monthly mean and median rent](figures/02_monthly_rent_trend.png)

![Monthly transaction counts](figures/03_monthly_record_count.png)

![Rent distribution by calendar month](figures/12_calendar_month_rent.png)

### Observation and modelling implication

The median rent in the last six observed months is **56.2%** different from the first six observed months. Monthly record counts range from **2,362** to **3,705**. The time trend is material enough that random train-validation splitting would mix earlier and later market regimes. Calendar-month differences remain descriptive because pooling years can confound seasonality with the long-term trend.

## 6. Property features and rent

### Flat type



|FLAT_TYPE | Records| Mean_rent| Median_rent|
|:---------|-------:|---------:|-----------:|
|5 room    |   35523|   2981.71|        3000|
|executive |    8234|   3069.33|        3000|
|4 room    |   54079|   2837.88|        2800|
|3 room    |   49024|   2386.55|        2350|
|2 room    |    3098|   1979.52|        2100|
|1 room    |      42|   1498.21|        1450|

![Rent by flat type](figures/04_rent_by_flat_type.png)

The highest median flat-type rent is **SGD 3,000** for `5 room`; the lowest is **SGD 1,450** for `1 room`. Flat type is therefore a strong candidate predictor, although part of its effect overlaps with floor area.

### Floor area

![Floor area and rent](figures/05_floor_area_vs_rent.png)

The Spearman correlation between floor area and rent is **0.318**. The binned median line should be used to judge non-linearity rather than relying only on a single correlation coefficient.

### Flat model



|FLAT_MODEL             | Records| Median_rent|
|:----------------------|-------:|-----------:|
|improved               |   44008|        2700|
|model a                |   42745|        2700|
|new generation         |   27380|        2450|
|premium apartment      |   12312|        2900|
|simplified             |    6794|        2500|
|standard               |    5903|        2500|
|apartment              |    4724|        3000|
|maisonette             |    2346|        3000|
|model a2               |    1553|        2550|
|dbss                   |    1090|        3400|
|type s1                |     299|        4600|
|2 room                 |     224|        2300|
|adjoined flat          |     218|        3325|
|type s2                |     160|        4950|
|model a maisonette     |     140|        3175|
|premium apartment loft |      51|        3800|
|premium maisonette     |      21|        3300|
|terrace                |      19|        4200|
|3gen                   |      12|        3500|
|improved maisonette    |       1|        3780|

![Median rent for major flat models](figures/07_flat_model_median_rent.png)

Model-level differences may reflect a combination of flat type, age and location. They motivate multivariate modelling but do not establish a causal model effect.

### Furnished status



|FURNISHED | Records| Missing_rent| Median_rent|
|:---------|-------:|------------:|-----------:|
|yes       |  150000|            0|        2700|

`FURNISHED` has one observed value in train and therefore cannot distinguish rental outcomes within this dataset.

### Fee



| FEE| Records| Median_rent|
|---:|-------:|-----------:|
|   0|  150000|        2700|

`FEE` is constant at zero in train and therefore cannot distinguish rental outcomes within this dataset. Its business meaning should still be confirmed before deciding how a later modelling pipeline handles the column.

## 7. Location features and rent



|TOWN            | Records| Mean_rent| Median_rent|
|:---------------|-------:|---------:|-----------:|
|bukit timah     |     438|   3030.74|        3000|
|central         |    2278|   3148.76|        3000|
|bishan          |    3306|   2980.24|        2900|
|bukit merah     |    8270|   3004.91|        2900|
|queenstown      |    6115|   2931.30|        2900|
|punggol         |    6668|   2777.37|        2850|
|pasir ris       |    4089|   2841.61|        2800|
|sengkang        |    9773|   2761.00|        2800|
|clementi        |    5429|   2797.21|        2700|
|jurong west     |   10482|   2768.43|        2700|
|kallang/whampoa |    5689|   2797.09|        2700|
|serangoon       |    3398|   2772.18|        2700|
|tampines        |    9942|   2764.71|        2700|
|bukit panjang   |    3717|   2603.98|        2600|
|choa chu kang   |    5300|   2646.13|        2600|
|jurong east     |    4024|   2722.30|        2600|
|marine parade   |    1637|   2690.83|        2600|
|sembawang       |    3553|   2665.40|        2600|
|toa payoh       |    5988|   2668.29|        2600|
|hougang         |    7115|   2621.42|        2550|
|ang mo kio      |    8283|   2528.25|        2500|
|bedok           |    8907|   2564.58|        2500|
|bukit batok     |    5605|   2579.20|        2500|
|geylang         |    4504|   2595.00|        2500|
|woodlands       |    7577|   2584.88|        2500|
|yishun          |    7913|   2515.43|        2500|

![Median rent by town](figures/06_town_median_rent.png)

Among towns with at least 100 records, `bukit timah` has the highest median rent at **SGD 3,000**, compared with **SGD 2,500** in `ang mo kio`. This supports retaining location information and later testing finer spatial features.

### Streets with the highest median rent (minimum 50 records)



|STREET                  | Records| Median_rent| IQR_rent|
|:-----------------------|-------:|-----------:|--------:|
|cantonment road         |     459|        4700|     1400|
|lim liak street         |      52|        4200|     1750|
|dawson road             |     258|        4000|     1200|
|lorong 1a toa payoh     |     135|        4000|     1700|
|telok blangah street 31 |     206|        4000|     1000|
|henderson road          |      65|        3900|      950|
|boon tiong road         |     319|        3800|     1500|
|ang mo kio street 51    |      55|        3600|      800|
|delta avenue            |      86|        3600|     1200|
|kim tian place          |     102|        3600|     1175|
|redhill road            |     288|        3600|     1200|
|simei lane              |      65|        3600|     1050|
|boon keng road          |     235|        3550|     1250|
|clarence lane           |      75|        3500|     1100|
|ghim moh link           |     273|        3500|     1600|

### Blocks with the highest median rent (minimum 30 records)



|TOWN        |STREET                  |BLOCK | Records| Median_rent| IQR_rent|
|:-----------|:-----------------------|:-----|-------:|-----------:|--------:|
|central     |cantonment road         |1a    |      50|        4900|   1450.0|
|central     |cantonment road         |1f    |      70|        4900|   1290.0|
|central     |cantonment road         |1d    |      71|        4800|   1131.0|
|central     |cantonment road         |1b    |      66|        4700|    962.5|
|central     |cantonment road         |1g    |      53|        4600|   1400.0|
|central     |cantonment road         |1e    |      72|        4550|   1325.0|
|queenstown  |ghim moh link           |22    |      31|        4500|   2050.0|
|central     |cantonment road         |1c    |      77|        4400|   1600.0|
|bukit merah |telok blangah street 31 |93b   |      33|        4300|    750.0|
|queenstown  |dawson road             |86    |      51|        4300|   1375.0|
|queenstown  |dawson road             |88    |      58|        4300|   1175.0|
|bukit merah |telok blangah street 31 |90a   |      38|        4200|    775.0|
|queenstown  |dover crescent          |28d   |      30|        4150|   1270.0|
|bukit merah |redhill road            |77a   |      36|        4075|   1025.0|
|bukit merah |kim tian road           |127d  |      30|        4050|   1200.0|

Street and block summaries are descriptive historical aggregates. Any target encoding based on them must be fitted only on the training portion of each validation fold to prevent leakage.

## 8. Property age

![Flat age and rent](figures/08_flat_age_vs_rent.png)

The Spearman correlation between derived flat age and rent is **-0.166**. The relationship should be treated as potentially nonlinear and confounded by town and flat type.

## 9. Interaction analysis

### Town and flat type

![Town and flat type interaction](figures/09_town_flat_type_heatmap.png)

The heatmap compares like-for-like flat types across towns and is more informative than a town-only median. Remaining differences may still reflect floor area, model, age and time composition.

### Time and flat type

![Time and flat type interaction](figures/10_time_by_flat_type.png)

This view tests whether flat types move together over time. Diverging trajectories would justify time-by-property interactions or flexible tree-based models.

### Flat type and floor area

![Floor area by flat type](figures/11_area_by_flat_type.png)

Flat type and floor area overlap strongly, but variation within flat types shows that they should not be treated as identical information at the EDA stage.

## 10. Validation recommendation

Use a primary temporal holdout of the final six months: **2024-10 to 2025-03**. Train only on earlier records.

Add two earlier six-month rolling backtests if computation permits. Compare models primarily on the competition metric and report MAE as a secondary diagnostic. Also break errors down by month, town, flat type and rent band. Do not calculate location target aggregates using validation rows.

## 11. Prioritised findings and next experiments

1. Time must be modelled explicitly because the dataset spans multiple market regimes.
2. Town, flat type and floor area are the first core predictors to test.
3. Flat model and property age may add nonlinear signal after controlling for the core predictors.
4. Street and block can be useful but require leakage-safe encoding and sample-size regularisation.
5. Establish baselines before adding auxiliary data: mean predictor, linear model, then a tree-based model.
6. Add auxiliary sources through separate ablation experiments: HDB block attributes, MRT, malls, schools, and only then macro variables.
7. Preserve high-rent observations unless a documented data error is identified; analyse their errors separately because RMSE weights them heavily.

## 12. Analysis limitations

- All relationships are descriptive and do not establish causality.
- Repeated records cannot be classified as erroneous without a transaction identifier or additional documentation.
- Apparent town, model and age effects may partly reflect differences in time, size and flat-type composition.
- `FEE` remains semantically unresolved.
- External geographic and macro datasets were intentionally excluded from this first analysis.

## 13. Reproducibility and source integrity

Generated by `scripts/run_eda.R` from `data/train.csv`. Source-file hashes were checked before and after execution. Report generated at 2026-09-25 17:27:11 +08.
