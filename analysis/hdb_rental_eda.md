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

### Task framing

This is a supervised tabular regression problem with property, location and temporal predictors. Individual flats cannot be tracked reliably across periods because the data lack a unique unit identifier. Time should therefore be modelled as a predictor, but the task should not be framed as pure time-series forecasting.

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

The median rent in the last six observed months is **56.2%** different from the first six observed months. Monthly record counts range from **2,362** to **3,705**. The strong temporal variation means approval time should be modelled explicitly. Standard random splitting or K-fold cross-validation remains appropriate for general model comparison, while a temporal holdout provides an additional robustness check for sensitivity to time-related distribution shift. Calendar-month differences remain descriptive because pooling years can confound seasonality with the long-term trend.

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

## 10. Temporal-location analysis

This section treats approval time as one dimension of a cross-sectional rental dataset, not as a standalone time series. `approval_quarter` is derived in memory from `RENT_APPROVAL_DATE`. Town labels use the same lower-case, whitespace-normalised convention as the existing EDA.

![Median monthly rent by town and approval quarter](figures/13_town_quarter_median_rent_heatmap.png)

The comparison windows are the first six observed months (**2021-01 to 2021-06**) and the last six observed months (**2024-10 to 2025-03**). The heatmap orders towns by their median rent in the latest observed quarter (**2025 Q1**).



|TOWN            | Records| Early median rent| Late median rent| Absolute change| Percentage change (%)|
|:---------------|-------:|-----------------:|----------------:|---------------:|---------------------:|
|central         |    2278|              2300|             3500|            1200|                  52.2|
|bishan          |    3306|              2350|             3500|            1150|                  48.9|
|bukit merah     |    8270|              2400|             3500|            1100|                  45.8|
|pasir ris       |    4089|              2100|             3400|            1300|                  61.9|
|jurong west     |   10482|              2100|             3350|            1250|                  59.5|
|bukit timah     |     438|              2350|             3350|            1000|                  42.6|
|tampines        |    9942|              2100|             3300|            1200|                  57.1|
|serangoon       |    3398|              2155|             3300|            1145|                  53.1|
|punggol         |    6668|              2000|             3200|            1200|                  60.0|
|sengkang        |    9773|              2000|             3200|            1200|                  60.0|
|queenstown      |    6115|              2150|             3200|            1050|                  48.8|
|kallang/whampoa |    5689|              2200|             3200|            1000|                  45.5|
|jurong east     |    4024|              2100|             3150|            1050|                  50.0|
|choa chu kang   |    5300|              1900|             3100|            1200|                  63.2|
|sembawang       |    3553|              2000|             3100|            1100|                  55.0|
|clementi        |    5429|              2160|             3100|             940|                  43.5|
|bukit panjang   |    3717|              1900|             3000|            1100|                  57.9|
|woodlands       |    7577|              1900|             3000|            1100|                  57.9|
|yishun          |    7913|              1900|             3000|            1100|                  57.9|
|bukit batok     |    5605|              1950|             3000|            1050|                  53.8|
|bedok           |    8907|              2000|             3000|            1000|                  50.0|
|geylang         |    4504|              2000|             3000|            1000|                  50.0|
|hougang         |    7115|              2000|             3000|            1000|                  50.0|
|marine parade   |    1637|              2000|             3000|            1000|                  50.0|
|toa payoh       |    5988|              2000|             3000|            1000|                  50.0|
|ang mo kio      |    8283|              1900|             2900|            1000|                  52.6|

### Interpretation

Most towns show similar temporal rent increases. All **26 towns** have a higher median in the late six-month window. The median town-level increase is **52.4%**, the middle 50% of towns lie between **50% and 57.9%**, and **25 of 26 towns** are within 10 percentage points of the median increase.

There are still noticeably different trajectories. `choa chu kang` (**63.2%**) and `pasir ris` (**61.9%**) rose fastest, while `bukit timah` (**42.6%**) and `clementi` (**43.5%**) rose slowest.

The common upward movement suggests a strong overall time effect, while the spread and quarter-to-quarter differences across towns suggest a possible time-by-location interaction. This is descriptive evidence rather than a causal or pure time-series conclusion: changing mixes of flat type, floor area, model and other property attributes within each town and period may explain part of the divergence. A multivariate model should therefore test a time-by-town interaction and compare it with a model containing only additive time and town effects.

## 11. Temporal composition analysis

This section compares the composition of rental records across approval quarters and between the same six-month windows used above: **2021-01 to 2021-06** and **2024-10 to 2025-03**. The pooled median monthly rent rose from **SGD 2,000** to **SGD 3,150** (**57.5%**). The diagnostics below assess whether changes in observed property mix are large enough to plausibly account for most of that difference.

For categorical variables, total-variation distance is half the sum of the absolute category-share changes. It can be read as the percentage of records that would need to move between categories to make the two distributions match.

### Flat-type composition

![Quarterly record share by major flat type](figures/14_flat_type_share_by_quarter.png)



|FLAT_TYPE | Early share (%)| Late share (%)| Change (percentage points)|
|:---------|---------------:|--------------:|--------------------------:|
|5 room    |           25.22|          22.72|                      -2.50|
|2 room    |            1.32|           2.77|                       1.44|
|3 room    |           31.88|          33.07|                       1.19|
|executive |            6.10|           4.95|                      -1.14|
|4 room    |           35.45|          36.45|                       0.99|
|1 room    |            0.02|           0.04|                       0.02|

The flat-type composition total-variation distance is **3.6 percentage points**. The largest individual shift is for `5 room`, changing by **-2.5 percentage points**. This is a modest shift relative to the rent increase.

### Town composition

![Quarterly representation of the ten largest towns](figures/15_largest_town_share_heatmap.png)

The heatmap is limited to the ten towns with the most records in the full training dataset. The table lists the ten largest early-to-late share changes across all 26 towns.



|TOWN        | Early share (%)| Late share (%)| Change (percentage points)|
|:-----------|---------------:|--------------:|--------------------------:|
|punggol     |            4.24|           4.79|                       0.55|
|sengkang    |            6.45|           6.91|                       0.46|
|bukit merah |            5.35|           5.73|                       0.39|
|tampines    |            6.76|           6.38|                      -0.38|
|clementi    |            3.62|           3.28|                      -0.34|
|bedok       |            6.01|           5.70|                      -0.31|
|queenstown  |            4.10|           3.82|                      -0.28|
|central     |            1.32|           1.58|                       0.26|
|yishun      |            5.42|           5.16|                      -0.26|
|sembawang   |            2.32|           2.58|                       0.26|

The town composition total-variation distance is **2.6 percentage points**. The largest individual town shift is `punggol` at **0.55 percentage points**, indicating relatively stable geographic representation.

### Floor-area composition

![Floor-area distribution by approval quarter](figures/16_floor_area_composition_by_quarter.png)



|Window | Records| Q1 (sqm)| Median (sqm)| Q3 (sqm)| Mean (sqm)|
|:------|-------:|--------:|------------:|--------:|----------:|
|Early  |   19250|       72|           93|      111|       94.7|
|Late   |   17198|       68|           92|      110|       92.3|

Median floor area changed from **93 sqm** to **92 sqm**. The interquartile range also moved slightly downward rather than toward systematically larger flats.

### Progress-report interpretation

The observed **57.5%** increase in pooled median rent is unlikely to be explained mainly by changes in the mix of records. Town representation is highly stable, median floor area falls by **1 sqm**, and the clearest categorical shift is only a modest change in flat-type shares. Flat type shows the largest composition movement: five-room flats lose share while two-room and three-room flats gain share. That movement is toward smaller flat types, so it does not provide an obvious compositional explanation for higher rents. These are descriptive diagnostics, not causal estimates; unmeasured or finer-grained changes in property composition may still contribute to the observed trend.

## 12. Validation recommendation

Treat the task primarily as supervised tabular regression with both cross-sectional and temporal dimensions. Use a standard random train-validation split or K-fold cross-validation as the main framework for general model comparison; the chronological ordering of the supplied split does not by itself make this a pure time-series forecasting task.

Add a holdout of the final six months (**2024-10 to 2025-03**) and, if computation permits, earlier rolling temporal checks as robustness analyses. These checks measure sensitivity to time-related distribution shift, not an assumption that the observations form a pure time series. Compare models primarily on the competition metric and report MAE as a secondary diagnostic, with errors broken down by month, town, flat type and rent band. Fit target encoding and every other target-derived aggregate using only the training portion of each split or fold, then apply the fitted mapping to validation records.

## 13. Prioritised findings and next experiments

1. Time must be modelled explicitly because the dataset spans multiple market regimes.
2. Town, flat type and floor area are the first core predictors to test.
3. Flat model and property age may add nonlinear signal after controlling for the core predictors.
4. Street and block can be useful but require leakage-safe encoding and sample-size regularisation.
5. Establish baselines before adding auxiliary data: mean predictor, linear model, then a tree-based model.
6. Add auxiliary sources through separate ablation experiments: HDB block attributes, MRT, malls, schools, and only then macro variables.
7. Preserve high-rent observations unless a documented data error is identified; analyse their errors separately because RMSE weights them heavily.

## 14. Analysis limitations

- All relationships are descriptive and do not establish causality.
- Repeated records cannot be classified as erroneous without a transaction identifier or additional documentation.
- Apparent town, model and age effects may partly reflect differences in time, size and flat-type composition.
- `FEE` remains semantically unresolved.
- External geographic and macro datasets were intentionally excluded from this first analysis.

## 15. Reproducibility and source integrity

Generated by `scripts/run_eda.R` from `data/train.csv`. The source dataset hash was checked before and after execution. Report generated at 2026-09-29 18:58:26 +08.
