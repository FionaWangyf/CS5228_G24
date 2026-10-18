## 1. Pipeline Overview

```text
load_data -> make_listing -> [load_auxiliary -> make_auxiliary -> attach_auxiliary]
          -> [scale_numerical] -> [onehot_encoding] -> encode_high_card(TODO)
```

| 函数 | 位置 | 作用 |
| --- | --- | --- |
| `load_data` | `src/validation.py` | 读取 train.csv、清洗、按固定月份切分 train / valid |
| `make_listing` | `src/features.py` | 主表 feature engineering：`RENT_APPROVAL_DATE` → `month_index`，`LEASE_COMMENCE_DATE` → `flat_age` |
| `load_auxiliary` | `src/auxiliary.py` | 读取并清洗 6 张副表 |
| `make_auxiliary` | `src/features.py` | 副表 feature engineering：聚合、去重、选列，输出可直接 join 的表 |
| `attach_auxiliary` | `src/auxiliary.py` | 按传入的 dict 把副表 left join 到主表；行数、index 不变 |
| `scale_numerical` | `src/features.py` | 中位数填补 + 标准化，只有线性模型需要 |
| `onehot_encoding` | `src/features.py` | 低基数类别 one-hot |

## 2. Feature Engineering

### 2.1 Listing（`make_listing`）

两个日期列经过 transformation：

- `RENT_APPROVAL_DATE` → `month_index`（2021-01 为 0）
- `LEASE_COMMENCE_DATE` → `flat_age` = 批准年份 − 租约起始年（负值记为缺失）

其余保留 `FLOOR_AREA_SQM`, `TOWN`, `FLAT_TYPE`, `FLAT_MODEL`, `BLOCK`, `STREET`。

### 2.2 Auxiliary（`make_auxiliary`）

`make_auxiliary` 只做副表内部的聚合、去重和选列，输出可以直接 join 的表；只返回传入的表。

| **Aux Table** | **处理** | **输出的列** |
| --- | --- | --- |
| HDB | 按 `TOWN` + `BLOCK` + `STREET` 去重，保留第一行 | 三个键 + `LATITUDE`, `LONGITUDE`, `MAX_FLOOR`, `SUBZONE`, `PLANNING_AREA` |
| COE | 每月两轮 `QUOTA_PREMIUM` 取平均，再把 a/b/c/e 四类相加 | `month_index`, `coe_sum` |
| Stock | 只用 2021-01 就有价格的 42 只 SGX（`.SI`）股票；月均 `ADJUSTED_CLOSE` 各自除以 2021-01 的值，再等权平均 | `month_index`, `stock_index` |
| MRT | 只保留 `open` 站点，按站名合并为 192 个站；`LINE_COUNT` = 不同线路代码前缀个数 | `NAME`, `LATITUDE`, `LONGITUDE`, `LINE_COUNT` |
| Schools | 按 `MAINLEVEL_CODE` 标记小学（含 `primary` / `p1`）和中学（含 `secondary` / `s1`），混合学段两者都算 | `LATITUDE`, `LONGITUDE`, `IS_PRIMARY`, `IS_SECONDARY` |
| Malls | 只选列 | `NAME`, `LATITUDE`, `LONGITUDE` |

几个处理的原因：

- **MRT**：231 条 open 记录只有 192 个唯一站点，换乘站每条线路占一行。原来最近站点的 `mrt_code` 由原表顺序决定（同一站可能是 CC13 或 NE12），且与 `mrt_name` 一一对应，删除后 RMSE 不变。因此先按站名聚合，用 `LINE_COUNT` 表达换乘站。
- **Schools**：有 4 组重复坐标但属性不同，“最近学校的类型”是任意挑中的。因此不再使用学校的类别属性，只区分小学 / 中学。
- **相关性**：

| 组合 | 事实 | 处理 |
| :-- | :-- | :-- |
| `YEAR_COMPLETED` 与 `flat_age` | Spearman −0.987；和 `LEASE_COMMENCE_DATE` 相关 0.991 | 删除 `YEAR_COMPLETED` |
| `REGION` 与 `TOWN` | 每个 `TOWN` 只有 1 个 `REGION` | 删除 `REGION` |
| `coe_a/b/c/e` 与 `month_index` | 四列两两 Spearman 0.89–0.99（`coe_b` 与 `coe_e` 0.993），与 `month_index` 0.82–0.87 | 合并为一个 `coe_sum` |

### 2.3 删除的列

除上面提到的列以外，主表和副表中还有很多列没有进入特征：连接键、只用于计算距离的坐标、常数列、自由文本、POI 自身的地址和行政区划等。每一列的原因见 [A.1](#a1-被-discarded-的列)；其中看起来可能有用的几列，我们加回后做了验证，RMSE 变化都在噪声范围内，见 [A.2](#a2-加回被-drop-的列的检查)。

## 3. Join auxiliary

`attach_auxiliary` 把 `make_auxiliary` 的输出 left join 到 `make_listing` 的结果上，行数和 index 不变。它只连接 dict 里出现的表，所以表级别的消融只需要少传一个 key。MRT / Schools / Malls 的距离依赖 HDB 的经纬度，必须和 `hdb` 一起传入。

| **Aux Table** | **连接方法** | **生成的列** |
| --- | --- | --- |
| HDB | 按 `TOWN` + `BLOCK` + `STREET` | `LATITUDE`, `LONGITUDE`, `MAX_FLOOR`, `SUBZONE`, `PLANNING_AREA` |
| COE | 按 `month_index` | `coe_sum` |
| Stock | 按 `month_index` | `stock_index` |
| MRT | 最近站点（haversine） | `mrt_distance_km`, `mrt_name`, `mrt_line_count` |
| Schools | 最近小学、最近中学；1 km / 2 km 内全部学校数 | `primary_school_distance_km`, `secondary_school_distance_km`, `schools_within_1km`, `schools_within_2km` |
| Malls | 最近商场 | `mall_distance_km`, `mall_name` |

all joined 后特征分成三类：

- **数值特征**（`NUMERIC_FEATURES`）：15 个
- **低基数类型特征**（`LOW_CARD_FEATURES`）：`TOWN`, `FLAT_TYPE`, `FLAT_MODEL`, `PLANNING_AREA`
- **高基数类型特征**（`HIGH_CARD_FEATURES`）：`BLOCK`, `STREET`, `SUBZONE`, `mrt_name`, `mall_name`

## 4. Compatibility

| Model | Scale | 低基数 |
| --- | --- | --- |
| Linear / Ridge | `scale_numerical` | `onehot_encoding` |
| RF / Extra Trees / HGB / LightGBM / CatBoost | 不需要 | `onehot_encoding` |

- Ridge 含正则，必须 scale；`scale_numerical` 同时做中位数填补（`flat_age`、`MAX_FLOOR` 有少量缺失）。
- 高基数特征对不同模型可能需要不同的编码（target encoding、frequency encoding、native category 等），留给后续实验在 `encode_high_card` 中扩展。

## 5. 表级别消融

`experiments/example_attach.ipynb` 固定编码方式，只改变 attach 的副表：

- listing only
- listing + all joined
- listing + all joined − 一个副表（HDB、MRT、Schools、Malls、COE、Stock）。由于 POI 依赖 HDB 坐标，**− HDB** 等价于 listing + COE + Stock。

### 5.1 Summary

验证集 RMSE（所有模型低基数 one-hot；线性模型额外 scale）：

| Model | listing only | all joined | − HDB | − MRT | − Schools | − Malls | − COE | − Stock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear | 615.65 | 518.78 | 524.07 | 520.60 | 518.80 | 519.53 | 608.79 | 530.53 |
| Ridge | 615.73 | 518.81 | 524.12 | 520.64 | 518.83 | 519.56 | 608.85 | 530.56 |
| Random Forest | 490.11 | 480.23 | 488.80 | 480.78 | 479.65 | 480.87 | 481.12 | 481.99 |
| Extra Trees | 493.76 | 480.04 | 492.59 | 480.95 | 480.49 | 479.81 | 480.80 | 482.10 |
| HGB | 481.90 | 476.11 | 481.78 | 476.72 | 476.73 | 475.90 | 476.78 | 478.24 |
| LightGBM | 481.17 | 476.15 | 481.45 | 476.79 | 476.46 | 475.93 | 475.23 | 476.91 |
| CatBoost | 483.03 | 476.72 | 482.75 | 478.21 | 476.34 | 477.16 | 475.99 | 477.94 |

- Join 对所有模型都有帮助：树模型下降约 5–14，线性模型下降约 97。
- **HDB** 是树模型最重要的副表（去掉后 +5.3 ~ +12.6）。注意 − HDB 同时去掉了依赖 HDB 坐标的 MRT / Schools / Malls 距离。
- **Stock** 是第二重要的副表：树模型 +0.8 ~ +2.1，线性模型 +11.8。
- **COE** 对线性模型极其重要（+90），但对树模型基本无用（−0.9 ~ +0.9）。listing only 的线性模型 Mean Error 约 +352，即 `month_index` 的线性外推严重高估；`coe_sum` 起到了修正时间趋势的作用。
- **MRT / Schools / Malls** 的单表贡献都在 ±2 以内（树模型 ±1.5 以内），可能与经纬度及彼此之间的信息重叠有关。

## Appendix

### A.1 被 discarded 的列

| Table | Column | Reason |
|---|---|---|
| listings | `RENT_APPROVAL_DATE` | 时间信息已经变成 `month_index`；COE / Stock 也按 `month_index` 连接。 |
| listings | `LEASE_COMMENCE_DATE` | 楼龄已经写成 `flat_age`。 |
| listings | `MONTHLY_RENT` | 预测目标 y。 |
| listings | `FURNISHED`, `FEE` | 清洗时发现是常数列。 |
| HDB | `YEAR_COMPLETED` | 与 `flat_age` Spearman −0.987。 |
| HDB | `REGION` | `TOWN` 的确定函数。 |
| HDB | `POSTAL_CODE` | 3% 是 `NIL`；邮编前两位（postal sector）加回后 RMSE 变化 < 0.5（见 A.2），经纬度已覆盖这部分位置信息。 |
| HDB | `TOWN`, `BLOCK`, `STREET` | 连接键，listing 里本来就有。 |
| MRT | `CODE` | 换乘站的代码是任意挑中的；只用代码前缀统计 `LINE_COUNT`。 |
| MRT | `STATUS`, `IS_OPEN` | 只用 open 站点后是常数。12 条 planned 记录全部落在已有的 open 站点上（未来换乘），到 planned 站点的距离加回后 RMSE 变化 < 1（见 A.2）。 |
| MRT | `LATITUDE`, `LONGITUDE` | 只用来算距离。 |
| MRT | `SUBZONE`, `PLANNING_AREA`, `REGION` | 车站的行政区划，不是房子的属性。 |
| Schools | `TYPE_CODE`, `MAINLEVEL_CODE`, `NATURE_CODE` | 不再取“最近学校”的属性（重复坐标导致任意挑选）；`MAINLEVEL_CODE` 只用于区分小学 / 中学。 |
| Schools | `NAME`, `URL`, `STREET`, `POSTAL_CODE` | 学校 ID / 地址，不是房子的属性。 |
| Schools | `MRT_STATIONS`, `BUS_LINES` | 自由文本。 |
| Schools | `SESSION_CODE` | 337 所里 319 所是 single session，几乎无区分度。 |
| Schools | `PLANNING_AREA`, `REGION` | 学校所在的行政区划，不是房子的属性；房子的区位已由 HDB 提供。学校的 `REGION` 只有 north/south/east/west 四类，与 HDB 的五个 region 也不是同一套划分。 |
| Schools | `SUSPECT_DUPLICATE_COORDS` | 清洗时的重复坐标标记（8 所为 True），不是特征。 |
| Malls | `LATITUDE`, `LONGITUDE` | 只用于算距离。 |
| COE | `ROUND` | 两轮已取平均。 |
| COE | `PREVAILING_QUOTA_PREMIUM`, `QUOTA`, `BIDS_RECEIVED` | 只用成交价作为月度指标。 |
| COE | `CATEGORY`, `YEAR_MONTH` | 已合并成 `coe_sum` 并按 `month_index` 连接。 |
| Stock | `NAME`, `SYMBOL`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOLUME` | 只用 `ADJUSTED_CLOSE` 合成月度指数（85 行缺价格，读取时已删除）。 |
| Stock | `DATE` | 聚合到月份后按 `month_index` 连接。 |
| Stock | 非 `.SI` 或 2021-01 没有价格的股票 | 非 `.SI` 的不在新加坡交易所；2021-01 之后才有数据的股票无法以 2021-01 为基准，中途加入也会让等权指数跳变。保留 42 只。 |

### A.2 加回被 drop 的列的检查

在 all joined 的基础上逐个加回（编码与第 5 节相同；基线即 all joined：LightGBM 476.15 / CatBoost 476.72 / Linear 518.78），RMSE 变化如下（负数 = 更好）：

| 加回的列 | LightGBM | CatBoost | Linear |
| --- | --- | --- | --- |
| postal sector（邮编前两位，作为低基数类别） | −0.27 | −0.11 | −0.48 |
| `REGION` | −0.30 | −0.18 | 0.00 |
| 到 planned MRT 的距离 | −0.97 | −0.02 | +0.02 |

改善都不超过 1；planned MRT 只有 LightGBM 改善接近 1，CatBoost 和 Linear 几乎不变，属于噪声范围，因此保持 drop。
