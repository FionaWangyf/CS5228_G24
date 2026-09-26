#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(scales)
  library(knitr)
})

args <- commandArgs(trailingOnly = TRUE)
project_dir <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = TRUE) else getwd()
data_dir <- file.path(project_dir, "data")
train_path <- file.path(data_dir, "train.csv")
output_dir <- file.path(project_dir, "analysis")
fig_dir <- file.path(output_dir, "figures")
report_path <- file.path(output_dir, "hdb_rental_eda.md")

if (!file.exists(train_path)) stop("Missing train.csv: ", train_path)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

source_files <- sort(list.files(data_dir, recursive = TRUE, full.names = TRUE))
source_files <- source_files[file.info(source_files)$isdir %in% FALSE]
hash_before <- tools::md5sum(source_files)

options(scipen = 999)
theme_set(theme_minimal(base_size = 12))

train <- read.csv(
  train_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA", "N/A", "null", "NULL")
)

required_columns <- c(
  "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE",
  "FLAT_MODEL", "FLOOR_AREA_SQM", "FURNISHED", "LEASE_COMMENCE_DATE",
  "FEE", "MONTHLY_RENT"
)
missing_columns <- setdiff(required_columns, names(train))
if (length(missing_columns) > 0) {
  stop("train.csv is missing required columns: ", paste(missing_columns, collapse = ", "))
}

# Derived objects exist only in memory. No source dataset is rewritten or exported.
normalise_case_space <- function(x) tolower(trimws(gsub("[[:space:]]+", " ", x)))
normalise_flat_type <- function(x) trimws(gsub("[[:space:]]+", " ", gsub("-", " ", tolower(x))))
eda <- train %>%
  mutate(
    approval_date = as.Date(paste0(RENT_APPROVAL_DATE, "-01")),
    approval_year = as.integer(format(approval_date, "%Y")),
    approval_month = as.integer(format(approval_date, "%m")),
    flat_age = approval_year - LEASE_COMMENCE_DATE,
    flat_type_std = normalise_flat_type(FLAT_TYPE),
    street_std = normalise_case_space(STREET),
    block_std = normalise_case_space(BLOCK)
  )

fmt_num <- function(x, digits = 0) format(round(x, digits), big.mark = ",", scientific = FALSE, trim = TRUE)
fmt_pct <- function(x, digits = 1) paste0(round(100 * x, digits), "%")
fmt_money <- function(x, digits = 0) paste0("SGD ", fmt_num(x, digits))
md_table <- function(x, digits = 2) {
  if (nrow(x) == 0) return("_No records._")
  paste(capture.output(knitr::kable(x, format = "pipe", digits = digits, row.names = FALSE)), collapse = "\n")
}
save_plot <- function(plot, filename, width = 10, height = 6) {
  path <- file.path(fig_dir, filename)
  ggsave(path, plot = plot, width = width, height = height, dpi = 160, bg = "white")
  invisible(path)
}
img <- function(filename, alt) paste0("![", alt, "](figures/", filename, ")")

column_overview <- data.frame(
  Column = names(train),
  R_type = vapply(train, function(x) class(x)[1], character(1)),
  Missing = vapply(train, function(x) sum(is.na(x)), integer(1)),
  Missing_pct = round(vapply(train, function(x) mean(is.na(x)) * 100, numeric(1)), 3),
  Unique = vapply(train, function(x) length(unique(x[!is.na(x)])), integer(1)),
  stringsAsFactors = FALSE
)

semantic_type <- c(
  RENT_APPROVAL_DATE = "Temporal (monthly)", TOWN = "Categorical location",
  BLOCK = "High-cardinality identifier", STREET = "High-cardinality location",
  FLAT_TYPE = "Categorical property", FLAT_MODEL = "Categorical property",
  FLOOR_AREA_SQM = "Numerical property", FURNISHED = "Categorical property",
  LEASE_COMMENCE_DATE = "Year / property age", FEE = "Numerical; meaning requires confirmation",
  MONTHLY_RENT = "Numerical target"
)
column_overview$Semantic_type <- unname(semantic_type[column_overview$Column])
column_overview <- column_overview[, c("Column", "Semantic_type", "R_type", "Missing", "Missing_pct", "Unique")]

exact_duplicates <- sum(duplicated(train))
near_keys <- c("RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLOOR_AREA_SQM")
near_duplicate_rows <- sum(duplicated(train[near_keys]))

quality_checks <- data.frame(
  Check = c(
    "Exact duplicate rows",
    "Repeated property-month key after the first record",
    "MONTHLY_RENT <= 0",
    "FLOOR_AREA_SQM <= 0",
    "Missing or invalid RENT_APPROVAL_DATE",
    "LEASE_COMMENCE_DATE later than approval year",
    "Derived flat age < 0",
    "Derived flat age > 99"
  ),
  Count = c(
    exact_duplicates,
    near_duplicate_rows,
    sum(!is.na(eda$MONTHLY_RENT) & eda$MONTHLY_RENT <= 0),
    sum(!is.na(eda$FLOOR_AREA_SQM) & eda$FLOOR_AREA_SQM <= 0),
    sum(is.na(eda$approval_date)),
    sum(!is.na(eda$LEASE_COMMENCE_DATE) & !is.na(eda$approval_year) & eda$LEASE_COMMENCE_DATE > eda$approval_year),
    sum(!is.na(eda$flat_age) & eda$flat_age < 0),
    sum(!is.na(eda$flat_age) & eda$flat_age > 99)
  ),
  stringsAsFactors = FALSE
)
quality_checks$Percent <- round(quality_checks$Count / nrow(eda) * 100, 3)

target_quantiles <- quantile(eda$MONTHLY_RENT, probs = c(0, .01, .05, .25, .5, .75, .95, .99, 1), na.rm = TRUE)
target_summary <- data.frame(
  Statistic = c("Count", "Mean", "Median", "Standard deviation", names(target_quantiles)),
  Value = c(
    sum(!is.na(eda$MONTHLY_RENT)), mean(eda$MONTHLY_RENT, na.rm = TRUE),
    median(eda$MONTHLY_RENT, na.rm = TRUE), sd(eda$MONTHLY_RENT, na.rm = TRUE),
    as.numeric(target_quantiles)
  ),
  stringsAsFactors = FALSE
)
target_summary$Value <- round(target_summary$Value, 1)

monthly <- eda %>%
  filter(!is.na(approval_date), !is.na(MONTHLY_RENT)) %>%
  group_by(approval_date) %>%
  summarise(
    records = n(), mean_rent = mean(MONTHLY_RENT),
    median_rent = median(MONTHLY_RENT), .groups = "drop"
  ) %>%
  arrange(approval_date)

flat_type_summary <- eda %>%
  filter(!is.na(flat_type_std), !is.na(MONTHLY_RENT)) %>%
  group_by(flat_type_std) %>%
  summarise(Records = n(), Mean_rent = mean(MONTHLY_RENT), Median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  arrange(desc(Median_rent))
names(flat_type_summary)[1] <- "FLAT_TYPE"

town_summary <- eda %>%
  filter(!is.na(TOWN), !is.na(MONTHLY_RENT)) %>%
  group_by(TOWN) %>%
  summarise(Records = n(), Mean_rent = mean(MONTHLY_RENT), Median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  arrange(desc(Median_rent))

model_summary <- eda %>%
  filter(!is.na(FLAT_MODEL), !is.na(MONTHLY_RENT)) %>%
  group_by(FLAT_MODEL) %>%
  summarise(Records = n(), Median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  arrange(desc(Records))

furnished_summary <- eda %>%
  group_by(FURNISHED) %>%
  summarise(Records = n(), Missing_rent = sum(is.na(MONTHLY_RENT)), Median_rent = median(MONTHLY_RENT, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(Records))

fee_summary <- eda %>%
  group_by(FEE) %>%
  summarise(Records = n(), Median_rent = median(MONTHLY_RENT, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(Records))

street_summary <- eda %>%
  filter(!is.na(street_std), !is.na(MONTHLY_RENT)) %>%
  group_by(street_std) %>%
  summarise(Records = n(), Median_rent = median(MONTHLY_RENT), IQR_rent = IQR(MONTHLY_RENT), .groups = "drop")
names(street_summary)[1] <- "STREET"

block_summary <- eda %>%
  filter(!is.na(TOWN), !is.na(block_std), !is.na(street_std), !is.na(MONTHLY_RENT)) %>%
  group_by(TOWN, street_std, block_std) %>%
  summarise(Records = n(), Median_rent = median(MONTHLY_RENT), IQR_rent = IQR(MONTHLY_RENT), .groups = "drop")
names(block_summary)[2:3] <- c("STREET", "BLOCK")

rare_categories <- bind_rows(lapply(c("TOWN", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED", "STREET", "BLOCK"), function(col) {
  counts <- table(train[[col]], useNA = "ifany")
  data.frame(
    Feature = col,
    Categories = length(counts),
    Categories_under_20_records = sum(counts < 20),
    Records_in_rare_categories = sum(counts[counts < 20]),
    stringsAsFactors = FALSE
  )
}))

format_audit <- data.frame(
  Feature = c("FLAT_TYPE", "STREET"),
  Raw_unique = c(n_distinct(train$FLAT_TYPE), n_distinct(train$STREET)),
  Normalised_unique = c(n_distinct(eda$flat_type_std), n_distinct(eda$street_std)),
  Difference = c(n_distinct(train$FLAT_TYPE) - n_distinct(eda$flat_type_std), n_distinct(train$STREET) - n_distinct(eda$street_std)),
  stringsAsFactors = FALSE
)

flat_type_collisions <- eda %>%
  distinct(FLAT_TYPE, flat_type_std) %>%
  group_by(flat_type_std) %>%
  summarise(Raw_forms = paste(sort(FLAT_TYPE), collapse = ", "), Raw_form_count = n(), .groups = "drop") %>%
  filter(Raw_form_count > 1)

high_cut <- quantile(eda$MONTHLY_RENT, .99, na.rm = TRUE)
low_cut <- quantile(eda$MONTHLY_RENT, .01, na.rm = TRUE)
tail_summary <- bind_rows(
  eda %>% filter(MONTHLY_RENT >= high_cut) %>% summarise(Tail = "Top 1%", Records = n(), Median_rent = median(MONTHLY_RENT), Median_area = median(FLOOR_AREA_SQM, na.rm = TRUE), Most_common_town = names(sort(table(TOWN), decreasing = TRUE))[1], Most_common_flat_type = names(sort(table(flat_type_std), decreasing = TRUE))[1]),
  eda %>% filter(MONTHLY_RENT <= low_cut) %>% summarise(Tail = "Bottom 1%", Records = n(), Median_rent = median(MONTHLY_RENT), Median_area = median(FLOOR_AREA_SQM, na.rm = TRUE), Most_common_town = names(sort(table(TOWN), decreasing = TRUE))[1], Most_common_flat_type = names(sort(table(flat_type_std), decreasing = TRUE))[1])
)

# Figure 1: target distribution.
p <- ggplot(eda, aes(MONTHLY_RENT)) +
  geom_histogram(bins = 50, fill = "#326891", color = "white", linewidth = 0.2) +
  geom_vline(xintercept = median(eda$MONTHLY_RENT, na.rm = TRUE), color = "#d95f02", linewidth = 0.9) +
  scale_x_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Monthly rent distribution", subtitle = "Orange line: median", x = "Monthly rent", y = "Records")
save_plot(p, "01_target_distribution.png")

# Figure 2: monthly mean and median rent.
monthly_long <- monthly %>% select(approval_date, mean_rent, median_rent) %>% pivot_longer(-approval_date, names_to = "Statistic", values_to = "Rent")
p <- ggplot(monthly_long, aes(approval_date, Rent, color = Statistic)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.3) +
  scale_color_manual(values = c(mean_rent = "#326891", median_rent = "#d95f02"), labels = c(mean_rent = "Mean", median_rent = "Median")) +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Monthly rent over time", x = NULL, y = "Monthly rent", color = NULL)
save_plot(p, "02_monthly_rent_trend.png")

# Figure 3: monthly record counts.
p <- ggplot(monthly, aes(approval_date, records)) +
  geom_col(fill = "#4c956c", width = 25) +
  scale_y_continuous(labels = comma) +
  labs(title = "Monthly transaction records", x = NULL, y = "Records")
save_plot(p, "03_monthly_record_count.png")

# Figure 4: rent by flat type.
flat_order <- flat_type_summary$FLAT_TYPE
p <- ggplot(eda, aes(factor(flat_type_std, levels = rev(flat_order)), MONTHLY_RENT)) +
  geom_boxplot(outlier.alpha = 0.08, fill = "#8ecae6") +
  coord_flip() + scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Rent by flat type", x = "Flat type", y = "Monthly rent")
save_plot(p, "04_rent_by_flat_type.png")

# Figure 5: floor area and rent. Sampled points are for legibility; binned medians use all data.
set.seed(5228)
plot_population <- eda %>% filter(!is.na(FLOOR_AREA_SQM), !is.na(MONTHLY_RENT))
plot_sample <- plot_population %>% slice_sample(n = min(7000, nrow(plot_population)))
area_bins <- eda %>%
  filter(!is.na(FLOOR_AREA_SQM), !is.na(MONTHLY_RENT)) %>%
  mutate(area_bin = floor(FLOOR_AREA_SQM / 5) * 5) %>%
  group_by(area_bin) %>%
  summarise(records = n(), median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  filter(records >= 30)
p <- ggplot(plot_sample, aes(FLOOR_AREA_SQM, MONTHLY_RENT)) +
  geom_point(alpha = 0.08, size = 0.8, color = "#326891") +
  geom_line(data = area_bins, aes(area_bin, median_rent), inherit.aes = FALSE, color = "#d95f02", linewidth = 1.1) +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Floor area and monthly rent", subtitle = "Points: reproducible sample; orange line: median by 5 sqm bin using all records", x = "Floor area (sqm)", y = "Monthly rent")
save_plot(p, "05_floor_area_vs_rent.png")

# Figure 6: town medians.
town_plot <- town_summary %>% mutate(TOWN = factor(TOWN, levels = rev(TOWN)))
p <- ggplot(town_plot, aes(TOWN, Median_rent)) +
  geom_col(fill = "#6a4c93") + coord_flip() +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Median monthly rent by town", subtitle = "All towns shown; use record counts when interpreting small groups", x = "Town", y = "Median rent")
save_plot(p, "06_town_median_rent.png", height = 8)

# Figure 7: major flat models.
top_models <- head(model_summary$FLAT_MODEL, 15)
model_plot <- model_summary %>% filter(FLAT_MODEL %in% top_models) %>% arrange(Median_rent) %>% mutate(FLAT_MODEL = factor(FLAT_MODEL, levels = FLAT_MODEL))
p <- ggplot(model_plot, aes(FLAT_MODEL, Median_rent)) +
  geom_col(fill = "#bc6c25") + coord_flip() +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Median rent for the 15 most common flat models", x = "Flat model", y = "Median rent")
save_plot(p, "07_flat_model_median_rent.png", height = 7)

# Figure 8: flat age relationship.
age_bins <- eda %>%
  filter(!is.na(flat_age), flat_age >= 0, flat_age <= 99, !is.na(MONTHLY_RENT)) %>%
  mutate(age_bin = floor(flat_age / 5) * 5) %>%
  group_by(age_bin) %>%
  summarise(records = n(), median_rent = median(MONTHLY_RENT), .groups = "drop")
p <- ggplot(age_bins, aes(age_bin, median_rent)) +
  geom_line(color = "#2a9d8f", linewidth = 1) + geom_point(color = "#2a9d8f") +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Median rent by derived flat age", subtitle = "Flat age = approval year - lease commencement year; 5-year bins", x = "Flat age (years)", y = "Median rent")
save_plot(p, "08_flat_age_vs_rent.png")

# Figure 9: town x flat type interaction.
town_flat <- eda %>%
  filter(!is.na(TOWN), !is.na(flat_type_std), !is.na(MONTHLY_RENT)) %>%
  group_by(TOWN, flat_type_std) %>%
  summarise(records = n(), median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  filter(records >= 30)
p <- ggplot(town_flat, aes(flat_type_std, factor(TOWN, levels = rev(town_summary$TOWN)), fill = median_rent)) +
  geom_tile(color = "white", linewidth = 0.25) +
  scale_fill_gradient(low = "#edf6f9", high = "#ae2012", labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Median rent by town and flat type", subtitle = "Cells with fewer than 30 records are omitted", x = "Flat type", y = "Town", fill = "Median rent") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p, "09_town_flat_type_heatmap.png", height = 9)

# Figure 10: time x flat type interaction.
time_flat <- eda %>%
  filter(!is.na(approval_date), !is.na(flat_type_std), !is.na(MONTHLY_RENT)) %>%
  group_by(approval_date, flat_type_std) %>%
  summarise(records = n(), median_rent = median(MONTHLY_RENT), .groups = "drop") %>%
  filter(records >= 20)
p <- ggplot(time_flat, aes(approval_date, median_rent, color = flat_type_std)) +
  geom_line(linewidth = 0.8) +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Monthly median rent by flat type", subtitle = "Groups with fewer than 20 monthly records are omitted", x = NULL, y = "Median rent", color = "Flat type") +
  theme(legend.position = "bottom")
save_plot(p, "10_time_by_flat_type.png", height = 7)

# Figure 11: floor area by flat type.
p <- ggplot(eda, aes(factor(flat_type_std, levels = rev(flat_order)), FLOOR_AREA_SQM)) +
  geom_boxplot(outlier.alpha = 0.06, fill = "#ffb703") + coord_flip() +
  labs(title = "Floor area by flat type", x = "Flat type", y = "Floor area (sqm)")
save_plot(p, "11_area_by_flat_type.png")

# Figure 12: calendar month rent. This is descriptive and does not establish seasonality.
p <- ggplot(eda, aes(factor(approval_month, levels = 1:12), MONTHLY_RENT)) +
  geom_boxplot(outlier.alpha = 0.05, fill = "#90be6d") +
  scale_y_continuous(labels = label_dollar(prefix = "SGD ")) +
  labs(title = "Rent distribution by calendar month", subtitle = "Pooled across years; year trend may confound apparent seasonality", x = "Calendar month", y = "Monthly rent")
save_plot(p, "12_calendar_month_rent.png")

area_spearman <- cor(eda$FLOOR_AREA_SQM, eda$MONTHLY_RENT, method = "spearman", use = "complete.obs")
age_spearman <- cor(eda$flat_age, eda$MONTHLY_RENT, method = "spearman", use = "complete.obs")
first_six <- head(monthly, 6)
last_six <- tail(monthly, 6)
trend_change <- median(last_six$median_rent) / median(first_six$median_rent) - 1
town_high <- town_summary %>% filter(Records >= 100) %>% slice_max(Median_rent, n = 1, with_ties = FALSE)
town_low <- town_summary %>% filter(Records >= 100) %>% slice_min(Median_rent, n = 1, with_ties = FALSE)
flat_high <- flat_type_summary %>% slice_max(Median_rent, n = 1, with_ties = FALSE)
flat_low <- flat_type_summary %>% slice_min(Median_rent, n = 1, with_ties = FALSE)
month_count_min <- min(monthly$records)
month_count_max <- max(monthly$records)
date_min <- min(monthly$approval_date)
date_max <- max(monthly$approval_date)
validation_start <- seq(date_max, by = "-5 months", length.out = 2)[2]

report <- c(
  "# HDB Rental Data Analysis",
  "",
  "## 1. Scope and safeguards",
  "",
  paste0("This report analyses only `data/train.csv` (", fmt_num(nrow(train)), " rows and ", ncol(train), " columns)."),
  "",
  "- The analysis does not overwrite, clean, impute, or export a modified dataset.",
  "- Derived variables and analytical subsets exist only in memory for descriptive analysis.",
  "- `test.csv` is not used for distributions, feature selection, or analytical conclusions.",
  "- Auxiliary datasets are not joined in this report.",
  "- Repeated property-month records are treated as review candidates, not automatically as errors, because multiple genuine rental approvals can share the same property characteristics.",
  "",
  "## 2. Dataset overview",
  "",
  paste0("The observed approval period is **", format(date_min, "%Y-%m"), " to ", format(date_max, "%Y-%m"), "**, covering **", nrow(monthly), " months**."),
  "",
  md_table(column_overview),
  "",
  "### Interpretation",
  "",
  "`BLOCK` and `STREET` are high-cardinality location fields. Their raw group medians should not be interpreted without minimum sample-size rules. `FEE` requires confirmation from the competition documentation because its business meaning cannot be inferred safely from the column name alone.",
  "",
  "## 3. Data quality audit",
  "",
  md_table(quality_checks),
  "",
  "### Rare-category exposure",
  "",
  md_table(rare_categories),
  "",
  "### Formatting consistency",
  "",
  md_table(format_audit),
  "",
  md_table(flat_type_collisions),
  "",
  "### Interpretation",
  "",
  paste0("There are **", fmt_num(exact_duplicates), " exact duplicate rows** and **", fmt_num(near_duplicate_rows), " repeated property-month keys after the first record**. The latter may represent parallel genuine transactions and should not be deleted without record-level evidence. `FLAT_TYPE` contains hyphen/space variants and `STREET` contains case variants. Subsequent grouped analysis uses disclosed in-memory standardised labels so the same business category is not split; the source values remain unchanged."),
  "",
  "## 4. Target analysis",
  "",
  md_table(target_summary),
  "",
  img("01_target_distribution.png", "Monthly rent distribution"),
  "",
  "### Tail comparison",
  "",
  md_table(tail_summary),
  "",
  "### Observation and modelling implication",
  "",
  paste0("The target median is **", fmt_money(median(eda$MONTHLY_RENT, na.rm = TRUE)), "**, while the mean is **", fmt_money(mean(eda$MONTHLY_RENT, na.rm = TRUE)), "**. The difference and upper tail mean RMSE will be sensitive to high-rent records. These records should be retained unless a concrete data error is found, then examined separately during model error analysis."),
  "",
  "## 5. Temporal analysis",
  "",
  img("02_monthly_rent_trend.png", "Monthly mean and median rent"),
  "",
  img("03_monthly_record_count.png", "Monthly transaction counts"),
  "",
  img("12_calendar_month_rent.png", "Rent distribution by calendar month"),
  "",
  "### Observation and modelling implication",
  "",
  paste0("The median rent in the last six observed months is **", fmt_pct(trend_change), "** different from the first six observed months. Monthly record counts range from **", fmt_num(month_count_min), "** to **", fmt_num(month_count_max), "**. The time trend is material enough that random train-validation splitting would mix earlier and later market regimes. Calendar-month differences remain descriptive because pooling years can confound seasonality with the long-term trend."),
  "",
  "## 6. Property features and rent",
  "",
  "### Flat type",
  "",
  md_table(flat_type_summary),
  "",
  img("04_rent_by_flat_type.png", "Rent by flat type"),
  "",
  paste0("The highest median flat-type rent is **", fmt_money(flat_high$Median_rent), "** for `", flat_high$FLAT_TYPE, "`; the lowest is **", fmt_money(flat_low$Median_rent), "** for `", flat_low$FLAT_TYPE, "`. Flat type is therefore a strong candidate predictor, although part of its effect overlaps with floor area."),
  "",
  "### Floor area",
  "",
  img("05_floor_area_vs_rent.png", "Floor area and rent"),
  "",
  paste0("The Spearman correlation between floor area and rent is **", round(area_spearman, 3), "**. The binned median line should be used to judge non-linearity rather than relying only on a single correlation coefficient."),
  "",
  "### Flat model",
  "",
  md_table(head(model_summary, 20)),
  "",
  img("07_flat_model_median_rent.png", "Median rent for major flat models"),
  "",
  "Model-level differences may reflect a combination of flat type, age and location. They motivate multivariate modelling but do not establish a causal model effect.",
  "",
  "### Furnished status",
  "",
  md_table(furnished_summary),
  "",
  "`FURNISHED` has one observed value in train and therefore cannot distinguish rental outcomes within this dataset.",
  "",
  "### Fee",
  "",
  md_table(head(fee_summary, 20)),
  "",
  "`FEE` is constant at zero in train and therefore cannot distinguish rental outcomes within this dataset. Its business meaning should still be confirmed before deciding how a later modelling pipeline handles the column.",
  "",
  "## 7. Location features and rent",
  "",
  md_table(town_summary),
  "",
  img("06_town_median_rent.png", "Median rent by town"),
  "",
  paste0("Among towns with at least 100 records, `", town_high$TOWN, "` has the highest median rent at **", fmt_money(town_high$Median_rent), "**, compared with **", fmt_money(town_low$Median_rent), "** in `", town_low$TOWN, "`. This supports retaining location information and later testing finer spatial features."),
  "",
  "### Streets with the highest median rent (minimum 50 records)",
  "",
  md_table(street_summary %>% filter(Records >= 50) %>% arrange(desc(Median_rent)) %>% head(15)),
  "",
  "### Blocks with the highest median rent (minimum 30 records)",
  "",
  md_table(block_summary %>% filter(Records >= 30) %>% arrange(desc(Median_rent)) %>% head(15)),
  "",
  "Street and block summaries are descriptive historical aggregates. Any target encoding based on them must be fitted only on the training portion of each validation fold to prevent leakage.",
  "",
  "## 8. Property age",
  "",
  img("08_flat_age_vs_rent.png", "Flat age and rent"),
  "",
  paste0("The Spearman correlation between derived flat age and rent is **", round(age_spearman, 3), "**. The relationship should be treated as potentially nonlinear and confounded by town and flat type."),
  "",
  "## 9. Interaction analysis",
  "",
  "### Town and flat type",
  "",
  img("09_town_flat_type_heatmap.png", "Town and flat type interaction"),
  "",
  "The heatmap compares like-for-like flat types across towns and is more informative than a town-only median. Remaining differences may still reflect floor area, model, age and time composition.",
  "",
  "### Time and flat type",
  "",
  img("10_time_by_flat_type.png", "Time and flat type interaction"),
  "",
  "This view tests whether flat types move together over time. Diverging trajectories would justify time-by-property interactions or flexible tree-based models.",
  "",
  "### Flat type and floor area",
  "",
  img("11_area_by_flat_type.png", "Floor area by flat type"),
  "",
  "Flat type and floor area overlap strongly, but variation within flat types shows that they should not be treated as identical information at the EDA stage.",
  "",
  "## 10. Validation recommendation",
  "",
  paste0("Use a primary temporal holdout of the final six months: **", format(validation_start, "%Y-%m"), " to ", format(date_max, "%Y-%m"), "**. Train only on earlier records."),
  "",
  "Add two earlier six-month rolling backtests if computation permits. Compare models primarily on the competition metric and report MAE as a secondary diagnostic. Also break errors down by month, town, flat type and rent band. Do not calculate location target aggregates using validation rows.",
  "",
  "## 11. Prioritised findings and next experiments",
  "",
  "1. Time must be modelled explicitly because the dataset spans multiple market regimes.",
  "2. Town, flat type and floor area are the first core predictors to test.",
  "3. Flat model and property age may add nonlinear signal after controlling for the core predictors.",
  "4. Street and block can be useful but require leakage-safe encoding and sample-size regularisation.",
  "5. Establish baselines before adding auxiliary data: mean predictor, linear model, then a tree-based model.",
  "6. Add auxiliary sources through separate ablation experiments: HDB block attributes, MRT, malls, schools, and only then macro variables.",
  "7. Preserve high-rent observations unless a documented data error is identified; analyse their errors separately because RMSE weights them heavily.",
  "",
  "## 12. Analysis limitations",
  "",
  "- All relationships are descriptive and do not establish causality.",
  "- Repeated records cannot be classified as erroneous without a transaction identifier or additional documentation.",
  "- Apparent town, model and age effects may partly reflect differences in time, size and flat-type composition.",
  "- `FEE` remains semantically unresolved.",
  "- External geographic and macro datasets were intentionally excluded from this first analysis.",
  "",
  "## 13. Reproducibility and source integrity",
  "",
  paste0("Generated by `scripts/run_eda.R` from `data/train.csv`. Source-file hashes were checked before and after execution. Report generated at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), ".")
)

writeLines(report, report_path, useBytes = TRUE)

hash_after <- tools::md5sum(source_files)
if (!identical(unname(hash_before), unname(hash_after))) {
  stop("Source integrity check failed: at least one file under data/ changed during analysis.")
}

cat("Report:", report_path, "\n")
cat("Figures:", length(list.files(fig_dir, pattern = "\\.png$")), "\n")
cat("Source files verified unchanged:", length(source_files), "\n")
