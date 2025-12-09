# Package imports
if (!requireNamespace("neuroCombat", quietly = TRUE)) {
  devtools::install_github("jfortin1/neuroCombat_Rpackage")
}
library(neuroCombat)
library(tidyverse)

# Read in data files of interest
feat_files <- list.files(here::here("data", "1_feature_extraction", "1_pyradiomics", "b_aggregated_pyradiomics"), pattern = "*.csv", full.names = TRUE) # nolint
feat_fp <- feat_files[length(feat_files)]
labels_fp <- here::here("data", "0_preprocessing", "0_LABELS", "radiomics_cohort_06-09-2025_w_demographics_clean.csv") # nolint

feat_df <- readr::read_csv(feat_fp) %>%
  select(
    where(~ !all(is.na(.x))) & # keeps those cols that aren't all NAs
      where(~ n_distinct(.x, na.rm = TRUE) > 1) # keep cols w/>= 1 distinct val
  )

labels_df <- readr::read_csv(labels_fp, col_types = "dfcccccdcdc") %>%
  rename(subject = `Subject Number`) %>%
  mutate(
    across(where(is.character), ~replace(., is.na(.), "Unknown")),
    MethylationSubgroup = as.factor(MethylationSubgroup),
    Chr22q = as.factor(Chr22q),
    Chr1p = as.factor(Chr1p),
    Chr9p = as.factor(Chr9p),
    TERT = as.factor(TERT),
    Sex = as.factor(Sex),
    Ethnicity = as.factor(Ethnicity),
  )

# Merge by subject
df <- merge(labels_df, feat_df, by = "subject")

# Prepare data for ComBat
feat_names <- colnames(df)[12:length(df)]
data <- df %>% select(all_of(feat_names))
subjects <- df$subject
batch <- as.numeric(df$Session)
mod <- model.matrix(~Age + Sex, labels_df)

# Find those features with enough data present for ComBat
data1 <- data[batch == 1, ] %>%
  select(
    where(~ !all(is.na(.x))) & # keeps those cols that aren't all NAs
      where(~ n_distinct(.x, na.rm = TRUE) > 1) # keep cols w/>= 1 distinct val
  )
data2 <- data[batch == 2, ] %>%
  select(
    where(~ !all(is.na(.x))) & # keeps those cols that aren't all NAs
      where(~ n_distinct(.x, na.rm = TRUE) > 1) # keep cols w/>= 1 distinct val
  )
feats_w_acceptable_missingness <- sort(
  intersect(colnames(data1), colnames(data2))
)

final_data <- t(data[, feats_w_acceptable_missingness])

# Run ComBat
combat <- neuroCombat(dat = final_data, batch = batch, mod = mod)
est <- combat$estimates
all.equal(combat$dat.combat, combat$dat.standardized)
head(t(est$delta.hat))
head(t(est$delta.star))
head(t(est$gamma.hat))
head(t(est$gamma.star))

# Post process ComBat results
data_combat_corrected <- t(combat$dat.combat) %>%
  as_tibble() %>%
  mutate(subject = subjects) %>%
  select(c(subject, all_of(feats_w_acceptable_missingness)))

# Save results
output_dir <- here::here("data", "1_feature_extraction", "1_pyradiomics", "c_combat_pyradiomics") # nolint
output_fp <- here::here(output_dir, tail(str_split(feat_fp, "/")[[1]], 1))
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
readr::write_csv(data_combat_corrected, output_fp)
