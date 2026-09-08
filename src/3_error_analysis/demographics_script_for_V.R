library(tidyverse)
library(here)

MeningiomaBiomarkerData_correct <- read_csv(here('data', '0_preprocessing', '0_LABELS', 'MeningiomaBiomarkerData-correct.csv'))
images_metadata_df <- read_csv(here(
    'src', '3_error_analysis', 'meningioma_cohort_imaging_metadata.csv'
  )) |> mutate(across(where(is.character), as.factor))

labels_df <- read_csv(here('data', '0_preprocessing', '0_LABELS', 'radiomics_cohort_06-09-2025_w_demographics_clean.csv'))

# All groups
labels_df |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

labels_df |> mutate(Ethnicity = as.factor(Ethnicity)) |> summary()

# All groups broken down by sex
labels_df |> 
  group_by(Sex) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Groups broken down by Methylation Subgroup
labels_df |> 
  select(`Subject Number`, MethylationSubgroup, Age, Sex) |>
  na.omit() |>
  mutate(MethylationSubgroup = MethylationSubgroup |> recode_values(0 ~ "Merlin Intact", 1 ~ "Immune Enriched", 2 ~ "Hypermetabolic")) |>
  group_by(MethylationSubgroup) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Groups broken down by Methylation Subgroup & Sex
labels_df |> 
  select(`Subject Number`, MethylationSubgroup, Age, Sex) |>
  na.omit() |>
  mutate(MethylationSubgroup = MethylationSubgroup |> recode_values(0 ~ "Merlin Intact", 1 ~ "Immune Enriched", 2 ~ "Hypermetabolic")) |>
  group_by(Sex, MethylationSubgroup) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Broken down by Chr22q
labels_df |> 
  select(`Subject Number`, Chr22q, Age, Sex) |>
  na.omit() |>
  mutate(Chr22q = Chr22q |> recode_values(0 ~ "Intact", 1 ~ "Lost")) |>
  group_by(Chr22q) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Broken down by Chr22q & Sex
labels_df |> 
  select(`Subject Number`, Chr22q, Age, Sex) |>
  na.omit() |>
  mutate(Chr22q = Chr22q |> recode_values(0 ~ "Intact", 1 ~ "Lost")) |>
  group_by(Sex, Chr22q) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Broken down by Chr1p
labels_df |> 
  select(`Subject Number`, Chr1p, Age, Sex) |>
  na.omit() |>
  mutate(Chr1p = Chr1p |> recode_values(0 ~ "Intact", 1 ~ "Lost")) |>
  group_by(Chr1p) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()

# Broken down by Chr1p & Sex
labels_df |> 
  select(`Subject Number`, Chr1p, Age, Sex) |>
  na.omit() |>
  mutate(Chr1p = Chr1p |> recode_values(0 ~ "Intact", 1 ~ "Lost")) |>
  group_by(Sex, Chr1p) |> 
  summarise(Min_Age = min(Age), Max_Age = max(Age), Mean_Age = mean(Age), Median_Age = median(Age), Count = n()) |> 
  View()





pulses <- c("AX_3D_T1_POST", "AX_ADC", "AX_DIFFUSION", "SAG_3D_FLAIR")

main_df <- images_metadata_df |> 
  filter(`Pulse Sequence` %in% pulses) |>
  filter(`Subject Number` %in% unique(labels_df$`Subject Number`)) |>
  filter(!((`Subject Number` == 8) & (Session == "original"))) |>
  filter(!((`Subject Number` == 54) & (Session == "initial_brainlab"))) |>
  filter(!((`Subject Number` == 64) & (Session == "brainlab2"))) |>
  filter(!((`Subject Number` == 107) & (Session == "brainlab_2")))

summary(main_df)

summary(labels_df |> mutate(across(where(is.character), as.factor)) |> select(Ethnicity))

main_df |>
  group_by(`Pulse Sequence`) |>
  summarise(
    across(
      c(EchoTime, RepetitionTime, FlipAngle),
      list(min = ~min(.x, na.rm = TRUE),
           max = ~max(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

main_df |>
  mutate(MagneticFieldStrength = as.factor(round(MagneticFieldStrength, 1))) |>
  group_by(`Pulse Sequence`, MagneticFieldStrength) |>
  summarise(n = n()) |> View()

main_df |>
  mutate(MagneticFieldStrength = as.factor(round(MagneticFieldStrength, 1))) |>
  group_by(MagneticFieldStrength) |>
  summarise(n = n()) |> View()

main_df |> 
  filter(`Pulse Sequence` %in% c('AX_ADC', 'AX_DIFFUSION')) |>
  group_by(`Subject Number`) |>
  summarise(count=n()) |> View()

write_csv(main_df, here('src', '3_error_analysis', 'meningioma_cohort_imaging_metadata_final_103subs.csv'))
