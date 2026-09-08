library(tidyverse)
library(here)

df <- read_csv(here('src', '3_error_analysis', 'meningioma_cohort_imaging_metadata_final_103subs.csv')) |> 
  mutate(
    MagneticFieldStrength = as.factor(round(MagneticFieldStrength, 1)),
    Chr22q = as.factor(Chr22q |> recode_values(0 ~ "Intact", 1 ~ "Lost")),
    Chr1p = as.factor(Chr1p |> recode_values(0 ~ "Intact", 1 ~ "Lost")),
    MethylationSubgroup = factor(MethylationSubgroup |> recode_values(0 ~ "Merlin Intact", 1 ~ "Immune Enriched", 2 ~ "Hypermetabolic"), levels=c("Merlin Intact", "Immune Enriched", "Hypermetabolic"))
  )

plot_confounding <- function(outcome) {
  df |> 
    group_by(`Pulse Sequence`, MagneticFieldStrength, !!sym(outcome)) |>
    count() |>
    na.omit() |>
    ggplot(aes(x=!!sym(outcome), y=n, fill=!!sym(outcome))) +
    geom_bar(stat='identity') +
    facet_grid(`Pulse Sequence`~MagneticFieldStrength) +
    theme_classic() +
    ylab("Count") + 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
}

plot_confounding('Chr22q')
plot_confounding('Chr1p')
plot_confounding('MethylationSubgroup')
