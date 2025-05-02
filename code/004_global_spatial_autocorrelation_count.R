library(tidyverse)

setwd("/Users/sds27/repos/llm-geoprobing-dev/results/spatial-autocorrelation")

df <-
  bind_rows(
    # GB
    read_csv("GB-l07_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "GB", layer =  7),
    read_csv("GB-l15_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "GB", layer = 15),
    read_csv("GB-l31_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "GB", layer = 31),
    # IT
    read_csv("IT-l07_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "IT", layer =  7),
    read_csv("IT-l15_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "IT", layer = 15),
    read_csv("IT-l31_activations-mean-pooling_spatial-autocorrelations_df.csv")      %>% mutate(area = "IT", layer = 31),
    # NYmetri
    read_csv("NYmetro-l07_activations-mean-pooling_spatial-autocorrelations_df.csv") %>% mutate(area = "NYmetro", layer =  7),
    read_csv("NYmetro-l15_activations-mean-pooling_spatial-autocorrelations_df.csv") %>% mutate(area = "NYmetro", layer = 15),
    read_csv("NYmetro-l31_activations-mean-pooling_spatial-autocorrelations_df.csv") %>% mutate(area = "NYmetro", layer = 31)
  ) %>% 
  arrange(layer, activation_idx, area) %>% 
  mutate(
    global_sa = if_else(moran_i>=0.3 & moran_p_sim<0.01, TRUE, FALSE)
  )

df_grouped <- 
  df %>% 
  group_by(layer, activation_idx) %>% 
  summarise(
    global_sa = any(global_sa)
  ) %>% 
  ungroup() 

df_grouped %>% 
  filter(global_sa) %>% 
  count()

# # A tibble: 1 × 1
# n
# <int>
#   1  1841
