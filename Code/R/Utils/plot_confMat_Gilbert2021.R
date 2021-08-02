
## plot_confMat_Gilbert2021.R
#
# Plots confusion matrix for behav data of study 1 in Gilbert (2021).
#
# 08/2021: Felix Klotzsche


library(tidyverse)
library(here)

# Data were originally downloaded from: https://osf.io/ku5nr/?view_only=516030d70a6c4096a513c0365f643e3c

# Get data frame:
path_data <- here('Data', 'Stimuli', 'Gilbert2021')

fname <- file.path(path_data, 'validation_emotions_facs_pofa.csv')
dat <- read_csv2(fname)

# sanity checks:
d_filt <- dat %>%  filter(button_pressed != 'NULL', 
                          !str_detect(stimulus_type, 'Emfacs')) %>% 
  group_by(subject) %>% 
  summarize(n = n())

print(d_filt)
# looks ok; 42 trials for most subs (for whatever reasons 2 subs have 43, but whatever)
# It's n=44; 1 sub did not complete bio information questionnaire; we don't care



# get mapping of keys to emotions:
sorted_emos_by_bu <- dat %>%  filter(button_pressed != 'NULL', 
                          !str_detect(stimulus_type, 'Emfacs')) %>% 
  separate(stimulus_type, sep = '_', into = c(NA, NA, 'emo_true')) %>% 
  group_by(emo_true) %>% 
  summarize(bu = mean(as.numeric(correct_response))) %>% 
  arrange(bu) %>% 
  pull(emo_true)

# Summarize data:
d_filt <- dat %>%  filter(button_pressed != 'NULL', 
                          !str_detect(stimulus_type, 'Emfacs')) %>% 
  separate(stimulus_type, sep = '_', into = c(NA, NA, 'emo_true')) %>% 
  mutate(emo_rated = as_factor(sorted_emos_by_bu[as.numeric(button_pressed) + 1]), 
         emo_true = as_factor(emo_true)) %>% 
  group_by(emo_true, emo_rated, .drop = FALSE) %>% 
  summarize(n = n()) %>% 
  mutate(freq = n/sum(n))


# plot it:

# Sequence in which we want to have the emos on the axes (same as in Smith & Smith (2019)):
emos_seq2plot <- c('neutral', 'happy', 'surprise', 'fear', 'disgust', 'anger', 'sad')

plt <- ggplot(d_filt, aes(x=emo_rated, y = emo_true, fill = freq)) + 
  geom_raster() + scale_fill_gradient2(low = "darkblue",
                                       mid = 'green', 
                                       high = "red", 
                                       midpoint = 0.50,
                                       na.value = "darkblue") + 
  scale_x_discrete(limits=(emos_seq2plot), 
                   position = 'top')  + 
  scale_y_discrete(limits=rev(emos_seq2plot)) + 
  coord_fixed(1) + 
  ylab("true emotion") + 
  xlab("rated emotion") + 
  theme_minimal()
plt 
