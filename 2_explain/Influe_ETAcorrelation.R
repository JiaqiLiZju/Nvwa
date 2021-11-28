setwd("/media/ggj/Files/mount/NvWA_Final/model_training_20211012/")

library(ggthemes)
library(ggplot2)

metric1 <- read.csv("Human_MAGIC_006/Motif/Influence/eta.csv")
metric1$species <- "Human"
dim(metric1);head(metric1)

metric2 <- read.csv("Mouse_MAGIC_label_006_batchsize_96_new/Motif_influence/Influence/eta.csv")
metric2$species <- "Mouse"
dim(metric2);head(metric2)

metric3 <- read.csv("Zebrafish_MAGIC_t5_new_006/Motif/Influence/eta.csv")
metric3$species <- "Zebrafish"
dim(metric3);head(metric3)

metric4 <- read.csv("Dmel_MAGIC_label_015_batchsize_32/Motif_Influence/Influence/eta.csv")
metric4$species <- "Dmel"
dim(metric4);head(metric4)

metric5 <- read.csv("Celegans_MAGIC_label_005_batchsize_32/Motif/Influence/eta.csv")
metric5$species <- "Celegans"
dim(metric5);head(metric5)

metric6 <- read.csv("Smed_MAGIC_label_020_batchsize_32/Motif_Influence/Influence/eta.csv")
metric6$species <- "Smed"
dim(metric6);head(metric6)

metric7 <- read.csv("Earthworm_MAGIC_t3_011/Motif/Influence/eta.csv")
metric7$species <- "Earthworm"
dim(metric7);head(metric7)

metric <- rbind(metric1, metric2, metric3, metric4, metric5, metric6, metric7)
# sort factor
metric$species <- factor(metric$species, levels = c("Human", "Mouse", "Zebrafish", "Dmel", "Earthworm", "Celegans", "Smed"))
metric <- metric[order(metric$species),]
# metric$value <- ifelse(metric$value > 0.5, metric$value, 1 - metric$value)
dim(metric); head(metric)

ggplot(metric) + 
  geom_boxplot(aes(x=species, y=eta, fill=type)) + theme_base() +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, size=10)) +
  ylab("ETA correlation with celltype")

