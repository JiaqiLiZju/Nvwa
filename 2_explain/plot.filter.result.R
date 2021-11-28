setwd("/media/ggj/Files/mount/NvWA_Final/model_training_20211030/")

library(ggplot2)
library(ggthemes)
library(ggrepel)
library(magrittr)
library(tidyverse)

data <- read.csv("Human_MAGIC_label_006_batchsize_96_20210930/Motif/explain_filter_result.csv", row.names = 1)
head(data)
data$influence_new <- 1-data$influence
data$Query_ID <- gsub("Motif", "Filter", data$Query_ID)
head(data)


data1 <- data[data$influence >= 1.1, ]
data2 <- data[data$influence <= 0.93, ]
data3 <- data[which(data$influence > 0.93 & data$influence < 1.1),]
data1$Tag <- "Negative"
data2$Tag <- "Positive"
data3$Tag <- "No change"
data_use <- rbind(data1,data2,data3)
head(data_use);dim(data_use)

data_use <- data_use[data_use$IC>4,]
head(data_use);dim(data_use)

data_use$Reductant_Tag <- "Reductant"
data_use[data_use$Reductant==0, ]$Reductant_Tag <- "Not_Reductant"

p<- ggplot(data_use,aes(IC, influence_new, color=Tag)) + theme_base() + 
  geom_point(aes(shape=Reductant_Tag, size=sum_cnt)) +
  scale_color_manual(values = c("blue","#999999","red")) +
  labs(title = '', x = 'IC', y= 'Influence score', color = 'Color', size = 'Reproducibility') +
  geom_label(
    data=data_use %>% filter(influence >= 1.1 | influence <= 0.93),
    aes(label=Query_ID),
    nudge_x = 0.3,
    nudge_y =0.06,
    check_overlap=T
  )
p

p_plot <- p + theme(axis.title = element_text(size = 14), axis.text = element_text(size = 11), title = element_text(size = 14), plot.title = element_text(hjust = 0.5), axis.line = element_line(colour = "black")) 

#legend.position = "none" 
pdf("Human_20211015.pdf", w=7.5, h=6)
p_plot
dev.off()

