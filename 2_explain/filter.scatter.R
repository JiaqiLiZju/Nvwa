setwd("/media/ggj/ggjlab_188/PeijingZhang/NvWA/Plot/tomtom/")
library(ggplot2)
library(ggthemes)
library(ggrepel)
library(magrittr)
library(tidyverse)

data <- read.csv("Earthworm_explain_filter_res_20211015ult.csv", row.names = 1)
#data <- read.csv("/media/Guo4T/PeijingZhang/NvWA/Plot/tomtom/Dmel_explain_filter_result.csv", row.names = 1)
head(data)
data$influence_new <- 1-data$influence
data$Query_ID <- gsub("Motif", "Filter", data$Query_ID)
head(data)


data1 <- data[data$influence >= 1.1, ]
data2 <- data[data$influence <= 0.9, ]
data3 <- data[which(data$influence > 0.9 & data$influence < 1.1),]
data1$Tag <- "Negative"
data2$Tag <- "Positive"
data3$Tag <- "No change"
data_use <- rbind(data1,data2,data3)
head(data_use);dim(data_use)

data_use <- data_use[data_use$IC>4,]
head(data_use);dim(data_use)




p<- ggplot(data_use,aes(IC, influence_new, color=Tag)) + theme_base() + geom_point(aes(size=sum_cnt)) +
  scale_color_manual(values = c("blue","#999999","red")) +
  labs(title = '', x = 'IC', y= 'Influence score', color = 'Color', size = 'Reproducibility') +
  geom_label(
    data=data_use %>% filter(influence >= 1.1 | influence <= 0.9),
    aes(label=Query_ID),
    nudge_x = 0.3,
    nudge_y =0.06,
    check_overlap=T
  )
p

p_plot <- p + theme(axis.title = element_text(size = 14), axis.text = element_text(size = 11), title = element_text(size = 14), plot.title = element_text(hjust = 0.5), axis.line = element_line(colour = "black")) 

#legend.position = "none" 
pdf("Earthworm_20211015.pdf", w=7.5, h=6)
p_plot
dev.off()

