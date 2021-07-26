setwd("/media/ggj/Files/NvWA/code_test/2_explain/")
library(ggplot2)
library(reshape2)
library(ggthemes)

data_HCL <- read.csv("./saliency_location_HCL.csv", stringsAsFactors = T, row.names = 1)
ggplot() + 
  geom_line(data = data_HCL, aes(x=location, y=saliancy), size=0.5, colour="black") +
  theme_base()+
  theme(axis.text.x = element_text(size = 10,vjust = 0.5,angle = 90)) +
  geom_vline(xintercept = 6500, col="red", size=1)


data_MCA <- read.csv("./saliency_location_MCA.csv", stringsAsFactors = T, row.names = 1)
ggplot() + 
  geom_line(data = data_MCA, aes(x=location, y=saliancy), size=0.5, colour="blue") + 
  theme_base()+
  theme(axis.text.x = element_text(size = 10,vjust = 0.5,angle = 90)) +
  geom_vline(xintercept = 6500, col="red", size=1)


ggplot() + 
  geom_line(data = data_HCL, aes(x=location, y=saliancy), size=0.5, colour="black") +
  geom_line(data = data_MCA, aes(x=location, y=saliancy), size=0.5, colour="blue") + 
  theme_base()+
  theme(axis.text.x = element_text(size = 10,vjust = 0.5,angle = 90)) +
  geom_vline(xintercept = 6500, col="red", size=1)


###################################################################################
setwd("/media/ggj/Files/NvWA/Yeast_stress/Train/ChromIV_train_test/Motif/")

data_HCL <- read.csv("./saliency_location.csv", stringsAsFactors = T, row.names = 1)
data_HCL$location <- data_HCL$location - 1000
p <- ggplot() + 
  geom_line(data = data_HCL, aes(x=location, y=saliancy), size=0.5, colour="black") +
  theme_base()+
  theme(axis.text.x = element_text(size = 10,vjust = 0.5, angle = 0)) +
  geom_vline(xintercept = 0, col="red", size=1)

pdf("./saliancy_location.pdf", width = 10, height = 4)
p
dev.off()