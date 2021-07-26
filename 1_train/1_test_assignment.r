library(ggthemes)
library(ggplot2)

############################### AUROC ##########################
setwd("/media/ggj/Files/NvWA/nvwa-imputation-official/Benchmark/")

metric1<-read.table("./HCL_NvWA_CellOut/Test/test_mode_roc.csv", sep=",", head=T)
colnames(metric1)<-c("ID", "value")
metric1$type <- "HCL_NvWA_CellOut"
head(metric1)

metric2<-read.table("./HCL_NvWA_Lineage/Test/test_mode_roc.csv", sep=",", head=T)
colnames(metric2)<-c("ID", "value")
metric2$type= "HCL_NvWA_Lineage"
head(metric2)

metric3<-read.table("./HCL_NvWA_Celltype/Test/test_mode_roc.csv", sep=",", head=T)
colnames(metric3)<-c("ID", "value")
metric3$type= "HCL_NvWA_Celltype"
head(metric3)

# rbind
metric <- rbind(metric1, metric2)
# sort factor
metric$type <- factor(metric$type, levels = c("HCL_NvWA_CellOut", "HCL_NvWA_Lineage", "HCL_NvWA_Celltype"))
metric <- metric[order(metric$type),]
dim(metric); head(metric)

################################ Correlation ##########################
setwd("/media/ggj/Files/NvWA/nvwa-imputation-official/Human_regression/")

metric1 <- read.csv("Chrom8_Train_test_byLineage/Test/test_mode_correlation.csv")
metric1$type <- "Chrom8_Train_test_byLineage"
colnames(metric1) <- c("ID", "value", "p", "type")
dim(metric1);head(metric1)

metric2 <- read.csv("Chrom8_Train_test_byCelltype/Test/test_mode_correlation.csv")
metric2$type <- "Chrom8_Train_test_byCelltype"
colnames(metric2) <- c("ID", "value", "p", "type")
dim(metric2);head(metric2)

# rbind
metric <- rbind(metric1, metric2)
# sort factor
metric$type <- factor(metric$type, levels = c("Chrom8_Train_test_byLineage", "Chrom8_Train_test_byCelltype"))
metric <- metric[order(metric$type),]
metric[1:5,]

####################################### Annotation ##########################################
Clu<-read.table("/media/ggj/Files/NvWA/nvwa-imputation-official/0_Annotation_Cellcluster/HCL_microwell_twotissue_preEmbry0.cellatlas.annotation.20201215.txt", head=T, sep = "\t")
dim(Clu); Clu[1:5,]

rownames(Clu)<-Clu$Cell
Clu_use<-Clu[as.character(metric$ID),]
Clu_use[1:2,]

metric$Cluster<-Clu_use$Cellcluster
metric$Celltype<-Clu_use$Celltype

metric<-metric[metric$Cluster != "Unkown",]
metric<-metric[metric$Cluster != "Other",]
metric<-metric[!is.na(metric$Cluster),]
metric <- metric[grep("Testis_Guo", metric$Celltype, invert = T),]

p <- ggplot(metric) + 
  geom_boxplot(aes(x=Cluster, y=value, fill=type)) + theme_base() +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, size=10))
  # xlab("") +
  # ylab(" value")

pdf("metric_cluster.pdf", width = 10, height = 4)
p
dev.off()

p <- ggplot(metric) + 
  geom_boxplot(aes(x=type, y=value, fill=type)) + theme_base() +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, size=10))

pdf("metric_type.pdf", width = 10, height = 4)
p
dev.off()

# auc <- auc1
# auc_high <- auc[auc$correlation>0.38,]
# dim(auc_high)
# table(auc_high$Cluster)
# write.csv(auc_high$ID, file = "high_roc_cell.csv")
