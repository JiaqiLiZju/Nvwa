library(pheatmap)
library(ggplot2)
library(dplyr)

####################################################
setwd("/media/ggj/Files/NvWA/nvwa-imputation-official/Human_regression/Chrom8_Train_test_byLineage/Test/")

target_orig <- read.csv("./target_sample.csv", sep=",", row.names = 1)
pred_orig <- read.csv("./pred_sample.csv", sep=",", row.names = 1)

# sample 1k cells
idx <- sample(colnames(target_orig), 1000) #colnames(target_orig)#
target <- target_orig[, idx]
pred <- pred_orig[, idx]

hvgs <- read.csv("../../hvg.csv", sep=",", row.names=1)
idx <- intersect(rownames(hvgs), rownames(pred))
target <- target[idx, ]
pred <- pred[idx, ]

dim(target);target[1:5,1:5]
dim(pred);pred[1:5,1:5]
# pred[pred<0] <- 0

####################################################
anno <- read.csv("/media/ggj/Files/NvWA/nvwa-imputation-official/0_Annotation_Cellcluster/HCL_microwell_twotissue_preEmbry0.cellatlas.annotation.20201215.txt", sep="\t", row.names = 1)

rownames(anno) <- gsub("-", ".", rownames(anno))
anno$Cellcluster <- gsub("-", "_", anno$Cellcluster)
dim(anno); head(anno)

anno <- anno[colnames(target),]
dim(anno); head(anno)

####################################################################################
idx = colnames(target) #sample(x = colnames(target), size = 1000) #

target_sample = target[,idx]
pred_sample = pred[,idx]
dim(target_sample);target_sample[1:5,1:5]
dim(pred_sample);pred_sample[1:5,1:5]

anno_sample <- as.data.frame(anno[idx, ]$Cellcluster)
anno_sample$cellID <- colnames(target_sample)
colnames(anno_sample) <- c("cellLineage","cellID")
row.names(anno_sample) <- anno_sample$cellID

dim(anno_sample);anno_sample[1:5,]
any(is.na(anno_sample))

## fc ## 
# target_sample_fc <- target_sample
target_sample_fc <- log(target_sample / (apply(target_sample, 1, FUN = mean) + 1e-3) + 1)
target_sample_fc[is.na(target_sample_fc)] <- 0

# pred_sample_fc <- pred_sample
pred_sample_fc <- log(pred_sample / (apply(pred_sample, 1, FUN = mean) + 1e-3) + 1)
pred_sample_fc[is.na(pred_sample_fc)] <- 0

dim(target_sample_fc); target_sample_fc[1:5,1:5]
dim(pred_sample_fc); pred_sample_fc[1:5,1:5]; all(pred_sample_fc==0)

# ggplot() + geom_density(data=pred_sample_fc, aes(x=cele.001.001.GCCAACGCCA)) + 
#   geom_density(data=target_sample_fc, aes(x=cele.001.001.GCCAACGCCA), color="red")

# ggplot() + geom_density(data=as.data.frame(t(pred_sample_fc)), aes(x=WBGene00044067)) + 
#   geom_density(data=as.data.frame(t(target_sample_fc)), aes(x=WBGene00044067), color="red")

# calculate correlation
corr_sample_fc_orig <- cor(pred_sample_fc, target_sample_fc, method = "pearson")
corr_sample_fc_orig[1:5,1:5]

idx = colnames(corr_sample_fc_orig) #sample(x = colnames(corr_sample_fc_orig), size = 1000)
anno_col_idx <- anno_sample[idx, ]
anno_col_idx <- anno_col_idx[anno_col_idx$cellLineage!="Other",]
anno_col_idx$cellLineage <- factor(anno_col_idx$cellLineage, levels = c("Endothelial", "Epithelial", "Erythroid", "Testis",
                                                          "Germline", "Hepatocyte", "Immune", "Muscle", "Stromal",
                                                          "Neuron", "Proliferating", "Secretory"))
anno_col_idx_sort <- anno_col_idx[order(anno_col_idx$cellLineage),]
anno_col_idx <- as.data.frame(anno_col_idx_sort$cellLineage)
rownames(anno_col_idx) <- rownames(anno_col_idx_sort)
colnames(anno_col_idx) <- "anno"
                                                          
table(anno_col_idx)

corr_sample_fc <- corr_sample_fc_orig[rownames(anno_col_idx), rownames(anno_col_idx)] #
corr_sample_fc[corr_sample_fc > 0.3] <- 0.3
corr_sample_fc[corr_sample_fc < -0.3] <- -0.3
dim(corr_sample_fc);corr_sample_fc[1:5,1:5]
corr_sample_fc[is.na(corr_sample_fc)] <- 0

# out <- pheatmap(corr_sample_fc, clustering_method = "ward.D2")
# row_idx <- rownames(anno_col_sample)#rownames(corr_sample_fc)[out$tree_row[["order"]]]
coloruse<-colorRampPalette(c('blue', 'white', 'red'))(100)
ann_colors=list(
  anno=c(Endothelial="#377EB8",
         Epithelial="#E41A1C",
         Erythroid="#3399FF",
         Testis="#4DAF4A",
         Germline="#A65628",
         Hepatocyte="#FCCDE5",
         Immune="#AAAAAA",
         Muscle="#B3DE69",
         Stromal="#6A3D9A",
         Neuron="#1B9E77",
         Proliferating="#7570B3",
         Secretory="#B3DE69"
         #Neuron="#D95F02",
         #Other="#FFFFB3",
         #Proliferating="#A6761D",
         #Proliferating.fetal="#BC80BD",
         #Secretory="#E6AB02",
         #Stromal.fetal="#CC99FF",
         #Stromal="#7570B3",
         #Unkown="#DCDCDC"
         )
)

p <- pheatmap(corr_sample_fc, scale = 'row', 
              cluster_cols = F, cluster_rows = F, 
              show_rownames = F, show_colnames = F, 
              annotation_colors = ann_colors, color = coloruse,
              annotation_col = anno_col_idx, annotation_row = anno_col_idx,
              clustering_method = "ward.D2")

png("pred_target_correlation.png", width = 1000, height = 1000)
p
dev.off()

###########################################################################
geneexpr <- target_orig
geneexpr[geneexpr > 6] = 6
geneexpr[geneexpr < -6] = -6

pred_orig[pred_orig < -4] <- -4 

coloruse<-colorRampPalette(c('blue', 'white', 'red'))(100)
out <- pheatmap(geneexpr, 
              cluster_cols = T, cluster_rows = T, 
              show_rownames = F, show_colnames = F, 
              # annotation_colors = ann_colors, 
              color = coloruse,
              annotation_col = anno_col_idx,# annotation_row = anno_col_idx,
              clustering_method = "ward.D2")

row_idx <- rownames(geneexpr)[out$tree_row[["order"]]]
col_idx <- colnames(geneexpr)[out$tree_col[["order"]]]
geneexpr[row_idx, col_idx][1:5,1:5]

pheatmap(pred_orig[row_idx, col_idx], 
         cluster_cols = F, cluster_rows = F, 
         show_rownames = F, show_colnames = F, 
         # annotation_colors = ann_colors, 
         color = coloruse,
         annotation_col = anno_col_idx,# annotation_row = anno_col_idx,
         clustering_method = "ward.D2")

pheatmap(pred_orig[row_idx, col_idx], 
         cluster_cols = T, cluster_rows = T, 
         show_rownames = F, show_colnames = F, 
         # annotation_colors = ann_colors, 
         color = coloruse,
         annotation_col = anno_col_idx,# annotation_row = anno_col_idx,
         clustering_method = "ward.D2")

###########################################################################
corr <- as.data.frame(diag(corr_sample_fc))
colnames(corr) <- "pearson_correlation"
corr$ID <- colnames(corr_sample_fc)
corr[1:5,]

Clu_use<-anno_sample[as.character(corr$ID),]
colnames(Clu_use) <- c("Cellcluster", "Cell")
Clu_use[1:2,]

corr$Cluster<-Clu_use$Cellcluster
# corr$Celltype<-Clu_use$Celltype
corr<-corr[corr$Cluster != "Unkown",]
corr<-corr[corr$Cluster != "Other",]
corr<-corr[!is.na(corr$Cluster),]
# corr <- corr[grep("Testis_Guo", corr$Celltype, invert = T),]
corr[1:5,]

ggplot(corr) + 
  geom_boxplot(aes(x=Cluster, y=pearson_correlation, fill=Cluster)) +
  theme(axis.text.x = element_text(angle = 90, vjust = .5, size=10))

write.csv(corr, file = "pred_target_correlation.csv")

######################### scatter smooth ##############################
library(cowplot)
target_orig[1:5,1:5]

i <- 24
result <- as.data.frame(cbind(target_orig[,i], pred_orig[,i]))
colnames(result) <- c("target", "prediction")

p.6 <- ggplot(result, aes(x=target, y=prediction)) +
  geom_point(size=0.5) +
  geom_smooth(method = 'lm') + 
  geom_rug() +
  labs(title = colnames(target_orig)[i]) +
  # theme_base() +
  theme(plot.title = element_text(hjust = 0.5, size = 10))
  
p.6

g <- ggdraw() + 
  draw_plot(p.1, 0, 0, 0.5, 0.33) +
  draw_plot(p.2, 0, 0.33, 0.5, 0.33) +
  draw_plot(p.3, 0, 0.66, 0.5, 0.33) +
  draw_plot(p.4, 0.5, 0.66, 0.5, 0.33) +
  draw_plot(p.5, 0.5, 0.33, 0.5, 0.33) +
  draw_plot(p.6, 0.5, 0, 0.5, 0.33)

g

# save.image("workspace.rdata")
