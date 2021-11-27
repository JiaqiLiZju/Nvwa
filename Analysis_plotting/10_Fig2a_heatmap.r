library(reshape2)
library(pheatmap)
library(ggplot2)
library(ggthemes)
library(RColorBrewer)
JSDR = read.table("TF_JSDR.out",sep = "\t",header = T)
colnames(JSDR) <- c("Gene","Celltype","JSDR")
JSDR_1 <- dcast(JSDR,Gene~Celltype) 
JSDR_1[1:2,1:2]
rownames(JSDR_1) <- JSDR_1$Gene
JSDR_1 <- JSDR_1[,-1]
JSDR_2 <- JSDR_1
Ann_new = read.table("./HCL_cellatlas.annotation.20210110.txt",sep = "\t",head=T)
Ann_new = Ann_new[,c(3,4)]
Ann_new = Ann_new[!duplicated(Ann_new$Celltype),]
rownames(Ann_new) = Ann_new$Celltype
Ann_new = Ann_new[as.character(colnames(JSDR_2)),]
Ann_new[1:2,]
Ann_new <- Ann_new[as.character(colnames(JSDR_2)),]
all.data <- t(JSDR_2)
JSDR_mean <- aggregate(all.data,list(Ann_new[,2]),mean)
rownames(JSDR_mean) <- JSDR_mean$Group.1
JSDR_mean <- JSDR_mean[,-1]
JSDR_mean<-t(JSDR_mean)
JSDR_mean <- scale(JSDR_mean)
JSDR_mean[JSDR_mean > 3]=3
JSDR_mean[JSDR_mean < (-3)]=(-3)
JSDR_mean <- JSDR_mean[,-8]   ##MARKER
Cluster = colnames(JSDR_mean)
ann_col = data.frame(Cluster)
rownames(ann_col) = colnames(JSDR_mean)
name_order <- c("Proliferating",
                "Germline",
                "Endothelial",
                "Muscle",
                "Stromal",
                "Neuron",
                "Secretory",
                "Epithelial",
                "Erythroid",
                "Immune"
)
ann_col$Cluster <- factor(ann_col$Cluster,levels = name_order)
JSDR_mean <- JSDR_mean[,name_order]
color_regions = c("#E6AB02", "#66A61E", "#D95F02", "#1B9E77", "#E7298A", "#E31A1C", "#A6761D", "#B2DF8A", "#FFFF99", "#7570B3", "#FF7F00", "#A65628", "#B3CDE3", "#BC80BD", "#A6CEE3","#984EA3", "#CCEBC5","#E41A1C","#4DAF4A","#BEBADA","#B3DE69","#CAB2D6","#FFFFB3","#33A02C","#B15928", "#6A3D9A","#FBB4AE","blue","#FB8072","#FFFF33","#CCEBC5","#A6761D","#2c7fb8","#fa9fb5","#BEBADA")
names(color_regions) = c("Secretory" ,"Muscle" ,"Neuron" ,"Immune" , "Epithelial","Glia","Proliferating","Other","Parenchymal","Stromal","Phagocytes","Pharynx","Rectum","Coelomocytes","Intestine","Hepatocyte","Germline","Endothelial","Erythroid","Testis","Unknown","Midgut","Hemocytes" ,"Hindgut","Embryo","Fat","SalivaryGland","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","Mesenchyme","FatBody","Female")
color_regions_use = color_regions[as.character(unique(Cluster))]
which(is.na(color_regions_use))
#color_regions_use <- color_regions_use[-which(is.na(color_regions_use))]
ann_colors=list(
  Cluster=color_regions_use
)

col_use = colorRampPalette(brewer.pal(9,"GnBu"))(20)
col_use=c("white","white","white",col_use)
jsd_p2 = pheatmap(JSDR_mean,
                  #clustering_method = "ward.D2",
                 color = col_use,
                  cluster_cols = FALSE,
                  annotation_col = ann_col,
                  annotation_colors = ann_colors,
                  show_colnames = F,
                  show_rownames   = F)





