library(reshape2)
library(Matrix)
library(readr)
library(tidyr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(ggdendro)
library(grid)
library(dendextend)
library(tidyverse)
library(RColorBrewer)
library(ape)
library(phylogram)
library(circlize)
message("")
aurocs<-read.csv("./AUROC.csv",row.names = 1,check.names = F)
hh<-length(aurocs[1,])/2
aurocs<-aurocs[1:hh,(hh+1):(hh*2)]

aurocs<-as.matrix(aurocs)
rownames(aurocs) <- gsub("Sample1[|]","",rownames(aurocs))
colnames(aurocs) <- gsub("Sample2[|]","",colnames(aurocs))
aurocs <- aurocs[,as.character(rownames(aurocs))]
total.dist<-as.dist(1-aurocs)
total.tree<-hclust(total.dist)
total.tree<-as.dendrogram(total.tree)

#---------add color-------------------------------------------------------------
Clu<-as.data.frame(rownames(aurocs))
rownames(Clu)<-rownames(aurocs)
Clu$Cluster<-gsub("[_]Drosophila[1-9]","",Clu[,1])
Clu$Cluster<-gsub("[0-9]","",Clu$Cluster)
colnames(Clu) <- c("Celltype","Cellcluster")
Clu$Species <- "Drosophila"
Cellcluster <- Clu$Cellcluster
Cellcluster<-as.factor(Cellcluster)

color_regions  =       c("#E6AB02",  "#66A61E", "#D95F02", "#1B9E77", "#E7298A",  "#E31A1C", "#A6761D"  , "#B2DF8A",   "#FFFF99",   "#7570B3", "#FF7F00",  "#A65628", "#B3CDE3", "#BC80BD",     "#A6CEE3","#984EA3",   "#CCEBC5",  "#E41A1C",    "#4DAF4A","#BEBADA", "#B3DE69", "#CAB2D6","#FFFFB3",   "#33A02C","#B15928", "#6A3D9A","#FBB4AE",    "blue",          "#FB8072",      "#FFFF33","#CCEBC5",      "#A6761D",   "#2c7fb8","#fa9fb5",  "#BEBADA","#E7298A", "#E7298A" )
names(color_regions) = c("Secretory" ,"Muscle" ,"Neuron" , "Immune" , "Epithelial","Glia",    "Proliferating","Other",  "Parenchymal","Stromal","Phagocytes","Pharynx","Rectum", "Coelomocytes","Intestine","Hepatocyte","Germline","Endothelial","Erythroid","Testis","Mesenchyme","Yolk", "Midgut" ,"Embryo","Hemocytes",  "Fat",  "Unknown","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","RP_high","FatBody","Female","Nephron", "Pancreatic")
color_regions_use = color_regions[as.character(levels(Cellcluster))]
which(is.na(color_regions_use))
Clu.order <- Clu[order.dendrogram(total.tree),]
celltype_color <- color_regions_use[as.character(Clu.order$Cellcluster)]
dend<-total.tree %>%
  color_branches(k=10) %>%
  set("labels_cex",0.5)%>%
  #set("branches_lwd",c=(5, 2, 1.5))%>%
  #set("branches_lty",c(1,1,3,1,1,2))%>%
  set("leaves_pch",19)%>%
  set("leaves_col",celltype_color) %>% 
  set("labels_colors",celltype_color)

circlize_dendrogram(dend,dend_track_height = 0.8)#,facing = "inside")
Cluster_1<-celltype_color
cluster.list <- unique(names(Cluster_1))
Celltype_bar<-matrix(nrow=length(cluster.list),ncol=length(Cluster_1))
for (i in 1:length(cluster.list)){
  Celltype_bar[i,]<-ifelse(names(Cluster_1) == as.character(cluster.list[i]),as.character(Cluster_1),"white")	
  
}
Celltype_bar<-t(Celltype_bar)
dend1<- color_branches(dend,k=10)

par(mar=c(10,3,10,3)+0.1,
    xpd=NA)
plot(dend)
colored_bars(colors=Celltype_bar,cex.rowLabels=0.7)



