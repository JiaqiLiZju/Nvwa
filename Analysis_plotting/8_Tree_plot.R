library(reshape2)
library(ggplot2)
library(ggdendro)
library(grid)
library(dendextend)
library(tidyverse)
library(RColorBrewer)
library(ape)
library(phylogram)
library(circlize)
total_merge <- read.table("./Result_AUROC_8species.merge_20210714_using.txt",sep = "\t",check.names = F)
Clu_total <- read.table("Result_CellAnn_8species_20210714.txt",sep = "\t",header = T)
rownames(Clu_total) <- Clu_total$Idents
Clu_total <- Clu_total[rownames(total_merge),]
Cellcluster <- Clu_total$Cellcluster
Cellcluster<-as.factor(Cellcluster)
color_regions  =       c("#fa9fb5","#e6ab02","#2c7fb8","#fb8070","#dcdcdc","#DCDCDC","#E6AB02",  "#66A61E", "#D95F02", "#1B9E77", "#E7298A",  "#E31A1C", "#A6761D"  ,       "#A6761D"  ,   "#A6761D"  ,   "#DCDCDC",  "#DCDCDC",   "#FFFF99", "#FFFF99",  "#7570B3", "#FF7F00",  "#A65628", "#DCDCDC","#DCDCDC", "#DCDCDC",     "#DCDCDC","#984EA3",   "#CCEBC5", "#CCEBC5", "#E41A1C",    "#4DAF4A","#BEBADA",   "#6A3D9A", "#6A3D9A","#CAB2D6","#FFFFB3",   "#33A02C","#B15928", "#6A3D9A","#FBB4AE",    "blue",          "#FB8072",      "#FFFF33","#CCEBC5",      "#A6761D",   "#DCDCDC","#DCDCDC",  "#BEBADA","#E7298A", "#E7298A" )
names(color_regions) = c("Notochord","Neuroendocrine","Mesenchyme","DigestiveGland","Follicle","MAG","Secretory" ,"Muscle" ,"Neuron" , "Immune" , "Epithelial","Glia",    "Proliferating",    "Precursors",  "Neoblast", "Other", "Protonephridia", "Parenchymal", "Cathepsin", "Stromal","Phagocytes", "Pharynx","Pharyn","Rectum", "Coelomocytes", "Intestine","Hepatocyte","Germ","Germline","Endothelial","Erythroid","Testis",  "Non-seam","Seam", "Yolk",    "Midgut" ,   "Embryo", "Hemocytes",  "Fat",  "Unknown","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","RP_high","FatBody","Female","Nephron", "Pancreatic")
color_regions_use = color_regions[as.character(levels(Cellcluster))]
which(is.na(color_regions_use))
Species <- Clu_total$Species
Species <- as.factor(Species)
color_species = structure(c("#2E8B57", "#FF4500","cyan2","#0073c2","#9bbb59","#e888bd","violetred","#f1a907"), 
                          names = c("Human","Mouse","Zebrafish","SeaSquirts","Drosophila","Earthworm","Celegans","Schmidtea")) 
color_species_use = color_species[as.character(levels(Species))]
which(is.na(color_regions_use))

total_merge[total_merge<0.8]=0
total.dist<-dist(1-total_merge)
total.tree<-hclust(total.dist,method = "ward.D")
total.tree<-as.dendrogram(total.tree)
Clu.order <- Clu_total[order.dendrogram(total.tree),]
celltype_color <- color_regions_use[as.character(Clu.order$Cellcluster)]
species_color <- color_species_use[as.character(Clu.order$Species)]

dend<-total.tree %>%
  color_branches(k=30) %>%
  set("labels_cex",0.00001)%>%
  set("branches_lwd",1)%>%
  #set("branches_lty",c(1,1,3,1,1,2))%>%
  set("leaves_pch",15)%>%
  set("leaves_cex",3)%>%
  set("leaves_col",celltype_color)%>%
  set("labels_col",celltype_color)


Cluster_1<-species_color
cluster.list <- unique(names(Cluster_1))
Celltype_bar<-matrix(nrow=length(cluster.list),ncol=length(Cluster_1))
for (i in 1:length(cluster.list)){
  Celltype_bar[i,]<-ifelse(names(Cluster_1) == as.character(cluster.list[i]),as.character(Cluster_1),"white")	
  
}
Celltype_bar<-t(Celltype_bar)

pdf("Figure_TotalMerge_tree.pdf",w=20,h=20)
par(mar=c(40,3,45,3)+0.1, 
    xpd=NA)
plot(dend)
colored_bars(colors=Celltype_bar,cex.rowLabels=1)
dev.off()

