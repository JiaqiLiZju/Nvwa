library(dplyr)
library(Seurat)
library(patchwork)
library(Matrix)
library(ggthemes)
library(reshape2)
library(ggplot2)
library(RColorBrewer)
library(harmony)

M6 <- readRDS("./20210513_dge/pbmc_merged_Singlet_used.rds")
M8_1 <- readRDS("./20210625_dge/1/pbmc_merged_Singlet_used.rds")
M8_2 <- readRDS("./20210625_dge/2/pbmc_merged_Singlet_used.rds")
pbmc <- merge(M6,M8_1)
pbmc <- merge(pbmc,M8_2)

Batch1 <- colnames(M6)
Batch2 <- colnames(M8_1)
Batch3 <- colnames(M8_2)
Batch1 <- as.data.frame(Batch1)
Batch2 <- as.data.frame(Batch2)
Batch3 <- as.data.frame(Batch3)
Batch1$Stage <- "M6"
Batch2$Stage <- "M8"
Batch3$Stage <- "M8"
colnames(Batch1) <- c("Cell","Stage")
colnames(Batch2) <- c("Cell","Stage")
colnames(Batch3) <- c("Cell","Stage")
Batch1$Sample <- "M6"
Batch2$Sample <- "M8_1"
Batch3$Sample <- "M8_2"
Batch <- rbind(Batch1,Batch2,Batch3)
rownames(Batch) <- Batch$Cell
UMI_USE <- GetAssayData(pbmc,assay = "RNA",slot = "counts")
data_pbmc<- CreateSeuratObject(counts = UMI_USE, project = "Earthworm")
Batch <- Batch[as.character(colnames(data_pbmc)),]
data_pbmc$Stage <- Batch$Stage
data_pbmc$Sample <- Batch$Sample

pbmc <- data_pbmc
options(repr.plot.height = 2.5, repr.plot.width = 6)
pbmc <- pbmc %>% 
  RunHarmony("Stage", plot_convergence = FALSE)

harmony_embeddings <- Embeddings(pbmc, 'harmony')
harmony_embeddings[1:5, 1:5]
options(repr.plot.height = 5, repr.plot.width = 12)
library(cowplot)
p1 <- DimPlot(object = pbmc, reduction = "harmony", pt.size = .1, group.by = "Stage")
p2 <- VlnPlot(object = pbmc, features = "harmony_1", group.by = "Stage", pt.size = .1)
pdf("Figure_harmony_batch.pdf",w=8,h=4)
plot_grid(p1,p2)
dev.off()
pbmc<-FindNeighbors(pbmc,reduction = "harmony",dims = 1:20)#,k.param = 100)# [dims]
pbmc<-RunTSNE(pbmc,reduction = "harmony", dims = 1:20,check_duplicates=FALSE)
batch_plot<-DimPlot(pbmc,reduction="tsne",group.by = "Stage",label = FALSE) + ggtitle("EarthWorm") + 
  theme(plot.title = element_text(hjust = 0.5)) +NoLegend()
pdf("Figure_harmony_Batch_plot.pdf",w=6,h=5)
batch_plot
dev.off()

pbmc<-FindClusters(pbmc,resolution = 0.6) #check
col_flg<-colorRampPalette(brewer.pal(12,"Paired"))(length(levels(pbmc$seurat_clusters)))
p1<-DimPlot(pbmc,reduction="tsne",cols = col_flg,label = TRUE,repel = TRUE) + ggtitle("EarthWorm") + 
  theme(plot.title = element_text(hjust = 0.5)) #+NoLegend()

subset.marker<-FindAllMarkers(pbmc,only.pos = TRUE,min.pct = 0.25,logfc.threshold = 0.25)
subset.marker %>% group_by(cluster) %>% top_n(n=100,wt=avg_log2FC)
write.table(subset.marker,file = "marker.txt",sep="\t",quote=F)
