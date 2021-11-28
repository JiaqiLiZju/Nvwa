library(Seurat)

setwd("/media/ggj/Files/mount/NvWA_Final/model_training_8000gene_20210207/Dmel_MAGIC_label_015_batchsize_32/Motif_influence_project_tSNE")
pbmc <- readRDS("./Dmel_merge_600gene.seurat_clustering.20210127.RDS")
pbmc

p <- DimPlot(pbmc, reduction = "tsne")
p

pdf("dim.pdf", width=10, height=10)
p
dev.off()

expr <- GetAssayData(pbmc, slot="data", assay="SCT")
dim(expr); expr[1:5,1:5]

t <- apply(expr, 1, quantile, probs=0.95)
length(t); t[1:5]

expr_bin <- expr
expr_bin[expr >= t] <- 1
expr_bin[expr < t] <- 0
dim(expr_bin); expr_bin[1:5,1:5]

expr_bin <- CreateAssayObject(expr_bin)
pbmc@assays$expr_bin <- expr_bin
DefaultAssay(pbmc) <- "expr_bin"
tfs <- c('Abd-B', 'Dfd', 'Dll', 'FoxP', 'Lim3', 'al', 'btn', 'cad', 'dati',
        'exex', 'eyg', 'l(1)sc', 'lab', 'lbe', 'lbl', 'lmd', 'lms', 'nau',
        'nub', 'pb', 'retn', 'rn', 'tj', 'toe', 'vvl', 'ovo')

for (tf in tfs){
    p <- FeaturePlot(pbmc, reduction = "tsne", features = tf)
    pdf(paste0("figures/features_", tf, ".pdf"), width=10, height=10)
    print(p)
    dev.off()
}

data <- read.csv("./Motif_influence.csv", header=1, row.names=1)
dim(data); data[1:5,1:5]

data_bin <- data
t <- apply(data_bin, 2, quantile, probs=0.99)
data_bin[data >= t] <- 1
data_bin[data < t] <- 0
dim(data_bin); data_bin[1:5,1:5]

# data_bin <- read.csv("./Motif_influence_bin.csv", header=1, row.names=1)
# dim(data_bin); data_bin[1:5,1:5]

influe <- CreateAssayObject(t(data_bin))
pbmc@assays$influe <- influe
DefaultAssay(pbmc) <- "influe"

motifs <- rownames(influe)
c('Motif-101', 'Motif-42', 'Motif-44', 'Motif-54', 'Motif-55',
  'Motif-7', 'Motif-72', 'Motif-73', 'Motif-85', 'Motif-87',
  'Motif-88', 'Motif-89', 'Motif-95', 'Motif-99', 'Motif-106')

for (m in motifs){
    p <- FeaturePlot(pbmc, reduction = "tsne", features = m)
    pdf(paste0("figures/features_", m, ".pdf"), width=10, height=10)
    print(p)
    dev.off()
}

anno <- read.csv("/media/ggj/Files/mount/NvWA_Final/0_Annotation_Cellcluster/Dmel_cellatlas.annotation.20210407.txt",
                 sep='\t', header=1, row.names=1)
dim(anno);head(anno)

for (c in unique(anno$Cellcluster)){
  p <- DimPlot(pbmc, cells.highlight = rownames(anno[anno$Cellcluster==c,]))
  pdf(paste0("figures/Dim_", c, ".pdf"), width=10, height=10)
  print(p)
  dev.off()
}

########################################################
anno <- read.csv("/media/ggj/Files/mount/NvWA_Final/0_Annotation_Cellcluster/Dmel_cellatlas.annotation.20210407.txt",
                 sep='\t', header=1, row.names=1)
dim(anno);head(anno)

pbmc@meta.data[, c('Celltype', 'Cluster')] <- anno[row.names(pbmc@meta.data), c('Celltype', 'Cellcluster')]
dim(pbmc@meta.data);head(pbmc@meta.data)

expr <- GetAssayData(pbmc, slot="data", assay="SCT")
dim(expr); expr[1:5,1:5]

expr.celltype <- aggregate(
  list(t(as.data.frame(expr))),
  list(name=pbmc@meta.data$Celltype),
  FUN=mean
)
dim(expr.celltype); expr.celltype[1:5,1:5]

influe <- GetAssayData(pbmc, slot="data", assay="influe")
influe.celltype <- aggregate(
  list(t(as.data.frame(influe))),
  list(name=pbmc@meta.data$Celltype),
  FUN=mean
)
dim(influe.celltype); influe.celltype[1:5,1:5]

library(ggplot2)
data <- rbind(expr.celltype[, c("FoxP")],
              influe.celltype[, c("Motif.85")])

data <- as.data.frame(t(data), row.names=influe.celltype$name)
colnames(data) <- c("tf", "motif")
data$celltype <- rownames(data)
dim(data); head(data)

p<- ggplot(data, aes(x=tf, y=motif)) +
  geom_point(shape=16) +
  geom_smooth(method='glm') +
  xlab("tf") +
  ylab("motif")
p

pdf("scatter.pdf", width=10, height=10)
print(p)
dev.off()

###########################
dat <- data
dat.lm <- lm(motif ~ tf, data = dat)
#edit the formula for the fitted line
formula <- sprintf("italic(y) == %.2f %+.2f * italic(x)",
                   round(coef(dat.lm)[1],2),round(coef(dat.lm)[2],2))
r2 <- sprintf("italic(R^2) == %.2f",summary(dat.lm)$r.squared)
labels <- data.frame(formula=formula,r2=r2,stringsAsFactors = FALSE)

sel <- (dat$tf>=.1)&(dat$motif>=.1)
#plot the simple scatterplot
p <- ggplot(dat,aes(x=tf, y=motif)) + 
  geom_point(shape=19, aes(x=tf, y=motif, colour=sel)) +
  xlab("TF Expression") + 
  ylab("Motif Influence")

p <- p + # geom_abline(intercept = coef(dat.lm)[1],slope = coef(dat.lm)[2]) + 
  geom_smooth(method='lm') +
  geom_text(data=labels, mapping=aes(x=0.075,y=0.32,label=formula),parse = TRUE,inherit.aes = FALSE, size = 6) +
  geom_text(data=labels, mapping=aes(x=0.05,y=0.3,label=r2),parse = TRUE,inherit.aes = FALSE,size = 6) + 
  annotate(geom="text", x=dat[sel,"tf"], y=dat[sel,"motif"]+0.01, label=dat[sel,"celltype"], size=4.0)

p <- p + theme(legend.position = "none") + 
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12,colour="black"))

p

pdf("scatter.expr.pdf", width=10, height=10)
print(p)
dev.off()

