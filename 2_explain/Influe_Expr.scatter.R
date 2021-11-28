setwd("/media/ggj/Files/mount/NvWA_Final/model_training_8000gene_20210207/Dmel_MAGIC_label_015_batchsize_32/Motif_influence_project_tSNE")

library(ggplot2)
library(reshape2)

JSD <- read.csv("/home/ggj/Desktop/JSD/dmel/TF_JSDR.out", sep='\t')
colnames(JSD) <- c("Gene","Celltype","JSDR")
dim(JSD);head(JSD)

JSD <- dcast(JSD, Gene ~ Celltype)
JSD <- as.data.frame(JSD)
rownames(JSD) <- JSD[,1]
JSD <- JSD[,-1]
JSD[is.na(JSD)] <- 0
JSD <- t(JSD)
dim(JSD); JSD[1:5,1:5]


# pbmc@meta.data[, c('Celltype', 'Cluster')] <- anno[row.names(pbmc@meta.data), c('Celltype', 'Cellcluster')]
# dim(pbmc@meta.data);head(pbmc@meta.data)

# expr <- GetAssayData(pbmc, slot="data", assay="SCT")
# dim(expr); expr[1:5,1:5]
# 
# expr.celltype <- aggregate(
#   list(t(as.data.frame(expr))),
#   list(name=pbmc@meta.data$Celltype),
#   FUN=mean
# )
# dim(expr.celltype); expr.celltype[1:5,1:5]

influe <- read.csv("./Motif_influence.csv", header=1, row.names=1)
dim(influe); influe[1:5,1:5]

anno <- read.csv("/media/ggj/Files/mount/NvWA_Final/0_Annotation_Cellcluster/Dmel_cellatlas.annotation.20210407.txt",
                 sep='\t', header=1, row.names=1)
anno <- anno[rownames(influe),]
dim(anno);head(anno)

# influe <- GetAssayData(pbmc, slot="data", assay="influe")
influe.celltype <- aggregate(
  list(influe),
  list(name=anno$Celltype),
  FUN=mean
)
dim(influe.celltype); influe.celltype[1:5,1:5]

data <- rbind(log(JSD[, c("FoxP")]+1e-2),
              influe.celltype[, c("Motif_89")])

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

sel <- (dat$tf >= -3.1)|(dat$motif >= 1)
#plot the simple scatterplot
p <- ggplot(dat,aes(x=tf, y=motif, colour=sel)) + 
  geom_point(shape=19) +
  xlab("tf") + 
  ylab("motif")

p <- p + geom_abline(intercept = coef(dat.lm)[1],slope = coef(dat.lm)[2]) + 
  geom_text(data=labels, mapping=aes(x=-4,y=2,label=formula),parse = TRUE,inherit.aes = FALSE, size = 6) +
  geom_text(data=labels, mapping=aes(x=-4,y=1.7,label=r2),parse = TRUE,inherit.aes = FALSE,size = 6) + 
  annotate(geom="text", x=dat[sel,"tf"], y=dat[sel,"motif"]+0.01, label=dat[sel,"celltype"], size=4.0)

p <- p + theme(legend.position = "none") + 
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12,colour="black"))

p

pdf("scatter.pdf", width=10, height=10)
print(p)
dev.off()