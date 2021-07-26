setwd("/media/ggj/Files/NvWA/nvwa-pse-official/train-mca/Motif/conv1_tomtom/")
library(ggplot2)
library(ggthemes)

#####################################################################
IC <- read.csv("/media/ggj/Files/NvWA/code_test/2_explain/freq.csv", stringsAsFactors = F, header=TRUE)
colnames(IC) <- c("Motif_id", "freq", "IC")
IC$Motif_id <- paste("Motif", IC$Motif_id, sep="_")
rownames(IC) <- IC$Motif_id

ggplot(IC, aes(x=IC)) +
  stat_density(alpha=0.3, colour="grey") +
  theme_base() +
  theme(axis.text.x = element_text(size = 10,vjust = 0.5,angle = 90))

#####################################################################
anno <- read.table("~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/JASPAR/motif.anno.tsv", stringsAsFactors = F)
colnames(anno) <- c("Motif", "Motif_ID", "TF")
rownames(anno) <- anno$Motif_ID

result <- read.table("./tomtom.tsv", stringsAsFactors = F, header=T)
result$TF <- anno[result$Target_ID, "TF"]
result$Name <- paste(result$Query_ID, result$TF, sep=" / ")

result <- result[result$q.value<0.05, ]

imp_d <- read.csv("/media/ggj/Files/NvWA/code_test/2_explain/importance_conv1.csv", header = T, row.names = 1)
imp_d <- t(imp_d)
rownames(imp_d) <- gsub("X", "Motif_", rownames(imp_d))

new <- aggregate(result, by=list(result$Query_ID), FUN=min)
rownames(new) <- new$Query_ID
idx <- rownames(imp_d) %in% new$Query_ID
rownames(imp_d)[idx] <- new[rownames(imp_d)[idx],"Name"]

Ann <- read.csv("/media/ggj/Files/NvWA/nvwa-pse-official/datasets-labels/MCA_pseudocell30W_774_ANN-new.csv", sep="\t", header = T)
rownames(Ann)<-Ann$Cell

Ann <- Ann[colnames(imp_d), ]

Cluster<-Ann$Cluster
annotation_col<-data.frame(Cluster)
rownames(annotation_col)<-colnames(imp_d)
ann_colors=list(
  #Stress=Stress_col,subStress=subStress_col,
  Cluster=c(Endothelial="#E41A1C",
            Epithelial="#377EB8",
            #Epithelial.fetal="#3399FF",
            Erythroid="#4DAF4A",
            
            ES="#A65628",
            Fat="#FCCDE5",
            Germline="#AAAAAA",#"#33A02C",#"#B3DE69",#"#8DD3C7",
            Hepatocyte="#6A3D9A",
            Immnue="#1B9E77",
            Muscle="#66A61E",
            #Muscle.fetal="#B3DE69",
            Neuron="#D95F02",
            Other="#FFFFB3",
            Proliferating="#A6761D",
            #Proliferating.fetal="#BC80BD",
            Secretory="#E6AB02",
            #Stromal.fetal="#CC99FF",
            Stromal="#7570B3"#,
            #Unkown="#DCDCDC"
  )
)

col<-c("white","red")

library(pheatmap)
pheatmap(imp_d, 
         cluster_cols = F, cluster_rows = T)

pheatmap(imp_d, 
         cluster_cols = F, cluster_rows = T,
         show_rownames = T, show_colnames = T, 
         annotation_col = annotation_col,
         annotation_colors = ann_colors,
         clustering_method = "ward.D2"
         #border_color = FALSE,
         )

#####################################################################
library(cowplot)
df <- as.data.frame(cbind(rowSums(imp_d), IC$IC))
rownames(df) <- rownames(imp_d)
colnames(df) <- c("imp", "IC")
df$anno <- "UnAnnotated"
df$anno[grep("/", rownames(df))] <- "Annotated"

pmain <- ggplot(df, aes(x=IC, y=imp, color=anno)) +
  geom_point(shape=19) +
  xlab("Information Content") + ylab("Importance") +
  theme_base() +
  theme(axis.text.x = element_text(size = 10,vjust = 0.5,angle = 90))

xdens <- axis_canvas(pmain, axis = "x") + 
  geom_density(data = df, aes(x=IC, fill=anno)) 

ydens <- axis_canvas(pmain, axis = "y", coord_flip = T) + 
  geom_density(data = df, aes(x=imp, fill=anno)) +
  coord_flip()

p <- insert_xaxis_grob(pmain, xdens, grid::unit(.2, "null"), position = "top")
p <- insert_yaxis_grob(p, ydens, grid::unit(.2, "null"), position = "right")
ggdraw(p)


################################################################################
library(ggrepel)
reproduc <- read.csv("../../../Motif_reproduce/motif_reproduce_inCV.csv", row.names = 1)[, c("sum_cnt","sum_match")]
reproduc$MotifID <- rownames(reproduc)
reproduc[1:5,]
df_m <-  merge(df, reproduc)
head(df_m)
df_m[df_m$imp < 1 & df_m$imp > -1, "Motif"] <- ""

df_m$sum_match <- factor(df_m$sum_match)
df_m$anno <- factor(df_m$anno)
ggplot(df_m, aes(x=IC, y=imp, color=sum_match)) +
  geom_point(size=2) + 
  xlab("Information Content") + ylab("Influence (Fold Change)") +
  # facet_wrap(~anno) +
  scale_color_manual(values = c("#AAAAAA", "#B3DE69", "#8DD3C7", "#4DAF4A", "#33A02C", 
                                "#FCCDE5", "#A65628", "#377EB8", "#6A3D9A", "#E41A1C")) +
  # geom_smooth(method = 'lm') +
  geom_text_repel(aes(label = Motif))
  # theme_base()


#################################################################
temp <- data.frame(
  value = imp_d$Stage_6,
  row.names = c(1:128)
)
temp$index <- df$Motif

temp <- temp[order(-temp$value), ]
temp[1,]; temp[128,]
temp$index[temp$value<0.001&temp$value>-0.002] <- ""

p.6 <- ggplot(temp, aes(x = c(1:128), y = value)) +
  geom_point() +
  labs(x = "rank", y = "importance value") +
  geom_text_repel(aes(label = index))

p.6

g <- ggdraw() + 
  draw_plot(p.1, 0, 0, 0.5, 0.33) +
  draw_plot(p.2, 0, 0.33, 0.5, 0.33) +
  draw_plot(p.3, 0, 0.66, 0.5, 0.33) +
  draw_plot(p.4, 0.5, 0.66, 0.5, 0.33) +
  draw_plot(p.5, 0.5, 0.33, 0.5, 0.33) +
  draw_plot(p.6, 0.5, 0, 0.5, 0.33)

g


pdf("influence.cell.pdf", width = 8, height = 10)
g
dev.off()