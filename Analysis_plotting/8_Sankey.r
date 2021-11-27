library(pheatmap)
library(networkD3)
library(reshape2)
CelltypePairs <- read.table("Result_SAMap_MappingTable_8Species_use.txt",header = T)
Result_use <- CelltypePairs[,c("Celltype1","Cellcluster1","Species1","Celltype2","Cellcluster2","Species2","Score","Gene_n")]
Result_use <- Result_use[Result_use$Gene_n >= 3,]
dim(Result_use)
data_temp <- Result_use

#--example1 vertebrate
data_temp <- data_temp[which(data_temp$Species1 == "hu" | data_temp$Species1 == "mo"),]
data_temp <- data_temp[which(data_temp$Species2 == "ze" | data_temp$Species2 == "mo"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Hepatocyte"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Hepatocyte"),]

#--example2 Drosophila and other species
data_temp <- data_temp[which(data_temp$Species1 == "dm" | data_temp$Species2 == "dm"),]
tem_use <- data_temp[which(data_temp$Species1 == "ci" ),]
tem_use <- tem_use[,c(4,5,6,1,2,3,7,8)]
colnames(tem_use) <- colnames(data_temp)
data_temp <- data_temp[-which(data_temp$Species1 == "ci" ),]
data_temp <- rbind(data_temp,tem_use)
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Hepatocyte"),]
#data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Hepatocyte"),]

#--example3 earthworm and other species
#data_temp <- data_temp[which(data_temp$Species1 == "ea" | data_temp$Species2 == "ea"),]
data_temp1 <- data_temp[which(data_temp$Species1 == "ea"),]
data_temp2 <- data_temp[which(data_temp$Species2 == "ea"),]
data_temp1$Cellcluster1 <- data_temp1$Celltype1
data_temp2$Cellcluster2 <- data_temp2$Celltype2
data_temp<- rbind(data_temp1,data_temp2)
tem_use <- data_temp[which(data_temp$Species1 == "ci" | data_temp$Species1 == "dm"),]
tem_use <- tem_use[,c(4,5,6,1,2,3,7,8)]
colnames(tem_use) <- colnames(data_temp)
data_temp <- data_temp[-which(data_temp$Species1 == "ci" | data_temp$Species1 == "dm"),]
data_temp <- rbind(data_temp,tem_use)
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Hepatocyte"),]

#--example4 zebrafish and other species
data_temp <- data_temp[which(data_temp$Species1 == "ze" | data_temp$Species2 == "ze"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Other"),]
data_temp <- data_temp[-which(data_temp$Cellcluster1 == "Hepatocyte"),]
data_temp <- data_temp[-which(data_temp$Cellcluster2 == "Hepatocyte"),]


# snanky plot
HT_OUT2 <- data_temp
HT_OUT2$Cellcluster1 <- paste(HT_OUT2$Cellcluster1,HT_OUT2$Species1,sep = "|")
HT_OUT2$Cellcluster2 <- paste(HT_OUT2$Cellcluster2,HT_OUT2$Species2,sep = "|")

edges<-cbind(as.character(HT_OUT2$Cellcluster1),as.character(HT_OUT2$Cellcluster2),HT_OUT2$Score)
#edges<-cbind(as.character(HT_OUT2$Celltype1),as.character(HT_OUT2$Celltype2),HT_OUT$Mean_AUROC)
edges<-as.data.frame(edges)
colnames(edges)<-c("N1","N2","Value")
edges$N1 = as.character(edges$N1)    
edges$N2 = as.character(edges$N2)  
d3links <- edges
d3nodes <- data.frame(name = unique(c(edges$N1, edges$N2)), stringsAsFactors = FALSE)
d3nodes$seq <- 0:(nrow(d3nodes) - 1)

d3links <- merge(d3links, d3nodes, by.x="N1", by.y="name")
names(d3links)[4] <- "source"
d3links <- merge(d3links, d3nodes, by.x="N2", by.y="name")
names(d3links)[5] <- "target"
names(d3links)[3] <- "value"
d3links <- subset(d3links, select=c("source", "target", "value"))
d3nodes <- subset(d3nodes, select=c("name"))

##color setting---------start---------------------------------------------------------
d3nodes1<-data.frame(do.call(rbind,strsplit(d3nodes$name,'[|]')))
colnames(d3nodes1)=c('name','species')
d3nodes$group<-as.factor(d3nodes1$name)

#color.list<-color.list[as.character(unique(d3nodes$group)),]
#color_regions = structure(rev(rainbow(length(all_regions))), names = as.character(all_regions))
color_regions = 'd3.scaleOrdinal()
                .domain(["Secretory" ,"Muscle" ,"Neuron" ,"Immune" , "Epithelial","Glia","Proliferating","Other","Parenchymal","Stromal","Phagocytes","Pharynx","Rectum","Coelomocytes","Intestine","Hepatocyte","Germ","Endothelial","Erythroid","Testis","Unknown","Midgut","Hemocytes" ,"Hindgut","Embryo","Fat","SalivaryGland","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","Mesenchyme","FatBody"])
                .range(["#E6AB02", "#66A61E", "#D95F02", "#1B9E77", "#E7298A", "#E31A1C", "#A6761D", "#DCDCDC", "#FFFF99", "#7570B3", "#FF7F00", "#A65628", "#B3CDE3", "#BC80BD", "#A6CEE3","#984EA3", "#CCEBC5","#E41A1C","#4DAF4A","#BEBADA","#B3DE69","#CAB2D6","#FFFFB3","#33A02C","#B15928", "#6A3D9A","#FBB4AE","blue","#FB8072","#FFFF33","#CCEBC5","#A6761D","#2c7fb8","#fa9fb5"])'
#.range(["#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#7570B3", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC","#DCDCDC", "#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC", "#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC","#DCDCDC", "#DCDCDC", "#DCDCDC", "#DCDCDC"])'


d3links$energy_type <- sub(' .*', '',
                           d3nodes1[d3links$source + 1, 'name'])

net<-sankeyNetwork(Links = d3links,
                   Nodes = d3nodes, 
                   Source = "source",
                   Target = "target", 
                   Value = "value", 
                   NodeID = "name",
                   NodeGroup = "group",
                  # colourScale = JS(color_regions),
                   LinkGroup = 'energy_type',
                   units = "votes",
                   height =800,
                   width =1000,
                   nodePadding = 20,
                   sinksRight =  FALSE, 
                   fontSize = 20, 
                   nodeWidth = 20
)
net
#---------------------------------------------------------------

