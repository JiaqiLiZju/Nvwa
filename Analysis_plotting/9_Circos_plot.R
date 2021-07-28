library(circlize)
library(dplyr)
library(RColorBrewer)
color_species = structure(c("#2E8B57", "#FF4500","cyan2","#0073c2","#9bbb59","#e888bd","violetred","#f1a907"), 
                          names = c("Human","Mouse","Zebrafish","SeaSquirts","Drosophila","Earthworm","Celegans","Schmidtea")) 

color_regions  =       c("#fa9fb5","#e6ab02","#2c7fb8","#fb8070","#dcdcdc","#DCDCDC","#E6AB02",  "#66A61E", "#D95F02", "#1B9E77", "#E7298A",  "#E31A1C", "#A6761D"  ,       "#A6761D"  ,   "#A6761D"  ,   "#DCDCDC",  "#DCDCDC",   "#FFFF99", "#FFFF99",  "#7570B3", "#FF7F00",  "#A65628", "#DCDCDC","#DCDCDC", "#DCDCDC",     "#DCDCDC","#984EA3",   "#CCEBC5", "#CCEBC5", "#E41A1C",    "#4DAF4A","#BEBADA",   "#6A3D9A", "#6A3D9A","#CAB2D6","#FFFFB3",   "#33A02C","#B15928", "#6A3D9A","#FBB4AE",    "blue",          "#FB8072",      "#FFFF33","#CCEBC5",      "#A6761D",   "#DCDCDC","#DCDCDC",  "#BEBADA","#E7298A", "#E7298A" )
names(color_regions) = c("Notochord","Neuroendocrine","Mesenchyme","DigestiveGland","Follicle","MAG","Secretory" ,"Muscle" ,"Neuron" , "Immune" , "Epithelial","Glia",    "Proliferating",    "Precursors",  "Neoblast", "Other", "Protonephridia", "Parenchymal", "Cathepsin", "Stromal","Phagocytes", "Pharynx","Pharyn","Rectum", "Coelomocytes", "Intestine","Hepatocyte","Germ","Germline","Endothelial","Erythroid","Testis",  "Non-seam","Seam", "Yolk",    "Midgut" ,   "Embryo", "Hemocytes",  "Fat",  "Unknown","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","RP_high","FatBody","Female","Nephron", "Pancreatic")
data_use <- read.table("Result_AUROC_best_hits_0.9_8Species_20210714.txt",sep = "\t",header = T)

OUT_File <- "Figure_circos_tophit.pdf"
data_use <- as.matrix(data_use)
data_use <- as.data.frame(data_use)
DF<-data_use
DF$Mean_AUROC =DF$AUROC
all_regions = unique(c(as.character(DF$Cellcluster1), as.character(DF$Cellcluster2)))
color_regions = color_regions[as.character(all_regions)]
which(is.na(color_regions))


df2 = data.frame(from=paste(DF$Species1,DF$Cellcluster1,sep="|"),to=paste(DF$Species2,DF$Cellcluster2,sep="|"),value=DF$Mean_AUROC)

combined = unique(data.frame(regions = c(as.character(DF$Cellcluster1), as.character(DF$Cellcluster2)), 
                             species = c(as.character(DF$Species1), as.character(DF$Species2)), stringsAsFactors = FALSE))



combined = combined[order(combined$species, combined$regions), ]
order = paste(combined$species, combined$regions, sep = "|")
grid.col = structure(color_regions[combined$regions], names = order)
gap = rep(1, length(order))
gap[which(!duplicated(combined$species, fromLast = TRUE))] = 5

pdf(OUT_File)
circos.par(gap.degree = gap,start.degree=180)
chordDiagram(df2, order = order, 
             annotationTrack = c("grid"),
             grid.col = grid.col, directional = FALSE,
             preAllocateTracks = list(
               track.height = 0.04,
               track.margin = c(0.05, 0)
             )
)
for(species in unique(combined$species)) {
  l = combined$species == species
  sn = paste(combined$species[l], combined$regions[l], sep = "|")
  highlight.sector(sn, track.index = 1, col = color_species[species], 
                   # text = species, 
                   niceFacing = TRUE)
}
circos.clear()

legend("bottomleft", pch = 15, col = color_regions, 
       legend = names(color_regions), cex = 0.3)
legend("bottomright", pch = 15, col = color_species, 
       legend = names(color_species), cex = 0.6)

dev.off()


