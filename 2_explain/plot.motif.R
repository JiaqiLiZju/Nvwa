library(BiocManager)
install("motifStack")

packageVersion("motifStack")
capabilities()["cairo"]

# ERROR: dependency ‘XML’ is not available for package ‘rtracklayer’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/rtracklayer’
# ERROR: dependency ‘XML’ is not available for package ‘grImport2’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/grImport2’
# ERROR: dependency ‘rtracklayer’ is not available for package ‘BSgenome’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/BSgenome’
# ERROR: dependency ‘BSgenome’ is not available for package ‘rGADEM’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/rGADEM’
# ERROR: dependency ‘rGADEM’ is not available for package ‘MotIV’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/MotIV’
# ERROR: dependencies ‘grImport2’, ‘MotIV’, ‘XML’ are not available for package ‘motifStack’
# * removing ‘/home/ggj/R/x86_64-pc-linux-gnu-library/3.6/motifStack’

library(MotifDb)
MotifDb[1]
MotifDb[[1]]

library(motifStack)
plotMotifLogo(MotifDb[[1]])

matrix.bin <- query(MotifDb, "bin_SANGER")
motif <- new("pfm", mat=matrix.bin[[1]], name=names(matrix.bin)[1])
motifStack(motif)


motifs <- importMatrix("/media/ggj/Files/NvWA/motif/meme_human_conv2.txt", format = "meme")

motifStack(motifs, layout = "tree")

motifStack(motifs, layout="phylog", f.phylog=.15, f.logo=0.25)


motifs2 <- importMatrix("/media/ggj/Files/NvWA/motif/meme_Celegan_conv1.txt", format = "meme", to="pfm")

## format the name
# names(motifs2) <- gsub("(_[\\.0-9]+)*_FBgn\\d+$", "", 
#                        elementMetadata(matrix.fly)$providerName)
names(motifs2) <- gsub("[^a-zA-Z0-9]", "_", names(motifs2))

motifs2 <- motifs2[unique(names(motifs2))]

pfms <- sample(motifs2, 50)

## use MotIV to calculate the distances of motifs
jaspar.scores <- MotIV::readDBScores(file.path(find.package("MotIV"), 
                                               "extdata", 
                                               "jaspar2010_PCC_SWU.scores"))
d <- MotIV::motifDistances(lapply(pfms, pfm2pwm))
hc <- MotIV::motifHclust(d, method="average")
## convert the hclust to phylog object
phylog <- hclust2phylog(hc)
## reorder the pfms by the order of hclust
leaves <- names(phylog$leaves)
pfms <- pfms[leaves]
## create a list of pfm objects
pfms <- mapply(pfms, names(pfms), 
               FUN=function(.pfm, .name){
                 new("pfm", mat=.pfm@mat, name=.name)})
## extract the motif signatures
motifSig <- motifSignature(pfms, phylog, groupDistance=0.01, min.freq=1)


## get the signatures from object of motifSignature
sig <- signatures(motifSig)
## get the group color for each signature
gpCol <- sigColor(motifSig)

library(RColorBrewer)
color <- brewer.pal(12, "Set3")
## plot the logo stack with pile style.
motifPiles(phylog=phylog, pfms=pfms, pfms2=sig, 
            col.tree=rep(color, each=5),
            col.leaves=rep(rev(color), each=5),
            col.pfms2=gpCol, 
            r.anno=c(0.02, 0.03, 0.04), 
            col.anno=list(sample(colors(), 50), 
                          sample(colors(), 50), 
                          sample(colors(), 50)),
            motifScale="logarithmic",
            plotIndex=TRUE,
            groupDistance=0.01)


## plot the logo stack with radial style.
plotMotifStackWithRadialPhylog(phylog=phylog, pfms=sig, 
                              circle=0.4, cleaves = 0.3, 
                              clabel.leaves = 0.5, 
                              col.bg=rep(color, each=5), col.bg.alpha=0.3, 
                              col.leaves=rep(rev(color), each=5),
                              col.inner.label.circle=gpCol, 
                              inner.label.circle.width=0.03,
                              angle=350, circle.motif=1.2, 
                              motifScale="logarithmic")

## plot the logo stack with cirsoc style.
motifCircos(phylog=phylog, pfms=pfms, pfms2=sig, 
            col.tree.bg=rep(color, each=5), col.tree.bg.alpha=0.3, 
            col.leaves=rep(rev(color), each=5),
            col.inner.label.circle=gpCol, 
            inner.label.circle.width=0.03,
            col.outer.label.circle=gpCol, 
            outer.label.circle.width=0.03,
            r.rings=c(0.02, 0.03, 0.04), 
            col.rings=list(sample(colors(), 50), 
                           sample(colors(), 50), 
                           sample(colors(), 50)),
            angle=350, motifScale="logarithmic")