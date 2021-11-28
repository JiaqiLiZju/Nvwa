setwd("/media/ggj/Files/mount/NvWA_Final/Predict_whole_genome/code_test/")

require(ChIPseeker)
require(clusterProfiler) 

require(TxDb.Mmusculus.UCSC.mm10.knownGene)
require(org.Mm.eg.db)
txdb <- TxDb.Mmusculus.UCSC.mm10.knownGene

###########################################################################
Peakfiles <- c('./Peak/peak_random_location.bed',
                '../resource_fantom/F5.mm10.enhancers.bed.gz'
               )
Peaks <- lapply(Peakfiles, function(i){
  df = read.csv(i, sep='\t', header = F)
  df = df[df$V1=='chr8',c(1,2,3)]
  colnames(df) <- c("chr","start","end")
  gr <- with(df, GRanges(chr, IRanges(start, end)))
  return(gr)})

# peak overlap venn
peakAnnoList <- lapply(Peaks, annotatePeak, TxDb=txdb)
names(peakAnnoList) <- c("peaks_quantile", "Mouse_sciATAC1-Neuron1") #names(peakfiles)
genes= lapply(peakAnnoList, function(i) as.data.frame(i)$geneId)
vennplot(genes)

###########################################################################
queryPeak <- readPeakFile('./Peak/peaks_quantile.bed')
target_files <- c('../Mouse_sciATAC1/mm10-peak-macs2/peak-Ex_neurons_CPN-clusters_5-cluster_1.mm10.bedGraph/Ex_neurons_CPN-clusters_5-cluster_1.mm10.bedGraph_peaks.narrowPeak',
                  '../Mouse_sciATAC1/mm10-peak-macs2/peak-Ex_neurons_CThPN-clusters_5-cluster_3.mm10.bedGraph/Ex_neurons_CThPN-clusters_5-cluster_3.mm10.bedGraph_peaks.narrowPeak',
                  '../resource_encode/cCRE-forbrain-ENCFF023NLD.bed.gz'
)
targetPeaks <- lapply(target_files, function(i){
  df = read.csv(i, sep='\t', header = F)
  # df = df[df$V1=='chr8',c(1,2,3)]
  df = df[,c(1,2,3)]
  colnames(df) <- c("chr","start","end")
  gr <- with(df, GRanges(chr, IRanges(start, end)))
  return(gr)})

# enrich peak overlap 
overlap_df = enrichPeakOverlap(queryPeak, targetPeaks, TxDb = txdb)
overlap_df
