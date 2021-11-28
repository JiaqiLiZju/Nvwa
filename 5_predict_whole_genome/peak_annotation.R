setwd("/media/ggj/Files/mount/NvWA_Final/Predict_whole_genome/code_test/")

# 然后再打开的命令中设置清华源信息
# options("repos" = c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
# options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor")

# 放在前面的话：一般要安装什么包直接搜索包名，对应的包的 manual 说怎么装就怎么装
# BiocManager::install("TxDb.Mmusculus.UCSC.mm10.knownGene")
# BiocManager::install("org.Mm.eg.db")
# BiocManager::install("ChIPseeker")
# BiocManager::install("clusterProfiler")

# 加载包
require(ChIPseeker)
require(clusterProfiler) 

require(TxDb.Mmusculus.UCSC.mm10.knownGene)
require(org.Mm.eg.db)

txdb <- TxDb.Mmusculus.UCSC.mm10.knownGene

# 读取当前目录 oldBedFiles 文件夹下的 Peak 文件
bedPeaksFile = './GSE60192_39_distal_CBP+H3K4Me1_from_Kim_et_al.bed.gz'; 
covplot(bedPeaksFile)

# 比如要画第4、5个文件（MACS生成的BED文件包含常规的5列）
# peak <- GenomicRanges::GRangesList(QU=readPeakFile('./peaks_quantile.bed'),
#                                 NP=readPeakFile('./peak_null_promoter.bed'))
# covplot(peak, weightCol="V3") + facet_grid(chr ~ .id)

# 查看 Peak 文件中染色体信息
peak <- readPeakFile( bedPeaksFile) 
seqlevels(peak)

# 过滤掉带有 Het 字眼的染色体
# 请留意 `grepl()` 函数
# keepChr = !grepl('Het',seqlevels(peak)) 
# seqlevels(peak, pruning.mode = "coarse") <- seqlevels(peak)[keepChr]

# 默认下游的范围是3kb，但是可以自己调整,调成500
# options(ChIPseeker.downstreamDistance = 500)
# 依然是设置options，用于总结结果
# options(ChIPseeker.ignore_1st_exon = T)
# options(ChIPseeker.ignore_1st_intron = T)
# options(ChIPseeker.ignore_downstream = T)
# options(ChIPseeker.ignore_promoter_subcategory = T)

# 使用 annotatPeak 进行注释，
peakAnno <- annotatePeak(peak, tssRegion = c(-5000, 5000), TxDb = txdb) 
peakAnno

# 转变成 data.frame 格式文件，方便查看与后续操作
peakAnno_df <- as.data.frame(peakAnno)
write.csv(peakAnno_df, quote = F, file = "./peaks_quantile.anno.Mouse.csv")

pdf("peaks_quantile.anno.Mouse.pdf", width = 10, height = 4)
plotAnnoPie(peakAnno)
dev.off()

pdf("peaks_quantile.anno.Mouse.Bar.pdf", width = 10, height = 4)
plotAnnoBar(peakAnno)
dev.off()

pdf("peaks_quantile.anno.Mouse.upsetplot.pdf", width = 10, height = 4)
upsetplot(peakAnno, vennpie=TRUE)
dev.off()

###########################################################################
# peak在某个窗口的结合谱图
promoter <- getPromoters(TxDb=txdb, upstream=3000, downstream=3000)
tagMatrix <- getTagMatrix(bedPeaksFile, windows=promoter)
tagHeatmap(tagMatrix, xlim=c(-3000, 3000), color="red")

plotAvgProf(tagMatrix, xlim=c(-3000, 3000),
            xlab="Genomic Region (5'->3')", 
            ylab = "Read Count Frequency")

# 支持多个数据比较
# tagMatrixList <- lapply(files, getTagMatrix, windows=promoter)

# 添加置信区间并分面
# plotAvgProf(tagMatrixList, xlim=c(-3000, 3000), 
#             conf=0.95,resample=500, facet="row")
