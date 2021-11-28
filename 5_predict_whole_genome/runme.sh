### predict whole genome
#for Mouse locus use "Chrom.8.Mouse.bed"
# 8 0   129401212
BASEFILE="Region.Chrom.8.Mouse"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf

# generate 1000nt step with 13Kb window size
bedtools makewindows -b Chrom.8.Mouse.bed -w 13000 -s 1000 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
# extract sequences from fasta of human or mouse genome
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

# predict whole genome
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

for x in `ls ./Pred_Mouse/predicted_tracks/Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Pred_Mouse/predicted_tracks/Predict_*.Minus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done

### compare peaks with ATAC
# grep -v track Predict_Neuron_Mouse3.Plus.bedGraph | cut -f 1-3 > Predict_Neuron_Mouse3.Plus.bed
# macs2 callpeak --nomodel --extsize 2000 -q 0.5 -g mm -f BED -t Predict_Neuron_Mouse3.Plus.bed -n Predict_Neuron_Mouse3.Plus.sorted --outdir ./Predict_Neuron_Mouse3.Plus.peaks/ &>log.macs2
# macs2 bdgpeakcall -c 1 -g 10 -i Predict_Neuron_Mouse3.Plus.bedGraph --outdir peaks -o ./Predict_Neuron_Mouse3.Plus.peaks &>log.macs2
python ./callpeak.py predicted_tracks/Predict_Neuron_Mouse12.Plus.bedGraph predicted_null
for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do python ../callpeak.py $x predicted_null; done
# forx in `ls Predict*`; do mv $x `echo ${x}|sed 's/Predict/Null/g'`; done

# # peak number
# totalPeaks=$(cat Peak/peak_null_promoter.bed |wc -l)
# echo '==> peaks:' $totalPeaks

# # peak overlap
# bedtools intersect -wo -a Peak/peak_null_promoter.bed -b ../resource_fantom/F5.mm10.enhancers.bed.gz > overlaps.bed
# Overlaped=$(cat overlaps.bed |wc -l)
# # Overlaped=$(bedtools intersect -a Peak/peak_null_promoter.bed -b ../resource_fantom/F5.mm10.enhancers.bed.gz |wc -l|awk '{print $1}')
# echo '==> Overlaped peaks:' $Overlaped
# echo '==> OR value:' $(bc <<< "scale=2;100*$Overlaped/$totalPeaks")'%'

# # Sort peak by -log10(p-value)
# sort -k8,8nr NAME_OF_INPUT_peaks.narrowPeak > macs/NAME_FOR_OUPUT_peaks.narrowPeak
# # idr
# idr --samples Peak/peak_null_promoter.bed ../resource_fantom/F5.mm10.enhancers.bed.gz --input-file-type bed \
#     --output-file Peak-idr --plot --log-output-file Peak.idr.log

# calculate correlation using deeptools
for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do bedGraphToBigWig $x ../Mouse_Region/chrom.sizes ${x}.bw; done
for x in `ls ./predicted_null/predicted_tracks/Null_*.Mean.bedGraph`; do bedGraphToBigWig $x ../Mouse_Region/chrom.sizes ${x}.bw; done
multiBigwigSummary bins -b \
    ../predicted_tracks/Predict_Neuron_Mouse3.Mean.bedGraph.bw \
    ../predicted_tracks/Predict_Neuron_Mouse74.Mean.bedGraph.bw \
    ../predicted_tracks/Predict_Neuron_Mouse12.Mean.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Neuron_Mouse3.Mean.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Neuron_Mouse74.Mean.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Neuron_Mouse12.Mean.bedGraph.bw \
    ../../resource_encode/Neuron/*_mm10_chr8.bigWig \
    ../../Mouse_sciATAC1/mm10-chr8-bw/CThPN-clusters_5-cluster_3.mm10.chr8.bw \
    ../../Mouse_sciATAC1/mm10-chr8-bw/SCPN-clusters_29-cluster_1.mm10.bw \
    ../../Mouse_sciATAC1/mm10-chr8-bw/CPN-clusters_5-cluster_1.mm10.bw \
    ../../Mouse_sciATAC1/mm10-chr8-bw/Inhibitory_neurons-clusters_15-cluster_2.mm10_chr8.bw \
    -o results_Neuron.npz

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Neuron.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts_Neuron.tab

# plotCorrelation \
#     -in results_Neuron.npz \
#     --corMethod spearman --skipZeros --removeOutliers \
#     --plotTitle "Spearman Correlation of Average Scores Per Transcript" \
#     --whatToPlot scatterplot \
#     -o scatterplot_Spearmanr_bigwigScores_Neuron.png   \
#     --outFileCorMatrix SpearmanCorr_bigwigScores.tab


multiBigwigSummary bins -b \
    ../predicted_tracks/Predict_Immune_Mouse*.Mean.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Immune_Mouse*.Mean.bedGraph.bw \
    ../../Mouse_sciATAC1/mm10-chr8-bw/Macrophages-clusters_16-cluster_2.mm10_chr8.bw \
    -o results_Macrophages.npz

plotCorrelation \
    -in results_Macrophages.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Macrophages.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts_Macrophages.tab

multiBigwigSummary bins -b \
    ../predicted_tracks/Predict_Epithelial_Mouse*.Mean.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Epithelial_Mouse*.Mean.bedGraph.bw \
    ../../Mouse_sciATAC1/mm10-chr8-bw/Type_II_pneumocytes-clusters_30-cluster_1.mm10_chr8.bw \
    -o results_Epithelial.npz

plotCorrelation \
    -in results_Epithelial.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Epithelial.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts_Epithelial.tab

###  show the prediction using USUC & IGV genome browser
# pyGenomeTracks
# pip install pyGenomeTracks
# track plot of Neuron
make_tracks_file --trackFiles \
    ../Mouse_sciATAC1/mm10-bw/Ex_neurons_CThPN-clusters_5-cluster_3.mm10.bw \
    ../Mouse_sciATAC1/mm10-bw/Ex_neurons_CPN-clusters_5-cluster_1.mm10.bw \
    ../Mouse_Epigenetic_Neuron_GSE60192/GSE60192_01_04_B1B2merged_un_H3K27Ac_ab4729_E300_track.mm10.bw \
    ../resource_encode/*.bw \
    ./predicted_tracks/Predict_Neuron_Mouse54.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse54.Minus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse74.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse74.Minus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse12.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse12.Minus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse3.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Mouse3.Minus.bedGraph \
    $GTF \
    ../resource_fantom/F5.mm10.enhancers.bed.gz \
    ../resource_fantom/mm10_liftover+new_CAGE_peaks_phase1and2.bed.gz \
    ../resource_encode/cCRE-forbrain-ENCFF023NLD.bed.gz \
    ../resource_encode/ChromHMM-ENCFF499HWY.bed.gz \
    ../database_TF/ORegAnno_mm10.bed \
    -o tracks_Neuron.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/# where = top/where = top/g' tracks_Neuron.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Neuron.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Neuron.ini
# sed -i 's/#tranform = log/transform = log1p/g' tracks_Neuron.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks_Neuron.ini
sed -i 's/height = 5/height = 3/g' tracks_Neuron.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Neuron.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Neuron.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_Neuron.ini
sed -i 's/labels = false/labels = True/g' tracks_Neuron.ini

pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:0-129400000 --outFileName tracks_Neuron.pdf --width 50 --fontSize 5 --dpi 100
pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:0-25880000 --outFileName tracks_Neuron_zoom1.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:25880000-51760000 --outFileName tracks_Neuron_zoom2.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:51760000-77640000 --outFileName tracks_Neuron_zoom3.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:77640000-129400000 --outFileName tracks_Neuron_zoom4.pdf --width 50 --fontSize 5 --dpi 300

pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:5050000-5090000 --outFileName tracks_Neuron_zoom_5srRNA.pdf --width 50 --fontSize 5 --dpi 300


###  Sox9,112 782 224,112787760,+,11
BASEFILE="Region.Sox9.Mouse"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf

# generate 1000nt step with 13Kb window size
bedtools makewindows -b Sox9.Mouse.bed -w 13000 -s 200 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
# extract sequences from fasta of human or mouse genome
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

make_tracks_file --trackFiles ./Pred_Sox9/Predict_Germline*.bedGraph \
    ./Mouse_sciATAC1/Sperm-clusters_14-cluster_*.mm10.bw.bw \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks_Sox9.ini

pyGenomeTracks --tracks tracks_Sox9.ini --region chr11:111100000-113000000 --outFileName tracks_Sox9.pdf --width 50 --fontSize 5 
pyGenomeTracks --tracks tracks_Sox9.ini --region chr11:112215000-112220000 --outFileName tracks_Sox9_zoom.pdf --width 50 --fontSize 5 










### for Human
# BASEFILE="Region.Chrom8.Human"
BASEFILE="Region.Human"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.fa
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.gtf

bedtools makewindows -b $BASEFILE -w 13000 -s 1000 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

split -l 100000 $BASEFILE.input.txt $BASEFILE.input.Split_

python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

cd Pred_EarthWorm/predicted_tracks
for x in `ls ./Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Predict_*.Minus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Predict_*.Mean.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done

python ./callpeak.py predicted_tracks/Predict_Neuron_Mouse12.Plus.bedGraph predicted_null
for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do python ../callpeak.py $x predicted_null; done

# calculate correlation using deeptools
for x in `ls ./predicted_tracks/Predict_Neuron*.Mean.bedGraph`; do bedGraphToBigWig $x ../Human_Region/chrom.sizes ${x}.bw; done
for x in `ls ./predicted_null/predicted_tracks/Null_*.Mean.bedGraph`; do bedGraphToBigWig $x ../Human_Region/chrom.sizes ${x}.bw; done
mkdir result_correlation/ && cd result_correlation/
multiBigwigSummary bins -b \
    ../predicted_tracks/Predict_Neuron*.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_Neuron*.bedGraph.bw \
    ../../resource_encode_hg38/*.hg38-chr8.bw    \
    -o results_Neuron.npz

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Neuron.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts_Neuron.tab

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod spearman --skipZeros --removeOutliers \
    --plotTitle "Spearman Correlation of Average Scores Per Transcript" \
    --whatToPlot scatterplot \
    -o scatterplot_Spearmanr_bigwigScores_Neuron.png   \
    --outFileCorMatrix PearsonCorr_bigwigScores.tab

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod pearson --skipZeros --removeOutliers \
    --plotTitle "pearson Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_pearsonCorr_readCounts_Neuron.pdf   \
    --outFileCorMatrix PearsonCorr_readCounts_Neuron.tab

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod pearson --skipZeros --removeOutliers \
    --plotTitle "Pearson Correlation of Average Scores Per Transcript" \
    --whatToPlot scatterplot \
    -o scatterplot_PearsonCorr_bigwigScores_Neuron.png   \
    --outFileCorMatrix PearsonCorr_bigwigScores_Neuron.tab


## show the prediction using USUC & IGV genome browser & pyGenomeTracks
make_tracks_file --trackFiles \
    ../predicted_tracks/Predict_Neuron*.bedGraph \
    ../../resource_encode_hg38/*.hg38-chr8.bw \
    $GTF \
    -o tracks_Neuron.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/# where = top/where = top/g' tracks_Neuron.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Neuron.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Neuron.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_Neuron.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_Neuron.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Neuron.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Neuron.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_Neuron.ini
sed -i 's/labels = false/labels = True/g' tracks_Neuron.ini

# track-plot
pyGenomeTracks --tracks tracks_Neuron.ini --region 8:0-145138636 --outFileName predicted_tracks_chr8.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region 8:20000000-25000000 --outFileName predicted_tracks_chr8_zoom.pdf --width 50 --fontSize 5 --dpi 300


###  BCL11A,60451167,60553567,-,2
BASEFILE="Region.BCL11A.Human"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.fa

# generate 1000nt step with 13Kb window size
bedtools makewindows -b BCL11A.Human.bed -w 13000 -s 200 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
# extract sequences from fasta of human or mouse genome
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

make_tracks_file --trackFiles ./Pred_BCL11A/Pred_BCL11A_Erythroid_*.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.gtf \
    -o tracks_BCL11A.ini
# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_BCL11A.ini
sed -i 's/# where = top/where = top/g' tracks_BCL11A.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_BCL11A.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_BCL11A.ini
# sed -i 's/#tranform = log/transform = log1p/g' tracks_BCL11A.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks_BCL11A.ini
sed -i 's/height = 5/height = 3/g' tracks_BCL11A.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_BCL11A.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_BCL11A.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_BCL11A.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_BCL11A.ini
sed -i 's/labels = false/labels = True/g' tracks_BCL11A.ini

pyGenomeTracks --tracks tracks_BCL11A.ini --region chr2:60400000-60600000 --outFileName tracks_BCL11A.pdf --width 50 --fontSize 5 
pyGenomeTracks --tracks tracks_BCL11A.ini --region chr2:60490000-60500000 --outFileName tracks_BCL11A_zoom.pdf --width 50 --fontSize 5 


### for Dmel
# cat Drosophila_melanogaster_genomic.fasta.fai |grep -v 2110000 |grep -v Unmapped |grep -v Scaffold |cut -f1,2 |awk -F'\t' '{print $1"\t0\t"$2}'> Region.Dmel
BASEFILE="Region.Dmel"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/D.mel/Drosophila_melanogaster_genomic.fasta
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/D.mel/Drosophila_melanogaster_genomic.gtf

bedtools makewindows -b $BASEFILE -w 13000 -s 1000 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

cd Pred_EarthWorm/predicted_tracks
for x in `ls ./Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-5000, $a[1]+6500+5000, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Predict_*.Minus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-5000, $a[1]+6500+5000, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done

## show the prediction using USUC & IGV genome browser & pyGenomeTracks
make_tracks_file --trackFiles \
    ./predicted_tracks/Predict_Neuron_Drosophila40.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Drosophila40.Minus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Drosophila10.Plus.bedGraph \
    ./predicted_tracks/Predict_Neuron_Drosophila10.Minus.bedGraph \
    ../resource_encode_Dmel/* \
    $GTF \
    -o tracks_lineage.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_lineage.ini
sed -i 's/# where = top/where = top/g' tracks_lineage.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_lineage.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_lineage.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_lineage.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_lineage.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_lineage.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_lineage.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_lineage.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_lineage.ini
sed -i 's/labels = false/labels = True/g' tracks_lineage.ini

# track-plot
pyGenomeTracks --tracks tracks_lineage.ini --region 4:0-1348131 --outFileName predicted_tracks_chr4.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region 4:548131-848131 --outFileName predicted_tracks_chr4_zoom.pdf --width 50 --fontSize 5 --dpi 300

python ../callpeak.py predicted_tracks/Predict_Neuron_Mouse12.Plus.bedGraph predicted_null



### for earthworm
cd /media/ggj/Files/mount/NvWA_Final/Predict_whole_genome/Pred_EarthWorm

# cat Earthworm.fasta.fai |cut -f1,2 |awk -F'\t' '{print $1"\t0\t"$2}'> Region.Earthworm
BASEFILE="Region.Earthworm"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/EarthWorm/Earthworm.fasta
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/EarthWorm/Earthworm.gtf

bedtools makewindows -b Region.Earthworm -w 13000 -s 10000 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

cd Pred_EarthWorm/predicted_tracks
for x in `ls ./Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-5000, $a[1]+6500+5000, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Predict_*.Minus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-5000, $a[1]+6500+5000, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done

## calculate correlation using deeptools
bedGraphToBigWig ./predicted_tracks_denoised/Predict_EarthWorm2.Plus.bedGraph ../EarthWorm_Region/chrom.sizes ./predicted_tracks_denoised/Predict_EarthWorm2.Plus.bw
bedGraphToBigWig ./predicted_tracks_denoised/Predict_EarthWorm25.Plus.bedGraph ../EarthWorm_Region/chrom.sizes ./predicted_tracks_denoised/Predict_EarthWorm25.Plus.bw
multiBigwigSummary bins \
    -b ./predicted_tracks_denoised/Predict_EarthWorm25.Plus.bw \
    ./predicted_tracks_denoised/Predict_EarthWorm2.Plus.bw \
    ../scATAC-EW/EW.SeqDepthNorm.bw \
    ../scATAC-EW/EW-CH.SeqDepthNorm.bw \
    ../scATAC-EW/EW.star_gene_exon_tagged.SeqDepthNorm.bw \
    -o results.npz

plotCorrelation \
    -in results.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_stride1k_ModelDenoise.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts.tab


## show the prediction using USUC & IGV genome browser & pyGenomeTracks
make_tracks_file --trackFiles \
    ./predicted_tracks_denoised/Predict_EarthWorm12.Plus.bedGraph \
    ./predicted_tracks_denoised/Predict_EarthWorm12.Minus.bedGraph \
    ./predicted_tracks_denoised/Predict_EarthWorm25.Plus.bedGraph \
    ./predicted_tracks_denoised/Predict_EarthWorm25.Minus.bedGraph \
    ../scATAC-EW/*.SeqDepthNorm.bw \
    -o tracks_lineage.ini
    # $GTF \

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_lineage.ini
sed -i 's/# where = top/where = top/g' tracks_lineage.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_lineage.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_lineage.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_lineage.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_lineage.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_lineage.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_lineage.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_lineage.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_lineage.ini
sed -i 's/labels = false/labels = True/g' tracks_lineage.ini

# track-plot
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000001:0-159027471 --outFileName predicted_tracks_denoised_chr1.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000002:0-139115577 --outFileName predicted_tracks_denoised_chr2.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000003:0-138077151 --outFileName predicted_tracks_denoised_chr3.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000004:0-123808561 --outFileName predicted_tracks_denoised_chr4.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000005:0-111976267 --outFileName predicted_tracks_denoised_chr5.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000006:0-88474218 --outFileName predicted_tracks_denoised_chr6.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000007:0-86638540 --outFileName predicted_tracks_denoised_chr7.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000008:0-78931488 --outFileName predicted_tracks_denoised_chr8.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000009:0-74019141 --outFileName predicted_tracks_denoised_chr9.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000010:0-73678751 --outFileName predicted_tracks_denoised_chr10.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000011:0-55753002 --outFileName predicted_tracks_denoised_chr11.pdf --width 50 --fontSize 5 --dpi 300

pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000001:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr1.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000002:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr2.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000003:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr3.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000004:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr4.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000005:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr5.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000006:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr6.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000007:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr7.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000008:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr8.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000009:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr9.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000010:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr10.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region GWHACBE00000011:20000000-25000000 --outFileName predicted_tracks_denoised_zoom_chr11.pdf --width 50 --fontSize 5 --dpi 300

python ./callpeak.py predicted_tracks/Predict_Neuron_Mouse12.Plus.bedGraph predicted_null


### for Smed
cd /media/ggj/Files/mount/NvWA_Final/Predict_whole_genome/Smed_Region
# cat SmedAsxl_genome_v1.1.nt.fai |cut -f1,2 |awk -F'\t' '{print $1"\t0\t"$2}'> Region.Smed
# cat Region.Smed |cut -f 1,3 >chrom.sizes
BASEFILE="Region.Smed.C15Marker"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Smed_genome/SmedAsxl_genome_v1.1.nt
GTF=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Smed_genome/smed.gtf

bedtools makewindows -b $BASEFILE -w 13000 -s 100 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt
split -l 100000 Region.Smed.input.txt Region.Smed.input.Split_

python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

cd Pred_Smed/predicted_tracks
for x in `ls ./Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in `ls ./Predict_*.Minus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done

for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do python ../callpeak.py $x predicted_null; done

## calculate correlation using deeptools
for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do xnew=`echo ${x}|sed 's/Mean/Mean_sort/g'` && sort -k1,1 -k2,2n $x > $xnew && bedGraphToBigWig $xnew ../Smed_Region/chrom.sizes ${xnew}.bw; done
for x in `ls ./predicted_null/predicted_tracks/Null_*.Mean.bedGraph`; do xnew=`echo ${x}|sed 's/Mean/Mean_sort/g'` && sort -k1,1 -k2,2n $x > $xnew && bedGraphToBigWig $xnew ../Smed_Region/chrom.sizes ${xnew}.bw; done
bedGraphToBigWig ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_treat_pileup.bdg ../../Smed_Region/chrom.sizes ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_treat_pileup.bdg.bw
bedGraphToBigWig ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_treat_pileup.bdg ../../Smed_Region/chrom.sizes ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_treat_pileup.bdg.bw
mkdir result_correlation/ && cd result_correlation/
multiBigwigSummary bins -b \
    ../../scATAC-Smed/WOC*.SeqDepthNorm.bw \
    ../predicted_tracks/Predict_C*_Neural.Mean_sort.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_C*_Neural.Mean_sort.bedGraph.bw \
    -o results_Neuron.npz

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Neuron.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts.tab

plotCorrelation \
    -in results_Neuron.npz \
    --corMethod pearson --skipZeros --removeOutliers \
    --plotTitle "pearson Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_pearsonCorr_readCounts_Neuron.pdf \
    --outFileCorMatrix PearsonCorr_readCounts_Neuron.tab

## show the prediction using USUC & IGV genome browser & pyGenomeTracks
make_tracks_file --trackFiles \
    ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_treat_pileup.bedgraph \
    ../../scATAC-Smed/snapATAC-Smed/Peaks/scATAC_Smed_C15_Neuron/scATAC_Smed_C15_Neuron_control_lambda.bedgraph \
    ../../scATAC-Smed/WOC1.SeqDepthNorm.bw \
    ../../scATAC-Smed/WOC2.SeqDepthNorm.bw \
    ../../scATAC-Smed/WOC3.SeqDepthNorm.bw \
    ../../scATAC-Smed/WOC4.SeqDepthNorm.bw \
    ../../scATAC-Smed/WOC_barcode_C15_sort.SeqDepthNorm.bedGraph \
    ../predicted_tracks/Predict_C1_Neural.*.bedGraph \
    $GTF \
    -o tracks_Neuron.ini 
    
    
# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/# where = top/where = top/g' tracks_Neuron.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Neuron.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Neuron.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_Neuron.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_Neuron.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Neuron.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Neuron.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_Neuron.ini
sed -i 's/labels = false/labels = True/g' tracks_Neuron.ini

# track-plot
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold4224:0-79514 --outFileName scaffold4224_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold4296:0-75533 --outFileName scaffold4296_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold5479:0-127070 --outFileName scaffold5479_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold3878:0-115171 --outFileName scaffold3878_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold1169:0-266986 --outFileName scaffold1169_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold4666:0-68914 --outFileName scaffold4666_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold10086:0-123513 --outFileName scaffold10086_Neuron.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neuron.ini --region scaffold10064:0-18556 --outFileName scaffold10064_Neuron.pdf --width 50 --fontSize 5 --dpi 300


multiBigwigSummary bins -b \
    ../../scATAC-Smed/WOC*.SeqDepthNorm.bw \
    ../predicted_tracks/Predict_C*_Neoblast.Mean_sort.bedGraph.bw \
    ../predicted_null/predicted_tracks/Null_C*_Neoblast.Mean_sort.bedGraph.bw \
    -o results_Neoblast.npz

plotCorrelation \
    -in results_Neoblast.npz \
    --corMethod spearman --skipZeros \
    --plotTitle "Spearman Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_SpearmanCorr_readCounts_Neoblast.pdf   \
    --outFileCorMatrix SpearmanCorr_readCounts.tab

plotCorrelation \
    -in results_Neoblast.npz \
    --corMethod pearson --skipZeros --removeOutliers \
    --plotTitle "pearson Correlation of Read Counts" \
    --whatToPlot heatmap --colorMap RdYlBu --plotNumbers \
    -o heatmap_pearsonCorr_readCounts_Neoblast.pdf \
    --outFileCorMatrix PearsonCorr_readCounts_Neoblast.tab

macs2 callpeak -t ./combined.bed -f BED -g 13000000000 --nomodel --shift 37 --ext 73 --qval 1e-2 -B --SPMR --call-summits -n scATAC_Smed_C15_Neuron

## show the prediction using USUC & IGV genome browser & pyGenomeTracks
make_tracks_file --trackFiles \
    ../../scATAC-Smed/WOC*.SeqDepthNorm.bw \
    ../predicted_tracks/Predict_C*_Neoblast.Mean_sort.bedGraph.bw \
    ../predicted_tracks/Predict_C*_Neoblast.Mean.bedGraph \
    ../predicted_tracks/Predict_C*_Neoblast.Plus.bedGraph \
    ../predicted_tracks/Predict_C*_Neoblast.Minus.bedGraph \
    -o tracks_Neoblast.ini 
    # $GTF 

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Neoblast.ini
sed -i 's/# where = top/where = top/g' tracks_Neoblast.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Neoblast.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Neoblast.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_Neoblast.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_Neoblast.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Neoblast.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Neoblast.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Neoblast.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_Neoblast.ini
sed -i 's/labels = false/labels = True/g' tracks_Neoblast.ini

# track-plot
pyGenomeTracks --tracks tracks_Neoblast.ini --region scaffold21:0-260345 --outFileName scaffold21_Neoblast.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neoblast.ini --region scaffold23:0-100000 --outFileName scaffold23_Neoblast.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_Neoblast.ini --region scaffold7446:0-561146 --outFileName scaffold7446_Neoblast.pdf --width 50 --fontSize 5 --dpi 300

for x in `ls ./predicted_tracks/Predict_*.Mean.bedGraph`; do python ../callpeak.py $x predicted_null; done


# findMotifsGenome.pl ./WOC.sorted_peaks.narrowPeak $GENOME ./Homer_Motif &>Homer_Motif/run.homer.log