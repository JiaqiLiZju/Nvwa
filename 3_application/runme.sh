### predict whole genome
#for Mouse locus use "Chrom.8.Mouse.bed"
# 8 0   129401212
BASEFILE="Region.Chrom.8.Mouse"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta

# generate 1000nt step with 13Kb window size
bedtools makewindows -b Chrom.8.Mouse.bed -w 13000 -s 2000 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
# extract sequences from fasta of human or mouse genome
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

# predict whole genome
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt
python ../predict_whole_genome.py ../Dataset/Dataset.ChromIV_train_test.h5 ./$BASEFILE.input.txt --reverse_comp

for x in `ls ./Pred-Random/Predict_*.Plus.txt`; do tail -n+2 ${x} | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > `echo ${x}|sed 's/\.txt//g'`.bedGraph; done
for x in {0..101}; do tail -n+2 test_predict_${x}.Minus.txt | perl -ne '@a=($_ =~ /(.*):(\d+)-(\d+)\t(.*)/); print "chr".join("\t", $a[0], $a[1]+6500-500, $a[1]+6500+500, $a[3]), "\n";' > test_predict_${x}.Minus.bedGraph; done
mv test_predict_* bedgraph/.

### compare peaks with ATAC
# grep -v track Predict_Neuron_Mouse3.Plus.bedGraph | cut -f 1-3 > Predict_Neuron_Mouse3.Plus.bed
# macs2 callpeak --nomodel --extsize 2000 -q 0.5 -g mm -f BED -t Predict_Neuron_Mouse3.Plus.bed -n Predict_Neuron_Mouse3.Plus.sorted --outdir ./Predict_Neuron_Mouse3.Plus.peaks/ &>log.macs2
# macs2 bdgpeakcall -c 1 -g 10 -i Predict_Neuron_Mouse3.Plus.bedGraph --outdir peaks -o ./Predict_Neuron_Mouse3.Plus.peaks &>log.macs2
python callpeak.py

# show the prediction using USUC & IGV genome browser
# pyGenomeTracks
# pip install pyGenomeTracks
make_tracks_file --trackFiles \
    ./Pred/Predict_Stromal_Mouse1.Plus.bedGraph \
    ./Pred/Predict_Secretory_Mouse59.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse12.Plus.bedGraph \
    ./Pred/Predict_Immune_Mouse11.Plus.bedGraph \
    ./Pred/Predict_Erythroid_Mouse16.Plus.bedGraph \
    ./Pred/Predict_Epithelial_Mouse13.Plus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse48.Plus.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks.ini
sed -i 's/# where = top/where = top/g' tracks.ini
# sed -i 's/min_value = 0/#min_value = 0/g' tracks.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks.ini
sed -i 's/#tranform = log/transform = log/g' tracks.ini
sed -i 's/#log_pseudocount = 2/log_pseudocount = 4/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks.ini
# sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks.ini
sed -i 's/labels = false/labels = True/g' tracks.ini

# track-plot
pyGenomeTracks --tracks tracks.ini --region chr8:0-129401212 --outFileName tracks.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks.ini --region chr8:10000000-15000000 --outFileName tracks_zoom_pred_celltype.pdf --width 50 --fontSize 5 --dpi 300

pyGenomeTracks --tracks tracks.ini --region chr8:23000000-24500000 --outFileName tracks_zoom_pred_celltype_Zmat.pdf --width 50 --fontSize 5 --dpi 300


make_tracks_file --trackFiles \
    ./Pred/Predict_Endothelial.Plus.bedGraph \
    ./Pred/Predict_Epithelial.Plus.bedGraph \
    ./Pred/Predict_Erythroid.Plus.bedGraph \
    ./Pred/Predict_Germline.Plus.bedGraph \
    ./Pred/Predict_Hepatocyte.Plus.bedGraph \
    ./Pred/Predict_Immune.Plus.bedGraph \
    ./Pred/Predict_Muscle.Plus.bedGraph \
    ./Pred/Predict_Proliferating.Plus.bedGraph \
    ./Pred/Predict_Secretory.Plus.bedGraph \
    ./Pred/Predict_Stromal.Plus.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
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
pyGenomeTracks --tracks tracks_lineage.ini --region chr8:0-129401212 --outFileName tracks_lineage.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_lineage.ini --region chr8:13000000-14000000 --outFileName tracks_zoom_pred_lineage.pdf --width 50 --fontSize 5 --dpi 300



make_tracks_file --trackFiles \
    ./Mouse_sciATAC1/Activated_B_cells-clusters_4-cluster_4.bw \
    ./Mouse_sciATAC1/Alveolar_macrophages-clusters_17-cluster_2.bw \
    ./Mouse_sciATAC1/Astrocytes-clusters_19-cluster_1.bw \
    ./Mouse_sciATAC1/B_cells-clusters_4-cluster_1.bw \
    ./Mouse_sciATAC1/Cardiomyocytes-clusters_7-cluster_1.bw \
    ./Mouse_sciATAC1/Cerebellar_granule_cells-clusters_8-cluster_1.bw \
    ./Mouse_sciATAC1/Dendritic_cells-clusters_17-cluster_1.bw \
    ./Mouse_sciATAC1/Endothelial_I_cells-clusters_22-cluster_1.bw \
    ./Mouse_sciATAC1/Enterocytes-clusters_6-cluster_1.bw \
    ./Mouse_sciATAC1/Ex_neurons_CPN-clusters_5-cluster_1.bw \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks_sciATAC1.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_sciATAC1.ini
sed -i 's/# where = top/where = top/g' tracks_sciATAC1.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_sciATAC1.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_sciATAC1.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_sciATAC1.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks_sciATAC1.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_sciATAC1.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_sciATAC1.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_sciATAC1.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_sciATAC1.ini
sed -i 's/labels = false/labels = True/g' tracks_sciATAC1.ini

# track-plot
pyGenomeTracks --tracks tracks_sciATAC1.ini --region chr8:0-129401212 --outFileName tracks.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks_sciATAC1.ini --region chr8:13000000-14000000 --outFileName tracks_zoom_sciATAC1.pdf --width 50 --fontSize 5 --dpi 300


make_tracks_file --trackFiles \
    ./Mouse_sciATAC1/Endothelial_I_cells-clusters_22-cluster_1.bw \
    ./Mouse_sciATAC1/Endothelial_I_cells-clusters_22-cluster_3.bw \
    ./Mouse_sciATAC1/Endothelial_II_cells-clusters_23-cluster_1.bw \
    ./Mouse_sciATAC1/Endothelial_II_cells-clusters_25-cluster_2.bw \
    ./Mouse_sciATAC1/Endothelial_II_cells-clusters_25-cluster_3.bw \
    ./Pred/Predict_Endothelial_Mouse48.Plus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse50.Plus.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks.ini

# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks.ini
sed -i 's/# where = top/where = top/g' tracks.ini
# sed -i 's/min_value = 0/#min_value = 0/g' tracks.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks.ini
# sed -i 's/#tranform = log/transform = log1p/g' tracks.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks.ini
sed -i 's/height = 5/height = 3/g' tracks.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks.ini
sed -i 's/labels = false/labels = True/g' tracks.ini

# track-plot
pyGenomeTracks --tracks tracks.ini --region chr8:0-129401212 --outFileName tracks.pdf --width 50 --fontSize 5 --dpi 300
pyGenomeTracks --tracks tracks.ini --region chr8:109500000-110200000 --outFileName tracks_zoom.pdf --width 50 --fontSize 5 --dpi 300


# track plot of Neuron
make_tracks_file --trackFiles \
    ./Mouse_sciATAC1/Ex_neurons_CThPN-clusters_5-cluster_3.mm10.bw \
    ./Mouse_sciATAC1/Ex_neurons_CPN-clusters_5-cluster_1.mm10.bw \
    ./Mouse_sciATAC1/Ex_neurons_SCPN-clusters_29-cluster_1.mm10.bw \
    ./Pred/Predict_Neuron_Mouse54.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse54.Minus.bedGraph \
    ./Pred/Predict_Neuron_Mouse74.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse74.Minus.bedGraph \
    ./Pred/Predict_Neuron_Mouse12.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse12.Minus.bedGraph \
    ./Pred/Predict_Neuron_Mouse3.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse3.Minus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse48.Plus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse48.Minus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse50.Plus.bedGraph \
    ./Pred/Predict_Endothelial_Mouse50.Minus.bedGraph \
    ./Pred/Predict_Epithelial_Mouse4.Plus.bedGraph \
    ./Pred/Predict_Epithelial_Mouse4.Minus.bedGraph \
    ./Pred/Predict_Epithelial_Mouse13.Plus.bedGraph \
    ./Pred/Predict_Epithelial_Mouse13.Minus.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
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

pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:0-129401212 --outFileName tracks_Neuron.pdf --width 50 --fontSize 5 --dpi 100
pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:10000000-15000000 --outFileName tracks_Neuron_zoom2.pdf --width 50 --fontSize 5 --dpi 300


make_tracks_file --trackFiles \
    ./Mouse_sciATAC1/Ex_neurons_CThPN-clusters_5-cluster_3.bw \
    ./Mouse_sciATAC1/Ex_neurons_CPN-clusters_5-cluster_1-mm10.bw \
    ./Pred/Predict_Neuron_Mouse12.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse3.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse54.Plus.bedGraph \
    ./Pred/Predict_Neuron_Mouse74.Plus.bedGraph \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks_Neuron.ini

sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/# where = top/where = top/g' tracks_Neuron.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Neuron.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Neuron.ini
sed -i 's/#tranform = log/transform = log1p/g' tracks_Neuron.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks_Neuron.ini
sed -i 's/height = 5/height = 3/g' tracks_Neuron.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Neuron.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Neuron.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Neuron.ini

pyGenomeTracks --tracks tracks_Neuron.ini --region chr8:13500000-14000000 --outFileName tracks_Neuron_zoom2_mm10.pdf --width 50 --fontSize 5 


# BCL11A,60451167,60553567,-,2
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

# Sox9,112 782 224,112787760,+,11
BASEFILE="Region.Sox9.Mouse"
GENOME=/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta

# generate 1000nt step with 13Kb window size
bedtools makewindows -b Sox9.Mouse.bed -w 13000 -s 200 | perl -ne '@a=split/\t/; print $_ if $a[2]-$a[1] == 13000;' | uniq >$BASEFILE.bed
# extract sequences from fasta of human or mouse genome
bedtools getfasta -tab -fi $GENOME -bed $BASEFILE.bed -fo $BASEFILE.input.txt

make_tracks_file --trackFiles ./Pred_Sox9/Predict_Germline*.bedGraph \
    ./Mouse_sciATAC1/Sperm-clusters_14-cluster_*.mm10.bw.bw \
    ../../../NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf \
    -o tracks_Sox9.ini
    
# modify tracks.ini
sed -i 's/#fontsize = 20/fontsize = 6/g' tracks_Sox9.ini
sed -i 's/# where = top/where = top/g' tracks_Sox9.ini
sed -i 's/min_value = 0/#min_value = 0/g' tracks_Sox9.ini
sed -i 's/# rasterize = true/rasterize = true/g' tracks_Sox9.ini
# sed -i 's/#tranform = log/transform = log1p/g' tracks_Sox9.ini
# sed -i 's/#log_pseudocount = 2/log_pseudocount = 2/g' tracks_Sox9.ini
sed -i 's/height = 5/height = 3/g' tracks_Sox9.ini
sed -i 's/fontsize = 10/fontsize = 6/g' tracks_Sox9.ini
sed -i 's/#style = UCSC/style = UCSC/g' tracks_Sox9.ini
sed -i 's/# prefered_name = gene_name/prefered_name = gene_name/g' tracks_Sox9.ini
sed -i 's/# merge_transcripts = true/merge_transcripts = true/g' tracks_Sox9.ini
sed -i 's/labels = false/labels = True/g' tracks_Sox9.ini

pyGenomeTracks --tracks tracks_Sox9.ini --region chr11:111100000-113000000 --outFileName tracks_Sox9.pdf --width 50 --fontSize 5 
pyGenomeTracks --tracks tracks_Sox9.ini --region chr11:112215000-112220000 --outFileName tracks_Sox9_zoom.pdf --width 50 --fontSize 5 
112,782,224 - 565,000 = 112,217,224
112,217,224 + 557     = 112,217,781