# run explainer on CV models
explainer.ipynb

# motif database
MOTIFDB=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/JASPAR/JASPAR2018_CORE_vertebrates_non-redundant.meme
MOTIF_ANNO=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/JASPAR/motif.anno.tsv
# cat $MOTIFDB |grep MOTIF >MOTIF_ANNO

# motif annotation
tomtom -oc tomtom_conv1 -thresh 0.5 ./meme_conv1.txt \
    $MOTIFDB &>log.tomtom_conv1 &

# motif location enrichment
centrimo ./meme_conv1.txt test_gene_pos1-2k.fasta

# featuremap sequence motif discovery 
# HOMER
findMotifs.pl test_gene_activate.fasta fasta motifResults_conv1 &>log.test_gene_pos1

# fimo enrichment
fasta-get-markov -m 1 Human_updown2k.fa > result/background.txt
fimo --thresh 1e-6 --max-stored-scores 500000 \
        --bgfile result/background.txt \
        --o result/out_Human_updown2k_jaspar \
        JASPAR.meme \
        Human_updown2k.fa &>result/log.Human_updown2k_jaspar.txt

# motif analysis
Rscript plot.motif.R

# saliancy location
Rscript plot.saliancy.R

# celltype-specific features 
# layer_conductance
Rscript plot.layer_conductance.R


############################ finetune explain ############################
