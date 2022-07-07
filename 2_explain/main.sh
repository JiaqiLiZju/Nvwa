## filter analyse
python 1_run_explain.py Dataset.h5

# motif annotation
MOTIF_ANNO=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/JASPAR/motif.anno.tsv
# cat $MOTIFDB |grep MOTIF >MOTIF_ANNO
MOTIFDB=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/JASPAR/JASPAR2018_CORE_vertebrates_non-redundant.meme
tomtom -oc tomtom_conv1_JASPAR -thresh 0.5 ./meme_conv1_thres95.txt $MOTIFDB &>log.tomtom_conv1 &

MOTIFDB=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/HUMAN/HOCOMOCOv9.meme
tomtom -oc tomtom_conv1_HOCOMOCO -thresh 0.5 ./meme_conv1_thres95.txt $MOTIFDB &>log.tomtom_conv1 &

MOTIFDB=~/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/MotifDB/MEME_Motif/motif_databases/CIS-BP/Homo_sapiens.meme
tomtom -oc tomtom_conv1_CISBP -thresh 0.5 ./meme_conv1_thres95.txt $MOTIFDB &>log.tomtom_conv1 &

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
python filter_annotation.py

# motif analysis
Rscript plot.motif.R

# saliancy location
Rscript plot.saliancy.R

## other validation analysis
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

# celltype-specific features 
# layer_conductance
Rscript plot.layer_conductance.R


## run explainer on CV models
for i in {1..10}
do
    cd CV_$i
    python ../1_run_explain_cv.py ../Dataset.cross_valid_${i}.h5 --gpu-device 0 --trails explainable
    
    cd ../../
    tomtom -oc tomtom_vs${i} -thresh 0.1 \
        ./Motif_filter/meme_conv1_thres9.txt \
        ./CV/CV_$i/Motif/meme_conv1_thres9.txt &>log.tomtom_conv1 &
done
# then calculate_reproduce on CV filters
python ./calculate_reproduce.py

# collect all filter-based result
python ./collect_result.py



############################ explain ############################
BASEDIR=/media/ggj/Files/mount/NvWA_Final
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/JASPAR2020/JASPAR2020_CORE_non-redundant_pfms_meme.txt

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Ciona_MAGIC_012_batchsize96_lr/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/EarthWorm_MAGIC_200gene_500UMI_label_011_batchsize_32_pt10_20211003/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Human_MAGIC_label_006_batchsize_96_20210930/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Zebrafish_MAGIC_600gene_mincell1_t5_label_006_batchsize_96_20211007/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Mouse_MAGIC_label_006_batchsize_96_new/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Celegans_MAGIC_label_005_batchsize_32/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Smed_MAGIC_label_020_batchsize_32/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python ../../filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python ../../filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python ../../filter.collect_result.py


cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Dmel_MAGIC_label_015_batchsize_32/Motif
tomtom -oc tomtom_conv1_JASPAR_t9 -thresh 0.1 ./meme_conv1_thres9.txt $MOTIFDB &>Log.tomtom_JASPAR 

for i in {1..10}; do tomtom -oc ../CrossValid/tomtom_vs$i -thresh 0.1 ./meme_conv1_thres9.txt ../CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_$i ; done
python $BASEDIR/model_training_20211030/filter.calculate_reproduce.py

tomtom -oc ../CrossValid/tomtom_Reductant -thresh 0.1 ./meme_conv1_thres9.txt ./meme_conv1_thres9.txt &>../CrossValid/Log.tomtom_Reductant 
python $BASEDIR/model_training_20211030/filter.reductant.py ../CrossValid/tomtom_Reductant/tomtom.tsv ./meme_conv1_thres9_IC_freq.csv

python $BASEDIR/model_training_20211030/filter.collect_result.py

# filter annotation
cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Ciona_MAGIC_012_batchsize96_lr/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Ciona_intestinalis/Ciona_intestinalis.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

# cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/EarthWorm_MAGIC_200gene_500UMI_label_011_batchsize_32_pt10_20211003/Motif
# MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Ciona_intestinalis/Ciona_intestinalis.meme
# tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Human_MAGIC_label_006_batchsize_96_20210930/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Homo_sapiens_2019_01/human.cisbp201901.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Zebrafish_MAGIC_600gene_mincell1_t5_label_006_batchsize_96_20211007/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Danio_rerio_2021_06_13_2/Danio_rerio_Cisbp.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Mouse_MAGIC_label_006_batchsize_96_new/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Mus_musculus_2019_01/mouse.cisbp201901.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Celegans_MAGIC_label_005_batchsize_32/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Caenorhabditis_elegans_2019_11_27_8-16_pm/Caenorhabditis_elegans_Cisbp.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Smed_MAGIC_label_020_batchsize_32/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Schmidtea_mediterranea_2021_06_13_2/Schmidtea_mediterranea_Cisbp.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Dmel_MAGIC_label_015_batchsize_32/Motif
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Drosophila_melanogaster/Dmel.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

## Influence Filter Celltype
# Influe.filter
cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Ciona_MAGIC_012_batchsize96_lr/Motif
python ../../Influe.filter.py /media/ggj/Files/mount/NvWA_Final//0_Dataset/Dataset.Ciona_train_test.h5 /media/ggj/Files/mount/NvWA_Final//0_Annotation_Cellcluster/Ciona_Larvae_cellatlas.annotation.20210222.txt 50000&

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/EarthWorm_MAGIC_200gene_500UMI_label_011_batchsize_32_pt10_20211003/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.EarthWorm_train_test.h5 ../../../0_Annotation_Cellcluster/EarthWorm_200Gene_500UMI_name.cellatlas.annotation.20210915.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Human_MAGIC_label_006_batchsize_96_20210930/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.Human_Chrom8_train_test.h5 ../../../0_Annotation_Cellcluster/HCL_cellatlas.annotation.20210110.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Zebrafish_MAGIC_600gene_mincell1_t5_label_006_batchsize_96_20211007/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.Zebrafish_train_test.h5 ../../../0_Annotation_Cellcluster/Zebrafish_cellatlas.annotation_600gene.20210923.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Mouse_MAGIC_label_006_batchsize_96_new/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.MCA_leave_chrom8.h5 ../../../0_Annotation_Cellcluster/MCA_cellatlas.annotation.20210125.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Celegans_MAGIC_label_005_batchsize_32/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.Cele_train_test.h5 ../../../0_Annotation_Cellcluster/Celegans_Cao2017_cellatlas.annotation.20201215.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Smed_MAGIC_label_020_batchsize_32/Motif
python ../../Influe.filter.py ../../../0_Dataset/Dataset.Smed_train_test.h5 ../../../0_Annotation_Cellcluster/Smed_GSE111764.cellatlas.annotation.20201215.txt 50000 &

cd /media/ggj/Files/mount/NvWA_Final/model_training_20211030/Dmel_MAGIC_label_015_batchsize_32/Motif
python $BASEDIR/model_training_20211030/Influe.filter.py ../../../0_Dataset/Dataset.Dmel_train_test.h5 ../../../0_Annotation_Cellcluster/Dmel_cellatlas.annotation.20210407.txt 50000 &




