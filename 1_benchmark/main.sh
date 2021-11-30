###########################################
## Xpresso
python ./xpresso_compare.py 

###########################################
## Expecto
wget http://deepsea.princeton.edu/media/code/expecto/resources_20190807.tar.gz

# Human
# datasets
python ./expecto_datasets.py ../HCL_pseudocell_bin.human.h5 ./resources/Xreducedall.2002.npy ./resources/geneanno.csv HCL_pseudocell_bin_Expecto.h5
# train
python ./expecto_compare.py 0 train HCL_pseudocell_bin_Expecto.h5
# test
python ./expecto_compare.py 0 test HCL_pseudocell_bin_Expecto.h5

# Mouse 
# transfer genomic features from human
python ./expecto_features_Xreduced.py 
# datasets
python ./expecto_datasets.py ../MCA_pseudocell_bin.mouse_noTestis.h5 ./resources/predictions_fwdonly.h5 ./resources/geneanno.mca.csv MCA_pseudocell_bin_Expecto.mouse_expecto.h5
# train
python ./expecto_compare.py 0 train MCA_pseudocell_bin_Expecto.mouse_expecto.h5
# test
python ./expecto_compare.py 0 test MCA_pseudocell_bin_Expecto.mouse_expecto.h5

python ../../expecto_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 ../Xreducedall.2002.npy ../geneanno.csv ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5
python ../../expecto_compare.py 0 train ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5 && python ../../expecto_compare.py 0 test ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5 &

python ../../expecto_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_train_test.h5 ../Xreducedall.2002.npy ../geneanno.csv ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5
python ../../expecto_compare.py 0 train ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5 && python ../../expecto_compare.py 0 test ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5 &

###########################################
## Fimo-SVM
python 0_proc_prom_region_seq.py database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.fa database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.gtf 500 500 >Human_updown500bp.fa 2>log.Species_updown500bp.fa
# run fimo
mkdir Fimo_Human_updown500 && cd Fimo_Human_updown500
fasta-get-markov -m 1 Human_updown500bp.fa > ./background.txt
fimo --thresh 1e-6 --max-stored-scores 500000 \
        --bgfile ./background.txt \
        --o ./out_Human_updown500bp_jaspar \
        ../../0_TFBS/JASPAR2020/JASPAR2020_CORE_non-redundant_pfms_meme.txt \
        Human_updown500bp.fa &>./log.Human_updown500bp.txt

# train and test
nvwa_dataset_fname=/media/ggj/Files/mount/NvWA_Final/0_Dataset/Dataset.Human_leave_chrom8.h5
motif_gene_fname=result/out_Human_updown2k_jaspar/human_updown1000_fimo.tsv
output_fname=Dataset.Human_leave_chrom8_updown1000_fimo.h5

python ./Fimo_datasets.py $nvwa_dataset_fname $motif_gene_fname $output_fname 
python ./Fimo_compare.py $output_fname 

###########################################
## random features baseline
cd ~/NvWA/Benchmark/HCL_random_feature
python ../random_feature_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 Dataset.Human_Chrom8_train_test_random_feature.h5
python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_feature.h5 --gpu-device 0 && python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_feature.h5 --gpu-device 0 --mode test &

## random label baseline
cd ~/NvWA/Benchmark/HCL_random_label
python ../random_label_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 Dataset.Human_Chrom8_train_test_random_label.h5
python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_label.h5 --gpu-device 0 && python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_label.h5 --gpu-device 0 --mode test &

###########################################
## DeepSEA
cd ~/NvWA/Benchmark/DeepSEA_HCL
python ../DeepSEA_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --lr 0.1 && python ../DeepSEA_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --lr 0.1 --mode test &

## Beluga
cd ~/NvWA/Benchmark/DeepSEA_Beluga_HCL
python ../DeepSEA_Beluga_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --lr 0.1 --gpu-device 1 && python ../DeepSEA_Beluga_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --mode test --gpu-device 1 &

###########################################
## Basset
cd ~/NvWA/Benchmark/Basset_HCL
python ../Basset_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --lr 0.1 --gpu-device 1 && python ../Basset_compare.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --mode test --gpu-device 1 &

## Basenji
python Basenji_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 Dataset.Basenji_HCL_Chr8.h5
Basenji_compare.sh


###########################################
cd /share/home/guoguoji/NvWA/Train/model_training_8000gene_20210207/Dmel_MAGIC_label_015_batchsize_32/benchmark_Arch

## Fimo-SVM
python 0_proc_prom_region_seq_Dmel.py database_genome/D.mel/Drosophila_melanogaster_genomic.fasta database_genome/D.mel/Drosophila_melanogaster_genomic.gtf 500 500 >Dmel_updown500bp.fa 2>log.Species_updown500bp.fa
# run fimo
mkdir Fimo_Dmel_updown500/ && cd Fimo_Dmel_updown500/
fasta-get-markov -m 1 Dmel_updown500bp.fa > ./background.txt
fimo --thresh 1e-6 --max-stored-scores 500000 \
        --bgfile ./background.txt \
        --o ./out_Dmel_updown500bp_jaspar \
        ../../0_TFBS/JASPAR2020/JASPAR2020_CORE_insects_non-redundant_pfms_meme.txt \
        Dmel_updown500bp.fa &>./log.Dmel_updown500bp.txt

mkdir Fimo_SVM_Dmel && cd Fimo_SVM_Dmel
python ../Fimo_datasets.py ../../Dataset.Dmel_train_test.h5 fimo.tsv Dataset.Dmel_train_test_Fimo_updown500bp_JASPAR.h5
python ../Fimo_compare.py Dataset.Dmel_train_test_Fimo_updown500bp_JASPAR.h5

## random features baseline
mkdir random_feature_Dmel && cd random_feature_Dmel
python ../random_feature_datasets.py ../../Dataset.Dmel_train_test.h5 Dataset.Dmel_train_test.random_feature.h5
python ../1_hyperopt_BCE_best.py Dataset.Dmel_train_test.random_feature.h5 --gpu-device 0 && python ../1_hyperopt_BCE_best.py Dataset.Dmel_train_test.random_feature.h5 --gpu-device 0 --mode test &

## random label baseline
mkdir random_label_Dmel && cd random_label_Dmel
python ../random_label_datasets.py ../../Dataset.Dmel_train_test.h5 Dataset.Dmel_train_test.random_label.h5
python ../1_hyperopt_BCE_best.py Dataset.Dmel_train_test.random_label.h5 --gpu-device 1 && python ../1_hyperopt_BCE_best.py Dataset.Dmel_train_test.random_label.h5 --gpu-device 1 --mode test &

## DeepSEA
mkdir DeepSEA_Dmel && cd DeepSEA_Dmel
python ../DeepSEA_compare.py ../../Dataset.Dmel_train_test.h5 --lr 0.1 --gpu-device 2 && python ../DeepSEA_compare.py ../../Dataset.Dmel_train_test.h5 --gpu-device 2 --mode test &

## Beluga
mkdir DeepSEA_Beluga_Dmel && cd DeepSEA_Beluga_Dmel
python ../DeepSEA_Beluga_compare.py ../../Dataset.Dmel_train_test.h5 --lr 0.1 --gpu-device 3 && python ../DeepSEA_Beluga_compare.py ../../Dataset.Dmel_train_test.h5 --gpu-device 3 --mode test &

## Basset
mkdir Basset_Dmel && cd Basset_Dmel
python ../Basset_compare.py ../../Dataset.Dmel_train_test.h5 --lr 0.1 --gpu-device 0 && python ../Basset_compare.py ../../Dataset.Dmel_train_test.h5 --gpu-device 0 --mode test &

## Basenji
mkdir Basenji_Dmel && cd Basenji_Dmel

export BASENJIDIR=/media/ggj/Files/Dev/basenji-master
export PATH=$BASENJIDIR/bin:$PATH
export PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH

mkdir data/tfrecords
python ./Basenji_datasets.py ../../0_Dataset/Dataset.Dmel_train_test.h5 data/

basenji_train.py -o models/Benchmark/ models/params.json data/ &>log.train_basenji &
basenji_test.py --ai 0,1,2 --save -o output/Benchmark/ models/params.json models/Benchmark/model_best.h5 data/ &>log.test
calculate_roc.py
# Basenji_compare.sh
# ggj@GGJLAB:/media/ggj/Files/DeepMind/Basenji/tutorials$