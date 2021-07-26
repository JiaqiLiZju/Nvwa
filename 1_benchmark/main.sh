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

# 
python ../../expecto_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 ../Xreducedall.2002.npy ../geneanno.csv ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5
python ../../expecto_compare.py 0 train ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5 && python ../../expecto_compare.py 0 test ./Dataset.HCL_Chrom8_imputation_bin_Expecto.h5 &

python ../../expecto_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_train_test.h5 ../Xreducedall.2002.npy ../geneanno.csv ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5
python ../../expecto_compare.py 0 train ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5 && python ../../expecto_compare.py 0 test ./Dataset.HCL_TrainTest_imputation_bin_Expecto.h5 &

###########################################
## SVM
# run fimo
mkdir result
fasta-get-markov -m 1 Human_updown2k.fa > result/background.txt
fimo --thresh 1e-6 --max-stored-scores 500000 \
        --bgfile result/background.txt \
        --o result/out_Human_updown2k_jaspar \
        JASPAR.meme \
        Human_updown2k.fa &>result/log.Human_updown2k_jaspar.txt

# train and test
python ./SVM_compare.py result/out_Human_updown2k_jaspar/fimo.tsv Human_v2.txt


###########################################
## random label baseline
cd ~/NvWA/Benchmark/HCL_random_feature
python ../random_feature_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 Dataset.Human_Chrom8_train_test_random_feature.h5
python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_feature.h5 --gpu-device 0 && python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_feature.h5 --gpu-device 0 --mode test &


###########################################
## random features baseline
cd ~/NvWA/Benchmark/HCL_random_label
python ../random_label_datasets.py ~/NvWA/Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 Dataset.Human_Chrom8_train_test_random_label.h5
python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_label.h5 --gpu-device 0 && python ~/NvWA/Train/1_hyperopt_BCE_best.py Dataset.Human_Chrom8_train_test_random_label.h5 --gpu-device 0 --mode test &
