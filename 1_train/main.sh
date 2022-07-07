############################ hyperopt ############################
# using Anneal-lr5-epoch10-EPOCH100
mkdir Anneal-lr5-epoch10-EPOCH100 && cd Anneal-lr5-epoch10-EPOCH100
python ./0_hyperopt_BCE.py 0 ../../Datasets/HCL_pseudocell_bin.h5 &>log.run
python ./print_losses.py hyperopt_trials/params.p > hyperopt_trials/params.txt

# using Anneal-lr5-epoch10
mkdir Anneal-lr5-epoch10 && cd Anneal-lr5-epoch10
python ./0_hyperopt_BCE.py 1 ../../Datasets/HCL_pseudocell_bin.h5 &>log.run
python ./print_losses.py hyperopt_trials/params.p > hyperopt_trials/params.txt

# using TPE-lr5-epoch10
mkdir TPE-lr5-epoch10 && cd TPE-lr5-epoch10
python ./0_hyperopt_BCE.py 0 ../../Datasets/HCL_pseudocell_bin.h5 &>log.run
python ./print_losses.py hyperopt_trials/params.p > hyperopt_trials/params.txt

# using TPE-lr4-epoch5
mkdir TPE-lr4-epoch5 && cd TPE-lr4-epoch5
python ./0_hyperopt_BCE.py 1 ../../Datasets/HCL_pseudocell_bin.h5  &>log.run
python ./print_losses.py hyperopt_trials/params.p > hyperopt_trials/params.txt

# plot trails
Rscript 0_hyperopt_trials.R

# get model parameter detail
0_tensorwatch.ipynb 

############################ train ############################
# python ./1_hyperopt_BCE_best.py device_id, mode, trails_fname, data, use_data_rc_augment
python ./1_hyperopt_BCE_best.py ../data/train.human.h5 --gpu-device 0 --mode train --trails hyperopt_trials/params.p --use_data_rc_augment
python ./1_hyperopt_BCE_best.py ../data/train.mouse.h5 --gpu-device 0 --mode train --trails hyperopt_trials/params.p --use_data_rc_augment

# if training failed, we can use resume mode to resume the model traing
# check the checkpoint.dir
python ./1_hyperopt_BCE_best.py ../data/train.human.h5 --gpu-device 0 --mode resume --trails hyperopt_trials/params.p --use_data_rc_augment True

## test
python ./1_hyperopt_BCE_best.py ../data/train.human.h5 --gpu-device 0 --mode test --trails hyperopt_trials/params.p
python ./1_hyperopt_BCE_best.py ../data/train.mouse.h5 --gpu-device 0 --mode test --trails hyperopt_trials/params.p

# plot test metric
Rscript 1_test_assignment.r
Rscript 1_test_cluster.r

# We used the model architectures considering model explaination
python ./1_hyperopt_BCE_best.py ../data/train.human.h5 --gpu-device 0 --trails explainable 
python ./1_hyperopt_BCE_best.py ../data/train.human.h5 --gpu-device 0 --trails explainable --mode test

######################## cross validation ########################
for x in {1..5};do
    python ./1_hyperopt_BCE_best.py ../data/train.human.{i}.h5 --trails hyperopt_trials/params.p && \
    python ./1_hyperopt_BCE_best.py ../data/train.human.{i}.h5 --trails hyperopt_trials/params.p --mode test
done
Rscript 1_test_assignment.r

# Human
cd ~/NvWA/Train/Human/Human_MAGIC/Train_test && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.Human_train_test.h5 --gpu-device 3 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.Human_train_test.h5 --gpu-device 3 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/Chrom8_Train_test && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.Human_Chrom8_train_test.h5 --gpu-device 2 --trails explainable --mode test &

for x in {1..10}; do mkdir ~/NvWA/Train/Human/Human_MAGIC/chrom8_$x; done
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_1; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_1.h5 --gpu-device 3 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_1.h5 --gpu-device 3 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_2; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_2.h5 --gpu-device 3 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_2.h5 --gpu-device 3 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_3; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_3.h5 --gpu-device 3 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_3.h5 --gpu-device 3 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_4; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_4.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_4.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_5; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_5.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_5.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_6; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_6.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_6.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_7; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_7.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_7.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_8; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_8.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_8.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_9; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_9.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_9.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Human/Human_MAGIC/chrom8_10; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_10.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Human/Human_MAGIC/Dataset.leave_chrom8_10.h5 --gpu-device 2 --trails explainable --mode test &

# Dmel
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/Train_test; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.Dmel_MAGIC_train_test.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.Dmel_MAGIC_train_test.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/chrom2R_1; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_1.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_1.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/chrom2R_2; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_2.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_2.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/chrom2R_3; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_3.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_3.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/chrom2R_4; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_4.h5 --gpu-device 0 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_4.h5 --gpu-device 0 --trails explainable --mode test &
cd ~/NvWA/Train/Dmel/Dmel_MAGIC/chrom2R_5; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_5.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Dmel/Dmel_MAGIC/Dataset.leave_chrom2R_5.h5 --gpu-device 1 --trails explainable --mode test &

# Celegans
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/train_test; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.Celegans_train_test.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.Celegans_train_test.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/chromIII_1; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_1.h5 --gpu-device 1 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_1.h5 --gpu-device 1 --trails explainable --mode test &
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/chromIII_2; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_2.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_2.h5 --gpu-device 2 --trails explainable --mode test &
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/chromIII_3; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_3.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_3.h5 --gpu-device 2 --trails explainable --mode test &
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/chromIII_4; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_4.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_4.h5 --gpu-device 2 --trails explainable --mode test &
cd ~/NvWA/Train/Celegan/Celegan_MAGIC/chromIII_5; python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_5.h5 --gpu-device 2 --trails explainable && python ../../../1_hyperopt_BCE_best.py ../../../../Dataset/Celegan/Celegans_MAGIC/Dataset.leave_chromIII_5.h5 --gpu-device 2 --trails explainable --mode test &
