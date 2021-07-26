mkdir /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_200gene_0001
cd /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_200gene_0001

python /share/home/guoguoji/NvWA/Train/1_gene_label_zzzzzz.py /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_200gene_0001.h5ad 0.0001 EarthWorm_MAGIC_200gene_0001.p

python /share/home/guoguoji/NvWA/code/0_preproc_dataset/2_propare_datasets.py train_test_split /share/home/guoguoji/NvWA/Dataset/Earthworm/onehot/EarthWorm_updown10k.rm_official.onehot.p EarthWorm_MAGIC_200gene_0001.p /share/home/guoguoji/NvWA/NvWA_Annotation/EarthWorm_cellatlas.annotation.20210602_mege.txt Dataset.EarthWorm_train_test.h5

python /share/home/guoguoji/NvWA/code/1_train/1_hyperopt_BCE_best.py Dataset.EarthWorm_train_test.h5 --patience 10 --gpu-device 0 --trails explainable --tower_by Celltype --batch_size 24 && python /share/home/guoguoji/NvWA/code/1_train/1_hyperopt_BCE_best.py Dataset.EarthWorm_train_test.h5 --gpu-device 0 --trails explainable --mode test --tower_by Celltype --batch_size 128 &



mkdir /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_300gene_0001
cd /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_300gene_0001

python /share/home/guoguoji/NvWA/Train/1_gene_label_zzzzzz.py /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_300gene_0001.h5ad 0.1 EarthWorm_MAGIC_300gene_01.p

python /share/home/guoguoji/NvWA/code/0_preproc_dataset/2_propare_datasets.py train_test_split /share/home/guoguoji/NvWA/Dataset/Earthworm/onehot/EarthWorm_updown10k.rm_official.onehot.p EarthWorm_MAGIC_300gene_01.p /share/home/guoguoji/NvWA/NvWA_Annotation/EarthWorm_cellatlas.annotation.20210602_mege.txt Dataset.EarthWorm_train_test.h5

python /share/home/guoguoji/NvWA/code/1_train/1_hyperopt_BCE_best.py Dataset.EarthWorm_train_test.h5 --patience 10 --gpu-device 2 --trails explainable --tower_by Celltype --batch_size 24 && python /share/home/guoguoji/NvWA/code/1_train/1_hyperopt_BCE_best.py Dataset.EarthWorm_train_test.h5 --gpu-device 2 --trails explainable --mode test --tower_by Celltype --batch_size 128 &



mkdir /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_500gene_100cells_0001
cd /share/home/guoguoji/NvWA/Train/model_training_official_20210412/Earthworm_20210602/EarthWorm_MAGIC_500gene_100cells_0001

python /share/home/guoguoji/NvWA/Train/1_gene_label_zzzzzz.py ./EarthWorm_MAGIC_500gene_100cell.h5ad 0.0001 EarthWorm_MAGIC_200gene_0001.p

python /share/home/guoguoji/NvWA/code/0_preproc_dataset/2_propare_datasets.py train_test_split /share/home/guoguoji/NvWA/Dataset/Earthworm/onehot/EarthWorm_updown10k.rm_official.onehot.p EarthWorm_MAGIC_200gene_0001.p /share/home/guoguoji/NvWA/NvWA_Annotation/EarthWorm_cellatlas.annotation.20210602_mege.txt Dataset.EarthWorm_train_test.h5

python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best_pos.py Dataset.EarthWorm_train_test.h5 --patience 10 --pos_upper 2000 --pos_lower 2000 --gpu-device 1 --trails explainable --tower_by Celltype --batch_size 16 && python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best_pos.py Dataset.EarthWorm_train_test.h5 --gpu-device 0 --trails explainable --mode test --tower_by Celltype --batch_size 128 &


cd train_pos1000/;ll -h
python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best.py ../Dataset.EarthWorm_train_test.h5 --patience 10 --pos 1000 --gpu-device 2 --trails explainable --tower_by Celltype --batch_size 16 && python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best.py ../Dataset.EarthWorm_train_test.h5 --gpu-device 2 --trails explainable --mode test --tower_by Celltype --batch_size 128 &

mkdir train_pos1000_lr4; cd train_pos1000_lr4/; ll -h
python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best.py ../Dataset.EarthWorm_train_test.h5 --patience 10 --pos 1000 --gpu-device 3 --lr 1e-4 --trails explainable --tower_by Celltype --batch_size 16 && python /share/home/guoguoji/NvWA/Train/1_hyperopt_BCE_best.py ../Dataset.EarthWorm_train_test.h5 --gpu-device 2 --trails explainable --mode test --tower_by Celltype --batch_size 128 &


##
cd CV_1; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_1.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_2; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_2.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_3; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_3.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_4; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_4.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_5; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_5.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_6; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_6.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_7; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_7.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_8; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_8.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_9; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_9.h5 --gpu-device 1 --trails explainable &
cd ..
cd CV_10; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_10.h5 --gpu-device 1 --trails explainable &
cd ..


for i in {1..10}
do
    tomtom -oc tomtom_vs${i} -thresh 0.1 \
    /media/ggj/Files/mount/NvWA_Final/model_training_8000gene_20210207/EarthWorm_MAGIC_300gene_500UMI_012/Motif/meme_conv1_thres9.txt \
    /media/ggj/Files/mount/NvWA_Final/model_training_8000gene_20210207/EarthWorm_MAGIC_300gene_500UMI_012/CrossValid/CV_$i/Motif/meme_conv1_thres9.txt &>log.tomtom_conv1 &
done

ln -s ../Motif/meme_conv1_thres9_IC_freq.csv .
ln -s ../Motif/influence_conv1_mean.csv .
ln -s ../Motif/tomtom_conv1_CisTarget_t1_allr/ .

python ./calculate_reproduce.py

# collect all filter-based result
python ./collect_result.py

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Caenorhabditis_elegans_2019_11_27_8-16_pm/Caenorhabditis_elegans_Cisbp.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
