###### TrainTest
python ../../1_hyperopt_BCE_best.py ./Dataset.Dmel_train_test.h5 --gpu-device 0 --trails explainable --pos 2000 
python ../../1_hyperopt_BCE_best.py ./Dataset.Dmel_train_test.h5 --gpu-device 0 --trails explainable --pos 2000 --batch_size 100 --mode test
python ../../1_hyperopt_BCE_best.py ./Dataset.Dmel_train_test.h5 --gpu-device 0 --trails explainable --pos 2000 --batch_size 100 --mode test_all
python ../../1_run_explain.py ./Dataset.Dmel_train_test.h5 --gpu-device 0 --trails explainable --pos 2000

# test metric plot
# python ./1_test_task_relationship.py
Rscript ./1_test_assignment.r

# TFBS
MOTIFDB=/media/ggj/Files/mount/NvWA_Final/TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

###### CV
mkdir CrossValid
python ../../../Dataset/2_propare_datasets.py cross_valid /share/home/guoguoji/NvWA/Dataset/Dmel/onehot/Dmel_updown10k.rm_official.onehot.p Dmel_MAGIC_015.p /share/home/guoguoji/NvWA/NvWA_Annotation/Dmel_cellatlas.annotation.20210127.txt ./CrossValid

for x in {1..10}; do mkdir CV_$x; done
cd CV_1; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_1.h5 --gpu-device 0 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_1.h5 --gpu-device 0 --trails explainable --mode test &
cd ..
cd CV_2; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_2.h5 --gpu-device 0 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_2.h5 --gpu-device 0 --trails explainable --mode test &
cd ..
cd CV_3; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_3.h5 --gpu-device 0 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_3.h5 --gpu-device 0 --trails explainable --mode test &
cd ..
cd CV_4; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_4.h5 --gpu-device 0 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_4.h5 --gpu-device 0 --trails explainable --mode test &
cd ..
cd CV_5; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_5.h5 --gpu-device 0 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_5.h5 --gpu-device 0 --trails explainable --mode test &
cd ..
cd CV_6; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_6.h5 --gpu-device 1 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_6.h5 --gpu-device 1 --trails explainable --mode test &
cd ..
cd CV_7; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_7.h5 --gpu-device 1 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_7.h5 --gpu-device 1 --trails explainable --mode test &
cd ..
cd CV_8; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_8.h5 --gpu-device 1 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_8.h5 --gpu-device 1 --trails explainable --mode test &
cd ..
cd CV_9; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_9.h5 --gpu-device 1 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_9.h5 --gpu-device 1 --trails explainable --mode test &
cd ..
cd CV_10; python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_10.h5 --gpu-device 1 --trails explainable && python ../../../../1_hyperopt_BCE_best.py ../Dataset.cross_valid_10.h5 --gpu-device 1 --trails explainable --mode test &
cd ..

python ../../../metric_CV.py CV_

##
cd CV_1; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_1.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_2; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_2.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_3; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_3.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_4; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_4.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_5; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_5.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_6; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_6.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_7; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_7.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_8; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_8.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_9; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_9.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_10; python ../../../../1_run_explain_cv.py ../Dataset.cross_valid_10.h5 --gpu-device 0 --trails explainable &
cd ..

for i in {1..10}
do
    tomtom -oc tomtom_vs${i} -thresh 0.1 \
    /media/ggj/Files/NvWA/NvWA_Final/Dmel/Motif_filter/meme_conv1_thres9.txt \
    /media/ggj/Files/NvWA/NvWA_Final/Dmel/CV/CV_$i/Motif/meme_conv1_thres9.txt &>log.tomtom_conv1 &
done

python ./calculate_reproduce.py

# collect all filter-based result
python ./collect_result.py

### benchmark
python ../../../../1_hyperopt_BCE_best_pos.py ../../Dataset.Dmel_train_test.h5 --lr 1e-3 --gpu-device 2 --trails explainable --pos 1000 
&& python ../../../../1_hyperopt_BCE_best_pos.py ../../Dataset.Dmel_train_test.h5 --lr 1e-3 --gpu-device 2 --trails explainable --pos 1000 --mode test &

python ../../../../1_hyperopt_BCE_best_pos.py ../../Dataset.Dmel_train_test.h5 --lr 1e-3 --gpu-device 2 --trails explainable --pos 500 
&& python ../../../../1_hyperopt_BCE_best_pos.py ../../Dataset.Dmel_train_test.h5 --lr 1e-3 --gpu-device 2 --trails explainable --pos 500 --mode test &

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Drosophila_melanogaster/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
