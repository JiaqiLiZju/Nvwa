##
cd CV_1; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_1.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_2; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_2.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_3; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_3.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_4; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_4.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_5; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_5.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_6; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_6.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_7; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_7.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_8; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_8.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_9; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_9.h5 --gpu-device 0 --trails explainable &
cd ..
cd CV_10; python ../../../../../1_run_explain_cv.py ../../Dataset.cross_valid_10.h5 --gpu-device 0 --trails explainable &
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

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Mus_musculus_2019_01/mouse.cisbp201901.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
