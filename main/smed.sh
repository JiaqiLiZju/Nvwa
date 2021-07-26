python ../../1_run_explain.py ./Dataset.Smed_train_test.h5 --gpu-device 3 --trails explainable &

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Schmidtea_mediterranea_2021_06_13_2/Schmidtea_mediterranea_Cisbp.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
