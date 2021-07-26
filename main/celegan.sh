python ../../1_run_explain.py ./Dataset.Cele_train_test.h5 --gpu-device 0 --trails explainable

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Caenorhabditis_elegans_2019_11_27_8-16_pm/Caenorhabditis_elegans_Cisbp.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
