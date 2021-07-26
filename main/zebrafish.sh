MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Danio_rerio_2021_06_13_2/Danio_rerio_Cisbp.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &
