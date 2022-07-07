python ../../1_run_explain.py ./Dataset.Human_leave_chrom8.h5 --gpu-device 2 --trails explainable

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/Motif_scenic/out.meme
tomtom -oc tomtom_conv1_CisTarget_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

MOTIFDB=/media/ggj/Files/mount/NvWA_Final/0_TFBS/CisBP/Homo_sapiens_2019_01/human.cisbp201901.meme
tomtom -oc tomtom_conv1_cisbp_t1_allr -thresh 0.1 -dist allr -no-ssc ./meme_conv1_thres9.txt $MOTIFDB &>log.tomtom_conv1 &

