# macs2 peak calling
bedtools bamtobed -i human.sorted.bam >human.sorted.bed
macs2 callpeak -g hs --nomodel \
    --shift -100 --extsize 200 \
    -t test.fragement.bed \
    -n test.fragement --outdir ./peaks/ &>log.macs2

# Step 1: Filter duplicates
macs2 filterdup -i test.fragement.bed --keep-dup=1 -o test.fragement_filterdup.bed 
# macs2 filterdup -i CTCF_Control_200K.bed.gz --keep-dup=1 -o CTCF_Control_200K_filterdup.bed
  
# Step 2: Decide the fragment length d
macs2 predictd -i test.fragement_filterdup.bed -g hs -m 5 50

# Step 3: Extend ChIP sample to get ChIP coverage track
macs2 pileup -f BED -i test.fragement_filterdup.bed -o test.fragement_filterdup.pileup.bdg --extsize 254

# Step 4: Build local bias track from control
macs2 pileup -f BED -i test.fragement_filterdup.bed -B --extsize 127 -o d_bg.bdg
macs2 pileup -f BED -i test.fragement_filterdup.bed -B --extsize 500 -o 1k_bg.bdg
macs2 bdgopt -i 1k_bg.bdg -m multiply -p 0.254 -o 1k_bg_norm.bdg
macs2 pileup -f BED -i test.fragement_filterdup.bed -B --extsize 5000 -o 10k_bg.bdg
macs2 bdgopt -i 10k_bg.bdg -m multiply -p 0.0254 -o 10k_bg_norm.bdg
macs2 bdgcmp -m max -t 1k_bg_norm.bdg -c 10k_bg_norm.bdg -o 1k_10k_bg_norm.bdg
macs2 bdgcmp -m max -t 1k_10k_bg_norm.bdg -c d_bg.bdg -o d_1k_10k_bg_norm.bdg
macs2 bdgopt -i d_1k_10k_bg_norm.bdg -m max -p .00001 -o local_bias_raw.bdg

# Step 5: Scale the ChIP and control to the same sequencing depth
macs2 bdgopt -i local_bias_raw.bdg -m multiply -p .99858 -o local_lambda.bdg

# Step 6: Compare ChIP and local lambda to get the scores in pvalue or qvalue
macs2 bdgcmp -t test.fragement_filterdup.pileup.bdg -c local_lambda.bdg -m qpois -o test.fragement_qvalue.bdg

# Step 7: Call peaks on score track using a cutoff
macs2 bdgpeakcall -i test.fragement_qvalue.bdg -c 1.301 -l 245 -g 100 -o test.fragement_peaks.bed

macs2 bdgcmp -m max -t Predict_Stromal_Mouse66.Plus.bedGraph -c Predict_Stromal_Mouse66.Minus.bedGraph -o 1k_10k_bg_norm.bdg
macs2 bdgcmp -m max -t 1k_10k_bg_norm.bdg -c Predict_Stromal_Mouse66.Minus.bedGraph -o d_1k_10k_bg_norm.bdg
macs2 bdgopt -i d_1k_10k_bg_norm.bdg -m max -p .0188023 -o local_bias_raw.bdg
macs2 bdgopt -i local_bias_raw.bdg -m multiply -p .99858 -o local_lambda.bdg

macs2 bdgcmp -t Predict_Neuron_Mouse12.Plus.bedGraph -c Pred.background.bedGraph -m qpois -o qvalue.bdg
macs2 bdgpeakcall -i qvalue.bdg -c 1.301 -l 20 -g 100 -o peaks.bed
head peaks.bed
