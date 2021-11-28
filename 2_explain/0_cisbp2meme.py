import glob
import os
import pandas as pd

out_str = ''
for fname in glob.glob("pwms_all_motifs/*.txt"):
    motif = fname.split('/')[-1].replace('.txt', '')
    out_str = out_str + '\n' + motif + '\n' + \
        pd.read_csv(fname, sep='\t', header=0, index_col=0) \
        .rename(columns={'A':'A:', 'T':'T:', 'G':'G:', 'C':'C:'}) \
        .T \
        .to_csv(sep='\t', header=False)

with open("tmp_transfac.txt", 'w') as out_fh:
    out_fh.write(out_str)

os.system("uniprobe2meme tmp_transfac.txt 1> out.meme 2>log.txt")