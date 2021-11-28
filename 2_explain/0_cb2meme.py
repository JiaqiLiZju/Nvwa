import glob
import os
import pandas as pd

out_str = ''
for fname in glob.glob("singletons/*.cb"):
    # motif = fname.split('/')[-1].replace('.cb', '')
    motif = open(fname,"r").readline().strip().replace('>', '')
    out_str = out_str + '\n' + motif + '\n' + \
        pd.read_csv(fname, sep='\t', skiprows=1, names=['A:', 'C:', 'G:', 'T:']) \
        .T \
        .to_csv(sep='\t', header=False)

with open("tmp_transfac.txt", 'w') as out_fh:
    out_fh.write(out_str)

os.system("uniprobe2meme tmp_transfac.txt 1> out.meme 2>log.txt")