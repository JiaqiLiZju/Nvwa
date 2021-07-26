import pickle
import numpy as np
import pandas as pd
# import pyfastx
# import logging
#logging.basicConfig(filename="logging.onehotGenome.txt", level=logging.DEBUG)


onehot_nuc = {'A':[1,0,0,0],
            'C':[0,1,0,0],
            'G':[0,0,1,0],
            'T':[0,0,0,1],
            'N':[0,0,0,0]}
            

def _onehot_seq(seq):
    return np.array([onehot_nuc[nuc] for nuc in str(seq).upper()])


def _onehot_genome_fastx(gfname):
	try:
		gf = pyfastx.Fasta(gfname, key_func=lambda x: x.split()[0])
		genome_dict = {}
		for s in gf:
			name, seq = s.name, s.seq 
			if name not in genome_dict:        
				genome_dict[name] = [seq, _onehot_seq(seq)]
	except TypeError:
		gf = pyfastx.Fasta(gfname)
		genome_dict = {}
		for s in gf:
			name, seq = s.name, s.seq 
			name = name.split()[0]
			if name not in genome_dict:        
				genome_dict[name] = [[seq, _onehot_seq(seq)]]
			else:
				genome_dict[name].append([seq, _onehot_seq(seq)])
	return genome_dict


def _onehot_genome(gfname):
    genome_dict = {}
    with open(gfname, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line.split()[0].replace(">", '')
            else:
                seq = line.rstrip()
                if name not in genome_dict:        
                    genome_dict[name] = [seq, _onehot_seq(seq)]
    
    return genome_dict


def _onehot_genome_motif(gfname, fimo_tsv, fill=999):
    '''
    genome_dict = _onehot_genome(gfname)
    fimo_df = pd.read_csv(fimo_tsv, sep='\t', header=0)
    assert all([item in fimo_df.columns for item in ['sequence_name', 'start', 'stop']])
    for gene in genome_dict.keys():
        selector = fimo_df['sequence_name']==gene
        if any(selector):
            onehot_seq = genome_dict[gene][0][-1]
            motif = np.zeros(shape=onehot_seq.shape[0])
            motif_select = fimo_df[selector][['start', 'stop']].astype(np.int8)
            for i in motif_select.index:
                start, end = motif_select.loc[i]
                motif[start:end] = fill
    '''
    raise NotImplementedError


if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 3
    _, gfname, out_fname = argv
    genome_dict = _onehot_genome(gfname)
    pickle.dump(genome_dict, open(out_fname, 'wb'))
