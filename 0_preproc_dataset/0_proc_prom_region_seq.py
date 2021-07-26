import sys
import pysam
import pandas as pd

def print_usage():
    print("\
    USAGE:\n \
    python 0_proc_prom_region_seq.py Genome.fasta Annotation.gtf 500 500 >Species_updown500bp.fa 2>log.Species_updown500bp.fa\n \
    python 0_proc_prom_region_seq.py Genome.fasta Annotation.gtf 2000 0 >Species_up2000.fa 2>log.Species_up2000.fa\n \
    python 0_proc_prom_region_seq.py Genome.fasta Annotation.gtf 0 2000 >Species_down2000.fa 2>log.Species_down2000.fa\n \
    ", file=sys.stderr)


def _check(w):
    if w < 0:
        w = 0
    return w


#rev_comp
def _rev_comp(seq):
    trantab = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')
    return seq.translate(trantab)[::-1]

def rev_comp(seq):
    #base complement dict
    basecomplemt = {
        "A":"T",
        "T":"A",
        "G":"C",
        "C":"G",
        "a":"t",
        "t":"a",
        "g":"c",
        "c":"g",
        "N":"N",
        "n":"N",
    }
    #reverse completement seq
    seq_list = list(seq)
    seq_list.reverse()
    seq_list = [ basecomplemt[base] for base in seq_list ]
    seq = ''.join(seq_list)
    return seq


def _proc_gtf(gtf):
    gene_pos_dict = {}
    with open(gtf, 'r') as fh:
        for line in fh:
            if not line.startswith('#'):
                (chrom, _, gtype, start, end, _, strand, _, anno) = line.rstrip().split('\t')
                if gtype == "gene":
                    gene_name = str(anno.split(';')[0].split()[-1]).strip('"')
                    # gene_name = str(anno.split(';')[1].split()[-1]).strip('"')
                    # gene_name = str(anno.split(';')[2].split()[-1]).strip('"')
                    gene_pos_dict[gene_name] = (start, end, strand, chrom)
    
    pd.DataFrame(gene_pos_dict, index=['start', 'end', 'strand', 'chrom']).T.to_csv("gtf_annotation.csv")
    return gene_pos_dict


def _write_gene_seq(gene_seq_dict):
    for key in gene_seq_dict.keys():
        gene, (start, end, strand, chrom), direct = key
        seq = gene_seq_dict[key]
        print(">%s\t%s:%s:%s:%s:%s" % (gene, start, end, strand, chrom, direct))
        print(seq)


def _get_prom_region(genome, gtf, forward=2000, backward=2000):
    length = forward + backward

    fa_handler = pysam.FastaFile(genome)
    gene_pos_dict = _proc_gtf(gtf)
    
    gene_seq_dict = {}
    for gene in gene_pos_dict:

        start, end, strand, chrom = gene_pos_dict[gene]

        if chrom not in fa_handler.references:
            print("GTF chrom: %s not in fasta" % chrom, file=sys.stderr)
            continue

        if strand == '+':
            tss = int(start)
            fp_start = tss - forward - 1 # -1 for pysam coord
            bp_end = tss + backward - 1
            bp_end, fp_start = _check(bp_end), _check(fp_start)
            seq = fa_handler.fetch(chrom, start=fp_start, end=bp_end)
            
        elif strand == '-':
            tss = int(end)
            bn_start = tss - backward - 1 # -1 for pysam coord
            fn_end = tss + forward - 1
            bn_start, fn_end = _check(bn_start), _check(fn_end)
            seq = fa_handler.fetch(chrom, start=bn_start, end=fn_end)
            seq = _rev_comp(seq)
        
        # pad zeroes
        seq_length = len(seq)
        if seq_length < length:
            pad_length = int((length - seq_length) / 2)
            seq = 'N'*pad_length + seq + 'N'*pad_length
        if len(seq) < length:
            seq = seq + 'N' # pad zeros to length

        gene_seq_dict[(gene, gene_pos_dict[gene], "Bidirect")] = seq

    _write_gene_seq(gene_seq_dict)
    
    fa_handler.close()

    return gene_seq_dict


if __name__ == "__main__":
    from sys import argv
    if len(argv) == 5:
        genome_fname = argv[1]
        gtf_fname = argv[2]
        forward = int(argv[3])
        backward = int(argv[4])
    else:
        print_usage()
        exit(1)

    print("up%ddown%d"%(forward, backward), file=sys.stderr)

    gene_seq_dict = _get_prom_region(genome_fname, gtf_fname, forward, backward)
