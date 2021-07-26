import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py
import math
import pyfasta

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

model = Beluga()
model.load_state_dict(torch.load('./resources/deepsea.beluga.pth'))
model.eval()
model.cuda()

def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.
    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output
    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize
    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    #dataflip = seqsnp[:, ::-1, ::-1]
    #seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp.astype("float32")


#now read in the gene file
gene_file = "/home/ggj/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf"
gene_chrom_tss_strand = []
# for i,line in enumerate(open(gene_file)):
#     gene_id,symbol,chrom,strand,TSS,CAGE_TSS,gene_type = line.rstrip().split(",")
for line in open(gene_file):
    if not line.startswith('#'):
        (chrom, _, gtype, TSS, end, _, strand, _, anno) = line.rstrip().split('\t')
        CAGE_TSS = TSS
        if gtype == "gene":
            gene_id, _, gene_name, _, gene_type = anno.split(';')[:5]
            gene_id = gene_id.split()[-1].strip('"')
            gene_chrom_tss_strand.append((gene_id, chrom, int(CAGE_TSS), (1 if strand=="+" else -1)) )

chrom_l = ['1 dna:chromosome chromosome:GRCm38:1:1:195471971:1 REF', '10 dna:chromosome chromosome:GRCm38:10:1:130694993:1 REF', '11 dna:chromosome chromosome:GRCm38:11:1:122082543:1 REF', '12 dna:chromosome chromosome:GRCm38:12:1:120129022:1 REF', '13 dna:chromosome chromosome:GRCm38:13:1:120421639:1 REF', '14 dna:chromosome chromosome:GRCm38:14:1:124902244:1 REF', '15 dna:chromosome chromosome:GRCm38:15:1:104043685:1 REF', '16 dna:chromosome chromosome:GRCm38:16:1:98207768:1 REF', '17 dna:chromosome chromosome:GRCm38:17:1:94987271:1 REF', '18 dna:chromosome chromosome:GRCm38:18:1:90702639:1 REF', '19 dna:chromosome chromosome:GRCm38:19:1:61431566:1 REF', '2 dna:chromosome chromosome:GRCm38:2:1:182113224:1 REF', '3 dna:chromosome chromosome:GRCm38:3:1:160039680:1 REF', '4 dna:chromosome chromosome:GRCm38:4:1:156508116:1 REF', '5 dna:chromosome chromosome:GRCm38:5:1:151834684:1 REF', '6 dna:chromosome chromosome:GRCm38:6:1:149736546:1 REF', '7 dna:chromosome chromosome:GRCm38:7:1:145441459:1 REF', '8 dna:chromosome chromosome:GRCm38:8:1:129401213:1 REF', '9 dna:chromosome chromosome:GRCm38:9:1:124595110:1 REF', 'MT dna:chromosome chromosome:GRCm38:MT:1:16299:1 REF', 'X dna:chromosome chromosome:GRCm38:X:1:171031299:1 REF']
chrom_d = {x.split(" ")[0]:x for x in chrom_l}
chrom_d

shifts = np.array(list(range(-20000,20000,200)))+100

hg19_path = "/home/ggj/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta"
genome = pyfasta.Fasta(hg19_path)

windowsize = 2000
predictions_fwdonly = []
predictions_withrc = []
maxshift=20000

print("shifts:\n",shifts)
assert len(shifts)==200
for gene,chrom,tss,strand in gene_chrom_tss_strand:
    print(gene)
    chrom = chrom_d[chrom]
    seqs_to_predict = []
    for shift in shifts:
        seq = genome.sequence({'chr': chrom, 'start': tss + shift*strand -
                               int(0.5*windowsize - 1), 'stop': tss + shift*strand + int(0.5*windowsize)})
        seqs_to_predict.append(seq)

    seqsnp = encodeSeqs(seqs_to_predict)
    
    model_input = torch.from_numpy(np.array(seqsnp)).unsqueeze(2)
    rc_model_input = torch.from_numpy(np.array(seqsnp[:,::-1,::-1])).unsqueeze(2)
    model_input = model_input.cuda()
    rc_model_input = rc_model_input.cuda()
    prediction = model.forward(model_input).cpu().data.numpy().copy()
    rc_prediction = model.forward(rc_model_input).data.cpu().numpy().copy()
    predictions_fwdonly.append(prediction)
    predictions_withrc.append(0.5*(prediction+rc_prediction))

predictions_fwdonly=np.array(predictions_fwdonly)
predictions_withrc=np.array(predictions_withrc)

pos_weight_shifts = shifts
pos_weights = np.vstack([
        np.exp(-0.01*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts <= 0),
        np.exp(-0.02*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts <= 0),
        np.exp(-0.05*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts <= 0),
        np.exp(-0.1*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts <= 0),
        np.exp(-0.2*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts <= 0),
        np.exp(-0.01*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts >= 0),
        np.exp(-0.02*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts >= 0),
        np.exp(-0.05*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts >= 0),
        np.exp(-0.1*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts >= 0),
        np.exp(-0.2*np.abs(pos_weight_shifts)/200)*(pos_weight_shifts >= 0)])

l = []
for i in range(0, predictions_fwdonly.shape[0], 128):
    l.append(np.sum(pos_weights[None,:,:,None]*predictions_fwdonly[i:i+128,None,:,:],axis=2))

reconstructed_expecto_fwdonly = np.vstackack(l)
reconstructed_expecto_fwdonly.shape

reconstructed_expecto_fwdonly = reconstructed_expecto_fwdonly.reshape(reconstructed_expecto_fwdonly.shape[0], -1)
reconstructed_expecto_fwdonly.shape

np.save("./reconstructed_expecto_fwdonly.npy", reconstructed_expecto_fwdonly)

gene_file = "/home/ggj/jiaqiLi/mount-1/ggj/jiaqiLi/general_global_soft/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.gtf"
geneanno = []
for line in open(gene_file):
    if not line.startswith('#'):
        (chrom, _, gtype, TSS, end, _, strand, _, anno) = line.rstrip().split('\t')
        CAGE_TSS = TSS
        if chrom == "Y":
            break
        if gtype == "gene":
            gene_id, _, symbol, _, gene_type = anno.split(';')[:5]
            gene_id = gene_id.split()[-1].strip('"')
            symbol = symbol.split()[-1].strip('"')
            gene_type = gene_type.split()[-1].strip('"')
            geneanno.append((gene_id,symbol,chrom,strand,TSS,CAGE_TSS,gene_type))

geneanno = pd.DataFrame(geneanno)
geneanno.columns=("id","symbol","chrom","strand","TSS","CAGE_representative_TSS","type")
geneanno.to_csv("./geneanno.mca.csv")