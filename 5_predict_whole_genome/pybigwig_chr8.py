import pyBigWig
import numpy as np
from sys import argv

bw_fname, bw_chr_fname = argv[1:]

bw = pyBigWig.open(bw_fname)
bw_chr = pyBigWig.open(bw_chr_fname, 'w')

# human
# intervals = np.array(bw.intervals("chr8"))
# bw_chr.addHeader([("chr8", 145138636)])
# chroms = np.array(["chr8"] * intervals.shape[0])

# mouse
intervals = np.array(bw.intervals("chr8"))
bw_chr.addHeader([("chr8", 129398500)])
chroms = np.array(["chr8"] * intervals.shape[0])

# bw = pyBigWig.open("./scATAC-EW/EW.SeqDepthNorm.bw")
# bw_chr = pyBigWig.open("./scATAC-EW/EW-GWHACBE00000001.SeqDepthNorm.bw", 'w')

# intervals = np.array(bw.intervals("GWHACBE00000001"))
# bw_chr.addHeader([("GWHACBE00000001", 159027471)])
# chroms = np.array(["GWHACBE00000001"] * intervals.shape[0])
starts = intervals[:,0].astype(np.int64)
ends = intervals[:,1].astype(np.int64)
values0 = intervals[:,2].astype(np.float64)

bw_chr.addEntries(chroms, starts, ends=ends, values=values0)
bw_chr.close()
bw.close()
