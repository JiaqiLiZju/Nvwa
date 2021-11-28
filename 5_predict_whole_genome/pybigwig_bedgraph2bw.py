import pyBigWig
import numpy as np
import pandas as pd
from sys import argv

# bedgraph, chromsize, bigwig = argv[1:]
bedgraph = pd.read_csv("./predicted_tracks/Predict_EarthWorm0.Plus.bedGraph", sep='\t', header=None, index_col=None)
bw = pyBigWig.open("./predicted_tracks/Predict_EarthWorm0.Plus.bw", 'w')

intervals = bedgraph.values #np.array(bw.intervals("chr8"))
bw.addHeader([("GWHACBE00000001", 159027471)])
chroms = np.array(["GWHACBE00000001"] * intervals.shape[0])
starts = intervals[:,0].astype(str)
ends = intervals[:,1].astype(np.int64)
values0 = intervals[:,2].astype(np.float64)

bw.addEntries(chroms, starts, ends=ends, values=values0)
bw.close()
