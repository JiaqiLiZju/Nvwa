from sys import argv
import pandas as pd

tomtom_fname, IC_fname = argv[1:]

tomtom = pd.read_csv(tomtom_fname, sep='\t', header=0, skipfooter=3, engine='python')
tomtom = tomtom[["Query_ID", "Target_ID", "q-value"]]
tomtom.columns = ["Filter1", "Filter2", "q_value"]
tomtom['Pair'] = '(' + tomtom.Filter1.map(lambda x:x.split('_')[-1]) + ',' + tomtom.Filter2.map(lambda x:x.split('_')[-1]) + ')'
tomtom = tomtom[tomtom["q_value"] < 0.01]
tomtom

IC = pd.read_csv(IC_fname, index_col=0)
IC.index = IC.index.map(lambda x: "Motif_"+str(x))
tomtom_ic = tomtom.merge(IC, left_on="Filter2", right_index=True, sort=False, how="left")
tomtom_ic["rank"] = tomtom_ic.groupby("Filter1", as_index=False, sort=False)["IC"].rank(ascending=False)
nondup_filter = tomtom_ic[tomtom_ic["rank"]==1]

IC["Reductant"] = 1
IC.loc[nondup_filter.Filter2.unique(), "Reductant"] = 0
IC.to_csv("./filter.IC_freq_Reductant.csv")
