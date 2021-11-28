import pandas as pd

ic = pd.read_csv("./filter.IC_freq_Reductant.csv", header=0, index_col=None)
ic.columns = ['ID', 'freq', 'IC', "Reductant"]
ic['ID'] = ic.ID.map(lambda x:x.split('_')[-1]).astype(int)
df = ic

influence = pd.read_csv("influence_conv1_mean.csv", header=None)
influence.columns = ['ID', 'influence']
df = df.merge(influence, on='ID', how='left', sort=False)

tomtom = pd.read_csv("./tomtom_conv1_JASPAR_t9/tomtom.tsv", sep='\t', header=0, skipfooter=3)
tomtom = tomtom[tomtom['q-value']<0.1]
tomtom = pd.DataFrame(tomtom.Query_ID.unique(), columns=['MotifName'])
tomtom['ID'] = tomtom.MotifName.map(lambda x:x.split('_')[-1]).astype(int)
tomtom['Anno'] = 1
df = df.merge(tomtom, on='ID', how='left', sort=False).fillna(0)

reproduce = pd.read_csv("./motif_reproduce_inCV.csv", header=0)[['Query_ID', 'sum_cnt', 'sum_match']]
reproduce['ID'] = reproduce.Query_ID.map(lambda x:x.split('_')[-1]).astype(int)
df = df.merge(reproduce, on='ID', how='left', sort=False)

df.to_csv("explain_filter_result.csv")
