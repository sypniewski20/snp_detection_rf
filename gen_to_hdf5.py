import pandas as pd
import h5py
print('Loading data...')

input_data = 'out.012'
snp_id = 'snp_list.txt'


X = pd.read_csv('../input/'+input_data ,sep='\t',header=None,index_col=0)
Xt = X.transpose()
snp_id = pd.read_csv('../input/'+snp_id,sep='\t',header=0)
Xt.insert(0,'rsid',snp_id.loc[:,'ID'])
Xt.insert(0,'chr',snp_id.loc[:,'#CHROM'])
Xt = Xt[Xt.rsid.str.contains('^rs*') == True]
Xt2 = Xt.transpose()
print(Xt2.shape)
print('Done '+str(input_data))

print('\nWriting '+str(input_data)+' to' ' h5')
Xt2.to_hdf('../input/'+ str(input_data)+'.h5', key='Xt2', mode='w')
print('Done '+str(input_data))