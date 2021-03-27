import pandas as pd
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier

print('Loading ensembl info...')
snp_info = pd.read_csv('../input/vep_snp.txt',sep='\t', skiprows=43)
snp_id = np.unique(snp_info.iloc[:,0])
print('Done')

print('\nLoading SNPs...')
X = pd.read_hdf('../input/out.012.h5')
snp = X
snp = snp.iloc[2:]
snp.columns=snp_id
columns = snp.columns[snp.columns.str.contains('^rs*')==True]
snp = snp.loc[:,columns]
X = np.array(snp)
print(X.shape)
print('Done')

labels = pd.read_csv('../input/labels.tsv',sep='\t',header=0) 
y = labels.iloc[:,2]

label_list = [0,1]
label_list2 = ['Healthy','Obese']
label_sum = 0

print('Number of animals per label:')
for i in range(len(label_list)):
    label_count=sum(y==label_list[i])
    print(label_list2[i]+': '+str(label_count))
    label_sum=label_sum+label_count
print('Total number of animals: '+str(label_sum))

n_estimators = [100,500,800,1500,2500]
max_features = ['auto', 'sqrt','log2']
max_depth = [10,20,30,40,50]
max_depth.append(None)
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15, 20]

grid_param = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf
             }
             
from sklearn.model_selection import RandomizedSearchCV
RFR = RandomForestClassifier(random_state=1)
RFR_random = RandomizedSearchCV(estimator = RFR,
                               param_distributions = grid_param, n_iter = 500,
                               cv = 5, verbose=2, random_state=42, n_jobs= 15,return_train_score=True)
                               

RFR_random.fit(X,y)
print(RFR_random.best_params_)

import csv
with open('../output/best_params_rsid_only.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in RFR_random.best_params_.items():
       writer.writerow([key, value])
