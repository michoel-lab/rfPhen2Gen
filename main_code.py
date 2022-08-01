# This script can be used to generate .npz file required for generating the figures in 'Results_21Feb2022.ipynb'

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from tqdm import tqdm
import seaborn as sns
import numpy as np
from pandas_plink import read_plink
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df_fs = pd.read_table('freesurfer_56rois_bl.csv', sep = ",") # Freesurfer phenotypes
df_apoe = pd.read_table('apoe_data_ceu.csv', sep=',') # APOE information

df_fs = df_fs[df_fs.PTID.isin(df_apoe.Subject_ID)]

(bim, fam, G) = read_plink('cleaned_new', verbose=False)
X = np.abs((G.compute())-2)

ceu_snp = pd.read_table('cleaned_new_ceu.snplist',header=None)


ptids = df_fs.PTID.unique()
X = X[:,fam[fam.iid.isin(ptids)].sort_values(by='iid').index.values]
X = X[np.where(bim.snp.isin(ceu_snp[0]))[0],:]


# Replace the NaN values for SNPs with mode values
for i in tqdm(range(X.shape[0])):
    mode = stats.mode(X[i,:])[0][0]
    X[i,np.isnan(X[i,:])] = mode

cov_data = df_fs.iloc[:,[2,3,4,5]]
cov_data.index = range(len(cov_data.index))

df = df_fs.drop(columns=['VISCODE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'ICV_bl'])


df_apoe = df_apoe[df_apoe.Subject_ID.isin(ptids)]

vol_corr = np.zeros([df.shape[0], df.shape[1]-1])

# Correcting for potential confounding factors like age, gender etc.
for i in tqdm(range(df.shape[1]-1)):
    y = pd.DataFrame(df.iloc[:,i+1].values, columns=['data'])
    DF = pd.concat([cov_data,y],axis=1)
    reg = smf.ols(formula="data ~ AGE + ICV_bl + C(PTGENDER) + C(PTEDUCAT)", data=DF).fit()
    vol_corr[:,i] = reg.resid


pheno = vol_corr.copy()
geno = np.hstack([X.copy().T, df_apoe['APOE'].values.reshape(-1,1)])

#geno[geno==2] = 1

#X = (pheno - pheno.mean(axis=0)) / (pheno.std(axis=0))
#y = (geno - geno.mean(axis=0)) / (geno.std(axis=0))

X = pheno.copy()
y = geno.copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train) 

X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test)

num_snps = y_train.shape[1]

RMSE_rf = np.empty([num_snps])
RMSE_las = np.empty([num_snps])
RMSE_rid = np.empty([num_snps])

AUPR_rf = np.empty([num_snps])
AUPR_rid = np.empty([num_snps])
AUPR_las = np.empty([num_snps])

AUROC_rf = np.empty([num_snps])
AUROC_rid = np.empty([num_snps])
AUROC_las = np.empty([num_snps])



for i in tqdm(range(num_snps)):
    regr = RandomForestRegressor(random_state=42, n_jobs=-1).fit(X_train, y_train[:,i])
    rid = Ridge(alpha=10.0).fit(X_train, y_train[:,i]) # new 1000, old 10
    las = Lasso(alpha=0.01).fit(X_train, y_train[:,i]) # new 0.1, old 0.01
    
    y_pred_rf = regr.predict(X_test)
    y_pred_rid = rid.predict(X_test)
    y_pred_las = las.predict(X_test)
    
    RMSE_rf[i] = (mean_squared_error(y_test[:,i].reshape(-1,1), y_pred_rf.reshape(-1,1), squared=False))

    RMSE_rid[i] = (mean_squared_error(y_test[:,i].reshape(-1,1), y_pred_rid.reshape(-1,1), squared=False))


    RMSE_las[i] = (mean_squared_error(y_test[:,i].reshape(-1,1), y_pred_las.reshape(-1,1), squared=False))

    
np.savez('results_ml_gwas_fs_21022022.npz', RMSE_rf=RMSE_rf, RMSE_rid=RMSE_rid, RMSE_las=RMSE_las)

    