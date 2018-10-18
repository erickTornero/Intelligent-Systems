from myregression import *
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
# Get appropiate URL's
import os
import numpy as np
url_train = os.path.join('Tarea 1', 'all','train.csv')
# Get the data through 'Pandas' API
import pandas as pd

data = load_data(url_train)
# Discard Outlier
## Deleting Outlier from GrLivArea
data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<300000)].index)
## Deleting Outlier from TotalBmstSF
data = data.drop(data[(data['TotalBsmtSF']>5000) & (data['SalePrice']<300000)].index)
## Deleting Outlier from 1stFlrSF
data = data.drop(data[(data['1stFlrSF']>4000) & (data['SalePrice']<300000)].index)
## Deleting Outlier from BsmtFinSF1
data = data.drop(data[(data['BsmtFinSF1']>5000) & (data['SalePrice']<300000)].index)
## Deleting Outlier from LotFrontage
data = data.drop(data[(data['LotFrontage']>300) & (data['SalePrice']<300000)].index)
## Deleting Outlier from LotArea
data = data.drop(data[(data['LotArea']>150000) & (data['SalePrice']<450000)].index)

#missing data
#dealing with missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
## Delet all missing data with more than 1 missin data
data = data.drop((missing_data[missing_data['Total'] > 300]).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)

# Normalization:
# Normalizing 'SalePrice' feature
#applying log transformation
data['SalePrice'] = np.log(data['SalePrice'])
data['GrLivArea'] = np.log(data['GrLivArea'])
data['1stFlrSF'] = np.log(data['1stFlrSF'])
data['TotRmsAbvGrd'] = np.log(data['TotRmsAbvGrd'])
data['YearBuilt'] = np.log(data['YearBuilt'])
data['LotArea'] = np.log(data['LotArea'])

dict_train = {'OverallQual': data['OverallQual'],'GrLivArea':data['GrLivArea'], 
              'GarageCars':data['GarageCars'], 'TotalBsmtSF': data['TotalBsmtSF'],
              '1stFlrSF':data['1stFlrSF'],'FullBath':data['FullBath'], 
              'TotRmsAbvGrd':data['TotRmsAbvGrd'], 'YearBuilt':data['YearBuilt'],
              'YearRemodAdd':data['YearRemodAdd'], 'Fireplaces':data['Fireplaces'],
              'BsmtFinSF1':data['BsmtFinSF1'], 'LotFrontage':data['LotFrontage'],
              'WoodDeckSF':data['WoodDeckSF'], 'OpenPorchSF':data['OpenPorchSF'],
              'HalfBath':data['HalfBath'], 'LotArea':data['LotArea'],
              'SalePrice':data['SalePrice']
             }
data_train = pd.DataFrame(data = dict_train, index = data.index)

imputer = Imputer(strategy='median')
imputer.fit(data_train)
X = imputer.transform(data_train)
data_tr = pd.DataFrame(X, columns=data_train.columns, index=list(data_train.index.values))
data_label = data_train['SalePrice'].copy()
data_tr.drop(['SalePrice'], axis = 1, inplace=True)
print("Data Training: Shape>",data_tr.shape)
print('Training ...')

reg = MyLinearRegression()
reg.fit(data_tr, data_label)
print(reg.coef)
pred = reg.predict(data_tr)

print('Self-testing>', pred)
mse_l = mean_squared_error(data_label, pred)
rmse_l = np.sqrt(mse_l)
print('srmse>', rmse_l)