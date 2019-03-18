#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
#cleaning data
dataset = pd.read_csv('autos.csv')
dataset = dataset[dataset.yearOfRegistration>1950]
dataset = dataset[dataset.price>100]
dataset = dataset[dataset.price<150000]
dataset = dataset[dataset.powerPS>50]
dataset = dataset[dataset.powerPS<500]
dataset = dataset[dataset.kilometer>5000]
dataset = dataset[dataset.kilometer<200000]
dataset.isnull().sum()
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
dataset.isnull().sum()
#datadivision
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,7:8].values
#label encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb1 = LabelEncoder()
X[:,4] = lb1.fit_transform(X[:,4])
X[:,5] = lb1.fit_transform(X[:,5])
X[:,6] = lb1.fit_transform(X[:,6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:9]
for i in range(0,284301):
    X[i,4] = 2018-X[i,4]
    X[i,3] = 4-X[i,3]
    if X[i,3]<0:
        X[i,3]=X[i,3]+12
        X[i,4]=X[i,4]-1
    X[i,3]=X[i,3]/12
    X[i,4]= X[i,4]+X[i,3]
X = np.delete(X,3,1)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#Y = sc_y.fit_transform(Y)
#1
X_test = X[0:56861,:]
X_train = X[56861:284301,:]
Y_test = Y[0:56861,:]
Y_train = Y[56861:284301,:]
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =20)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
i = [0,0,0,0,0]
i[0] = regressor.score(X_test,Y_test)

X_test = X[56861:113722,:]
X_train = X[113722:284301,:]
X_train=np.concatenate((X_train,X[0:56861,:]),axis = 0)
Y_test = Y[56861:113722,:]
Y_train = Y[113722:284301,:]
Y_train=np.concatenate((Y_train,Y[0:56861,:]),axis=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =20)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
i[1] = regressor.score(X_test,Y_test)


X_test = X[113722:170583,:]
X_train = X[170583:284301,:]
X_train=np.concatenate((X_train,X[0:113722,:]),axis = 0)
Y_test = Y[113722:170583,:]
Y_train = Y[170583:284301,:]
Y_train=np.concatenate((Y_train,Y[0:113722,:]),axis=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =20)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
i[2] = regressor.score(X_test,Y_test)

X_test = X[170583:227444,:]
X_train = X[227444:284301,:]
X_train=np.concatenate((X_train,X[0:170583,:]),axis = 0)
Y_test = Y[170583:227444,:]
Y_train = Y[227444:284301,:]
Y_train=np.concatenate((Y_train,Y[0:170583,:]),axis=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =20)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
i[3] = regressor.score(X_test,Y_test)


X_test = X[227444:284301,:]
X_train = X[0:227444,:]
Y_test = Y[227444:284301,:]
Y_train = Y[0:227444,:]
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =20)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
i[4] = regressor.score(X_test,Y_test)

#rf.score(Y_test,y_pred)
#visualisation
'''x = X[:,5:6]
regressor.fit(x, Y)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()'''