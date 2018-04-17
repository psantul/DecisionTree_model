#importing libraries
import pandas as pd
import numpy as np

#importing and preparing dataset
dataset_15 = pd.read_csv('2015.csv')
dataset_16 = pd.read_csv('2016.csv')
dataset_17 = pd.read_csv('2017.csv')

X_train = dataset_15.iloc[:, 5:].values
y_train = dataset_15.iloc[:, 3].values

X_test_16 = dataset_16.iloc[:,6:].values
y_test_16 = dataset_16.iloc[:,3].values

X_test_17 = dataset_17.iloc[:,5:].values
y_test_17 = dataset_17.iloc[:,2].values

#training the regressor
from sklearn.tree import DecisionTreeRegressor as DTR
regressor = DTR()
regressor = regressor.fit(X_train,y_train)

#predicting the result
y_pred_16 = regressor.predict(X_test_16)
y_pred_17 = regressor.predict(X_test_17)


# The mean squared error
print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test_16) - y_test_16) ** 2))
print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test_17) - y_test_17) ** 2))


# Variance score: 1 is perfect prediction
print('Variance score: %.2f' % regressor.score(X_test_16, y_test_16))
print('Variance score: %.2f' % regressor.score(X_test_17, y_test_17))


#calculating the loss and accuracy
loss_16 = (np.sum(np.abs(y_pred_16-y_test_16))/np.sum(y_test_16))*100
acc_16 = 100-loss_16

loss_17 = (np.sum(np.abs(y_pred_17-y_test_17))/np.sum(y_test_17))*100
acc_17 = 100-loss_17