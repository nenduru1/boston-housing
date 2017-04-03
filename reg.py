# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 11:57:02 2017

@author: Nitesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

df=pd.read_csv('Boston_Housing.csv')
df.head()


X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)


#performing backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
reg_ols=sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary()

#remove the column with P value > 0.05
X_opt=X[:,[0,1,2,3,5,6,7,8,9,10,11,12]]
reg_ols=sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary()

result=pd.DataFrame({'test':y_test,'predicted':y_pred})
result.to_csv('py_output.csv',index=False)
type(y_test)
type(y_pred)


#read all outputs
sas=pd.read_csv('sas_output.csv')
sas.head()

py=pd.read_csv('py_output.csv')
py.head()

r=pd.read_csv('r_output.csv')
r.head()
plt.show(py['test'],py['predicted'])
plt.scatter(py['test'],py['predicted'])

plt.scatter(r['test'],r['predicted'])

plt.hist(sas['test'])
