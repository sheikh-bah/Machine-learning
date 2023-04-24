# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:52:29 2023

@author: JE93867
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

import os
#%%
path = os.getcwd()
# load the data

data = pd.read_csv(path + '\\1.01.+Simple+linear+regression.csv')
#%%
# create the regression
# declare the dependent and independent variables

y = data['GPA'] # target

x = data['SAT'] # feature
x.shape
y.shape

# sklearb uses 2d array, so we reshape our data

x_matrix = x.values.reshape(-1,1)


reg = LinearRegression()

reg.fit(x_matrix, y)

#%%
# R_SQUARED
reg.score(x_matrix,y)

# coefi
reg.coef_

# intercept
reg.intercept_
#%%
# making predicitions
reg.predict([[1740]])

new_data = pd.DataFrame(data = [1740,1760],columns = ['SAT'])

reg.predict(new_data)

# create a new column
new_data['predicted gpa'] = reg.predict(new_data)


plt.scatter(x,y)
y_hat = reg.coef_ *x + reg.intercept_

fig = plt.plot(x,y_hat,lw = 4,c = 'orange', label = 'regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)


#%%

#%%


#%%
'''
Multiple linear regression
'''
multiple_data = pd.read_csv(path + '\\1.02.+Multiple+linear+regression.csv')
multiple_data.describe()

x = multiple_data[['Rand 1,2,3','SAT']]

y = multiple_data['GPA']

reg = LinearRegression()
reg.fit(x,y)
reg.coef_
reg.intercept_
reg.score(x,y)


#we can compute our adjusted R_squared
# 1-[1-r2]*[(n-1)/(n-p-1)]

r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjust_r2 = 1-(1-r2)*(n-1)/(n-p-1)


#%%

'''
FEATURE SELECTION
'''
from sklearn.feature_selection import f_regression

f_regression(x,y) # the 2nd array are the p_values, the first array is f_statistics

p_values = f_regression(x,y)[1]
p_values.round(4) #array([0.6763, 0.    ]) we can drop the first one with # p value 0.67

#%%
# creating a summary table

reg_summary = pd.DataFrame(data = x.columns.values, columns =['Feautures'])
reg_summary

reg_summary['Coefficients'] = reg.coef_
reg_summary['P-values'] = p_values.round(3)



#%%
'''
FEATURE SCALING ( STANDARZIZATION)
'''
# aim for about -1<= x <= 1 for each feature j


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x_scaled = scaler.transform(x)

#%%
# we do our regression
reg = LinearRegression()
reg.fit(x_scaled,y)
reg.coef_
reg.intercept_

# create a summary table

reg_summary = pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3']], columns = ['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[1],reg.coef_[0]

#%%
# make a prediction with the standardized coefficients (weights)

new_data = pd.DataFrame(data =[[1700,2],[1800,1]],columns = ['SAT','Rand 1,2,3'])

# if we do the prediction without standardizing the data, the results will not be good
reg.predict(new_data)
##now we standardize our new_data,we already created scaler
scaler.fit(new_data)
new_data_scaled = scaler.transform(new_data)
reg.predict(new_data_scaled)

#%%
'''
supposdely we removed the RAND 123, lets see the prediction
'''

reg_simple = LinearRegression()
reg_simple.fit(x_scaled[:,0].reshape(-1,1),y)

reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))


#%%%

'''
reducing overfit,splitting the data
'''

#%%
# generate data we are going to split

a = np.arange(1,101)
b = np.arange(501,601)

#%% 
# split the data
from sklearn.model_selection import train_test_split

# we want to make it a 80%, 20% size
a_train, a_test = train_test_split(a, test_size = 0.2, shuffle = True, random_state = 42)
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size = 0.2, shuffle = True, random_state = 42)