# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:41:29 2023

@author: JE93867
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()
import scipy
import matplotlib.pyplot as plt
import sklearn

import os
path = os.getcwd()

# load the data
data = pd.read_csv(path+'\\1.01.+Simple+linear+regression.csv')

# create our first regression

data.describe()

# linear regression between SAT and GPA
# Each time you create a linear regression, it should be meaninful

# the equation is y_hat = b_0 + b_1x_1

y = data['GPA']
x1 = data['SAT']

plt.scatter(x1,y)
plt.xlabel('SAT',fontsize = 20)
plt.ylabel('GPA',fontsize = 20)
plt.show()

# in linear regression to train a model we feed the training set to our learning algorithm
# then our learning algorithm wil produce a function f. the job of the function is to take a new
# input x and output an estimate y_hat.
# f can be represented as f(w,b) = wx + b
# the value chosen for w and b will determine the value y_hat based on the input feature x

#%%
# we want to add w values using stat model
w = sm.add_constant(x1)  #similar to np.ones
    
 # depending on the value we choose for w and b, we will get a different function f(w,b)
 # what we want to do is to choose values for the parameters w,b such that the straight line
 # we get from the function f, somehow fits our data well.
"""
 we want to fit a straight line to the training data, so that the model f(w,b) = wx +b
 depending on the values chosen for w,b we get different straight line. we want to find
 the best w,b so that the training example fit our straight line well
 
 cost fxn :
     how do we find w,b so the prediction y_hat(i) is close to y(i) for all x(i),y(i)
 we compute the square error ( how far off the prediction is to the target).
 this function goal is to minimise the error.
 for linear regression is to find the best w
 
"""
 
results = sm.OLS(y,w).fit()
results.summary()

#%%
# now we choose the best values for our prediction
# const          0.2750 
#SAT            0.0017 
plt.scatter(x1,y)
y_hat = 0.0017*x1 + 0.2750

fig = plt.plot(x1,y_hat,lw = 4, c ='orange',label = 'regression line')

plt.xlabel('SAT',fontsize = 20)
plt.ylabel('GPA',fontsize = 20)
plt.show()

#%%
real_estate_data = pd.read_csv(path+'\\real_estate_price_size.csv')

# linear regression
# relationship between price of a house and its size
# f(w,b) = wx+ b

x1 = real_estate_data['size']
y = real_estate_data['price']
plt.scatter(x1,y)
plt.xlabel('SIZE',fontsize = 20)
plt.ylabel('PRICE',fontsize = 20)
plt.show()

# We will initialize w_0 as a constant

w = sm.add_constant(x1)

# regression model
results = sm.OLS(y,w).fit()
results.summary()
#const       1.019e+05 
#size         223.1787 

y_hat = x1*223.1787 + 1.019e+05 

fig = plt.plot(x1,y_hat,lw = 4, c ='orange',label = 'regression line')

plt.xlabel('SIZE',fontsize = 20)
plt.ylabel('PRICE',fontsize = 20)
plt.show()

#%%
"""
Like the R-squared, the adjusted R-squared measures how well your model fits the data. 
However, it penalizes the use of variables that are meaningless for the regression.

Almost always, the adjusted R-squared is smaller than the R-squared.
 The statement is not true only in the extreme 
occasions of small sample sizes and a high number of independent variables.

"""
#%%
#multiple linear regression
# delaing with categorical variables
raw_data = pd.read_csv(path+'\\SAT_Dummies.csv')

data = raw_data.copy()
# now we transformed the categorical variables
data['Attendance'] = data['Attendance'].map({'Yes' : 1,'No': 0})
data.describe()
#%%
'''
regression model
'''

# Following the regression equation, our dependent variable (y) is the GPA
y = data ['GPA']
# Similarly, our independent variable (x) is the SAT score
x1 = data [['SAT','Attendance']]

x= sm.add_constant(x1)

results = sm.OLS(y,x).fit()
results.summary()
plt.scatter(data['SAT'],y)

#const          0.6439  
#SAT            0.0014 
#Attendance     0.2226 
# y_hat = wx1 + wx2 +... wx_n + b


# Define the two regression equations, depending on whether they attended (yes), or didn't (no)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
# Plot the two regression lines
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
# Name your axes :)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

#%%
# In this code I want to colour the points depending on attendance
# Note: This code would have been very easy in Seaborn

# Create one scatter plot which contains all observations
# Use the series 'Attendance' as color, and choose a colour map of your choice
# The colour map we've chosen is completely arbitrary
plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'],cmap='RdYlGn_r')

# Define the two regression equations (one with a dummy = 1, the other with dummy = 0)
# We have those above already, but for the sake of consistency, we will also include them here
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']

# Plot the two regression lines
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
#%%
# Same as above, this time we are including the regression line WITHOUT the dummies.

# In this code I want to color the points depending on attendance
# Note: This code would have been very easy in Seaborn

# Create one scatter plot which contains all observations
# Use the series 'Attendance' as color, and choose a colour map of your choice
# The colour map we've chosen is completely arbitrary
plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'],cmap='RdYlGn_r')

# Define the two regression equations (one with a dummy = 1, the other with dummy = 0)
# We have those above already, but for the sake of consistency, we will also include them here
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
# Original regression line
yhat = 0.0017*data['SAT'] + 0.275

# Plot the two regression lines
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837', label ='regression line1')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026', label ='regression line2')
# Plot the original regression line
fig = plt.plot(data['SAT'],yhat, lw=3, c='#4C72B0', label ='regression line')

plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

#%%



#%%



#%%


'''
multiple regression
'''
orig_data = pd.read_csv(path+'\\real_estate_price_size_year_view.csv')
# price of house depending on size,year and view
data = orig_data.copy()
data['view'] = data['view'].map({'No sea view' : 0, 'Sea view' : 1})

x1 = data[['size', 'year','view']]
y = data['price']

x = sm.add_constant(x1)

plt.scatter(data['size'],y)
plt.xlabel('size',fontsize = 20)
plt.ylabel('price',fontsize = 20)
plt.show()
# regression model

results = sm.OLS(y,x).fit()
results.summary()
#const      -5.772e+06
#size         227.7009
#year        2916.7853
#view        5.673e+04

y_hat = -5.772e+06 +  227.7009*data['size']+2916.7853*data['year'] + 5.673e+04*data['view']

yhat_yes_view = -5715270.0 + 227.7009*data['size']+2916.7853*data['year'] 

yhat_no_view =  -5.772e+06 +  227.7009*data['size']+2916.7853*data['year']

fig = plt.plot(data['size'],yhat_no_view, lw=2, c='#006837')
fig = plt.plot(data['price'],yhat_yes_view, lw=2, c='#a50026')
plt.xlabel('size', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.show()

# we can check the adjusted R with view and without view and see if it addition is useful

#%%
'''
make prediction
'''
new_data = pd.DataFrame({'const' : 1, 'SAT': [1700,1670],'Attendance' : [0,1]})
new_data = new_data[['const', 'SAT','Attendance']]
new_data.rename(index={0: 'Bob', 1:'Alice'})
prediction = results.predict(new_data)
predictionsdf = pd.DataFrame({'Predictions': prediction})
joined = new_data.join(predictionsdf)
joined.rename(index= {0:'Bob',1: 'Alice'})