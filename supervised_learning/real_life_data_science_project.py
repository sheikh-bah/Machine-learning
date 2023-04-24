# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:50:43 2023

@author: JE93867
"""
#%%
from lab_utils_common import dlc
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import os 
#%%
current_path = os.getcwd()
load_data = pd.read_csv(current_path + "\\1.04.+Real-life+example.csv")
load_data.shape #(4345, 9)
describe = load_data.describe(include = 'all')
# the bran count is 4345, price 4173,body 4345,mileage 4345,EngineV 4195,
# Engine Type 4345, Registration 4345,Year 4345, Model 4345,
# from this we can conclude that we have missing values for price, EngineV
load_data = load_data.drop(['Model'],axis = 1)
'''
check for missing values
'''
null_data = load_data.isnull().sum()
# we can remove these null values. The rule of thumb is if you are 
# removing less than 5% of the observations, you are free to just remove them
# total_price count is 4173, missing is 172 which is 4%
# enginve V 150 missing , that is 3%
data_no_mv = load_data.dropna(axis = 0)
new_describe = data_no_mv.describe(include = 'all')

#%%
'''
exploring the PDFs
'''
# we will be looking for a normal distribution
sns.distplot(data_no_mv['Price'])
# price has an exponential distribution
# it has a mean 19552, minimum 600,  max 3000000, the max is very far from the mean
# we tried to make the distribution normal by removing outliers
'''
dealing with outliers
'''
q = data_no_mv['Price'].quantile(0.99)
# we will remove the 99% quantile
data_1 = data_no_mv[data_no_mv['Price']<q]
sns.distplot(data_1['Price']) # the max is 129222 and the mean is 17837, that is far better
new_describe = data_1.describe(include = 'all')

sns.distplot(data_no_mv['Mileage'])
# we can remove the 99% quantile for mileage also
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]

sns.distplot(data_2['Mileage'])

# here we deal with engineVolume
sns.distplot(data_no_mv['EngineV'])

engV = pd.DataFrame(load_data['EngineV'])
engv = engV.dropna(axis = 0)
engv_sort = engv.sort_values(by = 'EngineV')
# engine volume is between 0.6 to 6.5
# so we can remove the incorrect entries

data_3  = data_2[data_2['EngineV'] <=6.5]

sns.distplot(data_3['EngineV'])


# YEAR feature

sns.distplot(data_3['Year'])
# we can keep the new cars and remove the outlier (old cars) from 1970 to 1987
# we will take the 1%
q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year']> q]
sns.distplot(data_4['Year'])
#%%
'''
reset the index
'''
data_cleaned = data_4.reset_index(drop = True)
description = data_cleaned.describe()

#%% LOOK FOR LINEARITY
'''
CHECKING THE OLS assumption
'''

f,(ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and Engine volume')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()

'''
lets look at price
'''
sns.distplot(data_cleaned['Price'])
# price is not normally distributed therefore its relationship with the other 
# variables that are normally distributed is not Linear
'''
log transformation is useful when dealing such relationships

'''
#%% RELAXING THE ASSUMPTIONS
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
f,(ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('log Price and Engine volume')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('log Price and Mileage')

plt.show()
#%%
# we drop the price since its no longer needed
data_cleaned = data_cleaned.drop(['Price'],axis = 1)
#%%
# NO ENDOGENEITY
#%% NORMALITY AND HOMOSCEDACITY
#%% No Autocorolleation ( this not a time series)
#%% MULTICOLLINEARITY
# variance inflation factor ( VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
# vif 10 < vif unacceptable
'''
VIF	                features
3.7922997858318643	Mileage
7.638285534314808	EngineV
10.332226439429983	Year

'''
# seems like year is too correlated with the other variables
# so, we will drop year
data_no_multicollinearity = data_cleaned.drop(['Year'], axis = 1)

#%%
'''
if we have N categories for a feature, we have to create N-1 dummies
'''
# create data with dummies
data_with_dummies = pd.get_dummies(data_no_multicollinearity,drop_first = True) # we will not create for Audi
#%%

'''
rearrange a bit
'''
# we will look for our column attributes
data_with_dummies.columns.values
'''
['Mileage', 'EngineV', 'log_price', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

'''
cols = ['log_price','Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]


#%%
'''
Linear regression model
'''


# we want to predict the price of a car based on the model type
# we will convert the Brand which are in categorical to numerical
# we will also convert the Engine type to numerical
#['Petrol', 'Diesel', 'Gas', 'Other']
#['BMW', 'Mercedes-Benz', 'Audi', 'Toyota', 'Renault', 'Volkswagen',
 #      'Mitsubishi'],
 # registration [yes, no]
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis = 1)
# we want to rescale( normalize our data)
scaler = StandardScaler()
scaler.fit(inputs)
x_scaled = scaler.transform(inputs)


# split the data to training and testing
x_train,x_test,y_train,y_test  = train_test_split(x_scaled,targets,test_size = 0.2, random_state = 42)

# build our regression model

reg = LinearRegression()
reg.fit(x_train,y_train) # log linear regression as our y_train is the logrithm of price

# we want to see if the model has learn something useful
# we will compare it to our y_train data

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)
plt.xlabel('targets(y_train)', size = 18)
plt.ylabel('predictions(y_hat)', size = 18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

sns.distplot(y_train - y_hat)  # errors must be noramilty distributed with a mean of 0
plt.title('Residual PDF', size = 18)
reg.score(x_train, y_train)
#%%
'''
weight and bias
'''
reg.coef_
reg.intercept_

reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])

reg_summary['Weights'] = reg.coef_  # a positive weight shows that as a feature increases
#in value, so do the log price  and price respectively
# a negative weight shows that as a feature increases in value, log_price and price decreases
# DUMMIES: Since we have drop one category for each discrete variable
#Audi is the drop one, therefore when ever all other dummies are zero
# Audi is 1, so Audi is the benchmark
# a positive weight for a dumm shows that the respective category (BRAND)
# is more expensive than the benchmark(Audi)
# vice versa for negative
# now we want to predict using our test data
 # mileage is the most important
 
 #%%
'''
TESTING
'''
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test,alpha = 0.2)
plt.xlabel('targets(y_train)', size = 18)
plt.ylabel('predictions(y_hat)', size = 18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#%% data frame performance
df_pf = pd.DataFrame(np.exp(y_hat_test), columns =['Predictions'])
#exp(ln(x)) = |x|
# log(exp(x)) = x
y_test = y_test.reset_index(drop = True)

df_pf['Target'] = np.exp(y_test)


df_pf['Residual'] = df_pf['Target'] -df_pf['Predictions']

df_pf['Differences'] = np.absolute(df_pf['Residual']/ df_pf['Target']* 100)


final_descriptions = df_pf.describe()
#%%
pd.set_option('display.precision', 2)
df_pf_sort = df_pf.sort_values(by = ['Differences'])


#%%
'''
we will try to improve our model.
we will use Stochastic gradient descent

'''
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_train, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
#%%
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
#%%
'''
Make predictions
Predict the targets of the training data.
Use both the predict routine and compute using w and b 
.
'''
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_test)

y_pred = np.dot(x_train, w_norm) + b_norm  
print(f'prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd)}')

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

#%%

# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,16,figsize=(8,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(inputs.columns.values[i])
    ax[i].scatter(x_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
    
ax[0].set_ylabel("log_price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()