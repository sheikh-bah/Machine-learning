# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:02:55 2023

@author: JE93867
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import os
#%%
path = os.getcwd()
# load the data
'''
The data is based on the marketing campaign efforts of a Portuguese banking
institution. The classification goal is to predict if 
the client will subscribe a term deposit (variable y).
'''
raw_data = pd.read_csv(path + '\\Example_bank_data.csv')
data = raw_data.copy()
# now we transformed the categorical variables
data['y'] = data['y'].map({'yes' : 1,'no': 0})
data.describe()


x1 = data['duration']
y = data['y']

plt.scatter(x1,y)

x = sm.add_constant(x1)



log_reg = sm.Logit(y,x).fit()

# Current function value shows the objective function
# the maximum iteration in stats model is 35
summary  = log_reg.summary()
# the bigger the Log Likelihood the better
# LL-null : of a model that has no independent variable
# LLR_p-value measures if our model is stastically different
#from the the Loglikelihood null ( i.e useless model)
# pseduo R_squared should be between 0.2 and 0.4
#const         -1.7001
#duration       0.0051 

x_test = data['Unnamed: 0']



plt.scatter(x1,y,color = 'C0')

# Don't forget to label your axes!
plt.xlabel('Duration', fontsize = 20)
plt.ylabel('Subscription', fontsize = 20)
plt.show()

#%% SKLEARN LOGISTIC REGRESSION
lr_model = LogisticRegression()
lr_model.fit(x, y)

y_pred = lr_model.predict(x)
print("Accuracy on training set:", lr_model.score(x, y))

#%%
'''
Binary predictors
'''

raw_data = pd.read_csv(path + '\\2.02.+Binary+predictors.csv')
# we map the categorical variable
data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({'No' :0, 'Yes' : 1})
data['Gender'] = data['Gender'].map({'Female' : 1, 'Male' : 0})

# male is the based line

#%%
# Declare the independent variable

x1 = data['Gender']
y = data['Admitted']
x = sm.add_constant(x1)
# create regression

log_reg = sm.Logit(y,x).fit()
log_reg.summary()

#LLR p-value:                 6.283e-10
# model is significant
#const         -0.6436
#Gender         2.0786
#Log-Likelihood:                -99.178
'''
our model is (log_odds_1) = -0.6436 + 2.0786*Gender1
             (log_oddd_2) = -0.6436 + 2.0786*Gender2
(log_odds_1) - (log_oddd_2) = 2.0786*(Gender2 - Gender1)
(log_odds_2)/(log_oddd_1) = 2.0786*(Gender2 - Gender1)
# gender has only two possible values, 1 or 0
male is 0 and female is 1
(log_odds_female)/(log_odds_male) = 2.0786*(1-0)

odds_female = e^(2.0786)*odds_male
odds_female = 7.99 * odds_male
'''
#%%
# new regression
x1 = data[['SAT', 'Gender']]
x = sm.add_constant(x1)
log_reg = sm.Logit(y,x).fit()
log_reg.summary()

#SAT           -0.0002
#Gender         1.8173 
#LLR p-value:                 1.423e-08
#Log-Likelihood:                -20.180
# log likelihood bigger, so SAT has significant
'''
odds_female = e^(1.8173)*odds_male
odds_female = 6.9*odds_male
a female has 7 times higher odds to get admitted
'''

#%%

raw_data = pd.read_csv(path + '\\Bank_data.csv')
data = raw_data.copy()
data['y'] = data['y'].map({'yes' : 1, 'no' : 0})

new_data = data.drop(['Unnamed: 0'],axis = 1)
description = new_data.describe()
x1 = new_data.drop(['y'], axis = 1)

y = new_data['y']

# build our model
x = sm.add_constant(x1)
logis_reg = sm.Logit(y,x).fit()

logis_reg.summary()

#LLR p-value:                 7.579e-77
#const            -0.1385
#interest_rate    -0.7802
#credit            2.4028 
#march            -1.8097
#may               0.1946 
#previous          1.2746  
#duration          0.0070

#%%
'''
ACCURACY OF OUR MODEL
'''
np.set_printoptions(formatter ={'float' : lambda x: "{0:.2f}".format(x)})
predicted_values =logis_reg.predict()

'''
confusion matrix
'''
results = logis_reg.pred_table()

cm_df = pd.DataFrame(results)
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index = {0:'Actual 0', 1 : 'Actual 1'})
 # the model made actual prediction in 448 cases out of 518
 # it was confused in 31 + 39 of the cases 
 # it has 86% accuracy


 #%%
cm = np.array(cm_df)
accuracy_train = (cm[0,0] + cm[1,1])/cm.sum()
#%%
def confusion_matrix(data,actual_values,model):
        
        # Confusion matrix 
        
        # Parameters
        # ----------
        # data: data frame or array
            # data is a data frame formatted in the same way as your input data (without the actual values)
            # e.g. const, var1, var2, etc. Order is very important!
        # actual_values: data frame or array
            # These are the actual values from the test_data
            # In the case of a logistic regression, it should be a single column with 0s and 1s
            
        # model: a LogitResults object
            # this is the variable where you have the fitted model 
            # e.g. results_log in this course
        # ----------
        
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and 
        return cm, accuracy
    
confusion_matrix(x,y,logis_reg)

#%%
'''
SKLEARN
i will also check the accuracy using sklearn
'''


lr_model = LogisticRegression()
lr_model.fit(x, y)

y_pred = lr_model.predict(x)
print("Accuracy on training set:", lr_model.score(x, y))