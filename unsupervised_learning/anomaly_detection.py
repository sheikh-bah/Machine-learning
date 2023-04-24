# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:10:54 2023

@author: JE93867
"""

import numpy as np
import matplotlib.pyplot as plt


X_train = np.load('X_part1.npy')
X_val = np.load('X_val_part1.npy')
y_val = np.load('y_val_part1.npy')
#%%
print('The first 5 element of X_train are :\n',X_train[:5])
#%%
print('The first 5 element of X_val are :\n',X_val[:5])

#%%
print('The first 5 element of y_val are :\n',y_val[:5])
#%%
#check the dimension of our variables
print('The shape of X_train is :',X_train.shape)
print('The shape of X_val is :',X_val.shape)
print('The shape of y_val is :',y_val.shape)
#%%
'''
Create a scatter plot of the data
'''
plt.scatter(X_train[:,0],X_train[:,1],marker ='x',c ='b')

#set the title
plt.title('The first dataset')

#set the y_axis label
plt.ylabel('Throughput (mb/s)')
#set the x-axis
plt.xlabel('Latency(ms)')

#set the axis range
plt.axis([0, 30, 0 , 30])
plt.show()

#%%
'''
compute the gaussian distribution
'''
# compute mu and then variance
def estimate_gaussian(X):
    m,n = X.shape
    mu = 1/m*np.sum(X,axis = 0)
    var = 1/m*np.sum( (X-mu)**2,axis = 0) 
    return mu,var
#%%

import scipy.linalg as linalg

def multivariateGaussian(X, mu, sigma2):
    #MULTIVARIATEGAUSSIAN Computes the probability density function of the
    #multivariate gaussian distribution.
    #    p = MULTIVARIATEGAUSSIAN(X, mu, sigma2) Computes the probability 
    #    density function of the examples X under the multivariate gaussian 
    #    distribution with parameters mu and sigma2. If sigma2 is a matrix, it is
    #    treated as the covariance matrix. If sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    #

    k = len(mu)

    # turns 1D array into 2D array
    if sigma2.ndim == 1:
        sigma2 = np.reshape(sigma2, (-1,sigma2.shape[0]))

    if sigma2.shape[1] == 1 or sigma2.shape[0] == 1:
        sigma2 = linalg.diagsvd(sigma2.flatten(), len(sigma2.flatten()), len(sigma2.flatten()))

    # mu is unrolled (and transposed) here
    X = X - mu.reshape(mu.size, order='F').T

    p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma2), -0.5) ) * \
        np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))

    return p

#%%
mu,var = estimate_gaussian(X_train)
p =multivariateGaussian(X_train,mu,var)
    
#%%
'''
select the best threshold epsilon
we will select the best e using the F1 score
'''

p_val = multivariateGaussian(X_val,mu,var)

#%%



def select_threshold(y_val,p_val):
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val)-min(p_val))/1000
    for epsilon in np.arange(min(p_val),max(p_val),step_size):
        arr_filters =( p_val < epsilon)
        
        tp = sum((arr_filters==1) & (y_val==1))
        fp = sum((y_val ==0) & (arr_filters ==1))
        fn = sum((y_val == 1) & (arr_filters ==0))
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1 = 2*precision*recall/(precision+recall)
        
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
            
    return best_epsilon,best_F1
    
#%%

# high dimensionality data set
     
X_train_high = np.load('X_part2.npy')
X_val_high = np.load('X_val_part2.npy')
y_val_high = np.load('y_val_part2.npy') 

   #%%
print('The shape of X_train is :',X_train_high.shape)
print('The shape of X_val is :',X_val_high.shape)
print('The shape of y_val is :',y_val_high.shape)

#%%
# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariateGaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high =  multivariateGaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))
      
     