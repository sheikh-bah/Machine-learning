# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:10:17 2023

@author: JE93867
"""

'''
the goal of clustering is to maximize the similarity
of observations within a cluster and maximize the dissimilarity between clusters
'''
# choose the number of clusters (k)
# specify the cluster seeds : starting centroid
# assign each point to a centroid
# adjust the centroid
# repeat the last two steps

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

import os
path = os.getcwd()

# load the data
data = pd.read_csv(path+'\\Countries-exercise_cluster.csv')



data.describe()

#%%
# plot the data
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180,180) # natural domain of the longitude
plt.ylim(-90, 90)
plt.show

#%% select the fatures

features = data.iloc[:,1:]
#%%
# clustering
kmeans = KMeans(7)
kmeans.fit(features)

#%%
# clustering results
identified_clusters = kmeans.fit_predict(features)
#%%
data_with_cluster = data.copy()
data_with_cluster['clusters'] = identified_clusters

#%%
plt.scatter(data_with_cluster['Longitude'],data_with_cluster['Latitude'],c =data_with_cluster['clusters'],cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90,90)
plt.show


#%%
'''
using catrgorical variables
'''
data = pd.read_csv(path+'\\Categorical_Kmeans.csv')



data.describe()

data_mapped = data.copy()
data_mapped['continent'] = data_mapped['continent'].map({'North America':0,'Europe':1,'Asia':2,'Africa':3,'South America':4, 'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})

#%%
x = data_mapped.iloc[:,3:4]

kmeans = KMeans(7)
kmeans.fit(x)
#%%
identified_clusters = kmeans.fit_predict(x)

#%%
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()


#%%
# Andrew Ng DON'T USE THE ELBOW METHOD
# ELBOW METHOD
'''
choosing the number of clusters
'''
# SELEFCTING THE NUMBER OF CLUSTERS

# WCSS
kmeans.inertia_

wcss = []

for i in range(1,10):
   
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
    
#%%
    
'''
Elbow method
'''
number_of_clusters = range(1,10)
plt.plot(number_of_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')

# 2 and 3 are the optimal values.
# you can also decide how you would like the cluster to be done
#%%

"""
Do remove outliers
"""