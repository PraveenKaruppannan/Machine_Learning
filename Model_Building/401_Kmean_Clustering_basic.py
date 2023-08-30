# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:20:14 2023

@author: PKA232
"""

#Imoprt package

import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#importing data 

my_df = pd.read_csv("data/sample_data_clustering.csv")

#plot the data 

plt.scatter(my_df["var1"],my_df["var2"])
plt.xlabel("var1")
plt.ylabel("var2")
plt.show()


# instantiate & fit the model

kmean = KMeans(n_clusters = 3,random_state=42)
kmean.fit(my_df)

# Add the cluster lable to our df

my_df["cluster"] = kmean.labels_
my_df["cluster"].value_counts()


# plot our cluster and centroid

centroids = kmean.cluster_centers_
print(centroids)

clusters = my_df.groupby("cluster")
for cluster , data in clusters :
    plt.scatter(data["var1"],data["var2"],marker="o",label = cluster)
    plt.scatter(centroids[cluster,0],centroids[cluster,1],marker="X",color = "black",s=300)
plt.legend()
plt.tight_layout()
plt.show()


