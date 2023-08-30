# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:22:09 2023

@author: PKA232
"""

#import Required package

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt 

#importing table 
transactions = pd.read_excel("data/grocery_database.xlsx",sheet_name="transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx",sheet_name="product_areas")

#maerge on product area name 

transactions = pd.merge(transactions,product_areas, how="inner",on = "product_area_id")

#drop the non_food category

transactions.drop(transactions[transactions["product_area_name"] =="Non-Food"].index,inplace = True)

# aggregate sales at customer level (by product area)

transactions_summay = transactions.groupby(["customer_id","product_area_name"])["sales_cost"].sum().reset_index()

#pivot data to place product area as column 

transactions_summay_pivot = transactions.pivot_table(index= "customer_id",
                                                     columns = "product_area_name",
                                                     values = "sales_cost",
                                                     aggfunc = "sum",
                                                     fill_value=0,
                                                     margins=True,
                                                     margins_name= "Total").rename_axis(None,axis =1)

# trun sales into % sales

transactions_summay_pivot = transactions_summay_pivot.div(transactions_summay_pivot["Total"],axis = 0)

#drop the "total" column 

data_for_clustring = transactions_summay_pivot.drop(["Total"],axis=1)


#data preparation & cleaning 
# check for missing value
data_for_clustring.isna().sum()

# normalise data
scale_norm = MinMaxScaler()
data_for_clustring_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustring),columns=data_for_clustring.columns)

# use WCSS to find a good value for k

k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(data_for_clustring_scaled)
    wcss_list.append(kmeans.inertia_)
    
plt.plot(k_values,wcss_list)
plt.title("Within Cluster Sum of Squares - by K")
plt.xlabel("K")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()

# instantiate & fit model 
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(data_for_clustring_scaled)

# use cluster informaton 
#add cluster lables to our data 
data_for_clustring["cluster"] = kmeans.labels_

#check cluster size 
data_for_clustring["cluster"].value_counts()

#profile our cluster

cluster_summary = data_for_clustring.groupby("cluster")[['Dairy', 'Fruit', 'Meat', 'Vegetables']].mean().reset_index()



data_for_clustring.columns








