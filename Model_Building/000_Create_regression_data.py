# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:53:50 2023

@author: PKA232
"""

#importing required package
import pandas as pd
import pickle

#mport data 

loyalty_score = pd.read_excel("data\grocery_database.xlsx", sheet_name = "loyalty_scores" )
transactions = pd.read_excel("data\grocery_database.xlsx", sheet_name = "transactions" )
customer_details = pd.read_excel("data\grocery_database.xlsx", sheet_name = "customer_details" )


#customer level of data
#combaining customer data and Loyelty score
data_for_regression = pd.merge(customer_details,loyalty_score,how = "left",on ="customer_id")

#cenerating sales summary
sale_summay = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                       "num_items" : "sum",
                                                       "transaction_id" : "count",
                                                       "product_area_id": "nunique"}).reset_index()

#Naming the Summary df for bette clarity
sale_summay.columns = ['customer_id', 'total_sale', 'total_items', 'transaction_count','product_area_count']

#Calculating average back per customer and addeding that as column to df
sale_summay["average_basket_value"] = sale_summay["total_sale"]/sale_summay["transaction_count"]

#combaining the customer data and sales summay for model preperation
data_for_regression = pd.merge(data_for_regression,sale_summay,how = "inner",on ="customer_id")

#data_for_regression.columns

#splitting the data into two df one for model praperation and another with loyelty of na 
regressor_modeling= data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regressor_scoring= data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

#Dropping Loyalty column as it does not have value which will predicted
regressor_scoring.drop(["customer_loyalty_score"], axis=1, inplace = True)

#saving the ouput as pickel file 

pickle.dump(regressor_modeling, open("data/regressor_modeling.p", "wb"))
pickle.dump(regressor_scoring, open("data/regressor_scoring.p", "wb"))




