# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:56:52 2023

@author: PKA232
"""

# Import required package 
from apyori import apriori
import pandas as pd


#import Data 
#Import 
alcohol_transaction = pd.read_csv("data/sample_data_apriori.csv")

# Drop ID column
alcohol_transaction.drop("transaction_id",axis = 1,inplace=True)

# Modify data for apriori algorithm 
transaction_list = []

for index, row in alcohol_transaction.iterrows():
    transaction = list(row.dropna())
    transaction_list.append(transaction)
    
# applying the Apriori Algorith 

apriori_rules = apriori(transaction_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2)

apriori_rules = list(apriori_rules)

apriori_rules[0]

#convert output to dataframe

product1 = [list(rule[2][0][0])[0] for rule in  apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in  apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "Support" : support,
                                 "Confident" : confidence,
                                 "Lift" : lift})

#sort rules by decending left

apriori_rules_df.sort_values(by = "Lift", ascending= False,inplace=True)

# search rules

apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]




