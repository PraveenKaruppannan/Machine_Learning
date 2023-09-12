# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:24:21 2023

@author: PKA232
"""

# Import Required Package
# pip install pycausalimpact

from causalimpact import CausalImpact
import pandas as pd

# import and Create data
#import data table

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name= "transactions")
campagine_date = pd.read_excel("data/grocery_database.xlsx", sheet_name= "campaign_data")

# Aggregate transactions data to customer, data level

customer_daily_sales = transactions.groupby(["customer_id","transaction_date"])["sales_cost"].sum().reset_index()

# merge on signup flag

customer_daily_sales = pd.merge(customer_daily_sales, campagine_date,how = "inner",on = "customer_id")

# pivot the data to aggregate daily sales by signup group

causal_impact_df = customer_daily_sales.pivot_table(index="transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc="mean")

# Provide a frequency for our DataTimeTindex
causal_impact_df.index.freq = "D"

# for causal impact we need to impacted group in the first column 

causal_impact_df = causal_impact_df[[1,0]]

# rename columns to something more meangingful
causal_impact_df.columns = ["member","non_member"]

# apply causal impact 

pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01","2020-09-30"]

ci = CausalImpact(causal_impact_df,pre_period,post_period
                  )

# plot the impact 

ci.plot()

# extraxct the summary statistics & report 

print(ci.summary())
print(ci.summary(output = "report"))






