# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:35:37 2023

@author: PKA232
"""
%reset -f
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

my_df = pd.DataFrame({"Height" : [1.98,1.77,1.76,1.80,1.64],
                      "Weight" : [99,81,70,86,82]})


#standrization method of for scaling data
scale_standard = StandardScaler()
scale_standard.fit_transform(my_df).round
my_df_standard = pd.DataFrame(scale_standard.fit_transform(my_df).round(2),columns=my_df.columns)


#normilization 
scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_norm = pd.DataFrame(scale_norm.fit_transform(my_df).round(2),columns=my_df.columns)