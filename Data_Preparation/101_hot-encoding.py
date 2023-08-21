# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:38:25 2023

@author: PKA232
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

my_df = pd.DataFrame({"input1" : [1,2,3,4,5],
                      "input2" : ["A","A","B","B","C"],
                      "input3" : ["X","X","X","Y","Y"]})

cat_var = ["input2","input3"]

one_hot_encoder = OneHotEncoder(sparse=False) # Sparse set to flase as we want output in Array than spase matrix

one_hot_encoder = OneHotEncoder(sparse=False,drop = 'first') #adding drop = first or last will help us avoide getting into dummy variable trap

encoder_var_array = one_hot_encoder.fit_transform(my_df[cat_var]) 

encoder_var_name = one_hot_encoder.get_feature_names(cat_var)#getting the column name from the dataframe

encoder_df = pd.DataFrame(encoder_var_array,columns=encoder_var_name) #creating DF and combaining the column name (encoded data)

my_df_encoder = pd.concat([my_df.reset_index(drop=True), encoder_df.reset_index(drop=True)],axis =1) #combaining the hot encoided data with souce Df

my_df_encoder.drop(cat_var,axis=1,inplace=(True)) #dropping the main column
