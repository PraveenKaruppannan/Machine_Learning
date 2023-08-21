# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:36:24 2023

@author: PKA232
"""

#Import package

import pandas as pd
import pickle


#import customer for scoring 
to_be_scored = pickle.load(open("data/regressor_scoring.p","rb"))


#import Model and model object 
regressor = pickle.load(open("data/random_forest_regression_mode.p","rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p","rb"))


#drop unused column 

to_be_scored.drop(["customer_id"],axis =1 , inplace = True)

# drop missing value
to_be_scored.dropna(how="any",inplace = True)

#apply one hot encoding 

cat_var = ["gender"]
encoder_var_array = one_hot_encoder.transform(to_be_scored[cat_var]) 
encoder_var_name = one_hot_encoder.get_feature_names(cat_var)#getting the column name from the dataframe
encoder_df = pd.DataFrame(encoder_var_array,columns=encoder_var_name) #creating DF and combaining the column name (encoded data)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_df.reset_index(drop=True)],axis =1) #combaining the hot encoided data with souce Df
to_be_scored.drop(cat_var,axis=1,inplace=(True)) #dropping the main column

# Making our prediction 

loyalty_predictions = regressor.predict(to_be_scored)

