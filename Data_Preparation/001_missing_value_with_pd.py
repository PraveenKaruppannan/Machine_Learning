# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:56:32 2023

@author: PKA232
"""

import pandas as pd
import numpy as np

#--------------------------------------------------------------------------------
my_df = pd.DataFrame({"A" : [1,2,3,np.nan,5,np.nan,7],
                      "B" : [4,np.nan,7,np.nan,1,np.nan,2]})

#finding missing value
my_df.isna()
my_df.isna().sum()

#dropping missing value

my_df.dropna()

my_df.dropna(how = "any", subset=  ["B"]) #subset is used to define the column

my_df.dropna(how = "all") # all define where all the column in Row has Na

my_df.dropna(inplace= True)

#----------------------------------------------------------------------------

#replacing missing value

my_df = pd.DataFrame({"A" : [1,2,3,np.nan,5,np.nan,7],
                      "B" : [4,np.nan,7,np.nan,1,np.nan,2]})

my_df.fillna(value=100)

my_df.fillna(value = my_df["A"].mean()) #replace mean of a column to all th na's in df

my_df.fillna(value = my_df.mean()) # repace missing value with mean of each column 

