# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:35:25 2023

@author: PKA232
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

my_df = pd.DataFrame({"A" : [1,4,7,10,13],
                      "B" : [3,6,9,np.nan,15],
                      "C" : [2,5,np.nan,11,np.nan]})


imputer = SimpleImputer(strategy="median") # can be replaced with mean(default), most_frequent,constant for static value 

imputer.fit(my_df)
imputer.transform(my_df)

my_df_1 = pd.DataFrame(imputer.fit_transform(my_df), columns=my_df.columns)