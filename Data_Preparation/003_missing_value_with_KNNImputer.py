# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:04:53 2023

@author: PKA232
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

my_df = pd.DataFrame({"A" : [1,2,3,4,5],
                      "B" : [1,1,3,3,4],
                      "C" : [1,2,9,np.nan,20]})

knn_imputer = KNNImputer()
knn_imputer = KNNImputer(n_neighbors=2)
knn_imputer = KNNImputer(n_neighbors=2, weights= "distance")
knn_imputer.fit_transform(my_df)

my_df_1 = pd.DataFrame(knn_imputer.fit_transform(my_df),columns=my_df.columns)