# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:37:17 2023

@author: PKA232
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#importing data

my_df = pd.read_csv("data/sample_data_regression.csv")

#splitting data into dependednt and independednt 

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]


#Instatiate our SridSearch Ogject 

gscv = GridSearchCV(estimator = RandomForestRegressor(random_state=42),
                    param_grid = {"n_estimators" : [10,50,100,500],
                                  "max_depth" : [1,2,3,4,5,6,7,8,9,10,None]},
                    cv = 5,
                    scoring="r2",
                    n_jobs=-1 # help us all the processing power of the computer
                    )

#Fit to data 

gscv.fit(X,y)


#get the best cv score (mean)

gscv.best_score_

#optimal Parameter 

gscv.best_params_

#create optinal model object
regressor = gscv.best_estimator_
