# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:20:12 2023

@author: PKA232
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#importing data

my_df = pd.read_csv("data/sample_data_regression.csv")

#splitting data into dependednt and independednt 

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]

#Splitting data into test set and train set

#regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Instantaion of Model
regressor = LinearRegression()

#training the model 
regressor.fit(X_train,y_train)

#assessing model accuracy 
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)











