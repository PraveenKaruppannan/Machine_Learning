# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:43:03 2023

@author: PKA232
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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
regressor = DecisionTreeRegressor(min_samples_leaf=7)

#training the model 
regressor.fit(X_train,y_train)

y_predred = regressor.predict(X_test)
r2_score(y_test, y_predred)

#assessing model accuracy 
y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)

#plottting the Tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True,fontsize=24)
















