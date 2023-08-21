# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:06:04 2023

@author: PKA232
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
regressor = RandomForestRegressor(random_state=42, n_estimators = 1000)

#training the model 
regressor.fit(X_train,y_train)

#assessing model accuracy 
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)


#Feature importance 

regressor.feature_importances_

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_name = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_name,feature_importance],axis = 1)
feature_importance_summary.columns = ["input_variable","Feature_importance"]
feature_importance_summary.sort_values(by = "Feature_importance",inplace = True)

import matplotlib.pyplot as plt

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["Feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()