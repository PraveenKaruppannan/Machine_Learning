# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:28:44 2023

@author: PKA232
"""

#model Validation 

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

mydf = pd.read_csv("feature_selection_sample_data.csv")

X = mydf.drop(["output"],axis =1)
y = mydf["output"]

#regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

#classification model
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#cross validation 

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

cv_score = cross_val_score(regressor, X,y,cv = 4,scoring="r2")
cv_score.mean()

#using regression 

cv = KFold(n_splits=4,shuffle=True,random_state=42)
cv_score = cross_val_score(regressor, X,y, cv =cv, scoring="r2")
cv_score.mean()

#using classification 
cv = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
cv_score = cross_val_score(clf, X,y, cv =cv, scoring="accuracy")
cv_score.mean()








