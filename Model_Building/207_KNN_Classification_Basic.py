# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:55:43 2023

@author: PKA232
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#importing data

my_df = pd.read_csv("data/sample_data_classification.csv")

#splitting data into dependednt and independednt 

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]

#Splitting data into test set and train set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

#Instantaion of Model
clf = KNeighborsClassifier()

#training the model 
clf.fit(X_train,y_train)

#assessing model accuracy
y_predred = clf.predict(X_test)
accuracy_score(y_test, y_predred)