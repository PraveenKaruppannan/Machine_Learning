# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:51:22 2023

@author: PKA232
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#importing data

my_df = pd.read_csv("data/sample_data_classification.csv")

#splitting data into dependednt and independednt 

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]

#Splitting data into test set and train set

#regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

#Instantaion of Model
clf = DecisionTreeClassifier(random_state = 42, min_samples_leaf=7)

#training the model 
clf.fit(X_train,y_train)

y_predred = clf.predict(X_test)
accuracy_score(y_test, y_predred)

#assessing model accuracy 
y_pred_training = clf.predict(X_train)
accuracy_score(y_train, y_pred_training)

#plottting the Tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(clf, feature_names=X.columns, filled=True, rounded=True,fontsize=24)
