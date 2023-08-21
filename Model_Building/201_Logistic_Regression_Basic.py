# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:59:49 2023

@author: PKA232
"""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
#importing data

my_df = pd.read_csv("data/sample_data_classification.csv")

#splitting data into dependednt and independednt 

y = my_df["output"]

#Splitting data into test set and train set

#regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

#Instantaion of Model
clf = LogisticRegression(random_state = 42)

#training the model 
clf.fit(X_train,y_train)

#assessing model accuracy 
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


y_pred_prob = clf.predict_proba(X_test)

#confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

import numpy as np
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion_matrix")
plt.ylabel("Actual class")
plt.xlabel("Predicted Class")
for (i,j),corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i,corr_value, ha = "center",va = "center",fontsize = 20)
plt.show()


0.25*0.25
#Amit Khandelwal







