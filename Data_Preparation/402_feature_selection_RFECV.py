# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:55:46 2023

@author: PKA232
"""

#recursive feature elimination with cross validation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

mydf = pd.read_csv("feature_selection_sample_data.csv")

#creating varibable to test the dependednt variable for deriving the output 
X = mydf.drop(["output"],axis=1)
y = pd.DataFrame(mydf["output"])

regressor = LinearRegression()
feature_selector= RFECV(regressor) #we can add CV(cross validation) value which is number of column required for creating this method default is 5

fit = feature_selector.fit(X, y)

#getting the number of variable required to predict out (we passed 4 column and below method will let how much variable is required to pridect output Correlation)
optimal_featuer_count = feature_selector.n_features_  
print(f"Optimal number of featue : {optimal_featuer_count}")

#creating new Datframe based on the model
X_new = X.loc[:,feature_selector.get_support()]

#plotting this to see the out code
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_featuer_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()