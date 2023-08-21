# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:14:09 2023

@author: PKA232
"""

import pandas as pd

mydf = pd.read_csv("feature_selection_sample_data.csv")

from sklearn.feature_selection import SelectKBest,f_regression

X =mydf.drop(["output"], axis = 1)
y = mydf["output"]

#regression
feature_selector = SelectKBest(f_regression, k= "all")
fit = feature_selector.fit(X, y)

fit.pvalues_
fit.scores_

p_value = pd.DataFrame(fit.pvalues_)
Score = pd.DataFrame(fit.scores_)
input_variable = pd.DataFrame(X.columns)
Summary = pd.concat([input_variable,p_value,Score],axis=1)
Summary.columns = ["input_variable","p_value","Score"]

Summary.sort_values(by ="p_value",inplace=True)

p_value_thresholde = 0.05
score_thresholde = 5

Selected_var = Summary.loc[(Summary["p_value"] <= p_value_thresholde) & (Summary["Score"] >= score_thresholde)]

Selected_var = Selected_var["input_variable"].tolist()

X_new = X[Selected_var]

# Classification 
from sklearn.feature_selection import SelectKBest,chi2

X =mydf.drop(["output"], axis = 1)
y = mydf["output"]

feature_selector = SelectKBest(chi2, k= "all")
fit = feature_selector.fit(X, y)

fit.pvalues_
fit.scores_

p_value = pd.DataFrame(fit.pvalues_)
Score = pd.DataFrame(fit.scores_)
input_variable = pd.DataFrame(X.columns)
Summary = pd.concat([input_variable,p_value,Score],axis=1)
Summary.columns = ["input_variable","p_value","chi1_Score"]

Summary.sort_values(by ="p_value",inplace=True)

p_value_thresholde = 0.05
score_thresholde = 5

Selected_var = Summary.loc[(Summary["p_value"] <= p_value_thresholde) & (Summary["chi1_Score"] >= score_thresholde)]

Selected_var = Selected_var["input_variable"].tolist()

X_new = X[Selected_var]







