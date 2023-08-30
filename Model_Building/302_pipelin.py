# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:50:00 2023

@author: PKA232
"""

#import required python package
import pandas as pd
import numpy as py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import  train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#import sample Data

my_df = pd.read_csv("data/pipeline_data.csv")

#splitting data into dependednt and independednt 

X = my_df.drop(["purchase"],axis = 1)
y = my_df["purchase"]

#Splitting data into test set and train set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


#specify numeric and categorical features 

numeric_feature = ["age","credit_score"]
categorical_feature = ["gender"]


#set up Pipelines 
#numerical feature Transformer 

numeric_transformer = Pipeline(steps = [("imputer",SimpleImputer()),
                                        ("scaler",StandardScaler())])

#categorical feature transformer 
categorical_transformer = Pipeline(steps = [("imputer",SimpleImputer(strategy="constant",fill_value="U")),
                                        ("OHE",OneHotEncoder(handle_unknown="ignore"))])

# preprocessing pipeline 

preprocessing_pipeline = ColumnTransformer(transformers=[("numeric",numeric_transformer,numeric_feature),
                                                         ('categorical',categorical_transformer,categorical_feature)])


#applying the pipeline
#logistic regression 

clf = Pipeline(steps=[("preprocessing_pipeline", preprocessing_pipeline),
                      ("classifier",LogisticRegression(random_state=42))])

clf.fit(X_train,y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)


#Random Forest
clf = Pipeline(steps=[("preprocessing_pipeline", preprocessing_pipeline),
                      ("classifier",RandomForestClassifier(random_state=42))])

clf.fit(X_train,y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

#Save the pipleline
import joblib
joblib.dump(clf,"data/mode.joblib")


#import piplline object  and predict on new data
#import required package
import joblib
import pandas as pd 
import numpy as np

#import pipeline 
clf = joblib.load("data/mode.joblib")
#create new Data 
new_data = pd.DataFrame({"age":[25,np.nan,50],
                         "gender": ["M","F",np.nan],
                         "credit_score": [200,100,500]})
#pass new data in and recive prdections
clf.predict(new_data)









