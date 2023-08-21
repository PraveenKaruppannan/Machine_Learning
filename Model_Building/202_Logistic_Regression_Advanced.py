# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:25:51 2023

@author: PKA232
"""

import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

################################################################################################
#import data 
data_for_model = pickle.load(open("data/abc_classification_modelling.p","rb"))

data_for_model.columns

data_for_model.drop(["customer_id"], axis = 1, inplace= True)


################################################################################################
#shuffling - This step is not mandatory but its best parctice

data_for_model = shuffle(data_for_model,random_state=42)

#class Balanace 

data_for_model["signup_flag"].value_counts(normalize = True)


################################################################################################
#dealing with missing value 
data_for_model.isna().sum()

data_for_model.dropna(how = "any",inplace=True)


################################################################################################
#outlier detection and clearing
outlier_summary = data_for_model.describe()

data_for_model.columns

outlier_col = ["distance_from_store","total_sales","total_items"]

for column in outlier_col:
    lower_quartail = data_for_model[column].quantile(0.25)
    upper_quartail = data_for_model[column].quantile(0.75)
    iqr = upper_quartail - lower_quartail
    iqr_extended = iqr * 2 #its default value this number can be adjuscted
    min_border = lower_quartail - iqr_extended
    max_border = upper_quartail + iqr_extended
    
    outlier = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outlier)} detected in column {column}")
    
    data_for_model.drop(outlier,inplace=True)

del outlier_summary,outlier_col,lower_quartail,upper_quartail,iqr,iqr_extended,min_border,max_border,outlier,column

################################################################################################
#Splitting Input and output variable 

X = data_for_model.drop(["signup_flag"],axis=1)
y = data_for_model["signup_flag"]


################################################################################################
#split test and train set data

X_train, X_test,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


################################################################################################
#Dealing with catagorical variable one hot encoder 
categorical_var = ["gender"]

one_hot_encoder = OneHotEncoder(sparse=False,drop = 'first') #adding drop = first or last will help us avoide getting into dummy variable trap

X_train_encoder = one_hot_encoder.fit_transform(X_train[categorical_var]) 
X_test_encoder = one_hot_encoder.transform(X_test[categorical_var]) 

X_train_var_name = one_hot_encoder.get_feature_names_out(categorical_var)#getting the column name from the dataframe

X_train_encoder = pd.DataFrame(X_train_encoder,columns=X_train_var_name) #creating DF and combaining the column name (encoded data)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoder.reset_index(drop=True)],axis =1) #combaining the hot encoided data with souce Df
X_train.drop(categorical_var,axis=1,inplace=(True)) #dropping the main column

X_test_encoder = pd.DataFrame(X_test_encoder,columns=X_train_var_name) #creating DF and combaining the column name (encoded data)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoder.reset_index(drop=True)],axis =1) #combaining the hot encoided data with souce Df
X_test.drop(categorical_var,axis=1,inplace=(True)) #dropping the main column


################################################################################################
#Feature selction 

clf = LogisticRegression(random_state=42,max_iter=1000)

feature_selector= RFECV(clf) #we can add CV(cross validation) value which is number of column required for creating this method default is 5

fit = feature_selector.fit(X_train, y_train)

#getting the number of variable required to predict out
optimal_featuer_count = feature_selector.n_features_  
print(f"Optimal number of featue : {optimal_featuer_count}")

#creating new Dataframe based on the model
X_train = X_train.loc[:,feature_selector.get_support()]
X_test = X_test.loc[:,feature_selector.get_support()]

#plotting this to see the out code
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_featuer_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()


################################################################################################
#Model Training

clf = LogisticRegression(random_state=42,max_iter=1000)
clf.fit(X_train,y_train)

################################################################################################
#model Assesment

#assessing model accuracy 
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

#confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)


plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion_matrix")
plt.ylabel("Actual class")
plt.xlabel("Predicted Class")
for (i,j),corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i,corr_value, ha = "center",va = "center",fontsize = 20)
plt.show()

#accuracy (the number correct classification out of all attempted classification)

accuracy_score(y_test, y_pred_class)

# precision (of all observation that where predicted as positive , how many actually positive)

precision_score(y_test, y_pred_class)

# Recall (of all ositive observation, hiw many did we predict as postive)

recall_score(y_test, y_pred_class)

#f1- Score (the harmonic mean of precision and recall)

f1_score(y_test, y_pred_class)


##############
#find the optimal threshold 
##################

thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    pred_class =(y_pred_prob >= threshold) * 1
    precision = precision_score(y_test, pred_class, zero_division = 0)
    precision_scores.append(precision)
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)

max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)

plt.style.use("seaborn-poster")
plt.plot(thresholds,precision_scores, label = 'precision', linestyle = "--")
plt.plot(thresholds,recall_scores,label = 'Recall', linestyle = "--")
plt.plot(thresholds,f1_scores,label = 'F1', linewidth = 5)
plt.title(f"Finding the optimal threshold for classifiaction Model \n Max F1 :{round(max_f1,2)} (Threshold = {round(thresholds[max_f1_idx],2)}")
plt.xlabel("Threshold")
plt.ylabel("Assessement Score")
plt.legend(low = "Lower left")
plt.tight_layout()
plt.show()


optimal_threshold = 0.44
y_pred_class_opt_thresh = (y_pred_prob >= optimal_threshold)*1























