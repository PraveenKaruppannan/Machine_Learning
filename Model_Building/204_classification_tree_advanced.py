# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:47:29 2023

@author: PKA232
"""

import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import OneHotEncoder


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
#Model Training

clf = DecisionTreeClassifier(random_state=42,max_depth=5)
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


#finding the max_depth

max_depth_list = list(range(1,15))
accuracy_score = []

for depth in max_depth_list:
    
    clf = DecisionTreeClassifier(max_depth=depth,random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test,y_pred)
    accuracy_score.append(accuracy)
    
max_accuracy = max(accuracy_score)
max_accuracy_idx = accuracy_score.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

#plot of max depth 

plt.plot(max_depth_list, accuracy_score)
plt.scatter(optimal_depth,max_accuracy,marker="X",color = "red")
plt.title(f"Accuracy (f1 score) by max Depth \n Optimal Tree Depth :{optimal_depth} (accuracy:{round(max_accuracy,4)})")
plt.xlabel("Max Depth of decision Tree")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(25,15))
tree = plot_tree(clf, feature_names=X.columns, filled=True, rounded=True,fontsize=15)