# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:28:15 2023

@author: PKA232
"""

import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


################################################################################################
#import data 
data_for_model = pickle.load(open("data/regressor_modeling.p","rb"))

data_for_model.columns

#dropping column
data_for_model.drop(["customer_id"], axis = 1, inplace= True)


################################################################################################
#shuffling - This step is not mandatory but its best parctice

data_for_model = shuffle(data_for_model,random_state=42)


################################################################################################
#dealing with missing value 
data_for_model.isna().sum()

data_for_model.dropna(how = "any",inplace=True)


################################################################################################
#Splitting Input and output variable 

X = data_for_model.drop(["customer_loyalty_score"],axis=1)
y = data_for_model["customer_loyalty_score"]


################################################################################################
#split test and train set data

X_train, X_test,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)


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

regressor = DecisionTreeRegressor(random_state=42,max_depth=4)
regressor.fit(X_train,y_train)

################################################################################################
#model Assesment
y_pred = regressor.predict(X_test)

#Calulate R2
r_squred = r2_score(y_test,y_pred)
print(r_squred)

#cross validation 
cv = KFold(n_splits=4,shuffle=True,random_state=42)
cv_score = cross_val_score(regressor,X_train,y_train,cv = cv,scoring ="r2")
cv_score.mean()

#calculate adjusted R2
num_data_points, num_input_var =X_test.shape
adjusted_r2 = 1 -(1-r_squred)*(num_data_points-1)/(num_data_points-num_input_var-1)
print(adjusted_r2)

#assessing model accuracy 
y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)

#finding the max_depth

max_depth_list = list(range(1,9))
accuracy_score = []

for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth=depth,random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_score.append(accuracy)
    
max_accuracy = max(accuracy_score)
max_accuracy_idx = accuracy_score.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

#plot of max depth 

plt.plot(max_depth_list, accuracy_score)
plt.scatter(optimal_depth,max_accuracy,marker="X",color = "red")
plt.title(f"Accuracy by max Depth \n Optimal Tree Depth :{optimal_depth} (accuracy:{round(max_accuracy,4)})")
plt.xlabel("Max Depth of decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(25,15))
tree = plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True,fontsize=15)


a = "abc"
ab = list(a)

print(ab[::-1])

l = ab.pop(len(ab)-1)

s=[]
for a in ab:
    s.append(a.pop(len(ab)))
print(s)


Rajat Agrawal
Harshit Chauhan















