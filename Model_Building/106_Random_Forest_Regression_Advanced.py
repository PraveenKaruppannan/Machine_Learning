# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:19:36 2023

@author: PKA232
"""

import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


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

regressor = RandomForestRegressor(random_state=42)
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


#Feature importance 

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_name = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_name,feature_importance],axis = 1)
feature_importance_summary.columns = ["input_variable","Feature_importance"]
feature_importance_summary.sort_values(by = "Feature_importance",inplace = True)


plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["Feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

#permutation Importance 

result = permutation_importance(regressor,X_test,y_test,n_repeats=10,random_state=42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_name = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_name,feature_importance],axis = 1)
permutation_importance_summary.columns = ["input_variable","Feature_importance"]
permutation_importance_summary.sort_values(by = "Feature_importance",inplace = True)


plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["Feature_importance"])
plt.title("permutation Importance of Random Forest")
plt.xlabel("permutation Importance")
plt.tight_layout()
plt.show()



#predictions under the hood 

y_pred[0]
new_date = [X_test.iloc[0]]
regressor.estimators_


predictions =[]
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_date)[0]
    predictions.append(prediction)
    tree_count += 1

print(predictions)
sum(predictions)/tree_count

import pickle

pickle.dump(regressor, open("data/random_forest_regression_mode.p","wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p","wb"))







