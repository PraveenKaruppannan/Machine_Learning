# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:02:35 2023

@author: PKA232
"""

import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

################################################################################################
#import data 
data_for_model = pickle.load(open("data/regressor_modeling.p","rb"))

data_for_model.columns

data_for_model.drop(["customer_id"], axis = 1, inplace= True)


################################################################################################
#shuffling - This step is not mandatory but its best parctice

data_for_model = shuffle(data_for_model,random_state=42)


################################################################################################
#dealing with missing value 
data_for_model.isna().sum()

data_for_model.dropna(how = "any",inplace=True)


################################################################################################
#outlier detection and clearing
outlier_summary = data_for_model.describe()

outlier_col = ["distance_from_store","total_sale","total_items"]

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
#Feature selction 

regressor = LinearRegression()
feature_selector= RFECV(regressor) #we can add CV(cross validation) value which is number of column required for creating this method default is 5

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

regressor = LinearRegression()
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

#extract mode coeffienct 
coeff = pd.DataFrame(regressor.coef_)
input_car = pd.DataFrame(X_train.columns)
summary_stat = pd.concat([input_car,coeff], axis = 1)
summary_stat.columns =["input_variable","coeff"]


#extract model intersept
regressor.intercept_

