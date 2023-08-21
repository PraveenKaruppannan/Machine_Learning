# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:21:46 2023

@author: PKA232
"""

import pandas as pd


my_df = pd.DataFrame({"input1" : [15,41,44,47,50,53,56,59,99],
                      "input2" : [29,41,44,47,50,53,56,59,66]})

#outlier detection and processing the data using BOX Plot method

my_df.plot(kind="box",vert=False)

outlier_col = ["input1","input2"]

for column in outlier_col:
    lower_quartail = my_df[column].quantile(0.25)
    upper_quartail = my_df[column].quantile(0.75)
    iqr = upper_quartail - lower_quartail
    iqr_extended = iqr*1.15 #its default value
    min_border = lower_quartail - iqr_extended
    max_border = upper_quartail + iqr_extended
    
    outlier = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outlier)} detected in column {column}")
    
    my_df.drop(outlier,inplace=True)
    
%reset -f

#Standared deviation methoder to process outlier

my_df = pd.DataFrame({"input1" : [15,41,44,47,50,53,56,59,99],
                      "input2" : [29,41,44,47,50,53,56,59,66]})


outlier_col = ["input1","input2"]
for column in outlier_col:

    mean = my_df[column].mean()
    std_dev = my_df[column].std()
    
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    outlier = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outlier)} detected in column {column}")
    
    my_df.drop(outlier,inplace=True) 
    
    