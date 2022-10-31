# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:21:46 2017

@author: Administrator
"""


import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_mutual_info_score


os.chdir(r'C:\Users\Yung\Desktop\사상체질')

df = pd.read_excel('raw)체질확진자_자료(만나이추가).xlsx', index_col='dummy ID', 
                  sheet_name = '분석대상_최종_disc')
feature_list1 = list(df.columns)

df_class = pd.read_excel('feature_to_class.xlsx')
dict_class = df_class.set_index('feature')['class'].to_dict()



mat_MI = pd.DataFrame(np.zeros((len(feature_list1),len(feature_list1))))

for i in range (len(feature_list1)):
    for j in range(len(feature_list1)):
        if i==j:
            mat_MI[i][j] = 1
            
        else:
            df1 = df[feature_list1[i]]
            df2 = df[feature_list1[j]]        
            MI_score = adjusted_mutual_info_score(df1,df2)        

            mat_MI[i][j] = MI_score

mat_MI.to_excel('mat_MI(5bin).xlsx')


df_fre = pd.DataFrame(columns = ['feature1','feature2','adj_MI_score'])
for k in range (len(feature_list1)):
    mi_score = pd.DataFrame(np.zeros((249, 3)), columns = ['feature1','feature2',
                                 'adj_MI_score'])
    mi_score['feature1'] = feature_list1[k]
    mi_score['feature2'] = feature_list1
    mi_score['adj_MI_score'] = mat_MI.iloc[:,k]
    
    df_fre = pd.concat([df_fre, mi_score])
        
df_fre.to_excel('MI_score_table(5bin).xlsx')

# MI_score 중 같은 subgraph 밖에 있는 edge만 추출하기 - 불필요!
#df_fre = pd.read_excel('MI_score_table.xlsx')
#other_mat = pd.DataFrame(np.zeros((249,249)), index=feature_list1 , columns=feature_list1)
#MI_table = df_fre.reset_index(drop=True)
#
#class1 = MI_table['class1']
#class2 = MI_table['class2']
#
#other_edge = (class1 != class2)
#MI_table_other = MI_table[:][other_edge]
#
#for i in range(len(MI_table_other)):
#    a,feature1,b,feature2,c,d = MI_table_other.iloc[i]
#    other_mat.loc[feature1][feature2] = c
#
#other_mat.to_excel('MI_score_edge(5bin).xlsx')