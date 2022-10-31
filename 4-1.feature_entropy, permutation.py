# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:09:36 2018

@author: Yung
"""

import time
import os
 # target prediction 폴더로 변경
os.chdir(r'C:\Users\Yung\Desktop\kdc 사상체질\KDC analysis code\feature_network_result')

import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# AMI rank matrix class 

MI_mat = pd.read_excel('mat_MI_nondi.xlsx')
node_list = pd.read_excel('node_list.xlsx')




# rank 넣을 heatmap 설정
rank_mat = pd.DataFrame(np.zeros((248,249)))

for i in range(MI_mat.shape[0]):
    MI = pd.DataFrame(MI_mat[i])
    MI = pd.concat([MI,node_list['node_class']], axis=1)
    MI = MI.drop(MI.index[i])
    sort_class = list(MI.sort_values(i, ascending=False)['node_class'])
    rank_mat[i] = sort_class
    
e = 10**(-6)

## wide range rank thresholded Entropy calculation

# class all combination list (except SC type feature)
class_list = list(set(node_list['node_class']))
class_list.remove('진단')
class_com = []

for q in range(2, len(class_list)):
    for subset in combinations(class_list,q):
        class_com.append(subset)
class_com = list(class_com)

entropy_combination = np.zeros((len(class_com),25))

# by threshold
for j in range(25):
    
    start_time = time.time()
    
    fre_mat = pd.DataFrame(np.zeros((9,249)), index=list(set(node_list['node_class'])))

    rank = rank_mat.iloc[:(j+1)*10]
    fre_class = pd.DataFrame(np.ones((len(rank.index),2)))    
    
    # feature 별 frequency pivot
    for k in range(len(rank.columns)):
        
        fre_class[0] = rank[k]
        fre_pivot = fre_class.pivot_table(values=1, index=0, aggfunc = 'sum')
        fre_pivot.columns = [k]
        
        fre_mat[k] = fre_pivot
        
    # by each combination
    for p in range(len(class_com)):
        
        combi_fre = fre_mat.loc[list(class_com[p])]
       
                   
        array_fre = np.array(combi_fre)
        array_fre = np.nan_to_num(array_fre)

        array_prob = array_fre/len(rank.index)
        array_entropy = - array_prob * np.where(array_prob>0, np.log(array_prob), array_prob)
        
        entropy_sum = array_entropy.sum(axis=0, dtype='float32')
        
        number_over_SC = np.sum((entropy_sum > (entropy_sum[-1] + e)))
        
        entropy_combination[p,j] = number_over_SC
    
    print('%s 번째 threshold 계산 완료'%j)
    print('----- %s s -----' % (time.time()- start_time))
    
combination_prob = (entropy_combination/249)

entropy_combination = pd.DataFrame(combination_prob , index=class_com)


# RdBu_r palette 사용 (AMI heatmap과 style 통일)

sns.heatmap(combination_prob, yticklabels=False, cmap = "RdBu_r", vmin=0, vmax=1, center =0)
#
#entropy_combination.to_excel('SC_type over prob by combination.xlsx')


# combination class mapping by prob
    
entropy_mean = entropy_combination.mean(axis=1)
entropy_sort = entropy_mean.sort_values()

sorted_com = entropy_sort.index

class_mapping = pd.DataFrame((np.zeros((len(sorted_com),len(class_list)))), columns=['인구학적조사', '성격1', '소증', '한열', '병증', '질병', '건강', '체형사진음성'])

for z in range(len(sorted_com)):
    com = sorted_com[z]
    
    for x in range(len(com)):
        com_class = com[x]
        class_mapping[com_class][z] = 1


sns.heatmap(class_mapping, xticklabels=False, yticklabels=False, cmap = "RdBu_r", vmin=0, vmax=1, center =0, cbar=None)
        
class_mapping.to_excel('class_composition by prob.xlsx')        
    
    
    

# edge distribution proportion 및 entropy 계산
Fre_pivot = edge_MI_table.pivot_table(values = 1, index='feature1', aggfunc='sum', columns='class2')
Fre_array = np.array(Fre_pivot)
Fre_array = np.nan_to_num(Fre_array)
Feature_prob = Fre_array/Fre_array.sum(axis=1, keepdims=True)

Feature_entropy = - Feature_prob * np.where(Feature_prob>0, np.log(Feature_prob), Feature_prob)

MI_weighted_entropy = MI_array * Feature_entropy

MI_weighted_entropy_sum = MI_weighted_entropy.sum(axis=1)

entropy = pd.DataFrame(MI_weighted_entropy_sum, columns=['MI_weighted_entropy'], index=MI_pivot.index)









## 모든 조합에서의 entropy by class rank

entropy_rank_df = pd.DataFrame(np.zeros((25,1)))

for j in range(25):
    
    fre_mat = pd.DataFrame(np.zeros((9,249)), index=list(set(node_list['node_class'])))

    rank = rank_mat.iloc[:(j+1)*10]
    fre_class = pd.DataFrame(np.ones((len(rank.index),2)))    
    
    # by feature
    for k in range(len(rank.columns)):
        
        fre_class[0] = rank[k]
        fre_pivot = fre_class.pivot_table(values=1, index=0, aggfunc = 'sum')
        df_pivot = fre_pivot.to_frame()
        df_pivot.columns = [k]
        
        fre_mat[k] = df_pivot
    
            
    fre_mat=fre_mat.drop('진단')
    
    array_fre = np.array(fre_mat)
    array_fre = np.nan_to_num(array_fre)
    
    array_prob = array_fre/len(rank.index)
    array_entropy = - array_prob * np.where(array_prob>0, np.log(array_prob), array_prob)
    entropy_sum = array_entropy.sum(axis=0, dtype='float32')
    entropy_rank = entropy_sum.argsort().argsort()

    SC_type_rank = entropy_rank[-1]

    entropy_rank_df.iloc[j] = SC_type_rank 

entropy_rank_df.to_excel('SC_type entropy_rank.xlsx')
    

## 낮은 prob combination의 class mapping




###################################

## Entropy (Adjusted MI weight)

import time
import os
 # target prediction 폴더로 변경
os.chdir(r'C:\Users\Yung\Desktop\kdc 사상체질\KDC analysis code\feature_network_result')

import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# AMI rank matrix class 

MI_mat = pd.read_excel('mat_MI_nondi.xlsx')
node_list = pd.read_excel('node_list.xlsx')


## Entropy (Adjusted MI weight)

# class all combination list (except SC type feature)
class_list = list(set(node_list['node_class']))
class_list.remove('진단')
class_com = []

for q in range(2, len(class_list)+1):
    for subset in combinations(class_list,q):
        class_com.append(subset)

class_com = list(class_com)

edge_MI_table = pd.read_excel('0. MI_score_table(0over).xlsx')

# feature combi_rank 
SC_rank = pd.DataFrame(np.zeros((len(class_com))),index = class_com)

# feature class 판단 by feature class combination

is_class = pd.DataFrame(np.zeros((247,8)), index = class_com, columns = class_list)

for i in range(len(is_class.index)):
    for j in range(len(is_class.columns)):
        
        if class_list[j] in is_class.index[i]:
            is_class[class_list[j]][is_class.index[i]] = 1


# MI score 계산 및 normalization
MI_pivot = edge_MI_table.pivot_table(values = 'adj_MI_score', index='feature1', aggfunc='sum', columns='class2')

MI_pivot.drop('진단', axis=1, inplace = True)

MI_array = np.array(MI_pivot)
MI_array = np.nan_to_num(MI_array)
  # feature / feature class별 entropy값의 합


# by each combination

for p in range(len(class_com)):
    
    feature_combi = MI_pivot[list(class_com[p])]  
    SC_index = list(feature_combi.index).index('최종진단')
    array_fre = np.array(feature_combi)
    array_prob = array_fre/array_fre.sum(axis=1,keepdims=True)
   
    array_entropy = - array_prob * np.where(array_prob>0, np.log(array_prob), array_prob)
    entropy_sum = array_entropy.sum(axis=1, dtype='float32')
    number_over_SC = np.sum((entropy_sum > (entropy_sum[SC_index] + e)))
    
    SC_rank.iloc[p] = number_over_SC

SC_rank_feature_class = pd.concat([SC_rank, is_class],axis=1)

## SC type rank + 각 feature class 포함여부 dataframe 작성
#SC_rank_feature_class.to_excel('SC_type_rank_class.xlsx')
# 다시 정렬함

SC_rank_class = pd.read_excel('SC_type_rank_class.xlsx')

plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x='Feature class', y='Rank', hue='Yes_no', data=SC_rank_class, width= 0.75)







####### SC type feature permutation


import os
 # target prediction 폴더로 변경
os.chdir(r'C:\Users\Yung\Desktop\kdc 사상체질\KDC analysis code')

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


df = pd.read_excel('raw)체질확진자_자료(비만_12로).xlsx', index_col='dummy ID', 
                  sheet_name = '분석대상_최종_disc')

df_array= np.array(df, dtype='int32')
permut_MI = np.array(np.zeros((100000,len(df.columns)-1)), dtype='float32')
    
for i in range(permut_MI.shape[0]):
    if i%10 == 0:
        print('%s th permutation' %i)
    
    SC_type = df_array[:,-1]
    np.random.shuffle(SC_type)
    
    for j in range(permut_MI.shape[1]):
        feature_vector = df_array[:,j]
        
        MI_score = adjusted_mutual_info_score(SC_type,feature_vector)        
        
        permut_MI[i,j] = MI_score 
    
permut_df = pd.DataFrame(permut_MI, columns=df.columns[:-1])
permut_df['Adj_MI_sum'] = permut_MI.sum(axis=1)

os.chdir(r'feature_network_result')

permut_df.to_excel('SC_type_permutation.xlsx')