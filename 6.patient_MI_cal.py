# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:21:46 2017

@author: Administrator
"""


import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_mutual_info_score
from multiprocessing import Process, Queue
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\yung\Desktop\kdc 사상체질\KDC analysis code')

df = pd.read_excel('raw)체질확진자_자료(비만_12로).xlsx', index_col='dummy ID', 
                  sheet_name = '분석대상_최종')
df_class = pd.read_excel('feature_to_class.xlsx')
dict_class = df_class.set_index('feature')['class'].to_dict()

sasang_feature = ['최종진단','몸무게', '성격_남성적_여성적', '성격_대범_섬세', 
                  '성격_동적_정적', '성격_외향_내성', '성격_적극_소극', 
                  '성격_행동빠름_행동느림', '땀_더울때', '땀_운동할때', 
                  '땀_일상생활', '땀정도', '복통', '소화입맛', '음수온도', '음수정도',
                  '한열 민감도', '한열_발', '한열_손', '한열증상_1', '한열증상_3', 
                  '한열증상_4', '한열증상_7', '한열증상_8', '컨디션_소화', 
                  '고혈압_진단', '내분비_비만', '내분비_없다', '소화기_지방간', 
                  '5부위_1', '5부위_2', '5부위_3', '5부위_4', '5부위_5', 
                  '8부위_1', '8부위_2', '8부위_3', '8부위_4', '8부위_5', 
                  '8부위_6', '8부위_7', '8부위_8', '늑골각도']

continuous_set = set(['키','몸무게','만나이(/365)','식사시간(분)','대변횟수(일)','대변시간(분)',
                   '소변횟수(회)','소변야간횟수(회)','음수양(잔)','수면시간(시간)',
                   '8부위_1','8부위_2','8부위_3','8부위_4','8부위_5','8부위_6',
                   '8부위_7','8부위_8','5부위_1','5부위_2','5부위_3','5부위_4','5부위_5',
                   ])

sasang_cat = list(set(sasang_feature)-continuous_set)
sasang_cont = list(set(sasang_feature)-set(sasang_cat))

# 남성, 여성을 나누어서 선택
gender = 1    # 1 : man 2: female
gender_index = ( df['성별'] == gender)
gender_df = df[sasang_feature][gender_index]


# 사상관련 feature만 가진 df 뽑아 사상체질 순서 (1234)대로 정렬
df_patient = gender_df.sort_values('최종진단').T
#df_patient.to_excel('raw)df_patient(woman).xlsx')

df_cat_feature = df_patient.loc[sasang_cat]
df_conti_feature = df_patient.loc[sasang_cont]

patient_list = list(df_patient.columns)

patient_sim_mat = pd.DataFrame(np.zeros((len(patient_list),len(patient_list))))

# conti feature normalize 

array_conti_feature = np.array(df_conti_feature)
norm_conti_feature = 2*(array_conti_feature - array_conti_feature.min(axis=1, keepdims=True))/(array_conti_feature.max(axis=1, keepdims=True) - array_conti_feature.min(axis=1, keepdims=True))-1
df_conti_feature_norm = pd.DataFrame(norm_conti_feature, columns=df_conti_feature.columns, index=df_conti_feature.index)

# patient similarity measure

for i in range(len(patient_list)):
    for j in range(len(patient_list)):

        if i==j:
            patient_sim_mat[i][j] =1  
        
        else:
            dot_product = np.sum(df_conti_feature_norm[patient_list[i]]*df_conti_feature_norm[patient_list[j]]) + np.sum(df_cat_feature[patient_list[i]] == df_cat_feature[patient_list[j]])             
            euclidian_dist1 = (np.sum(df_conti_feature_norm[patient_list[i]]*df_conti_feature_norm[patient_list[i]]) + df_cat_feature[patient_list[0]].size) ** (1/2)
            euclidian_dist2 = (np.sum(df_conti_feature_norm[patient_list[j]]*df_conti_feature_norm[patient_list[j]]) + df_cat_feature[patient_list[0]].size) ** (1/2)

            PSM = dot_product / (euclidian_dist1 * euclidian_dist2) 
                  
            patient_sim_mat[i][j] = PSM


patient_sim_mat.to_excel('patient_sim_mat_%s(overlap).xlsx' %(gender))

## Mean PSM calculation by each SC type

for i in range(1,3):
    
    if i == 1:
        TE = 191
        SE = 320
        SY = 492
    else:
        TE = 312
        SE = 575
        SY = 842
    
    PSM_mat = np.array(pd.read_excel('patient_sim_mat_%s(overlap).xlsx' %(i)))
    
    TE_TE = np.mean(PSM_mat[:TE, :TE])
    TE_SE = np.mean(PSM_mat[:TE, TE:SE])
    TE_SY = np.mean(PSM_mat[:TE, SE:SY])
    TE_TY = np.mean(PSM_mat[:TE, SY:])

    SE_TE = np.mean(PSM_mat[TE:SE, :TE])
    SE_SE = np.mean(PSM_mat[TE:SE, TE:SE])
    SE_SY = np.mean(PSM_mat[TE:SE, SE:SY])
    SE_TY = np.mean(PSM_mat[TE:SE, SY:])

    SY_TE = np.mean(PSM_mat[SE:SY, :TE])
    SY_SE = np.mean(PSM_mat[SE:SY, TE:SE])
    SY_SY = np.mean(PSM_mat[SE:SY, SE:SY])
    SY_TY = np.mean(PSM_mat[SE:SY, SY:])

    TY_TE = np.mean(PSM_mat[SY:, :TE])
    TY_SE = np.mean(PSM_mat[SY:, TE:SE])
    TY_SY = np.mean(PSM_mat[SY:, SE:SY])
    TY_TY = np.mean(PSM_mat[SY:, SY:])

    mean_PSM = np.array([[TE_TE, TE_SE, TE_SY, TE_TY],[SE_TE, SE_SE, SE_SY, SE_TY], [SY_TE, SY_SE, SY_SY, SY_TY], [TY_TE, TY_SE, TY_SY, TY_TY]])

    mean_PSM = pd.DataFrame(mean_PSM, columns = ['TE','SE','SY','TY'], index = ['TE','SE','SY','TY'])
    fig = sns.heatmap (mean_PSM, cmap = "RdBu_r", vmin=0, vmax=1, center =0, cbar=True)
    plt.show(fig)
    plt.close()
    mean_PSM.to_excel('PSM_mean_by_SC_%s.xlsx' % (i))
    
    

# MI score table 만들기
sim_mat_index = pd.DataFrame(np.array(patient_sim_mat), columns= patient_list, index= patient_list)
patient_sim_table = pd.DataFrame(columns=['node1','node2','PSM'])

for i in range(len(patient_list)):
    node1 = pd.DataFrame(np.zeros((len(df_patient.columns),1)))
    node1[0] = patient_list[i]

    node2 = pd.DataFrame(sim_mat_index.index)
    
    MI_score = sim_mat_index[patient_list[i]]
    MI_score  = MI_score.reset_index(drop=True)

    df = pd.concat([node1, node2, MI_score],axis=1)
    df = pd.DataFrame(np.array(df), columns=['node1','node2','PSM'])
    
    patient_sim_table = pd.concat([patient_sim_table,df])

patient_sim_table = patient_sim_table.reset_index(drop=True)


# digonal value 제외하기

digonal = [((len(df_patient.columns)+1)*n) for n in range(0,len(df_patient.columns))]

patient_sim_table = patient_sim_table.drop(patient_sim_table.index[digonal])
patient_sim_table.to_excel('patient_sim_table_%s(overlap).xlsx' %(gender)) 




## sim mat heatmap 각 체질별 평균 구하기
#
#patient_mat = np.array(pd.read_excel('patient_MI_mat_%s.xlsx') % (gender))
#sasang_vec = pd.read_excel('constitution_vec.xlsx')
#
#TE = (sasang_vec[1]  == 1)
#SE = (sasang_vec[1]  == 2)
#SY = (sasang_vec[1]  == 3)
#TY = (sasang_vec[1]  == 4)
#
#sa_list= ['TE','SE','SY','TY']
#vec_list = [TE, SE, SY,TY]
#mean_value = pd.DataFrame(columns=sa_list, index = sa_list)
#
#for i in range(4):
#    for j in range(4):
#        vec = patient_mat[vec_list[i]][vec_list[j]]
#        mean_value[sa_list[i]][sa_list[j]]  vec




########### MI score ver.


## matrixs에 MI score 채워넣기
#for i in range(0,len(df_patient.columns)):
#    for j in range(len(patient_list)):
#        if i==j:
#            patient_sim_mat[i][j] =1  
#        
#        else:
#            df1 = df_feature[patient_list[i]]
#            df2 = df_feature[patient_list[j]]
#            MI_score = adjusted_mutual_info_score(df1, df2)
#                    
#            patient_sim_mat[i][j] = MI_score
#
#patient_sim_mat.to_excel('patient_MI_mat_woman.xlsx')
#
#
## MI score table 만들기
#MI_mat_index = pd.DataFrame(np.array(patient_sim_mat), columns= patient_list, index= patient_list)
#patient_MI_table = pd.DataFrame(columns=['node1','node2','adj_MI_score'])
#
#for i in range(len(patient_list)):
#    node1 = pd.DataFrame(np.zeros((len(df_patient.columns),1)))
#    node1[0] = patient_list[i]
#
#    node2 = pd.DataFrame(MI_mat_index.index)
#    
#    MI_score = MI_mat_index[patient_list[i]]
#    MI_score  = MI_score.reset_index(drop=True)
#
#    df = pd.concat([node1, node2, MI_score],axis=1)
#    df = pd.DataFrame(np.array(df), columns=['node1','node2','adj_MI_score'])
#    
#    patient_MI_table = pd.concat([patient_MI_table,df])
#    patient_MI_table = patient_MI_table.reset_index(drop=True)
#
## digonal value 제외하기
#
#digonal = [((len(df_patient.columns)+1)*n) for n in range(0,len(df_patient.columns))]
#
#patient_MI_man = patient_MI_table.drop(patient_MI_table.index[digonal])
#patient_MI_man.to_excel('patient_MI_woman.xlsx')
#
## female MI heatmap 각 체질별 평균 구하기
#
#patient_mat = np.array(pd.read_excel('patient_MI_mat_2.xlsx'))
#sasang_vec = pd.read_excel('constitution_vec.xlsx')
#
#TE = (sasang_vec[1]  == 1)
#SE = (sasang_vec[1]  == 2)
#SY = (sasang_vec[1]  == 3)
#TY = (sasang_vec[1]  == 4)
#
#sa_list= ['TE','SE','SY','TY']
#vec_list = [TE, SE, SY,TY]
#mean_value = pd.DataFrame(columns=sa_list, index = sa_list)
#
#for i in range(4):
#    for j in range(4):
#        vec = patient_mat[vec_list[i]][vec_list[j]]
#        mean_value[sa_list[i]][sa_list[j]] = vec
