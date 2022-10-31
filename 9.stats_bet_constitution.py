# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:37:05 2017

@author: Yung
"""



import numpy as np
import pandas as pd
import os
import networkx as nx
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Yung\Desktop\사상체질')
sns.set_style("whitegrid")

df = pd.read_excel('raw)체질확진자_자료(만나이추가).xlsx', sheetname='분석대상_최종')

p_value_table = pd.DataFrame(columns=['feature','test','p_value'])

TE_vec = (df['최종진단'] == 1)
SE_vec = (df['최종진단'] == 2)
SY_vec = (df['최종진단'] == 3)
TY_vec = (df['최종진단'] == 4)

df['최종진단'][TE_vec] = 'TE'
df['최종진단'][SE_vec] = 'SE'
df['최종진단'][SY_vec] = 'SY'
df['최종진단'][TY_vec] = 'TY'


# 연속형 변수의 ANOVA 검정
conti_variable = ['몸무게', '5부위_1', '5부위_2', '5부위_3', '5부위_4', '5부위_5', 
                  '8부위_1', '8부위_2', '8부위_3', '8부위_4', '8부위_5', 
                  '8부위_6', '8부위_7', '8부위_8']

conti_variable_eng = ['kg', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm', 'cm']

# x는 column과 체질명, y는 column을 영어로 바꿀

my_pal = {'TE' :'b', 'SE' : 'cyan','SY' :'yellow','TY':'r'}

for i in range(len(conti_variable)):
    df_feature = df[[conti_variable[i],'최종진단']]

    ax = sns.boxplot(x=df_feature['최종진단'],y=df_feature[conti_variable[i]], palette=my_pal)
    ax.set(xlabel = '', ylabel = conti_variable_eng[i])
    plt.savefig(conti_variable[i]+'_boxplot.png')
    plt.gcf().clear()
    
    TE_value = df[conti_variable[i]][TE_vec]
    SE_value = df[conti_variable[i]][SE_vec]
    SY_value = df[conti_variable[i]][SY_vec]
    TY_value = df[conti_variable[i]][TY_vec]
   
    f, p_value = stats.f_oneway(TE_value, SE_value, SY_value , TY_value)

    p_value_table.loc[-1] = (conti_variable[i], 'ANOVA', p_value)
    p_value_table.index = p_value_table.index+1
    

# 이산형 변수의 chisquare
disc_variable = ['성격_남성적_여성적', '성격_대범_섬세', 
                  '성격_동적_정적', '성격_외향_내성', '성격_적극_소극', 
                  '성격_행동빠름_행동느림', '땀_더울때', '땀_운동할때', 
                  '땀_일상생활', '땀정도', '복통', '소화입맛', '음수온도', '음수정도',
                  '한열 민감도', '한열_발', '한열_손', '한열증상_1', '한열증상_3', 
                  '한열증상_4', '한열증상_7', '한열증상_8', '컨디션_소화', 
                  '고혈압_진단', '내분비_비만', '내분비_없다', '소화기_지방간', '늑골각도']

for j in range(len(disc_variable)):
    df_feature = df[[disc_variable[j],'최종진단']]

    # boxplot
    sns.boxplot(x=df_feature['최종진단'],y=df_feature[disc_variable[j]])
    plt.savefig(disc_variable[j]+'_boxplot.png')
    plt.gcf().clear()
    
    #cross table
    crosstab = pd.crosstab(index=df['최종진단'],columns=df[disc_variable[j]])
    crosstab.to_excel(disc_variable[j]+'_체질.xlsx')
    
    # crosstable 이용해서 통계분석
    a, p_value, b, c  = stats.chi2_contingency(crosstab)

    p_value_table.loc[-1] = (disc_variable[j], 'chisq', p_value)
    p_value_table.index = p_value_table.index+1
    

# p-value log scale로 바꾸어서 dist plot에 그리기

p_value_table = pd.read_excel('feature_p_value.xlsx')
p_value = p_value_table['adj_p_log']

g = sns.distplot(p_value, color = 'r',  kde=False, bins=10)
plt.xlabel('')
