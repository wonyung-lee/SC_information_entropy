# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:13:38 2018

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

df = pd.read_excel('raw)체질확진자_자료(비만_12로).xlsx', sheetname='분석대상_최종')

# 결과를 저장해놓을 df 생성
p_value_table = pd.DataFrame(columns=['disease','feature','test','p_value'])

# disease와 관련 feature들의 dict 생성
disease_list = ['고지혈증_진단', '고혈압_진단', '당뇨_진단', '내분비_비만', '종양_종류']

list_0 = ['고혈압_진단', '소화기_지방간', '당뇨_진단', '결혼여부']
list_1 = ['순환기_중풍', '결혼여부', '고지혈증_진단', '호흡기_없다']
list_2 = ['순환기_중풍', '소변_거품뇨', '고혈압_진단', '결혼여부', '컨디션_소화', '순환기_없다', '머리양상_메스꺼우면서아픔']
list_3 = ['소화기_지방간', '최종진단']
list_4 = ['수술력', '결혼여부', '성별', '기타증상_없다', '기타증상_건망', '한열_배', '땀_손', '내분비_갑상선저하', '근골격_디스크', '소변_요실금']

list_0_cont = ['만나이(/365)', '8부위_4', '8부위_5', '8부위_6', '8부위_3', '8부위_7']
list_1_cont = ['만나이(/365)', '8부위_5', '8부위_6', '8부위_4', '8부위_7', '5부위_4']
list_2_cont = ['만나이(/365)', '소변야간횟수(회)', '8부위_5']
list_3_cont = ['8부위_4', '8부위_6', '5부위_4', '8부위_7', '5부위_2', '몸무게', '5부위_3', '8부위_3']
list_4_cont = []

feature_list = [list_0, list_1, list_2, list_3, list_4]
cont_list = [list_0_cont, list_1_cont, list_2_cont, list_3_cont, list_4_cont]

# continuous variable에 대한 ANOVA
for i in range(5):
    yes = (df[disease_list[i]] == 2)
    no = (df[disease_list[i]] == 1)
    
    related_feature = cont_list[i]

    for j in range(len(related_feature)):
        
        yes_value = df[related_feature[j]][yes]
        no_value = df[related_feature[j]][no]
        f, p_value = stats.f_oneway(yes_value, no_value)


        p_value_table.loc[-1] = (disease_list[i], related_feature[j], 'ANOVA',p_value)
        p_value_table.index = p_value_table.index+1


# discrete variable에 대한 chi-square 검정
for i in range(5):
    related_feature = feature_list[i]
    
    for j in range(len(related_feature)):
        crosstab = pd.crosstab(index=df[disease_list[i]], columns=df[related_feature[j]])
        a, p_value, b, c = stats.chi2_contingency(crosstab)
        p_value_table.loc[-1] = (disease_list[i], related_feature[j], 'chisq', p_value)
        p_value_table.index = p_value_table.index+1

