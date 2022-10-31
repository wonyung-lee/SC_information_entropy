# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:08:55 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import os

os.chdir(r'C:\Users\Administrator\Desktop\kdc 사상체질')


# 엑셀파일을 xls_file로 불러와 모든 sheet를 df에 통합
xls_file = pd.ExcelFile('raw)체질확진자_자료(땀생리종양뺴고 0채움).xlsx')
sheet_names = xls_file.sheet_names

df_fre = pd.DataFrame(columns=['sheet','feature','frequency'])

df_OS = xls_file.parse('소증', index_col='dummy ID')

# 한열 값을 가지는 환자 1855명의 vector만 
col_HC_null = pd.isnull(df_OS['한열증상_4'])
correc_HC = (col_HC_null == False)

# 각 sheet별로 data 가져오기
for i in range(len(sheet_names)):
    df1 = xls_file.parse(sheet_names[i], index_col='dummy ID')
    col_list = list(df1.columns)
    
    # 각 column별로 유효한 data 길이/전체 길이 구해 df_fre에 추가
    for j in range(len(col_list)):
        
        col = col_list[j]
        col_data = df1[col][correc_HC]
        col_fre = len(col_data.dropna())/len(col_data)
        
        df_fre.loc[-1] = (sheet_names[i], col, col_fre)
        df_fre.index = df_fre.index +1        
            
df_fre.to_excel('feature_frequency11.xlsx')


df = pd.DataFrame()
for i in range(len(sheet_names)):
    df_data =xls_file.parse(sheet_names[i], index_col='dummy ID')ㅏ
    df_data_HC = df_data[:][correc_HC]
    df= pd.concat([df,df_data_HC],axis=1)
