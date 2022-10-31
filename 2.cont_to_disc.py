# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:13:51 2017

@author: Administrator
"""



import numpy as np
import pandas as pd
import os

os.chdir(r'C:\Users\Yung\Desktop\사상체질')

df = pd.read_excel('raw)체질확진자_자료(비만_12로).xlsx',sheet_name='분석대상_최종')

continuous_list = ['키','몸무게','만나이(/365)','식사시간(분)','대변횟수(일)','대변시간(분)',
                   '소변횟수(회)','소변야간횟수(회)','음수양(잔)','수면시간(시간)',
                   '8부위_1','8부위_2','8부위_3','8부위_4','8부위_5','8부위_6',
                   '8부위_7','8부위_8','5부위_1','5부위_2','5부위_3','5부위_4','5부위_5',
                   ]

# 3등분으로 나누어서 group으로 나눔

for i in range(len(continuous_list)):
    array = np.array(df[continuous_list[i]])    
    array.sort()
    cutoff_val_1 = array[len(array)//5]
    cutoff_val_2 = array[len(array)*2//5]
    cutoff_val_3 = array[len(array)*3//5]
    cutoff_val_4 = array[len(array)*4//5]
    cutoff_val_5 = array[len(array)*5//5]
    
    array = np.array(df[continuous_list[i]])      

    group_vec_1 = ( array < cutoff_val_1 )
    group_vec_2 = ((cutoff_val_1 <= array) & (array < cutoff_val_2))
    group_vec_3 = ((cutoff_val_2 <= array) & (array < cutoff_val_3))
    group_vec_4 = ((cutoff_val_3 <= array) & (array < cutoff_val_4))
    group_vec_5 = ( cutoff_val_4 <= array )

    array[group_vec_1] = 1
    array[group_vec_2] = 2
    array[group_vec_3] = 3
    array[group_vec_4] = 4
    array[group_vec_5] = 5
    
    df[continuous_list[i]] = array
