# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:13:45 2017

@author: Administrator
"""


import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_mutual_info_score
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)


os.chdir(r'C:\Users\Yung\Desktop\사상체질\feature_network_result')


df = pd.read_excel('raw)체질확진자_자료(만나이추가).xlsx', index_col='dummy ID', 
                  sheet_name = '분석대상_최종_disc')
MI_score_df = pd.read_excel('MI_score_table.xlsx')

feature_list1 = list(df.columns)

df_class = pd.read_excel('feature_to_class.xlsx')
dict_class = df_class.set_index('feature')['class'].to_dict()

# node df, edge df 생성

df_node = pd.DataFrame(columns=['node','class'])

for i in range(len(feature_list1)):
    df_node.loc[i]= (feature_list1[i], dict_class[feature_list1[i]])
    
MI_score_cutoff = 0.01
MI_score_vec = (MI_score_df['adj_MI_score'] >= MI_score_cutoff)

edge1 = MI_score_df['feature1'][MI_score_vec]
edge2 = MI_score_df['feature2'][MI_score_vec]
MI_score = MI_score_df['adj_MI_score'][MI_score_vec]

df_edge = pd.DataFrame({'edge1':edge1,'edge2':edge2,'MI_score':MI_score})

#df_node.to_excel('node_list.xlsx')
#df_edge.to_excel('edge_cutoff_'+str(MI_score_cutoff)+'.xlsx')

# network x에 node, edge 생성
df_node = pd.read_excel('node_list.xlsx')
df_edge = pd.read_excel('edge_cutoff_0.01.xlsx')

G= nx.from_pandas_dataframe(df_edge,'edge1', 'edge2')

node_color_dict = {'인구학적조사':'b','성격1':'g','소증':'r','병증':'c','질병':'m',
                   '건강':'y','한열':'teal','체형사진음성':'orange','진단':'indigo'}
node_color = []


## network fully_connected 조사

df_edge = pd.read_excel('MI_score_table(connected).xlsx')
df_edge = df_edge.reset_index(drop=True)

MI_score_cutoff = 0.010

MI_score_vec = (df_edge['adj_MI_score'] >= MI_score_cutoff)

edge_1 = df_edge['feature1'][MI_score_vec]
edge_2 = df_edge['feature2'][MI_score_vec]

df_edge_cutoff = pd.DataFrame({'edge1':edge_1,'edge2':edge_2})

G= nx.from_pandas_dataframe(df_edge_cutoff,'edge1', 'edge2')

print(nx.is_connected(G))

edges_connected = pd.DataFrame(list(G.edges()))
edges_connected.to_excel('edges_connected(0.017).xlsx')

# edge distribution 구하기
MI_table = pd.read_excel('MI_score_table.xlsx')
MI_score = MI_table['adj_MI_score']

plt.xlim(0, 0.1)
sns.distplot(MI_score, bins = 400, kde=False, color='r')

# matrix visualization
MI_mat = pd.read_excel('mat_MI_nondi.xlsx')
ax = sns.heatmap(MI_mat, xticklabels=False, yticklabels=False, vmin = -0.01, vmax =0.05, cmap="Blues")


