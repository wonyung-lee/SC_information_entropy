# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:59:24 2017

@author: Yung
"""


import numpy as np
import pandas as pd
import os
import networkx as nx
from networkx.algorithms import community

os.chdir(r'C:\Users\Yung\Desktop\사상체질\patient_network')


# node, edge 불러오고 network construction, 체질 attr 추가
df_node = pd.read_excel('patient_node.xlsx')
df_edge = pd.read_excel('patient_edge_1(0.2822 to edge density 통일).xlsx')

G = nx.from_pandas_dataframe(df_edge,0,1)

constitute_attr = pd.DataFrame(np.array(df_node['최종진단']),index=df_node['dummy ID'])
nx.set_node_attributes(G, constitute_attr[0].to_dict(), '체질')

# 체질 제외한 class 

feature_class = [1,2,3,4]
# 1=태음  2=소음  3=소양  4=태양

SD_mean = {}

# 각 class 내부의 최단거리 구하기
for i in range(len(feature_class)):
    feature_node = [n for n,d in G.nodes(data=True) if d['체질']==feature_class[i]]
    SD_in_network = []
    
    # j번째 node 선정하여 min(해당 노드 제외 나머지 노드들과의 SD)= node와 network간의 SD
    for j in range(len(feature_node)):
        nodes = [n for n,d in G.nodes(data=True) if d['체질']==feature_class[i]]
        nodes.remove(feature_node[j])
        SD_node = []
        
        for k in range(len(nodes)):
            SD_node.append(nx.shortest_path_length(G, nodes[k], feature_node[j]))
        
        SD_in_network.append(min(SD_node))
            
    SD_mean[feature_class[i]] = sum(SD_in_network)/len(SD_in_network)

# 각 class간 최단거리 구하기

SD_bet_net = {}
for k in range(len(feature_class)):    
    for l in range(len(feature_class)):
        SD_network = []
        
        node_classA = [n for n,d in G.nodes(data=True) if d['체질'] == feature_class[k]]
        node_classB = [n for n,d in G.nodes(data=True) if d['체질'] == feature_class[l]]
        
        for o in range(len(node_classA)):
            SD_score = []
            for p in range(len(node_classB)):
                SD_score.append(nx.shortest_path_length(G, node_classA[o], node_classB[p]))
            SD_network.append(min(SD_score))
        
        SD_bet_net[feature_class[k]+feature_class[l]] = sum(SD_network)/len(SD_network)

# network seperation 계산식에 대입 : (A, B 최단거리)-(A 내부 최단거리 + B 내부 최단거리) /2

net_sep_mat = pd.DataFrame(np.zeros((len(feature_class),len(feature_class))), 
                            columns = feature_class, index = feature_class)
net_sep_score = pd.DataFrame(columns=['class1','class2','SP_score'])

for q in range(len(feature_class)):
    for w in range(len(feature_class)):
        
        feat_a = feature_class[q]
        feat_b = feature_class[w]
        
        if feat_a == feat_b:
            SP_score = 0
        else:         
            SP_score = SD_bet_net[feat_a+feat_b] - (SD_mean[feat_a]+SD_mean[feat_b])/2
        
        net_sep_mat[feat_a][feat_b] = SP_score
        net_sep_score.loc[-1] = (feat_a, feat_b, SP_score)
        net_sep_score.index = net_sep_score.index +1



net_sep_mat.to_excel('patient_seperation_mat_1(edge_modi).xlsx')
net_sep_score.to_excel('patient_seperation_table_1(edge_modi).xlsx')
    
