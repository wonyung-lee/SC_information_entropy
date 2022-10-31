# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:03:50 2018

@author: Yung
"""

## female network에서 n수 맞추기 (360개)
# 1. network construction & random sampling -> df
# 2. edge density threshold 3. network construction 4. fully connected
import networkx as nx
import random
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\yung\Desktop\kdc 사상체질\KDC analysis code\patient_network')

gender = 1   # 1 = male / 2 = female


#df = pd.read_excel('raw)df_patient(woman).xlsx')
PSM_score_df = pd.read_excel("patient_sim_%s(overlap).xlsx" % (gender))



# network construction
PSM_score_df = PSM_score_df.loc[PSM_score_df['PSM']>=0.1]
G= nx.from_pandas_edgelist(PSM_score_df,'node1', 'node2', 'PSM')
list_node = G.nodes()

## (optional) Randomly removing nodes to match the number of nodes
#sample_node = random.sample(list_node, 500)
#G.remove_nodes_from(sample_node)

edge_density = 0.046
# 0.046 - both female and male network가 fully connected되는 가장 낮은 edge density

patient_network = nx.to_pandas_edgelist(G)
patient_network.sort_values('PSM', ascending=False, inplace =True)
node_fre = len(list_node)
edge_fil = patient_network.iloc[:int(edge_density*node_fre*(node_fre-1)*(1/2))]

G_fil = nx.from_pandas_edgelist(edge_fil, 'source','target','PSM')
if nx.is_connected(G_fil) == True:
    print ('Network is fully connected in edge density %s' % (edge_density))
else:
    print ('Network is not fully connected in edge density %s' % (edge_density))


# node, edge 불러오고 network construction, 체질 attr 추가
df_node = pd.read_excel('patient_node.xlsx')
df_edge = edge_fil

#df_edge.to_excel("edge_list_%s.xlsx" % (gender))
# save edge list to excel

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

sns.heatmap(net_sep_mat,cmap="RdBu_r", vmax = 0.1, vmin = -0.1, xticklabels=['TE','SE','SY','TY'], yticklabels=['TE','SE','SY','TY'])

net_sep_mat.to_excel('permut)separat_mat_1.xlsx')
net_sep_score.to_excel('permut)separat_score_1.xlsx')
    
