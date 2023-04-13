# -*- coding: UTF-8 -*-
from libtiff import TIFF
import openslide
# import matplotlib.pyplot as plt
# from scipy import misc
import numpy as np
saverootdir='/media/tx-deepocean/5271ceaf-10dc-456f-b54c-3165996444a530/GCN/GCN_clinical'
import os
os.chdir(saverootdir)
# from PIL import Image
import csv
import pandas as pd
flag1='train_'
flag='test_'
flag2='merge_'
allK=[5,10,20,30];#KNN parameter
datafile=flag1+'radiomics_feature.csv'
train_data = pd.read_csv(datafile)
train_data_np1=train_data.to_numpy()
datafile=flag+'radiomics_feature.csv'
train_data = pd.read_csv(datafile)
train_data_np2=train_data.to_numpy()
train_data_np=np.vstack((train_data_np1,train_data_np2))
train_mask=np.hstack((np.ones(train_data_np1.shape[0]),np.zeros(train_data_np2.shape[0])))
test_mask=np.hstack((np.zeros(train_data_np1.shape[0]),np.ones(train_data_np2.shape[0])))
train_label=train_data_np[:,0]
node_feature=train_data_np[:,1:]



datafile=flag1+'clinical_feature.csv'
train_data = pd.read_csv(datafile)
train_data['age']=pd.to_numeric(train_data['age'])
train_data['age']=train_data['age'].fillna(train_data['age'].median())
train_data['sex']=pd.to_numeric(train_data['sex'])
train_data['sex']=train_data['sex'].fillna(0)
train_data['attr1']=pd.to_numeric(train_data['attr1'])
train_data['attr1']=train_data['attr1'].fillna(0)
train_data['attr2']=pd.to_numeric(train_data['attr2'])
train_data['attr2']=train_data['attr2'].fillna(0)
train_data['attr3']=pd.to_numeric(train_data['attr3'])
train_data['attr3']=train_data['attr3'].fillna(0)
train_data['attr4']=pd.to_numeric(train_data['attr4'])
train_data['attr4']=train_data['attr4'].fillna(0)
print(train_data.isnull().values.any())


train_clinical_data_np1=train_data.to_numpy()
clinical_node_feature1=train_clinical_data_np1
datafile=flag+'clinical_feature.csv'
train_data = pd.read_csv(datafile)
train_data['age']=pd.to_numeric(train_data['age'])
train_data['age']=train_data['age'].fillna(train_data['age'].median())
train_data['sex']=pd.to_numeric(train_data['sex'])
train_data['sex']=train_data['sex'].fillna(0)
train_data['attr1']=pd.to_numeric(train_data['attr1'])
train_data['attr1']=train_data['attr1'].fillna(0)
train_data['attr2']=pd.to_numeric(train_data['attr2'])
train_data['attr2']=train_data['attr2'].fillna(0)
train_data['attr3']=pd.to_numeric(train_data['attr3'])
train_data['attr3']=train_data['attr3'].fillna(0)
train_data['attr4']=pd.to_numeric(train_data['attr4'])
train_data['attr4']=train_data['attr4'].fillna(0)
print(train_data.isnull().values.any())


train_clinical_data_np2=train_data.to_numpy()
clinical_node_feature2=train_clinical_data_np2
node_feature1=np.vstack((clinical_node_feature1,clinical_node_feature2))



adjacent_matrix=np.corrcoef(node_feature1)
row_number=node_feature.shape[0]
for K in allK:
    adjacent_info = []
    for index1 in range(row_number):
        for index3 in range(0,index1+1):
            adjacent_matrix[index1,index3]=0
        temp=adjacent_matrix[index1,:]
        position=np.argpartition(temp,-K)[-K:]
        for index4 in range(K):
            if temp[position[index4]]!=0:
                adjacent_info.append([index1,position[index4],temp[position[index4]]])
    adjacent_info=np.array(adjacent_info)
    adjacent_info=adjacent_info.T
    txtfilename=flag2+'adjacent_info_'+str(K)+'.txt'
    np.savetxt(os.path.join(saverootdir,txtfilename),adjacent_info,fmt='%.8f')

txtfilename=flag2+'node_feature'+'.txt'
np.savetxt(os.path.join(saverootdir,txtfilename),node_feature,fmt='%.8f')
txtfilename=flag2+'label'+'.txt'
np.savetxt(os.path.join(saverootdir,txtfilename),train_label,fmt='%d')
txtfilename=flag1+'mask'+'.txt'
np.savetxt(os.path.join(saverootdir,txtfilename),train_mask,fmt='%d')
txtfilename=flag+'mask'+'.txt'
np.savetxt(os.path.join(saverootdir,txtfilename),test_mask,fmt='%d')