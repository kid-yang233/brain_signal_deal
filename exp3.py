import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import preprocessing

datafile = 'subj_0001_FC.mat'
orin_data = scio.loadmat(datafile)['corrmatrix']

plt.imshow(orin_data,cmap=plt.cm.hot)
plt.colorbar()
plt.savefig('0.jpg',dpi=750)
plt.clf()

"""1.二值化图"""
orin_data = np.abs(orin_data)
a = np.median(orin_data)

for i in range(116):
    for j in range(116):
        if orin_data[i,j]>=a:
            orin_data[i,j] = 1
        else:
            orin_data[i,j] = 0


fig,axes = plt.subplots(1,1)
plt.imshow(orin_data,cmap=plt.cm.hot)
plt.colorbar()
plt.savefig('1.jpg',dpi=750)
plt.clf()


"""聚类系数的计算和最短距离"""

C = np.zeros(116)

for i in range(116):
    q = []
    for j in range(116):
        if orin_data[i,j]==1:
            q.append(j)
    lengths = len(q)
    near = 0
    for x in range(lengths):
        for y in range(x,lengths):
            if orin_data[q[x],q[y]] ==1:
                near+=1
    cn2 = lengths*(lengths-1)/2
    C[i] = near/cn2

fig,axes = plt.subplots(1,1)
plt.bar(range(116),C)
plt.savefig('4.jpg',dpi=750)
plt.clf()

d = orin_data.copy()
L = np.zeros(116)

for i in range(116):
    for j in range(116):
        if d[i,j] == 0 and i!=j:
            d [i,j] = 116

for k in range(116):
    for i in range(116):
        for j in range(116):
            if d[i][j]>d[i][k]+d[k][j]:
                d[i][j]=d[i][k]+d[k][j]

l = np.sum(d/116/115)
print(l)#1.495

"""聚类方式和结果"""
min_max_scaler = preprocessing.MinMaxScaler()
data_M = min_max_scaler.fit_transform(d)
plt.figure(figsize=(12,6))
Z = linkage(data_M, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
plt.savefig('2.jpg',dpi=750)
plt.clf()

ac = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
ac.fit(data_M)
labels = ac.fit_predict(data_M)