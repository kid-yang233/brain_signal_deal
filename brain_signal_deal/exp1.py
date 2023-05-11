import nibabel as nb 
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.ndimage.interpolation as inter
import sys

path_mask = os.path.abspath('..')
data = []
mask = []
data_name = ['../dataset/exp1/re_DMN.nii',
        '../dataset/exp1/re_LCCN.nii',
        '../dataset/exp1/re_RCCN.nii',
        '../dataset/exp1/re_SN.nii']


mask_name = [
    '../dataset/exp1/LCCN_mask.nii',
    '../dataset/exp1/RCCN_mask.nii'
]

u = '../dataset/exp1/4DfMRI.nii'



g_data = nb.load(u).get_fdata()
for i in data_name:
    data.append(nb.load(i).get_fdata()) #已经采好样的数据
    
for i in mask_name:
    k = nb.load(i).get_fdata() #还没重采样的数据
    g = inter.zoom(k,[61/k.shape[0],73/k.shape[1],61/k.shape[2]],order = 3)
    mask.append(g)
"""   1重采样部分    """
fig,axes = plt.subplots(3,2)
for i in range(3):
    axes[i,0].imshow(np.sum(data[1],axis=i))
    axes[i,1].imshow(np.sum(mask[0],axis=i))
plt.savefig('../result/exp1/0.jpg')
plt.clf()

"""   2时间序列展示   """
a = np.zeros((4,170),dtype=np.float32)
for i in range(4):
    for j in range(170):
        a[i,j] = np.sum(g_data[:,:,:,j]*data[i])/np.sum(data[i])

fig,axes = plt.subplots(1,1)
axes.plot(a[0],color='red')
plt.savefig('../result/exp1/1.jpg')
plt.clf()


"""   3皮尔逊相关系数计算  """
r = np.corrcoef(a)
r = r/np.max(r)
fig,axes = plt.subplots(1,1)
plt.imshow(r,cmap=plt.cm.hot)
plt.colorbar()
plt.savefig('../result/exp1/2.jpg')
plt.clf()


"""    4 AAL模板多脑区计算  """
aal_path = '../dataset/exp1/aal.nii'
aal = nb.load(aal_path).get_fdata()  #116类
tem = np.zeros((116,170),dtype=np.float32)
aal = inter.zoom(aal,[61/aal.shape[0],73/aal.shape[1],61/aal.shape[2]],order = 3)
place = np.array(aal,dtype=int)
for i in range(116):
    u = np.zeros_like(aal)
    u[place==i] = 1
    for j in range(170):
        tem[i,j] = np.sum(g_data[:,:,:,j]*u)


r_new = np.corrcoef(tem)
r_new = r_new / np.max(r_new)
plt.imshow(r_new,cmap=plt.cm.hot)
plt.colorbar()
plt.savefig('../result/exp1/3.jpg')

