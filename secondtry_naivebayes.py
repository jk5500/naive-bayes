# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 10:06:35 2016

@author: kamal
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib import colors as c

iris = datasets.load_iris()
X = iris.data[:, 1:3]  
Y = iris.target
c1=np.matrix(X[Y==0])
c2=np.matrix(X[Y==1])
c3=np.matrix(X[Y==2])
c1m=np.matrix(np.mean(c1,axis=0)).T
c2m=np.matrix(np.mean(c2,axis=0)).T
c3m=np.matrix(np.mean(c3,axis=0)).T

'''
cov1=np.matrix(np.cov(c1.T,bias=1))
cov2=np.matrix(np.cov(c2.T,bias=1))
cov3=np.matrix(np.cov(c3.T,bias=1))

'''
# we have to assume, features are independent in naive bayes
# so covariance matrix should be diagonal so that two features are 
#independent
std1,std11=np.var(c1[:,0], ddof=1),np.var(c1[:,1], ddof=1)
std2,std22=np.var(c2[:,0], ddof=1),np.var(c2[:,1], ddof=1)
std3,std33=np.var(c3[:,0], ddof=1),np.var(c3[:,1], ddof=1)
cov1=np.matrix([[std1,0],[0,std11]])
cov2=np.matrix([[std2,0],[0,std22]])
cov3=np.matrix([[std3,0],[0,std33]])


def gaussian_2d(x,mean,sigma):
    exponent=np.exp(-0.5*(x-mean).T*np.linalg.inv(sigma)*(x-mean))
    ones=1.0/(2*np.pi*np.sqrt(np.linalg.det(sigma)))
    return ones*exponent
xmin,xmax=X[:,0].min()-1,X[:,0].max()+1
ymin,ymax=X[:,1].min()-1,X[:,1].max()+1
#xmin,xmax=4,8
#ymin,ymax=2,4.5
h=0.02
XX,YY=np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
Z=np.matrix(np.c_[XX.ravel(),YY.ravel()])
labels=[]
m=Z.shape[0]
for i in range(m):
    g0=1.0/3*gaussian_2d(Z[i].T,c1m,cov1).item(0,0)
    g1=1.0/3*gaussian_2d(Z[i].T,c2m,cov2).item(0,0)
    g2=1.0/3*gaussian_2d(Z[i].T,c3m,cov3).item(0,0)
    if g0 ==max(g0,g1,g2):
        label=1
    elif g1 ==max(g0,g1,g2):
        label=2
    elif g2==max(g0,g1,g2):
        label=3
    labels.append(label)
import collections
counter=collections.Counter(labels)
print counter
labels=np.array(labels)
labels=labels.reshape(XX.shape)
cMap = c.ListedColormap(['m','c','y'])# first color label 1,second label 2 and so on
plt.pcolormesh(XX,YY,labels,cmap=cMap)
plt.scatter(c1[:,0],c1[:,1],color='g',label='setosa')
plt.scatter(c2[:,0],c2[:,1],color='r',label='versicolor')
plt.scatter(c3[:,0],c3[:,1],color='b',label='virginica')
plt.axis([XX.min(), XX.max(), YY.min(), YY.max()])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


