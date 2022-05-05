#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:15:51 2022

@author: pavelvazquez

"""




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD,PCA
#import numpy as np
from modules import cluster_labels,test
#import igraph
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,pairwise_distances
import glob, os

def wm2df(wm, feat_names):

    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)
def slope(x2, x1):
    m = (x2-x1)
    return m

#for file in glob.glob('*.tsv'):
path='./'
#file_path=path+file
file_path='Canarypox.tsv'
print(file_path)
df=pd.read_csv(file_path, sep='\t')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)



data=df.abstract


countvectorizer = CountVectorizer(stop_words='english')
vector = countvectorizer.fit_transform(data)
vnames=countvectorizer.get_feature_names()
X=wm2df(vector,vnames)


#M=pairwise_distances(X,metric='cosine')



#Dimensionality reduction by Truncated Singular Value Decomposition used only to Plot


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
score=[]

pca = PCA().fit(X)
pca_n=list(pca.explained_variance_ratio_)
for variance in pca_n:
     m=slope(variance,pca_n[pca_n.index(variance)+1])
     if m<0.0001:
         break


ncomponents=pca_n.index(variance)
tsvd = TruncatedSVD(n_components=ncomponents)
X_sparse = csr_matrix(X)
X_tsvd = tsvd.fit(X_sparse).transform(X_sparse)

range_n_clusters = list (range(2,20))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(X_tsvd)
    centers = clusterer.cluster_centers_
    score.append(silhouette_score(X_tsvd, preds))
    #clusters.append(n_clusters)
    #print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

ncluster=range_n_clusters[score.index(max(score))]



#G=np.dot(X,X.T)
#labels=cluster_labels(ncluster,X_tsvd,df.pmid)
#for k in range (0,len(G)):   
 #   G[k][k]=0

#from sklearn.cluster import AgglomerativeClustering
#cluster=AgglomerativeClustering(affinity='precomputed',n_clusters=3,linkage='complete').fit(M)
#cluster.fit(M)
#labels=cluster.labels_

labels=cluster_labels(ncluster,X_tsvd,df.pmid)
#test(tsvd.explained_variance_ratio_,X_tsvd,ncomponents)

import matplotlib.pyplot as plt
plt.scatter(x=X_tsvd[:,1], y=X_tsvd[:,2],c=labels)
plt.axis('off')
plt.show()
#plt.savefig(file_path+'.png')


#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X_tsvd[:,0],X_tsvd[:,1],X_tsvd[:,2], c=labels)
#for i in range(0,len(pmid)):
 #   ax.text(Xs[i,0],Xs[i,1],Xs[i,2],pmid[i], size=8)

#plt.show()









