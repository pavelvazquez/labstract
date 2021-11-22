#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pavelvazquez
"""
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#import igraph
#import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pandas as pd

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help= 'path to input file', default='./onchip_summary.csv')
parser.add_argument('-o','--output', help= 'path to output file of abstracts', default='./data_summary.json')
parser.add_argument('-t','--type', help= 'vectorizer type. Use: count or tfidf', default='tfidf')
parser.add_argument('-n','--components', help= 'Number of components to use', default=5)
parser.add_argument('-c','--cluster', help= 'Number of clusters to use', default=3)
parser.add_argument('-m','--mode', help= 'Test, kcluster (K-Means Clustering) or acluster(Agglomerative Clustering *)')
parser.add_argument('-a','--affinity', help= 'To be used allong with Agglomerative Clustering. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, default is cosine', default='cosine')
parser.add_argument('-v','--verbose', help="Optional. Plot 3d")
args = parser.parse_args()



def wm2df(wm, feat_names):
    
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

###PCA componentsevaluation
def test(variance,X):
    fig, (ax1, ax2) = plt.subplots(2,1)
    PC = range(1, tsvd.n_components+1)
    ax1.bar(PC, variance, color='red')
 
    
    ##Kmeans inertias evaluation
    from sklearn.cluster import KMeans 
    inertias = []
    for k in range(1,10):
        model = KMeans(n_clusters=k) 
        model.fit(X[:,:3])
        inertias.append(model.inertia_)
    ##Plot
    ax2.plot(range(1,10), inertias, '-p', color='red')
    ax1.set_title('PCA Components')
    ax2.set_title('Clusters')
    plt.show()
    return()

###Cluster by Kmeans
def cluster(ncluster,X):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=ncluster)
    model.fit(X[:,:2])
    labels = model.predict(X[:,:2])  

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],X[:,2], c=labels)
    plt.show()
    return()

####Cosine similarity
def cosinef(X,Xs,ncluster):
    from sklearn.cluster import AgglomerativeClustering
    model=AgglomerativeClustering(affinity='cosine',n_clusters=ncluster,linkage='complete').fit(X)
    labels=model.labels_
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Xs[:,0],Xs[:,1],Xs[:,2], c=labels)
    plt.show()
    return()


##Variable parser
##Load 
##put an if so I can also download by pmid
filename=args.input
output=args.output
vtype=args.type
mode=args.mode
ncomponents=int(args.components)
ncluster=int(args.cluster)


#File read. TODO: other formats and search query
df = pd.read_csv (filename)

####Drop the papers with no DOI
df.doi.dropna(axis='rows')





#### Abstract counter #########
if vtype=='count':
    countvectorizer = CountVectorizer(stop_words='english')
elif vtype=='tfidf':
    countvectorizer = TfidfVectorizer(stop_words='english')
else:
    print('Invalid vectorizer type')
    exit()
data=df.abstract

vector = countvectorizer.fit_transform(data)
vnames=countvectorizer.get_feature_names()
X=wm2df(vector,vnames)



#Dimensionality reduction by Truncated Singular Value Decomposition
tsvd = TruncatedSVD(n_components=ncomponents)
X_sparse = csr_matrix(X)
X_sparse_tsvd = tsvd.fit(X_sparse).transform(X_sparse)


#### 2d Plots #############
#sns.scatterplot(
 #   x=X_sparse_tsvd[:,1], y=X_sparse_tsvd[:,2],
  #  alpha=0.3
#)

#########################

if mode=='test':
    ###PCA componentsevaluation
    test(tsvd.explained_variance_ratio_,X_sparse_tsvd)


if mode=='kcluster':
    #Cluster by Kmeans
    cluster(ncluster,X_sparse_tsvd)

 
#Calculate the distance matrix
if mode=="acluster":
    cosinef(X,X_sparse_tsvd,ncluster)




if args.verbose:
    print('Largest words \n')
    largest=X.sum(axis=0).nlargest(40)
    print(largest)
    print('Variance',tsvd.explained_variance_ratio_.sum())

#Plot the conectivity graph 
#g = igraph.Graph.Weighted_Adjacency(G.tolist())

#out=igraph.plot(g, edge_width=0.1,edge_arrow_size=0.001,edge_color=(200, 200, 200), vertex_size=12, layout='kk')
#out.save('./results/%s.png' % month)
