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

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pandas as pd

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help= 'path to input file', default='data_input.csv')
parser.add_argument('-o','--output', help= 'path to output file of abstracts', default='./data_output.csv')
parser.add_argument('-q','--query', help= 'Pubmed query as string')
parser.add_argument('-e','--email', help="Optional. Email for entrez")
parser.add_argument('-t','--type', help= 'vectorizer type. Use: count or tfidf', default='tfidf')
parser.add_argument('-n','--components', help= 'Number of components to use', default=5)
parser.add_argument('-c','--cluster', help= 'Number of clusters to use', default=3)
parser.add_argument('-m','--mode', help= 'Test, kcluster (K-Means Clustering) or acluster(Agglomerative Clustering *)')
parser.add_argument('-a','--affinity', help= 'To be used allong with Agglomerative Clustering. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, default is cosine', default='cosine')
parser.add_argument('-v','--verbose', help="Optional. Plot 3d")
args = parser.parse_args()

##Variable parser
##Load 
##put an if so I can also download by pmid
filename=args.input
output=args.output
vtype=args.type
mode=args.mode
ncomponents=int(args.components)
ncluster=int(args.cluster)

from modules import pubmed,wm2df,test,cluster,cosinef






#File read. TODO: other formats and search query
if args.query:
    df=pubmed(args.query,args.email)
    df.to_csv(args.output,index=False)
else:
    print('reading file', filename)    
    df = pd.read_csv (filename)

####Drop the papers with no DOI
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)




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
    test(tsvd.explained_variance_ratio_,X_sparse_tsvd,ncomponents)


if mode=='kcluster':
    #Cluster by Kmeans
    cluster(ncluster,X_sparse_tsvd,df.pmid)
    
 
#Calculate the distance matrix
if mode=="acluster":
    cosinef(X,X_sparse_tsvd,ncluster,df.pmid)




if args.verbose:
    print('Largest words \n')
    largest=X.sum(axis=0).nlargest(40)
    print(largest)
    print('Variance',tsvd.explained_variance_ratio_.sum())

#Plot the conectivity graph 
#g = igraph.Graph.Weighted_Adjacency(G.tolist())

#out=igraph.plot(g, edge_width=0.1,edge_arrow_size=0.001,edge_color=(200, 200, 200), vertex_size=12, layout='kk')
#out.save('./results/%s.png' % month)
