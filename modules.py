#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 03:15:58 2022

@author: pavelvazquez
"""
def pubmed(query,email):
    import pandas as pd
    from Bio import Entrez
    from metapub import PubMedFetcher
    fetch = PubMedFetcher()
    pmids = fetch.pmids_for_query(query, retmax=2500000)
    df = pd.DataFrame(index=range(len(pmids)), columns=('pmid','date','abstract','title','type','doi'))
    k=0
    Entrez.email = email
    for pmid in pmids:
        
        handle = Entrez.esummary(db="pubmed", id=pmid)
        record = Entrez.read(handle)
        
        handle.close()
        date=record[0]["PubDate"]
        df.loc[k].pmid=pmid
        df.loc[k].date=date
        df.loc[k].abstract=fetch.article_by_pmid(pmid).abstract
        df.loc[k].type=fetch.article_by_pmid(pmid).publication_types
        df.loc[k].title=fetch.article_by_pmid(pmid).title
        df.loc[k].doi=fetch.article_by_pmid(pmid).doi
        print('Quering',k+1,'of',len(pmids))
        k=k+1
    return(df)

def wm2df(wm, feat_names):
    import pandas as pd
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

###PCA componentsevaluation
def test(variance,X,ncomponents):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    tsvd = TruncatedSVD(n_components=ncomponents)
    fig, (ax1, ax2) = plt.subplots(2,1)
    PC = range(1, tsvd.n_components+1)
    ax1.bar(PC, variance, color='red')
 
    
    ##Kmeans inertias evaluation
    from sklearn.cluster import KMeans 
    inertias = []
    for k in range(1,20):
        model = KMeans(n_clusters=k) 
        model.fit(X[:,:3])
        inertias.append(model.inertia_)
    ##Plot
    ax2.plot(range(1,20), inertias, '-p', color='red')
    ax1.set_title('PCA Components')
    ax2.set_title('Clusters')
    plt.show()
    return()

###Cluster by Kmeans
def cluster(ncluster,X,pmid,ns):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    model = KMeans(n_clusters=ncluster)
    model.fit(X[:,:2])
    labels = model.predict(X[:,:2])  
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],X[:,2], c=labels)
    #for i in range(0,len(pmid)):
     #   ax.text(X[i,0],X[i,1],X[i,2],pmid[i], size=8)
    if ns==1:
        plt.show()     
    else:
        plt.savefig('kcluster.png')

    
    return()

def cluster_labels(ncluster,X,pmid):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=ncluster)
    model.fit(X[:,:2])
    labels = model.predict(X[:,:2])  
    return(labels)

####Cosine similarity
def cosinef(X,Xs,ncluster,pmid,ns):
    from sklearn.cluster import AgglomerativeClustering
    import matplotlib.pyplot as plt
    model=AgglomerativeClustering(affinity='cosine',n_clusters=ncluster,linkage='complete').fit(X)
    labels=model.labels_
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Xs[:,0],Xs[:,1],Xs[:,2], c=labels)
    #for i in range(0,len(pmid)):
     #   ax.text(Xs[i,0],Xs[i,1],Xs[i,2],pmid[i], size=8)
    if ns=='show':
        plt.show()
    else:
        plt.savefig('acluster.png')
    return()