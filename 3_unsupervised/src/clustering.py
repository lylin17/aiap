import pandas as pd 
import numpy as np
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA

class Clustering:
    
    def __init__(self,data):
    """
    Initialize Clustering class
    Params:
        data: ndarry of shape (n_samples, n_features) for data with outlier removed and scaled
    Returns: None
    """   
        self.data = data
       
    def best_K(self,nrefs=10,maxClusters=20,random_state=42,seed =1):
    """
    Plot Graph of Gap Statistics Vs no. of clusters K to determine opitmal K
    Params:
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
        random_state: random_state of kmeans
        seed: np.random.seed for generating random datasets for reference
    Returns: gapdf
    """    
    
        def optimalK(data, nrefs=3, maxClusters=15):
            """
            Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
            Params:
                data: ndarry of shape (n_samples, n_features)
                nrefs: number of sample reference datasets to create
                maxClusters: Maximum number of clusters to test for
            Returns: (gaps, optimalK)
            """
            gaps = np.zeros((len(range(1, maxClusters+1)),))
            resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
            for gap_index, k in enumerate(range(1, maxClusters+1)):

                np.random.seed(seed)
                # Holder for reference dispersion results
                refDisps = np.zeros(nrefs)

                # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                for i in range(nrefs):

                    # Create new random reference set
                    randomReference = copy.deepcopy(self.data)
                    for j in range(data.shape[1]):
                        np.random.shuffle(randomReference[:,j])
                    # Fit to it
                    km = KMeans(k,random_state = random_state)
                    km.fit(randomReference)

                    refDisp = km.inertia_
                    refDisps[i] = refDisp

                # Fit cluster to original data and create dispersion
                km = KMeans(k,random_state = 42)
                km.fit(self.data)

                origDisp = km.inertia_

                # Calculate gap statistic
                gap = np.mean(np.log(refDisps)- np.log(origDisp))
                gap_sd = np.std(np.log(refDisps)- np.log(origDisp))

                # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap

                resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap,'gap_sd':gap_sd}, ignore_index=True)

            return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

        k, gapdf = optimalK(self.data, nrefs=nrefs, maxClusters=maxClusters)

        gapdf2 = gapdf.iloc[3:,:]
        f, ax = plt.subplots(figsize=(7, 5))
        plt.errorbar(gapdf2['clusterCount'],gapdf2['gap'], yerr=gapdf2['gap_sd'], capsize=4,color = 'orange',ecolor='blue')
        ax.set_title('Plot of Gap Statistics Vs K')
        x_axis = plt.xlabel('No. of Clusters, K')
        y_axis = plt.ylabel('Gap statistics')
        plt.show()     
        
        return gapdf

    def kmeans(self,unscaled_data,k,random_state=42):
    """
    Get feature mean and feature description for each cluster
    Params:
        unscaled_data: pandas dataframe of shape (n_samples, n_features) for data with outlier removed and not scaled
        k: number of clusters
        random_state: random_state of kmeans
    Returns: num_out, sum2
    """    
        kmean = KMeans(n_clusters=k, random_state=random_state)
        k_cluster = kmean.fit_predict(self.data)
        
        unscaled_data['Cluster'] = k_cluster
        num_out = unscaled_data.groupby('Cluster').mean()
        
        sum2 = pd.DataFrame(np.zeros(unscaled_data.groupby('Cluster').mean().shape),columns = unscaled_data.groupby('Cluster').mean().columns, index = unscaled_data.groupby('Cluster').mean().index)
        for i in range(unscaled_data.groupby('Cluster').mean().shape[0]):
            for j in range(unscaled_data.groupby('Cluster').mean().shape[1]):
                var = unscaled_data.columns[j] 
                if (ttest(unscaled_data.loc[unscaled_data['Cluster']==i,var],unscaled_data.loc[:,var],equal_var=False)[1] <0.001) and (unscaled_data.loc[unscaled_data['Cluster']==i,var].mean() > unscaled_data.loc[:,var].mean()):
                    sum2.iloc[i,j] = 'High'
                elif (ttest(unscaled_data.loc[unscaled_data['Cluster']==i,var],unscaled_data.loc[:,var],equal_var=False)[1] <0.001) and (unscaled_data.loc[unscaled_data['Cluster']==i,var].mean() < unscaled_data.loc[:,var].mean()):
                    sum2.iloc[i,j] = 'Low'
                else:
                    sum2.iloc[i,j] = 'Average'
        
        return num_out,sum2
    
    def visualize(self,k,method='tsne',random_state=42):
    """
    Visualize clusters with data projected to 2-D space
    Params:
        k: number of clusters
        method: Method to project data to 2-D space. Either 'tsne' or 'pca'
        random_state: random_state of kmeans
    Returns: None
    """    
        #Best Clustering by K means
        kmean = KMeans(n_clusters=k, random_state=random_state)
        k_cluster = kmean.fit_predict(self.data)
        color_labels = np.unique(k_cluster)

        # List of RGB triplets
        rgb_values = sns.color_palette("Set2", len(color_labels))

        # Map label to RGB
        color_map = dict(zip(color_labels, rgb_values))
        
        if method == 'tsne':
        #t-sne transformation
            X_embedded = TSNE(n_components=2,random_state=1).fit_transform(self.data)

            # Finally use the mapped values
            f, ax = plt.subplots(figsize=(8, 8))
            plt.scatter(X_embedded[:,0],X_embedded[:,1], c=pd.Series(k_cluster).map(color_map))
            plt.show()
            
        if method =='pca':
            pca = PCA(n_components=2, random_state=random_state)
            pca_x = pca.fit_transform(self.data)

            # Finally use the mapped values
            f, ax = plt.subplots(figsize=(8, 8))
            plt.scatter(pca_x[:,0],pca_x[:,1], c=pd.Series(k_cluster).map(color_map))
            plt.show()
    
        