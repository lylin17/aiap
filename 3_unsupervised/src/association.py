import pandas as pd 
import numpy as np
import copy
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Association_Rules:
    
    def __init__(self,cleaned_data,unscaled_data):
    """
    Initialize Association_Rules class
    Params:
        cleaned_data: pandas dataframe of shape (n_samples, n_features) for cleaned data without outlier removed and unscaled
        unscaled_data: pandas dataframe of shape (n_samples, n_features) for cleaned data with outlier removed and unscaled
    Returns: None
    """     
        self.clean = cleaned_data
       
        self.unscaled = unscaled_data
        scaler = StandardScaler()
        self.scaled = scaler.fit_transform(self.unscaled.values)       
       
    def rules(self,k,min_support,metric='confidence', min_threshold=0.8,sort = 'lift',random_state=42):
    """
    Get association rules for each kmeans cluster in a list
    Params:
        k: number of clusters
        min_support: get itemsets with specified min_support using apriori algorithm
        metric:metric to set threshold to find association rules
        min_threshold: min threshold of metric to find association rules
        sort: metric to sort association rules found
        random_state: random_state of kmeans
    Returns: rule_list
    """   
        rule_list = []
        
        kmean = KMeans(n_clusters=k, random_state=random_state)
        self.unscaled['Cluster'] = kmean.fit_predict(self.scaled)
        
        for n  in range(k):
            cluster0 = self.unscaled.loc[self.unscaled['Cluster'] == n,:]
            print('Number of rows in cluster '+str(n)+' :',cluster0.shape[0])

            cluster0_df = self.clean.loc[self.clean['CustomerID'].isin(cluster0.index.tolist()),:]
            cluster0_df = cluster0_df.iloc[:,:2]

            #Convert data to Invoice-StockCode matrix
            enc = OneHotEncoder()
            cluster0_df2 = pd.DataFrame(enc.fit_transform(cluster0_df['StockCode'].values.reshape(-1, 1)).toarray(),columns = [name.split('_')[1] for name in enc.get_feature_names()])
            cluster0_df2.index = cluster0_df['InvoiceNo']

            cluster0_df3 = cluster0_df2.groupby('InvoiceNo').sum()
            cluster0_df3 = cluster0_df3.clip(0,1)

            #Association Rule Mining
            supp0 = apriori(cluster0_df3, min_support=min_support, use_colnames=True,n_jobs=-1)
            rconf0 = association_rules(supp0,metric=metric, min_threshold=min_threshold, support_only=False)

            print('Number of rules above threshold:',rconf0.shape[0])
            rule = rconf0.sort_values(sort,ascending=False).head()
            print('\n')
            rule_list.append(rule)
     
        return rule_list
            