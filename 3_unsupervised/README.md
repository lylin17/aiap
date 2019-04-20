# aiap_2.0_unsuperv
Materials for AIAP 2.0 Week of unsupervised learning

This projects clusters ecommerce transactions in the form of invoices from a UK-based online retails store. 
The transactions recorded occured between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers

The dataset is available on Kaggle (https://www.kaggle.com/carrie1/ecommerce-data/home) or the UCI Machine Learning Repository. (/data/raw)

Objective
1. Clustering the data at customers level to discover customer segments.
2. Suggesting simple reccomendation tailored to each customer segment using Association Rule Mining

Audience
Business teams looking to design customized marketing/promotion campaigns for customer segments 

Getting Started

Data Exploration, Model Selection 

Customer Segments 

Preprocessing 
Preprocessing was performed using the Cleaning class in the preprocessing module 
A pandas DataFrame of unprocessed data was taken as input and a pandas DataFrame of the preprocessed data was returned.

1.Cleaning
a. Remove NAs
b. Drop Duplicates
c. Remove non-positive values 
d. Remove non-integer values 
e. Change feature data type

2. Feature Engineering - create useful numerical features

Clustering
Clustering by hierarchical clustering, K means clustering and Gaussian Mixture Models(hard cluster) were used.
To determine the optimal number of clusters, 
1. An elbow plot of Within-sum-of-squares Vs Number of cluster was plotted. The best number of clusters is at the 'kink' in the plot.
2. A plot of Gap statistics Vs Number of cluster was plotted. The best number of clusters is the local maximum in the plot with the smallest number of clusters.

Cluster Visualization
For hierarchical clustering, a truncated dendrogram was plotted to visualize the clusters
For K means and Gaussian mixture model, the data was projected to a 2D plane using t-sne(and PCA) to visualize if the clusters were well-seperated

Association Rule Mining (Part 1 Notebook II.ipynb)

The data was formatted to a binary InvoiceNo (transactions), StockCode (Unique Items) matrix. A minimum support of 1% (item sets appearing in at least 1% of the invoices) were used

For each customer segments, 
1. Isolate the data to rows corresponding to customers in the customer segment
2. Perform association rule mining to find the top 5 association rule according to lift score
3. Customers belonging to this segement would likely be interested to purchase the consequent itemset if they had already bought the antecedent itemset.

Final deliverable (preprocessing.py and clustering.py)
Preprocessing
1. Cleaning
2. FeatEng

Clustering using K means
1. Find opitmal number of clusters (Clustering.best_K)
2. Get description of customer segments (Clustering.kmeans)
3. Visualize cluster (Clustering.visualize)

Association Rule Mining




