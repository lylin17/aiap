from src.decision_tree import DecisionTree
import numpy as np
import pandas as pd

class RandomForest:
    def __init__(self,n_trees=5,subsample_size=1,feature_proportion=1,max_depth=np.Inf,random_state=42):
        self.ensemble = [[],[]]
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.feature_proportion = feature_proportion
        self.max_depth = max_depth
        self.random_state = random_state
    
    def fit(self,x_train,y_train):
        np.random.seed(self.random_state)
        idx =  list(range(x_train.shape[0]))
        var_idx = list(range(x_train.shape[1]))
        size = int(x_train.shape[0]*self.subsample_size)
        mtry = int(x_train.shape[1]*self.feature_proportion)
        for n in range(self.n_trees):
            sub_idx = np.random.choice(idx,size)
            sub_var = np.random.choice(var_idx,mtry,replace=False)
            self.ensemble[0].append(sub_var)
    
            x_trainsub = x_train[sub_idx,:]
            x_trainsub = x_trainsub[:,sub_var]
            y_trainsub = y_train[sub_idx]
    
            dt = DecisionTree(max_depth = self.max_depth)
            dt.fit(x_trainsub,y_trainsub)
    
            self.ensemble[1].append(dt.tree)
            
    def predict(self,x_test):
        y_list = []
        for i in range(len(self.ensemble[1])):
            sub_var =  self.ensemble[0][i]
            x_testsub = x_test[:,sub_var]
            
            dt = DecisionTree(max_depth = self.max_depth)
            dt.tree =  self.ensemble[1][i]
            
            y_pred = dt.predict(x_testsub)
            y_list.append(y_pred)
        
        y_df = pd.DataFrame(y_list).T
        y_df['prob'] =y_df.sum(axis=1)/len(self.ensemble[1])
        y_pred = [round(x) for x in y_df['prob']]
        
        return y_pred
            
        

