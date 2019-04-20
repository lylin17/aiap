import numpy as np

class DecisionTree:
    def __init__(self,max_depth=np.Inf):
        self.tree = None
        self.max_depth = max_depth
        
    def gini(self,br_arr1, br_arr2):
        arrays = [br_arr1, br_arr2]
        n_instances = float(sum([len(array) for array in arrays]))
        # sum weighted Gini impurity for each branch
        gini_score= 0.0
        for array in arrays:
            size = float(len(array))
            if size == 0:continue # avoid divide by zero
            
            count_1=0
            count_0=0 
            for i in array:
                if i ==1: count_1+=1
                if i ==0: count_0+=1
            p1 = count_1/size
            p0 = count_0/size
            branch_score = 1 - (p1*p1) - (p0*p0)
            
            gini_score += branch_score * (size / n_instances)
            
        return gini_score
    
    def get_split(self,x_train,y_train):
        best_gini = 1
        best_var = 0
        best_val = 0
        best_group = None
    
        for i in range(x_train.shape[1]):
            val = list(set(x_train[:,i]))
            for value in val:
                left, right = list(), list()
                for j in range(x_train.shape[0]):
                    if x_train[j,i] < value: left.append(j)
                    else :right.append(j)
    
                if val == [0,1] and value == 0: continue
                left_targets = [y_train[k,] for k in left]  
                right_targets = [y_train[k,] for k in right]  
    
                if self.gini(left_targets,right_targets)<best_gini:
                    best_gini = self.gini(left_targets,right_targets)
                    best_var =i
                    best_val =value
                    best_group = [left,right]
        
        group_L, group_R= list(), list()
        
        if len(best_group[0]) !=0:            
            group_L = [x_train[best_group[0]],y_train[best_group[0]]]
            
        if len(best_group[1]) !=0: 
            group_R = [x_train[best_group[1]],y_train[best_group[1]]]
            
        group = [group_L,group_R]
        
        return {'index':best_var,'value':best_val,'groups':group}
    
    # Create a terminal node value
    def to_terminal(self,group):
        outcomes = [row for row in group]
        return max(set(outcomes), key=outcomes.count)
     
    # Create child splits for a node or make terminal
    def split(self,node,min_size, depth,max_depth=np.Inf):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if len(left)==0:
            node['left'] = node['right'] = self.to_terminal(right[1])
            return
        if len(right)==0:
            node['left'] = node['right'] = self.to_terminal(left[1])
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left[1]), self.to_terminal(right[1])
            return
        # process left child
        if len(left[0]) <= min_size:
            node['left'] = self.to_terminal(left[1])
        else:
            node['left'] = self.get_split(left[0],left[1])
            self.split(node['left'], min_size, depth+1,max_depth = max_depth)
        # process right child
        if len(right[0]) <= min_size:
            node['right'] = self.to_terminal(right[1])
        else:
            node['right'] = self.get_split(right[0],right[1])
            self.split(node['right'], min_size, depth+1,max_depth = max_depth)
    
    def fit(self,x_train,y_train):
        #initialize first split
        self.tree = self.get_split(x_train,y_train)
        #recursive splitting
        self.split(self.tree,1,1,max_depth=self.max_depth)
        
    # Make a prediction with a decision tree
    def predict(self,x_test):
        prediction =[]
        def predict_row(tree, row):
            if row[tree['index']] < tree['value']:
                if isinstance(tree['left'], dict):
                    return predict_row(tree['left'], row)
                else:
                    return tree['left']
            else:
                if isinstance(tree['right'], dict):
                    return predict_row(tree['right'], row)
                else:
                    return tree['right']
        
        for i in range(len(x_test)):
            prediction.append(predict_row(self.tree,x_test[i]))
        return prediction
