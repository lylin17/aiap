import numpy as np
from keras.utils import to_categorical
np.random.seed(1)

def relu(x):
    return np.maximum(0,x)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

class MLPTwoLayers:
    def __init__(self, input_size,hidden_size,output_size):
        self.w1 = np.random.normal(0,np.sqrt(2/input_size),size = (input_size*hidden_size)).reshape(input_size,hidden_size) #he_normal initialization
        self.b1 = np.zeros((1,hidden_size))
        
        self.w2 = np.random.normal(0,np.sqrt(2/hidden_size),size = (hidden_size*output_size)).reshape(hidden_size,output_size) #he_normal initialization
        self.b2 = np.zeros((1,output_size))
        
        self.w1_grad = np.zeros(self.w1.shape)
        self.b1_grad = np.zeros(self.b1.shape)
        
        self.w2_grad = np.zeros(self.w2.shape)
        self.b2_grad = np.zeros(self.b2.shape)
        
        self.inputs1 = None
        self.inputs2 = None
        self.targets = None
        self.prob = None
        
        
    def forward(self, inputs):      
        inputs = inputs.reshape(1,self.w1.shape[0])
        self.inputs1 = inputs
        
        #hidden layer
        inputs2 = np.zeros((1,self.w1.shape[1]))
        inputs2 = np.dot(inputs,self.w1)+self.b1
        inputs2 = relu(inputs2)
        
        self.inputs2 = inputs2        
      
        #output layer
        outputs = np.dot(inputs2,self.w2)+self.b2
        unscaled_probs = np.exp(outputs)
        probs_sum = np.sum(unscaled_probs)+0.000000001
        probs = unscaled_probs/probs_sum
        
        self.prob=probs
        return probs[0]
    
    def loss(self,preds,actual):
        self.target = to_categorical(actual,10)
        return -np.log(preds[actual]) 
    
    def backward(self, loss):
        in_grads = self.prob-self.target        
        
        #output layer
        self.w2_grad = np.dot(self.inputs2.T,in_grads)
        self.b2_grad = in_grads
        out_grads = np.dot(in_grads,self.w2.T)      
        out_grads = (self.inputs2 >=0 ) * out_grads
        
        #hiddden layer
        self.w1_grad = np.dot(self.inputs1.T,out_grads)
        self.b1_grad = out_grads
        
        #Update weights
        self.w1 = self.w1-(10e-3*self.w1_grad)
        self.w2 = self.w2-(10e-3*self.w2_grad)
        self.b1 = self.b1-(10e-3*self.b1_grad)
        self.b2 = self.b2-(10e-3*self.b2_grad)     
