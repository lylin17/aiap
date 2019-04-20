import torch

class RNN:
    def __init__(self, in_features, units):       
        self.kernel =  torch.distributions.normal.Normal(0,np.sqrt(2/in_features)).sample((in_features,units)).cuda()
        self.recurrent_kernel =  torch.distributions.normal.Normal(0,np.sqrt(2/units)).sample((units,units)).cuda()
        self.bias = torch.zeros(units).cuda()

    def forward(self, inputs):               
        n,t = inputs.shape[0],inputs.shape[1]                
        k = self.recurrent_kernel.shape[0]
        
        h = torch.zeros((n,t+1,k)).cuda()
        outputs = torch.zeros((n,t,k)).cuda()               
                        
        for i in range(t):
                outputs[:,i,:] = torch.tanh(torch.matmul(inputs[:,i,:].clone(),self.kernel)+torch.matmul(h[:,i,:].clone(),self.recurrent_kernel)+self.bias.reshape(1,-1))
                h[:,i+1,:] = outputs[:,i,:]
        return outputs