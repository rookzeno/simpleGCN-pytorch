import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias= Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias",None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv=1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        self.weight2.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self,input,adj):
        out = []
        e = torch.eye(input.shape[1],input.shape[1])
        for i in range(len(input)):
            support = torch.mm(input[i].view(input.shape[1],-1),self.weight)
            output = torch.spmm(adj[i],support) + torch.spmm(e,torch.mm(input[i].view(input.shape[1],-1),self.weight2))
            out.append(output)
        out = torch.stack(out,dim=0)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Readout(Module):

    def __init__(self,in_features):
        super(Readout,self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features,in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv=1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
    
    def forward(self,input):
        input = torch.sum(input,2)
        return torch.mm(input,self.weight)
